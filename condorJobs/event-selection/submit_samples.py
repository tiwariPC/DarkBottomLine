#!/usr/bin/env python3
"""
Submit condor jobs for event selection (NanoAOD -> EVENTSELECTION.root).

One cluster per samplelist .txt in data/samplelist/<year>/, one job per slice of
BATCH ROOT files. Each job runs its files sequentially (runanalysis.sh) and
xrdcp's every output to:
    <OUTDIR>/<year>/eventsel/<sample>/<sample>_I<fileIdx>C<ClusterId>P<ProcId>_EVENTSELECTION.root

Files whose output is already on EOS are never submitted, so rerunning the same
command after a round finishes resubmits only what is missing. Samples that still
have jobs in the queue are skipped, so nothing runs twice.

Batch size per sample is either fixed (--batch) or derived from a pilot timing
file (--timing, written by pilot_report.py) to give jobs of ~--target-minutes.

Examples (run from anywhere):
    # Dry run: show clusters/jobs, write slice files, submit nothing
    python3 condorJobs/event-selection/submit_samples.py --year 2024 --dry-run

    # Pilot: first file of a few samples, at 2 and 8 cores
    python3 condorJobs/event-selection/submit_samples.py --year 2024 \\
        --samples 'WtoLNu-2Jets_Bin-2J-PTLNu-40to100*' 'Muon0-Run2024G*' \\
        --files 0:1 --batch 1 --cpus 2 --memory 8000

    # Production with pilot timing
    python3 condorJobs/event-selection/submit_samples.py --year 2024 \\
        --timing condorJobs/event-selection/timing_2024.json --target-minutes 90
"""

import argparse
import fnmatch
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

CONDOR_DIR = Path(__file__).resolve().parent
REPO_DIR = CONDOR_DIR.parent.parent
DEFAULT_OUTDIR = "/eos/cms/store/group/phys_susy/sus-23-008/run3_bbMET/20260917-849194d_10102026"

# Schedd limits on lxplus (condor_config_val -schedd, 2026-10-10).
MAX_JOBS_PER_SUBMISSION = 20000
MAX_JOBS_PER_OWNER = 100000


def read_files(sample_txt: Path):
    """ROOT entries of a samplelist; same filter as runanalysis.sh."""
    with open(sample_txt) as f:
        return [s for s in (line.strip() for line in f) if s and not s.startswith("#")]


# Collision data primary datasets this analysis uses; any other data PD (e.g. Muon0/1)
# is skipped. Data samples are named like JetMET0-Run2024G-MINIv6NANOv15-v2.
DATA_PRIMARY_DATASETS = ("EGamma", "JetMET")


def is_data_sample(sample_name: str) -> bool:
    return re.search(r"-Run20\d{2}", sample_name) is not None


def is_signal_sample(sample_name: str) -> bool:
    """Same rule as darkbottomline.cli._is_signal_sample (fastsim 2HDMa signal)."""
    return "2hdma" in sample_name.lower()


def sample_kind(sample_name: str) -> str:
    """DATA, SIG or MC; SKIP for data primary datasets not in DATA_PRIMARY_DATASETS."""
    if is_data_sample(sample_name):
        pd = re.sub(r"\d*$", "", sample_name.split("-Run")[0])
        return "DATA" if pd in DATA_PRIMARY_DATASETS else "SKIP"
    return "SIG" if is_signal_sample(sample_name) else "MC"


# <sample>_I<fileIdx>C<ClusterId>P<ProcId>_EVENTSELECTION.root, plus the older
# <sample>_<fileIdx>[_<ClusterId>-<ProcId>]_EVENTSELECTION.root so those are not redone.
OUTPUT_RES = (
    re.compile(r"^(.*)_I(\d+)C\d+P\d+_EVENTSELECTION\.root$"),
    re.compile(r"^(.*)_(\d+)(?:_\d+-\d+)?_EVENTSELECTION\.root$"),
)


def parse_output_name(name: str):
    """(sample, fileIdx) of an EVENTSELECTION output name, or None."""
    for rx in OUTPUT_RES:
        m = rx.match(name)
        if m:
            return m.group(1), int(m.group(2))
    return None


def existing_outputs(outdir_year: Path):
    """(sample, fileIdx) pairs that already have an output on EOS.

    Outputs live in <outdir_year>/<sample>/; files directly in <outdir_year> (the
    earlier flat layout) are counted too.
    """
    try:
        entries = list(os.scandir(outdir_year))
    except FileNotFoundError:
        return set()
    names = []
    for e in entries:
        if e.is_dir():
            names += os.listdir(e.path)
        else:
            names.append(e.name)
    return {key for key in map(parse_output_name, names) if key}


def queued_batch_names():
    """JobBatchName of every job this user has in the queue."""
    try:
        out = subprocess.run(["condor_q", "-af", "JobBatchName"], capture_output=True,
                             text=True, check=True, timeout=120).stdout
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        sys.exit(f"✗ Could not query condor_q ({e}); pass --allow-duplicates to skip this check")
    return {line.strip() for line in out.splitlines() if line.strip()}


def queued_job_count():
    try:
        out = subprocess.run(["condor_q", "-totals"], capture_output=True, text=True,
                             check=True, timeout=120).stdout
    except (subprocess.SubprocessError, FileNotFoundError):
        return 0
    m = re.search(r"Total for \S+: (\d+) jobs", out)
    return int(m.group(1)) if m else 0


def batch_for(sample, is_data, timing, args):
    """Files per job: from pilot timing when available, else --batch."""
    if not timing:
        return args.batch
    entry = timing.get("samples", {}).get(sample) or timing.get("default", {}).get("data" if is_data else "mc")
    if not entry:
        return args.batch
    batch = round(args.target_minutes * 60 / max(entry["sec_per_file"], 1))
    return max(1, min(args.max_batch, batch))


def parse_file_range(spec, n):
    if not spec:
        return range(n)
    start, _, end = spec.partition(":")
    return range(int(start or 0), min(int(end) if end else n, n))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--year", required=True, choices=["2022", "2022EE", "2023", "2024"])
    p.add_argument("--outdir", default=DEFAULT_OUTDIR, help="EOS base dir; outputs go to <outdir>/<year>/eventsel/<sample>/")
    p.add_argument("--input-dir", type=Path, default=None, help="Samplelist dir (default: data/samplelist/<year>/)")
    p.add_argument("--config", default=None, help="Config YAML, repo-relative (default: configs/<year>.yaml)")
    p.add_argument("--samples", nargs="+", default=None, metavar="GLOB", help="Only samples matching these globs")
    p.add_argument("--exclude", nargs="+", default=[], metavar="GLOB", help="Skip samples matching these globs")
    p.add_argument("--files", default=None, metavar="START:END", help="Only file indices [START, END) of each sample")
    p.add_argument("--batch", type=int, default=10, help="Files per job when no timing is available (default: 10)")
    p.add_argument("--timing", type=Path, default=None, help="Pilot timing JSON from pilot_report.py")
    p.add_argument("--target-minutes", type=float, default=90, help="Target job length with --timing (default: 90)")
    p.add_argument("--max-batch", type=int, default=50, help="Upper bound on files per job (default: 50)")
    p.add_argument("--cpus", type=int, default=2, help="request_cpus = executor workers (default: 2)")
    p.add_argument("--memory", type=int, default=None, help="request_memory in MB (default: 2000 per cpu)")
    p.add_argument("--flavour", default="workday",
                   choices=["espresso", "microcentury", "longlunch", "workday", "tomorrow", "testmatch", "nextweek"])
    p.add_argument("--chunk-size", default="50000", help="Executor chunk size, integer or 'auto' (default: 50000)")
    p.add_argument("--max-events", default="", help="Max events per file (testing only)")
    p.add_argument("--proxy", default=None, help="x509 proxy (default: ~/private/x509up_u<uid> on AFS)")
    p.add_argument("--no-skip-existing", action="store_true", help="Submit files even if their output is on EOS")
    p.add_argument("--allow-duplicates", action="store_true", help="Submit samples that still have jobs queued")
    p.add_argument("--dry-run", action="store_true", help="Write slice files and print the plan, submit nothing")
    args = p.parse_args()

    if args.chunk_size != "auto" and not args.chunk_size.isdigit():
        sys.exit(f"✗ --chunk-size must be an integer or 'auto' (got {args.chunk_size})")
    memory = args.memory or 2000 * args.cpus
    config = args.config or f"configs/{args.year}.yaml"
    if not (REPO_DIR / config).is_file():
        sys.exit(f"✗ Config not found: {REPO_DIR / config}")
    input_dir = (args.input_dir if args.input_dir and args.input_dir.is_absolute()
                 else REPO_DIR / (args.input_dir or Path("data/samplelist") / args.year)).resolve()
    if not input_dir.is_dir():
        sys.exit(f"✗ Samplelist dir not found: {input_dir}")

    user = os.environ.get("USER") or os.getlogin()
    proxy = Path(args.proxy or f"/afs/cern.ch/user/{user[0]}/{user}/private/x509up_u{os.getuid()}")
    if not proxy.is_file():
        sys.exit(f"✗ Proxy not found: {proxy}\n  voms-proxy-init --voms cms --valid 192:00 && cp /tmp/x509up_u{os.getuid()} {proxy.parent}/")
    left = subprocess.run(["voms-proxy-info", "-file", str(proxy), "-timeleft"], capture_output=True, text=True)
    hours_left = int(left.stdout.strip() or 0) / 3600 if left.returncode == 0 else 0
    if hours_left < 48:
        msg = f"proxy {proxy} has {hours_left:.0f}h left"
        if args.dry_run:
            print(f"⚠  {msg} (would refuse to submit with < 48h)")
        else:
            sys.exit(f"✗ {msg}; renew it (voms-proxy-init --voms cms --valid 192:00) and copy it to {proxy.parent}/")

    timing = json.loads(args.timing.read_text()) if args.timing else None
    outdir_year = Path(args.outdir) / args.year / "eventsel"
    done = set() if args.no_skip_existing else existing_outputs(outdir_year)
    queued = set() if (args.dry_run or args.allow_duplicates) else queued_batch_names()

    log_dir = CONDOR_DIR / "logs" / args.year
    slice_dir = CONDOR_DIR / "submitfiles" / args.year
    for d in (log_dir / "output", log_dir / "error", slice_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Build the per-sample plan.
    plan = []
    n_selected = 0
    skipped_pds = []
    for txt in sorted(input_dir.glob("*.txt")):
        sample = txt.stem
        if args.samples and not any(fnmatch.fnmatch(sample, g) for g in args.samples):
            continue
        if any(fnmatch.fnmatch(sample, g) for g in args.exclude):
            continue
        kind = sample_kind(sample)
        if kind == "SKIP":
            skipped_pds.append(sample)
            continue
        n_selected += 1
        batch_name = f"{sample}_{args.year}"
        files = read_files(txt)
        todo = [i for i in parse_file_range(args.files, len(files)) if (sample, i) not in done]
        if batch_name in queued:
            print(f"⚠  {sample}: jobs still queued, skipping (rerun once they finish)")
            continue
        if not todo:
            continue
        data = kind == "DATA"
        batch = batch_for(sample, data, timing, args)
        slices = [todo[i:i + batch] for i in range(0, len(todo), batch)]
        plan.append(dict(txt=txt, sample=sample, batch_name=batch_name, data=data, kind=kind,
                         n_files=len(files), todo=todo, batch=batch, slices=slices))

    if skipped_pds:
        print(f"⚠  Skipping {len(skipped_pds)} data samplelist(s) not in DATA_PRIMARY_DATASETS "
              f"{DATA_PRIMARY_DATASETS}: {', '.join(skipped_pds)}")
    if not n_selected:
        sys.exit(f"✗ No samplelist in {input_dir} matches --samples {args.samples} / --exclude {args.exclude}")
    if not plan:
        print("Nothing to submit: every selected file already has an output on EOS (or is queued).")
        return

    # Largest clusters first so their tails start earliest.
    plan.sort(key=lambda s: len(s["todo"]), reverse=True)
    total_jobs = sum(len(s["slices"]) for s in plan)
    biggest = max(len(s["slices"]) for s in plan)
    already_queued = 0 if args.dry_run else queued_job_count()
    if biggest > MAX_JOBS_PER_SUBMISSION:
        sys.exit(f"✗ A cluster would have {biggest} jobs > MAX_JOBS_PER_SUBMISSION={MAX_JOBS_PER_SUBMISSION}; raise --batch")
    if already_queued + total_jobs > MAX_JOBS_PER_OWNER:
        sys.exit(f"✗ {already_queued} queued + {total_jobs} new > MAX_JOBS_PER_OWNER={MAX_JOBS_PER_OWNER}; raise --batch")

    print("=" * 72)
    print(f"Event selection {args.year}: {len(plan)} clusters, {total_jobs} jobs, "
          f"{sum(len(s['todo']) for s in plan)} files ({len(done)} outputs already on EOS)")
    print(f"Output: {outdir_year}/")
    print(f"Per job: {args.cpus} cpus, {memory} MB, flavour {args.flavour}, chunk size {args.chunk_size}")
    print("=" * 72)
    print(f"{'sample':<75} {'type':<4} {'files':>6} {'batch':>5} {'jobs':>5}")
    for s in plan:
        print(f"{s['sample'][:75]:<75} {s['kind']:<4} {len(s['todo']):>6} {s['batch']:>5} {len(s['slices']):>5}")

    submitted = 0
    for s in plan:
        slicefile = slice_dir / f"{s['sample']}.slices"
        slicefile.write_text("".join(f"{i} {':'.join(map(str, sl))}\n" for i, sl in enumerate(s["slices"])))
        if args.dry_run:
            continue
        appends = {
            "EXEC": CONDOR_DIR / "runanalysis.sh",
            "REPO_DIR": REPO_DIR,
            "CONFIG": config,
            "YEAR": args.year,
            "SAMPLE_TXT": s["txt"].resolve(),
            "OUTDIR": args.outdir,
            "IS_DATA": "true" if s["data"] else "false",
            "WORKERS": args.cpus,
            "CHUNK_SIZE": args.chunk_size,
            "MAX_EVENTS": args.max_events,
            "PROXY_PATH": proxy,
            "PROXY_FILE": proxy.name,
            "LOGDIR": log_dir,
            "BATCH_NAME": s["batch_name"],
            "CPUS": args.cpus,
            "MEMORY": memory,
            "FLAVOUR": args.flavour,
            "SLICEFILE": slicefile,
        }
        cmd = ["condor_submit", str(CONDOR_DIR / "submit.sub")]
        for k, v in appends.items():
            cmd += ["-append", f"{k}={v}"]
        res = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL)
        if res.returncode != 0:
            print(f"✗ {s['sample']}: condor_submit failed\n{res.stderr.strip()}")
            continue
        m = re.search(r"submitted to cluster (\d+)", res.stdout)
        print(f"✓ {s['sample']}: cluster {m.group(1) if m else '?'}, {len(s['slices'])} jobs")
        submitted += len(s["slices"])

    if args.dry_run:
        print(f"\n[DRY RUN] slice files in {slice_dir}/, nothing submitted")
    else:
        print(f"\nSubmitted {submitted} jobs. Monitor: condor_q -batch    Logs: {log_dir}/")


if __name__ == "__main__":
    main()
