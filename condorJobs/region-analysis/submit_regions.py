#!/usr/bin/env python3
"""
Submit the region-analysis stage on HTCondor: merged <sample>_EVENTSELECTION.root ->
per-sample region PKL with DNN scores -> region plots.

One job per merged sample file (run_region.sh), then one plot job (run_plots.sh) as the
DAGMan child of all region jobs. Everything is submitted with one condor_submit_dag.

  <outdir>/per_sample/<stem>_DNNAPPLIED.pkl    one per sample (job output)
  <outdir>/plots/<version>/...                     plot job output
  <outdir>/config/<year>_lumi<lumi>.yaml           the config the jobs used (lumi set here)
  <outdir>/config/run_info.json                    git version, lumi, inputs, command

Samples whose PKL is already on EOS are not resubmitted, so rerunning the same command
after a round finishes submits only what is missing (and the plot job again, as the child
of whatever runs; with nothing missing it is plot only).

Examples (run from anywhere):
    # dry run: write job list, DAG and config copy locally, print the commands, submit nothing
    python3 condorJobs/region-analysis/submit_regions.py --lumi 109.627 --dry-run \\
        --input-dir /eos/cms/store/group/phys_susy/sus-23-008/run3_bbMET/20261014-d110783_14102026/2024/eventsel_merged \\
        --outdir    /eos/cms/store/group/phys_susy/sus-23-008/run3_bbMET/20261014-d110783_14102026/2024/region_analysis
"""

import argparse
import fnmatch
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

CONDOR_DIR = Path(__file__).resolve().parent
REPO_DIR = CONDOR_DIR.parent.parent
EOS_HOST = "eoscms.cern.ch"
SUFFIX = "_EVENTSELECTION.root"


def _event_selection_module():
    """submit_samples.py (event-selection) for sample_kind(): one rule for data/MC/signal."""
    path = REPO_DIR / "condorJobs" / "event-selection" / "submit_samples.py"
    spec = importlib.util.spec_from_file_location("dbl_submit_samples", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def git_version() -> str:
    text = (REPO_DIR / "darkbottomline" / "_version.py").read_text()
    return re.search(r'__version__\s*=\s*"([^"]+)"', text).group(1)


def default_plot_version() -> str:
    """YYYYMMDD_<sha7>, same convention as darkbottomline.cli._default_version."""
    v = git_version()
    sha = v.split("+", 1)[1] if "+" in v else "nogit"
    return f"{datetime.now():%Y%m%d}_{sha}"


def xrd_ls(path: str):
    """Names in an EOS directory via xrdfs (FUSE listings can lag); empty if it does not exist."""
    res = subprocess.run(["xrdfs", EOS_HOST, "ls", path], capture_output=True, text=True, timeout=300)
    if res.returncode != 0:
        return set()
    return {os.path.basename(line.strip()) for line in res.stdout.splitlines() if line.strip()}


def request_memory_mb(size_bytes: int, args) -> int:
    """Linear in input size, clamped: --mem-min for small files up to --mem-max for the largest."""
    gb = size_bytes / 1e9
    return int(min(args.mem_max, max(args.mem_min, args.mem_min + args.mem_per_gb * gb)))


def job_cpus(memory_mb: int, args) -> int:
    """Cores scale with memory at --mb-per-cpu (default 3000: lxplus gives at most 3000 MB per core,
    so fewer cores than memory/3000 would be raised anyway)."""
    return max(args.cpus, math.ceil(memory_mb / args.mb_per_cpu))


def write_lumi_config(base: Path, lumi: float, dest: Path) -> None:
    text = base.read_text()
    new, n = re.subn(r"^lumi:[ \t]*[0-9.eE+-]+[^\n]*$",
                     f"lumi: {lumi}   # set by condorJobs/region-analysis/submit_regions.py (--lumi)",
                     text, flags=re.M)
    if n != 1:
        sys.exit(f"✗ expected exactly one top-level 'lumi:' line in {base}, found {n}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(new)


def xrdcp_to_eos(src: Path, dest: str) -> None:
    subprocess.run(["xrdfs", EOS_HOST, "mkdir", "-p", os.path.dirname(dest)], check=False,
                   capture_output=True)
    subprocess.run(["xrdcp", "--nopbar", "--force", "--posc", "--cksum", "adler32", str(src),
                    f"root://{EOS_HOST}/{dest}"], check=True)


def queued_batch_names():
    try:
        out = subprocess.run(["condor_q", "-af", "JobBatchName"], capture_output=True,
                             text=True, check=True, timeout=120).stdout
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        sys.exit(f"✗ Could not query condor_q ({e}); pass --allow-duplicates to skip this check")
    return {line.strip() for line in out.splitlines() if line.strip()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", required=True, type=Path,
                   help="Directory of merged <sample>_EVENTSELECTION.root (scripts/merge_rootfiles.py output)")
    p.add_argument("--outdir", required=True,
                   help="EOS regions directory; PKLs go to <outdir>/per_sample/, plots to <outdir>/plots/")
    p.add_argument("--lumi", required=True, type=float,
                   help="Integrated luminosity in /fb written into the config copy "
                        "(2024 production: 109.627 JetMET brilcalc; EGamma is 108.460)")
    p.add_argument("--year", default="2024", choices=["2022", "2022EE", "2023", "2024"])
    p.add_argument("--base-config", default=None, help="Config YAML to copy lumi into (default: configs/<year>.yaml)")
    p.add_argument("--regions-config", default="configs/regions.yaml")
    p.add_argument("--xsection-json", default="data/cross-section/xsection_background_run3.json")
    p.add_argument("--xsection-signal-json", default="data/cross-section/xsection_signal.json")
    p.add_argument("--plot-config", default="configs/plotting.yaml")
    p.add_argument("--dnn-model", default="data/dnn/dnn_model.pt")
    p.add_argument("--dnn-config", default="configs/dnn.yaml")
    p.add_argument("--signal-scale", type=float, default=10)
    p.add_argument("--version", default=None, help="Plot version tag (default: YYYYMMDD_<sha>)")
    p.add_argument("--samples", nargs="+", default=None, metavar="GLOB", help="Only samples matching these globs")
    p.add_argument("--exclude", nargs="+", default=[], metavar="GLOB", help="Skip samples matching these globs")
    p.add_argument("--cpus", type=int, default=1, help="Minimum request_cpus per region job (default: 1; "
                   "raised to ceil(memory / --mb-per-cpu))")
    p.add_argument("--mb-per-cpu", type=int, default=3000,
                   help="request_cpus = ceil(memory / this) (default: 3000, the lxplus maximum per core: "
                        "12000 MB -> 4 cores; lower it to ask for more cores)")
    p.add_argument("--mem-min", type=int, default=4000, help="request_memory (MB) of the smallest files (default: 4000)")
    p.add_argument("--mem-per-gb", type=float, default=500, help="Extra MB per GB of input (default: 500)")
    p.add_argument("--mem-max", type=int, default=12000, help="request_memory cap in MB (default: 12000)")
    p.add_argument("--flavour", default="workday",
                   choices=["espresso", "microcentury", "longlunch", "workday", "tomorrow", "testmatch", "nextweek"])
    p.add_argument("--no-plot", action="store_true", help="Region jobs only: no plot node in the DAG")
    p.add_argument("--plot-only", action="store_true",
                   help="Submit only the plot job; refuses if any selected sample has no PKL on EOS yet")
    p.add_argument("--plot-cpus", type=int, default=4)
    p.add_argument("--plot-memory", type=int, default=16000, help="Plot job request_memory in MB (default: 16000)")
    p.add_argument("--plot-flavour", default="tomorrow",
                   choices=["espresso", "microcentury", "longlunch", "workday", "tomorrow", "testmatch", "nextweek"])
    p.add_argument("--proxy", default=None, help="x509 proxy (default: ~/private/x509up_u<uid> on AFS)")
    p.add_argument("--no-skip-existing", action="store_true", help="Submit samples even if their PKL is on EOS")
    p.add_argument("--allow-duplicates", action="store_true", help="Submit even if jobs of this stage are queued")
    p.add_argument("--dry-run", action="store_true",
                   help="Write job list, DAG and config copy locally and print the commands; touch nothing on EOS, submit nothing")
    args = p.parse_args()

    es = _event_selection_module()
    outdir = args.outdir.rstrip("/")
    version = args.version or default_plot_version()
    base_config = REPO_DIR / (args.base_config or f"configs/{args.year}.yaml")
    for f in (base_config, REPO_DIR / args.regions_config, REPO_DIR / args.xsection_json,
              REPO_DIR / args.xsection_signal_json, REPO_DIR / args.plot_config,
              REPO_DIR / args.dnn_model, REPO_DIR / args.dnn_config):
        if not f.is_file():
            sys.exit(f"✗ Not found: {f}")
    if not args.input_dir.is_dir():
        sys.exit(f"✗ Input dir not found: {args.input_dir}")

    user = os.environ.get("USER") or os.getlogin()
    proxy = Path(args.proxy or f"/afs/cern.ch/user/{user[0]}/{user}/private/x509up_u{os.getuid()}")
    if not proxy.is_file():
        sys.exit(f"✗ Proxy not found: {proxy}\n  voms-proxy-init --voms cms --valid 192:00 && cp /tmp/x509up_u{os.getuid()} {proxy.parent}/")
    left = subprocess.run(["voms-proxy-info", "-file", str(proxy), "-timeleft"], capture_output=True, text=True)
    hours_left = int(left.stdout.strip() or 0) / 3600 if left.returncode == 0 else 0
    if hours_left < 24:
        msg = f"proxy {proxy} has {hours_left:.0f}h left"
        if args.dry_run:
            print(f"⚠  {msg} (would refuse to submit with < 24h)")
        else:
            sys.exit(f"✗ {msg}; renew it (voms-proxy-init --voms cms --valid 192:00) and copy it to {proxy.parent}/")

    batch_name = f"regions_{args.year}"
    plot_batch = f"regionplots_{args.year}"
    if not (args.dry_run or args.allow_duplicates):
        queued = queued_batch_names()
        if queued & {batch_name, plot_batch}:
            sys.exit(f"✗ jobs of this stage are still queued ({batch_name}); wait for them or pass --allow-duplicates")

    # Plan: one job per merged file.
    pkl_dir = f"{outdir}/per_sample"
    done = set() if args.no_skip_existing else xrd_ls(pkl_dir)
    plan, n_done, skipped_pds = [], 0, []
    inputs = sorted(f for f in args.input_dir.iterdir() if f.name.endswith(SUFFIX))
    if not inputs:
        sys.exit(f"✗ No *{SUFFIX} in {args.input_dir}")
    for f in inputs:
        stem = f.name[:-len(".root")]
        sample = stem[:-len("_EVENTSELECTION")]
        if args.samples and not any(fnmatch.fnmatch(sample, g) for g in args.samples):
            continue
        if any(fnmatch.fnmatch(sample, g) for g in args.exclude):
            continue
        kind = es.sample_kind(sample)
        if kind == "SKIP":
            skipped_pds.append(sample)
            continue
        if f"{stem}_DNNAPPLIED.pkl" in done:
            n_done += 1
            continue
        size = f.stat().st_size
        mem = request_memory_mb(size, args)
        plan.append(dict(path=str(f), stem=stem, kind=kind, size=size, mem=mem, cpus=job_cpus(mem, args)))
    if skipped_pds:
        print(f"⚠  Skipping {len(skipped_pds)} data sample(s) not in DATA_PRIMARY_DATASETS: {', '.join(skipped_pds)}")
    if args.plot_only and args.no_plot:
        sys.exit("✗ --plot-only and --no-plot contradict each other")
    if args.plot_only and plan:
        sys.exit(f"✗ --plot-only: {len(plan)} sample(s) have no PKL in {pkl_dir} yet "
                 f"(first: {plan[0]['stem']}); run the region jobs first (--no-plot) or drop --plot-only")
    if not plan and args.no_plot:
        print("Nothing to submit: every selected sample already has a PKL on EOS.")
        return
    plan.sort(key=lambda j: j["size"], reverse=True)  # biggest first so their tails start earliest

    # Local files (AFS): job list, DAG, config copy. Jobs read the config copy from here.
    work = CONDOR_DIR / "submitfiles" / args.year
    log_dir = CONDOR_DIR / "logs" / args.year
    for d in (work, log_dir / "output", log_dir / "error"):
        d.mkdir(parents=True, exist_ok=True)
    lumi_tag = f"{args.lumi:g}"
    config_copy = work / f"{args.year}_lumi{lumi_tag}.yaml"
    write_lumi_config(base_config, args.lumi, config_copy)
    joblist = work / "regions.jobs"
    joblist.write_text("".join(
        f"{j['path']} {j['stem']} {'true' if j['kind'] == 'DATA' else 'false'} {j['mem']} {j['cpus']}\n"
        for j in plan))

    common = dict(
        REPO_DIR=REPO_DIR, CONFIG=config_copy, REGIONS_CONFIG=args.regions_config,
        XSEC_JSON=args.xsection_json, OUTDIR=outdir, LOGDIR=log_dir,
    )
    region_vars = dict(common, EXEC=CONDOR_DIR / "run_region.sh", DNN_MODEL=args.dnn_model,
                       DNN_CONFIG=args.dnn_config, PROXY_PATH=proxy, PROXY_FILE=proxy.name,
                       BATCH_NAME=batch_name, FLAVOUR=args.flavour, JOBLIST=joblist)
    plot_vars = dict(common, EXEC=CONDOR_DIR / "run_plots.sh", XSEC_SIGNAL_JSON=args.xsection_signal_json,
                     PLOT_CONFIG=args.plot_config, INPUT_DIR=args.input_dir.resolve(),
                     PKL_DIR=pkl_dir, VERSION=version, SIGNAL_SCALE=args.signal_scale,
                     BATCH_NAME=plot_batch, PLOT_CPUS=args.plot_cpus, PLOT_MEMORY=args.plot_memory,
                     PLOT_FLAVOUR=args.plot_flavour)

    def vars_line(node, d):
        return f"VARS {node} " + " ".join(f'{k}="{v}"' for k, v in d.items())

    dag = []
    if plan:
        dag += [f"JOB regions {CONDOR_DIR / 'submit.sub'}", vars_line("regions", region_vars)]
    if not args.no_plot:
        dag += [f"JOB plot {CONDOR_DIR / 'plot.sub'}", vars_line("plot", plot_vars)]
        if plan:
            dag += ["PARENT regions CHILD plot"]
    dag_file = work / "regions.dag"
    dag_file.write_text("\n".join(dag) + "\n")

    total_gb = sum(j["size"] for j in plan) / 1e9
    print("=" * 72)
    print(f"Region analysis {args.year}: {len(plan)} jobs ({total_gb:.1f} GB input), "
          f"{n_done} PKLs already on EOS, plot {'off' if args.no_plot else 'on'}")
    print(f"Input : {args.input_dir}")
    print(f"Output: {pkl_dir}/   plots: {outdir}/plots/")
    print(f"Config: {config_copy} (lumi {args.lumi:g}/fb)")
    print(f"DNN   : {args.dnn_model} + {args.dnn_config}")
    print("=" * 72)
    print(f"{'sample':<80} {'type':<4} {'GB':>6} {'MB':>6} {'cpu':>3}")
    for j in plan:
        print(f"{j['stem'][:80]:<80} {j['kind']:<4} {j['size'] / 1e9:>6.1f} {j['mem']:>6} {j['cpus']:>3}")
    if not plan:
        print("(no region jobs: every selected sample has a PKL; DAG is plot only)")

    stage = "plots" if args.plot_only else ("pkls" if args.no_plot else "pkls+plots")
    dag_batch = f"region_{stage}_{args.year}_lumi{lumi_tag}_{git_version().replace('+', '-')}"
    cmd = ["condor_submit_dag", "-f", "-batch-name", dag_batch, str(dag_file)]
    if args.dry_run:
        print(f"\n[DRY RUN] wrote {joblist}, {dag_file}, {config_copy}; nothing submitted, nothing written to EOS")
        print("Submit with:\n  " + " ".join(cmd))
        return

    # Record the run next to its outputs.
    run_info = work / "run_info.json"
    run_info.write_text(json.dumps(dict(
        git_version=git_version(), lumi_fb=args.lumi, year=args.year, input_dir=str(args.input_dir),
        outdir=outdir, plot_version=version, dnn_model=args.dnn_model, dnn_config=args.dnn_config,
        xsection_json=args.xsection_json, regions_config=args.regions_config,
        submitted=f"{datetime.now():%Y-%m-%d %H:%M}", command=" ".join(sys.argv)), indent=2) + "\n")
    xrdcp_to_eos(config_copy, f"{outdir}/config/{config_copy.name}")
    xrdcp_to_eos(run_info, f"{outdir}/config/run_info.json")
    res = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    print(res.stdout.strip())
    if res.returncode != 0:
        sys.exit(f"✗ condor_submit_dag failed\n{res.stderr.strip()}")
    print(f"\nMonitor: condor_q -batch    DAG status: tail {dag_file}.dagman.out    Logs: {log_dir}/")


if __name__ == "__main__":
    main()
