#!/usr/bin/env python3
"""
Verify golden JSON lumi masking works the same way in iterative mode and
via the coffea futures/dask executors, on the same real data file, in a
single run.

Runs DarkBottomLineProcessor.apply_lumi_mask twice on the same file:
  1. iterative: raw uproot.open(...).arrays() -> ak.Array (same code path as
     `darkbottomline analyze` without --executor)
  2. executor: coffea Runner + FuturesExecutor/DaskExecutor, which builds
     events via NanoEventsFactory (same code path as
     `darkbottomline analyze --executor futures/dask`)
Then compares the two (run, luminosityBlock) sets that survive the mask and
reports PASS/FAIL.

With --full-chain, instead runs the real `darkbottomline analyze --mode
event-selection --data` CLI once per (file, mode) for each --modes entry and
checks every (run, luminosityBlock) in the written EVENTSELECTION.root
against the golden JSON. This covers the path that actually writes the
output (the coffea wrapper's own apply_selection call), which the
mask-only comparison above does not exercise. --repo-dir picks which
checkout's code runs, so a fixed and an unfixed tree can be compared.

Usage:
  python scripts/test_golden_json_executor.py <data_file.root> <config.yaml> [--executor futures|dask] [--workers N]
  python scripts/test_golden_json_executor.py <file1.root> [<file2.root> ...] <config.yaml> --full-chain \\
      --out-dir <dir> [--modes iterative futures dask] [--repo-dir <checkout>] [--jobs N]

Example:
  python scripts/test_golden_json_executor.py \\
      "root://cms-xrd-global.cern.ch//store/data/Run2024I/EGamma0/NANOAOD/MINIv6NANOv15-v1/2530000/0f2ec74c-22ac-4431-9a72-c31d4313e995.root" \\
      configs/2024.yaml --executor futures
"""
import argparse
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def run_iterative(data_file, config):
    import awkward as ak
    import uproot

    from darkbottomline.processor import DarkBottomLineProcessor

    events = ak.Array(uproot.open(f"{data_file}:Events").arrays(["run", "luminosityBlock"]))
    proc = DarkBottomLineProcessor(config)
    filtered = proc.apply_lumi_mask(events)
    pairs = set(zip(filtered.run.tolist(), filtered.luminosityBlock.tolist()))
    return len(events), pairs


def run_executor(data_file, config, executor_name, workers, chunksize):
    from coffea.nanoevents import BaseSchema
    from coffea.processor import ProcessorABC, Runner, dict_accumulator, set_accumulator

    from darkbottomline.processor import DarkBottomLineProcessor

    class GoldenJsonCheckProcessor(ProcessorABC):
        def __init__(self, cfg):
            self.base = DarkBottomLineProcessor(cfg)

        @property
        def accumulator(self):
            return dict_accumulator({"n_total": 0, "pairs": set_accumulator(set())})

        def process(self, events):
            n_total = len(events)
            filtered = self.base.apply_lumi_mask(events)
            # Must be a set_accumulator, not a plain set: coffea's
            # dict_accumulator merges values with += across the executor's
            # multi-worker merge tree, and plain Python sets don't support
            # += (only |=), so a bare set() blows up with
            # "TypeError: unsupported operand type(s) for +=: 'set' and 'set'"
            # as soon as there's more than one chunk to merge.
            pairs = set_accumulator(set(zip(filtered.run.tolist(), filtered.luminosityBlock.tolist())))
            return dict_accumulator({"n_total": n_total, "pairs": pairs})

        def postprocess(self, accumulator):
            return accumulator

    fileset = {"dataset": {"treename": "Events", "files": [data_file]}}
    processor_instance = GoldenJsonCheckProcessor(config)

    if executor_name == "futures":
        from coffea.processor import FuturesExecutor

        executor = FuturesExecutor(workers=workers)
    else:
        from coffea.processor import DaskExecutor
        from dask.distributed import Client

        client = Client(n_workers=workers)
        executor = DaskExecutor(client=client)

    runner = Runner(executor=executor, chunksize=chunksize, schema=BaseSchema)
    result = runner(fileset, processor_instance)
    return result["n_total"], result["pairs"]


def load_golden_pairs(config):
    """Set of certified (run, lumi) pairs from the config's data.golden_json."""
    from pathlib import Path

    gjp = Path(config["data"]["golden_json"])
    if not gjp.exists():
        gjp = Path(__file__).resolve().parent.parent / gjp
    with open(gjp) as f:
        golden = json.load(f)
    return {(int(r), l) for r, ranges in golden.items() for a, b in ranges for l in range(a, b + 1)}


def run_cli(data_file, config_path, mode, out_root, log_path, repo_dir, workers, chunksize):
    """Run the real event-selection CLI for one file in one mode; return (rc, seconds)."""
    cmd = [sys.executable, "-m", "darkbottomline.cli", "analyze", "--mode", "event-selection",
           "--config", os.path.abspath(config_path), "--input", data_file,
           "--event-selection-output", out_root, "--data"]
    if mode != "iterative":
        cmd += ["--executor", mode, "--workers", str(workers), "--chunk-size", str(chunksize)]
    # cwd=repo_dir puts that checkout first on sys.path for `python -m`, ahead
    # of the editable install, so --repo-dir really selects which code runs.
    t0 = time.time()
    with open(log_path, "w") as log:
        rc = subprocess.call(cmd, cwd=repo_dir, stdout=log, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL)
    return rc, time.time() - t0


def count_output(out_root, golden_pairs):
    """(n_selected, n_out_of_golden, out_of_golden_runs) for an EVENTSELECTION.root."""
    import uproot

    with uproot.open(out_root) as f:
        if "Events" not in f or f["Events"].num_entries == 0:
            return 0, 0, []
        a = f["Events"].arrays(["run", "luminosityBlock"], library="np")
    bad_runs = set()
    n_bad = 0
    for r, l in zip(a["run"].tolist(), a["luminosityBlock"].tolist()):
        if (r, l) not in golden_pairs:
            n_bad += 1
            bad_runs.add(r)
    return len(a["run"]), n_bad, sorted(bad_runs)


def run_full_chain(args, config):
    golden_pairs = load_golden_pairs(config)
    os.makedirs(args.out_dir, exist_ok=True)
    repo_dir = os.path.abspath(args.repo_dir)

    def one(task):
        data_file, mode = task
        tag = os.path.basename(data_file).removesuffix(".root")
        out_root = os.path.join(os.path.abspath(args.out_dir), f"{tag}__{mode}_EVENTSELECTION.root")
        log_path = os.path.join(args.out_dir, f"{tag}__{mode}.log")
        rc, secs = run_cli(data_file, args.config, mode, out_root, log_path, repo_dir,
                           args.workers, args.chunksize)
        row = {"file": tag, "mode": mode, "rc": rc, "seconds": round(secs, 1)}
        if rc == 0 and os.path.exists(out_root):
            row["selected"], row["out_of_golden"], row["bad_runs"] = count_output(out_root, golden_pairs)
        print(f"[{mode:9s}] {tag[:60]:60s} rc={rc} sel={row.get('selected')} "
              f"out_of_golden={row.get('out_of_golden')} ({secs:.0f}s)", flush=True)
        return row

    tasks = [(f, m) for f in args.data_files for m in args.modes]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        rows = list(pool.map(one, tasks))

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump({"repo_dir": repo_dir, "rows": rows}, f, indent=1)

    print("\n=== SUMMARY (repo: %s) ===" % repo_dir)
    failed = [r for r in rows if r["rc"] != 0 or "selected" not in r]
    leaking = [r for r in rows if r.get("out_of_golden", 0) > 0]
    for mode in args.modes:
        mr = [r for r in rows if r["mode"] == mode and "selected" in r]
        print(f"{mode:9s}: files={len(mr)} selected={sum(r['selected'] for r in mr)} "
              f"out_of_golden={sum(r['out_of_golden'] for r in mr)}")
    # Same file must select the same number of events in every mode.
    mismatched = []
    for f in args.data_files:
        tag = os.path.basename(f).removesuffix(".root")
        counts = {r["mode"]: r.get("selected") for r in rows if r["file"] == tag}
        if len(set(counts.values())) > 1:
            mismatched.append((tag, counts))
    for tag, counts in mismatched:
        print(f"  MODE MISMATCH {tag}: {counts}")
    for r in failed:
        print(f"  FAILED {r['mode']} {r['file']} rc={r['rc']}")

    ok = not failed and not leaking and not mismatched
    print("\nPASS" if ok else "\nFAIL", f"(failed={len(failed)}, leaking={len(leaking)}, "
          f"mode_mismatch={len(mismatched)})")
    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("paths", nargs="+",
                        help="NanoAOD data ROOT file(s) (local path or root:// URL), then the "
                             "year config YAML (e.g. configs/2024.yaml) last; more than one "
                             "file only with --full-chain")
    parser.add_argument("--executor", choices=["futures", "dask"], default="futures")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--chunksize", type=int, default=50000)
    parser.add_argument("--full-chain", action="store_true",
                        help="Run the real event-selection CLI and check its output file")
    parser.add_argument("--modes", nargs="+", choices=["iterative", "futures", "dask"],
                        default=["iterative", "futures", "dask"])
    parser.add_argument("--out-dir", help="--full-chain: where outputs, logs and summary.json go")
    parser.add_argument("--repo-dir", default=os.path.join(os.path.dirname(__file__), ".."),
                        help="--full-chain: checkout whose darkbottomline code is run")
    parser.add_argument("--jobs", type=int, default=2,
                        help="--full-chain: (file, mode) runs in parallel")
    args = parser.parse_args()
    if len(args.paths) < 2:
        parser.error("need at least one data file and the config YAML")
    args.data_files, args.config = args.paths[:-1], args.paths[-1]

    import yaml

    with open(args.config) as f:
        config = yaml.safe_load(f)
    config.setdefault("data", {})["is_data"] = True

    if args.full_chain:
        if not args.out_dir:
            parser.error("--full-chain needs --out-dir")
        sys.exit(run_full_chain(args, config))
    if len(args.data_files) != 1:
        parser.error("more than one data file needs --full-chain")
    args.data_file = args.data_files[0]

    print(f"=== Iterative run ({args.data_file}) ===")
    n_total_iter, pairs_iter = run_iterative(args.data_file, config)
    print(f"Total events read: {n_total_iter}")
    print(f"Events passing golden JSON mask: {len(pairs_iter)}")

    print(f"\n=== {args.executor.capitalize()} executor run ===")
    try:
        n_total_exec, pairs_exec = run_executor(
            args.data_file, config, args.executor, args.workers, args.chunksize
        )
    except Exception:
        logging.exception(
            "Executor run raised an exception (this is exactly what the "
            "uproot/coffea RNTuple bug looks like if it is not fixed)"
        )
        sys.exit(1)
    print(f"Total events read: {n_total_exec}")
    print(f"Events passing golden JSON mask: {len(pairs_exec)}")

    print("\n=== COMPARISON ===")
    same_total = n_total_iter == n_total_exec
    same_pairs = pairs_iter == pairs_exec
    print(f"Same total event count: {same_total} ({n_total_iter} vs {n_total_exec})")
    print(f"Same surviving (run, lumi) set: {same_pairs}")

    if same_total and same_pairs:
        print("\nPASS: golden JSON masking is consistent between iterative and "
              f"{args.executor} executor.")
        sys.exit(0)
    else:
        only_iter = pairs_iter - pairs_exec
        only_exec = pairs_exec - pairs_iter
        print(f"\nFAIL: mismatch between iterative and {args.executor} executor.")
        if only_iter:
            print(f"  In iterative only ({len(only_iter)}): {sorted(only_iter)[:10]}")
        if only_exec:
            print(f"  In {args.executor} only ({len(only_exec)}): {sorted(only_exec)[:10]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
