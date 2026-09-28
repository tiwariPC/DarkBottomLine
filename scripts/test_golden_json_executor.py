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

Usage:
  python scripts/test_golden_json_executor.py <data_file.root> <config.yaml> [--executor futures|dask] [--workers N]

Example:
  python scripts/test_golden_json_executor.py \\
      "root://cms-xrd-global.cern.ch//store/data/Run2024I/EGamma0/NANOAOD/MINIv6NANOv15-v1/2530000/0f2ec74c-22ac-4431-9a72-c31d4313e995.root" \\
      configs/2024.yaml --executor futures
"""
import argparse
import logging
import sys

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
    from coffea.processor import ProcessorABC, Runner, dict_accumulator

    from darkbottomline.processor import DarkBottomLineProcessor

    class GoldenJsonCheckProcessor(ProcessorABC):
        def __init__(self, cfg):
            self.base = DarkBottomLineProcessor(cfg)

        @property
        def accumulator(self):
            return dict_accumulator({"n_total": 0, "pairs": set()})

        def process(self, events):
            n_total = len(events)
            filtered = self.base.apply_lumi_mask(events)
            pairs = set(zip(filtered.run.tolist(), filtered.luminosityBlock.tolist()))
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


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("data_file", help="NanoAOD data ROOT file (local path or root:// URL)")
    parser.add_argument("config", help="Year config YAML (e.g. configs/2024.yaml)")
    parser.add_argument("--executor", choices=["futures", "dask"], default="futures")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--chunksize", type=int, default=50000)
    args = parser.parse_args()

    import yaml

    with open(args.config) as f:
        config = yaml.safe_load(f)
    config.setdefault("data", {})["is_data"] = True

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
