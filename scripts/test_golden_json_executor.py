#!/usr/bin/env python3
"""
Verify golden JSON lumi masking works correctly through the coffea
futures/dask executors (not just iterative mode).

Runs DarkBottomLineProcessor.apply_lumi_mask on a real local data ROOT file,
via coffea's Runner + FuturesExecutor/DaskExecutor (the same NanoEventsFactory
code path used by `darkbottomline analyze --executor futures/dask`), and
prints which (run, luminosityBlock) pairs survive.

Usage:
  python scripts/test_golden_json_executor.py <data_file.root> <config.yaml> [--executor futures|dask] [--workers N]

Example:
  python scripts/test_golden_json_executor.py \\
      /path/to/some/Run2024_data_file.root configs/2024.yaml --executor futures
"""
import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_file", help="Local NanoAOD data ROOT file (not xrootd)")
    parser.add_argument("config", help="Year config YAML (e.g. configs/2024.yaml)")
    parser.add_argument("--executor", choices=["futures", "dask"], default="futures")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--chunksize", type=int, default=50000)
    args = parser.parse_args()

    import yaml
    from coffea.nanoevents import BaseSchema
    from coffea.processor import ProcessorABC, Runner, dict_accumulator

    with open(args.config) as f:
        config = yaml.safe_load(f)
    config.setdefault("data", {})["is_data"] = True

    from darkbottomline.processor import DarkBottomLineProcessor

    class GoldenJsonCheckProcessor(ProcessorABC):
        def __init__(self, cfg):
            self.base = DarkBottomLineProcessor(cfg)

        @property
        def accumulator(self):
            return dict_accumulator({"n_total": 0, "n_passed": 0, "sample_pairs": []})

        def process(self, events):
            n_total = len(events)
            filtered = self.base.apply_lumi_mask(events)
            n_passed = len(filtered)
            sample = list(
                zip(
                    filtered.run.tolist()[:5],
                    filtered.luminosityBlock.tolist()[:5],
                )
            )
            return dict_accumulator(
                {"n_total": n_total, "n_passed": n_passed, "sample_pairs": sample}
            )

        def postprocess(self, accumulator):
            return accumulator

    fileset = {"dataset": {"treename": "Events", "files": [args.data_file]}}
    processor_instance = GoldenJsonCheckProcessor(config)

    if args.executor == "futures":
        from coffea.processor import FuturesExecutor

        executor = FuturesExecutor(workers=args.workers)
    else:
        from coffea.processor import DaskExecutor
        from dask.distributed import Client

        client = Client(n_workers=args.workers)
        executor = DaskExecutor(client=client)

    runner = Runner(executor=executor, chunksize=args.chunksize, schema=BaseSchema)

    print(f"\n=== Running golden JSON check via {args.executor} executor ===")
    result = runner(fileset, processor_instance)

    print("\n=== RESULT ===")
    print(f"Total events read:     {result['n_total']}")
    print(f"Events passing mask:   {result['n_passed']}")
    print(f"Removed:               {result['n_total'] - result['n_passed']}")
    print(f"Sample surviving (run, lumi) pairs: {result['sample_pairs']}")

    if result["n_total"] == 0:
        print("\nFAIL: no events were read from the file at all.")
        sys.exit(1)

    print(
        "\nIf this ran without the "
        "\"AttributeError: module 'uproot.behaviors' has no attribute 'RNTuple'\" crash, "
        "the uproot/coffea fix worked. Compare n_passed against an iterative-mode run "
        "on the same file to confirm the golden JSON mask itself is consistent."
    )


if __name__ == "__main__":
    main()
