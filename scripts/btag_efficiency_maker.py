#!/usr/bin/env python3
"""
B-tag (UParT, 2024) MC-truth tagging-efficiency MAKER.

Usage:
    python scripts/btag_efficiency_maker.py \
        --inputs "outputs/btageff/2024/*.root" \
        --output outputs/btageff/2024/heavyflavor_efficiency_maps.root
"""

from __future__ import annotations

import argparse
import glob
from typing import Dict, List, Tuple

import numpy as np
import uproot

FLAVORS = ("b", "c", "light")
WORKING_POINTS = ("lwp", "mwp", "twp")

# Must match scripts/btag_efficiency_slimmer.py's ETA_EDGES/PT_EDGES exactly
# (real Run2 production binning: bTagEffs_{2016,2017,2018}.root).
ETA_EDGES = np.array([-2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5])  # 6 bins, signed
PT_EDGES = np.array([20., 50., 80., 120., 200., 300., 400., 500., 700., 1000.])  # 9 bins


def resolve_inputs(patterns: List[str]) -> List[str]:
    files: List[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No files matched: {pattern}")
        files.extend(matches)
    return files


def sum_counts(files: List[str]) -> Tuple[Dict[str, np.ndarray], "set[str]"]:
    """Sum the slimmer's denominator/pass count histograms for every flavor
    and working point across every input file (the `hadd`-equivalent step,
    done on raw counts only — never sum/average an already-divided efficiency).

    Each file is all-or-nothing: every key it needs is read before any of it
    is added to the totals. A file with even one unreadable key is added to
    `bad_files` and contributes nothing — otherwise a file whose `mwp` key is
    corrupted but `lwp` key is fine would be counted in the lwp sums only,
    leaving the per-WP denominators (which must be identical) inconsistent.
    condor's run_slimmer.sh verifies its own output before exiting, but a file
    from an older run (or one that slipped past that check) can still land
    here; better to report every bad file up front than crash on the first one
    and force a fix-one-rerun-find-the-next cycle (see
    condorJobs/btag-efficiency/run_slimmer.sh for the EOS write-corruption
    background: an uproot.recreate() block can exit cleanly while a subset of
    keys in the file on disk decompress to garbage)."""
    keys = [f"hist_{flavor}_efficiency_denominator" for flavor in FLAVORS] + [
        f"hist_{flavor}_efficiency_pass_{wp}"
        for flavor in FLAVORS for wp in WORKING_POINTS
    ]
    expected_shape = (len(ETA_EDGES) - 1, len(PT_EDGES) - 1)
    totals: Dict[str, np.ndarray] = {}
    bad_files: "set[str]" = set()
    for path in files:
        try:
            with uproot.open(path) as f:
                counts = {key: f[key].values() for key in keys}
        except Exception as e:
            print(f"WARNING: skipping corrupted file {path}: {e}")
            bad_files.add(path)
            continue
        for key, values in counts.items():
            if values.shape != expected_shape:
                raise ValueError(
                    f"{path}: {key} has shape {values.shape}, expected "
                    f"{expected_shape} — was it produced with a different "
                    f"ETA_EDGES/PT_EDGES than the current "
                    f"btag_efficiency_slimmer.py?"
                )
        for key, values in counts.items():
            if key in totals:
                totals[key] += values
            else:
                totals[key] = values.copy()
    if not totals:
        raise RuntimeError("No readable input files — nothing to sum.")
    return totals, bad_files


def build_efficiency_map(files: List[str]) -> Tuple[Dict[str, np.ndarray], "set[str]"]:
    hists: Dict[str, np.ndarray] = {}
    totals, bad_files = sum_counts(files)
    for flavor in FLAVORS:
        for wp in WORKING_POINTS:
            den = totals[f"hist_{flavor}_efficiency_denominator"]
            pas = totals[f"hist_{flavor}_efficiency_pass_{wp}"]

            eff = np.zeros_like(den)
            nonzero = den > 0
            eff[nonzero] = pas[nonzero] / den[nonzero]

            n_eta, n_pt = eff.shape
            for i in range(n_eta):
                for j in range(n_pt):
                    if den[i, j] > 0 and (eff[i, j] == 0.0 or eff[i, j] == 1.0):
                        print(
                            f"WARNING: {flavor}-jet {wp} efficiency bin "
                            f"(eta_bin={i}, pt_bin={j}) is {eff[i, j]:.0%} "
                            f"with only {den[i, j]:.0f} MC-truth jets — "
                            f"low-stat bin, check before using in the SF "
                            f"reweighting formula."
                        )

            hists[f"hist_{flavor}_denominator_{wp}"] = den
            hists[f"hist_{flavor}_pass_{wp}"] = pas
            hists[f"hist_{flavor}_efficiency_{wp}"] = eff

    return hists, bad_files


def write_histograms(output_path: str, hists: Dict[str, np.ndarray]) -> None:
    with uproot.recreate(output_path) as rf:
        for name, values in hists.items():
            rf[name] = (values, ETA_EDGES, PT_EDGES)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Glob pattern(s) for per-file slimmer ROOT outputs")
    parser.add_argument("--output", required=True, help="Output efficiency-map ROOT file")
    args = parser.parse_args()

    files = resolve_inputs(args.inputs)
    print(f"Summing counts across {len(files)} slimmer output file(s):")
    for f in files:
        print(f"  {f}")

    hists, bad_files = build_efficiency_map(files)

    for flavor in FLAVORS:
        for wp in WORKING_POINTS:
            den = hists[f"hist_{flavor}_denominator_{wp}"]
            pas = hists[f"hist_{flavor}_pass_{wp}"]
            eff = pas.sum() / den.sum() if den.sum() else float('nan')
            print(f"{flavor} {wp}: total denominator={den.sum():.0f}, "
                  f"total pass={pas.sum():.0f}, overall efficiency={eff:.3f}")

    import os
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    write_histograms(args.output, hists)
    print(f"Wrote {args.output}")

    if bad_files:
        print(f"\n✗ {len(bad_files)}/{len(files)} input file(s) were corrupted and "
              f"EXCLUDED from the sums above (map was still written from the "
              f"remaining good files). Re-run the corresponding condor job(s) "
              f"and re-run this maker before trusting the output:")
        for path in sorted(bad_files):
            print(f"  {path}")
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()
