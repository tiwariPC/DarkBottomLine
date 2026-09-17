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


def sum_counts(files: List[str], flavor: str, wp: str) -> Tuple[np.ndarray, np.ndarray]:
    """Sum the slimmer's denominator/pass count histograms for one flavor and
    working point across every input file (the `hadd`-equivalent step, done
    on raw counts only — never sum/average an already-divided efficiency)."""
    den_total = None
    pass_total = None
    for path in files:
        with uproot.open(path) as f:
            den = f[f"hist_{flavor}_efficiency_denominator"].values()
            pas = f[f"hist_{flavor}_efficiency_pass_{wp}"].values()
        if den.shape != (len(ETA_EDGES) - 1, len(PT_EDGES) - 1):
            raise ValueError(
                f"{path}: hist_{flavor}_efficiency_denominator has shape "
                f"{den.shape}, expected {(len(ETA_EDGES) - 1, len(PT_EDGES) - 1)} "
                f"— was it produced with a different ETA_EDGES/PT_EDGES than "
                f"the current btag_efficiency_slimmer.py?"
            )
        if den_total is None:
            den_total = den.copy()
            pass_total = pas.copy()
        else:
            den_total += den
            pass_total += pas
    return den_total, pass_total


def build_efficiency_map(files: List[str]) -> Dict[str, np.ndarray]:
    hists: Dict[str, np.ndarray] = {}
    for flavor in FLAVORS:
        for wp in WORKING_POINTS:
            den, pas = sum_counts(files, flavor, wp)

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

    return hists


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

    hists = build_efficiency_map(files)

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


if __name__ == "__main__":
    main()
