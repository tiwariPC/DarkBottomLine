#!/usr/bin/env python3
"""
Standalone cross-check plot: Pileup_nTrueInt per region, WITH vs WITHOUT the
pileup reweight (all other SFs kept). Reads an EVENTSELECTION.root file, applies
region cuts via RegionManager (flat-branch path), and overlays:

  * with PU reweight    : weighted by full_event_weight
  * without PU reweight : weighted by weight_noPileup (full_event_weight / pu_central)

Does NOT touch the main analysis/plotting workflow.

Usage:
  python scripts/plot_nTrueInt_noPU.py \
      --input outputs/eventsel/SAMPLE_EVENTSELECTION.root \
      --regions-config configs/regions.yaml \
      --outdir outputs/eventsel/nTrueInt_noPU
"""
import argparse
import os
import logging

import numpy as np
import awkward as ak
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from darkbottomline.regions import RegionManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="EVENTSELECTION.root file")
    ap.add_argument("--regions-config", default="configs/regions.yaml")
    ap.add_argument("--tree", default="Events")
    ap.add_argument("--bins", type=int, default=80)
    ap.add_argument("--low", type=float, default=0.0)
    ap.add_argument("--high", type=float, default=80.0)
    ap.add_argument("--outdir", default="outputs/eventsel/nTrueInt_noPU")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    events = uproot.open(args.input)[args.tree].arrays()
    n_ev = len(events)
    logging.info("Loaded %d events from %s", n_ev, args.input)

    for req in ("Pileup_nTrueInt", "full_event_weight", "weight_noPileup"):
        if req not in events.fields:
            raise SystemExit(
                f"Branch '{req}' missing. Re-run event-selection with the updated "
                f"code so weight_noPileup is written."
            )

    nt = np.asarray(ak.to_numpy(events["Pileup_nTrueInt"]), dtype=float)
    w_with = np.asarray(ak.to_numpy(events["full_event_weight"]), dtype=float)
    w_wo = np.asarray(ak.to_numpy(events["weight_noPileup"]), dtype=float)

    bins = np.linspace(args.low, args.high, args.bins + 1)
    cen = 0.5 * (bins[:-1] + bins[1:])

    rm = RegionManager(args.regions_config)
    # empty objects dict -> Region._get_variable_value uses the flat-branch fallback
    region_masks = rm.apply_regions(events, {})

    for region_name, mask in region_masks.items():
        m = np.asarray(ak.to_numpy(mask)).astype(bool)
        if m.sum() == 0:
            logging.warning("Region %s: 0 events, skipping", region_name)
            continue

        h_with, _ = np.histogram(nt[m], bins=bins, weights=w_with[m])
        h_wo, _ = np.histogram(nt[m], bins=bins, weights=w_wo[m])

        fig, (ax, axr) = plt.subplots(
            2, 1, figsize=(9, 8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
        )
        ax.step(cen, h_wo, where="mid", color="black", lw=1.6, label="without PU reweight")
        ax.step(cen, h_with, where="mid", color="green", lw=1.6, label="with PU reweight")
        ax.set_ylabel("weighted events")
        ax.legend()
        ax.set_title(f"Pileup_nTrueInt  [{region_name}]  (N={int(m.sum())})")
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(h_wo > 0, h_with / h_wo, np.nan)
        axr.step(cen, ratio, where="mid", color="green", lw=1.2)
        axr.axhline(1, color="black", ls=":", lw=1)
        axr.set_ylabel("with/without")
        axr.set_xlabel("Pileup_nTrueInt")
        axr.set_ylim(0, 3)
        plt.tight_layout()

        safe = region_name.replace(":", "_")
        out = os.path.join(args.outdir, f"nTrueInt_noPU_{safe}.png")
        plt.savefig(out, dpi=150)
        plt.close(fig)
        logging.info("Saved %s (integral with=%.1f without=%.1f)", out, h_with.sum(), h_wo.sum())


if __name__ == "__main__":
    main()
