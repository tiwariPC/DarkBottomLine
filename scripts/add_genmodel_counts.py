#!/usr/bin/env python3
"""Per-mass-point generated weighted events for a multi-point signal sample.

A multi-point signal dataset (e.g. BBDM-2HDMa-5f, 29 mass points flagged by GenModel_* branches)
has one file-level weighted_total_events that covers every point. Each point must be normalised to
its own generated events, so event selection now writes one one-bin TH1 per point named
    weighted_total_events_<full GenModel_... branch name>
next to the file-level histogram. This script computes those counts directly from the NanoAOD
files (sum of sign(genWeight) over the events whose GenModel_<point> flag is set, before any
selection) and adds them to an already-produced (merged) EVENTSELECTION.root, so the signal does
not have to go through event selection again.

Usage:
    python scripts/add_genmodel_counts.py \\
        --filelist data/samplelist/2024/BBDM-2HDMa-5f_TuneCP5_13p6TeV_madgraph-pythia8.txt \\
        --target /eos/.../eventsel_merged/BBDM-2HDMa-5f_..._EVENTSELECTION.root \\
        [--json-out counts.json] [--workers 8] [--dry-run] [--force]

The NanoAOD files are read with uproot (root:// URLs need a valid grid proxy; run on lxplus).
Without --target only the counts are written (--json-out / stdout). The sum over all points is
compared with the target's weighted_total_events as a consistency check.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import uproot

PREFIX = "weighted_total_events_"


def count_file(url: str, tree: str = "Events", step: int = 200_000) -> Tuple[int, float, Dict[str, float]]:
    """(n_events, sum sign(genWeight), {GenModel_<point>: sum sign(genWeight) of its events})."""
    with uproot.open(url) as f:
        t = f[tree]
        gm = sorted(k for k in t.keys() if k.startswith("GenModel_"))
        if "genWeight" not in t.keys():
            raise RuntimeError(f"{url}: no genWeight branch")
        n, total, per = 0, 0.0, {k: 0.0 for k in gm}
        for chunk in t.iterate(["genWeight"] + gm, step_size=step, library="np"):
            sign = np.sign(chunk["genWeight"].astype(float))
            n += len(sign)
            total += float(sign.sum())
            for k in gm:
                per[k] += float(sign[chunk[k].astype(bool)].sum())
    return n, total, per


def read_filelist(path: str) -> List[str]:
    return [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip() and not ln.startswith("#")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--filelist", required=True, help="text file with one NanoAOD file (path or URL) per line")
    ap.add_argument("--target", help="merged EVENTSELECTION.root to add the counters to")
    ap.add_argument("--json-out", help="also write the counts to this JSON file")
    ap.add_argument("--tree", default="Events")
    ap.add_argument("--workers", type=int, default=4, help="files read in parallel (default 4)")
    ap.add_argument("--dry-run", action="store_true", help="count and report, do not modify --target")
    ap.add_argument("--force", action="store_true", help="overwrite counters already present in --target")
    a = ap.parse_args()

    files = read_filelist(a.filelist)
    print(f"{len(files)} NanoAOD files", file=sys.stderr)
    n_tot, w_tot, per_tot = 0, 0.0, {}
    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        for i, (n, w, per) in enumerate(ex.map(lambda u: count_file(u, a.tree), files), 1):
            n_tot += n
            w_tot += w
            for k, v in per.items():
                per_tot[k] = per_tot.get(k, 0.0) + v
            print(f"  [{i}/{len(files)}] {n} events", file=sys.stderr)
    if not per_tot:
        print("ERROR: the input files have no GenModel_* branches (not a multi-point sample)", file=sys.stderr)
        return 2

    s = sum(per_tot.values())
    print(f"events: {n_tot}   sum sign(genWeight): {w_tot:.1f}   sum over {len(per_tot)} mass points: {s:.1f}")
    cnt = np.array(list(per_tot.values()))
    print(f"per point: min {cnt.min():.0f}  max {cnt.max():.0f}  mean {cnt.mean():.1f}"
          f"  ({'equal' if cnt.min() == cnt.max() else 'NOT equal'} across points)")
    if abs(s - w_tot) > 0.5:
        print("WARNING: points do not partition the events (an event in several or in no point)", file=sys.stderr)

    if a.json_out:
        Path(a.json_out).write_text(json.dumps({"n_events": n_tot, "weighted_total_events": w_tot,
                                                "per_genmodel": per_tot}, indent=1))
    if not a.target or a.dry_run:
        return 0

    with uproot.open(a.target) as f:
        have = {str(k).split(";")[0] for k in f.keys()}
        try:
            wte_target = float(f["weighted_total_events"].values()[0])
        except Exception:
            wte_target = None
    if wte_target is not None and abs(wte_target - w_tot) > 0.5:
        print(f"ERROR: target weighted_total_events {wte_target:.1f} != NanoAOD sum {w_tot:.1f}: the filelist "
              "is not the one this file was made from; nothing written", file=sys.stderr)
        return 3
    new = {PREFIX + k: v for k, v in per_tot.items()}
    clash = sorted(set(new) & have)
    if clash and not a.force:
        print(f"ERROR: {len(clash)} counters already in the target (use --force to overwrite)", file=sys.stderr)
        return 4
    edges = np.array([0.0, 1.0])
    with uproot.update(a.target) as f:
        for name, v in new.items():
            f[name] = (np.array([v]), edges)
    print(f"added {len(new)} counters to {a.target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
