#!/usr/bin/env python3
"""
Compute the EFFECTIVE processed luminosity for event-selection-level MC
normalization, from the (run, luminosityBlock) pairs actually present in
processed data EVENTSELECTION.root files, via brilcalc.

This is deliberately NOT the same as the full golden-JSON luminosity (the
number usually quoted for an analysis, e.g. in a production README). A
lumisection only shows up here if at least one event from it survived the
FULL event-selection pipeline (trigger, object/event cuts — not just the
golden-JSON mask) and got written to EVENTSELECTION.root. Lumisections with
zero surviving events are silently absent, so this number comes out slightly
*below* the full golden-JSON lumi (observed ~1% lower on the 2024 dataset).
That's expected and correct for this script's purpose: it answers "how much
luminosity did the data that actually reached the selection contribute,"
which is the right normalization denominator when cross-checking MC-to-data
scaling at the event-selection level — not "what's the official recorded
luminosity for this run range" (use the golden-JSON file directly for that).

The analysis's golden-JSON lumi mask (DarkBottomLineProcessor.apply_lumi_mask)
is applied before events are saved to EVENTSELECTION.root, so the run/
luminosityBlock pairs found there are already golden-JSON-filtered on top of
the event-selection-survival filtering described above — this script just
extracts the unique pairs, writes a brilcalc-compatible JSON, and calls
brilcalc per dataset (EGamma, JetMET). Results are written to a single
brilcalc_summary.txt (one "=== <stream> ===" section per dataset, keeping
only brilcalc's #Summary block — not the full --byls per-lumisection table).

Usage:
    python scripts/lumi_calculator.py \
        --input-dir /eos/cms/store/group/phys_susy/sus-23-008/run3_bbMET/goldenfix_642f62c/03102026/2024/results_55p214fb/eventsel \
        --output-dir outputs/lumi/2024
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import uproot

NORMTAG = "/cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags/normtag_PHYSICS.json"

# Which filename substrings belong to which trigger stream (same split used
# by the analysis: EGamma -> electron regions, JetMET -> muon/SR regions).
STREAMS = {
    "EGamma": "EGamma",
    "JetMET": "JetMET",
}


def find_data_files(input_dir: str) -> Dict[str, List[str]]:
    """Group EVENTSELECTION.root files under input_dir by data stream."""
    by_stream: Dict[str, List[str]] = defaultdict(list)
    for fname in sorted(os.listdir(input_dir)):
        if not fname.endswith(".root"):
            continue
        for stream, tag in STREAMS.items():
            if tag in fname:
                by_stream[stream].append(os.path.join(input_dir, fname))
                break
    return by_stream


def unique_run_lumis(root_files: List[str]) -> Dict[int, List[int]]:
    """Read run/luminosityBlock from each file, return {run: sorted unique lumis}."""
    run_lumis: Dict[int, set] = defaultdict(set)
    for fn in root_files:
        tree = uproot.open(fn)["Events"]
        arrs = tree.arrays(["run", "luminosityBlock"], library="np")
        runs = arrs["run"]
        lumis = arrs["luminosityBlock"]
        for r in np.unique(runs):
            mask = runs == r
            run_lumis[int(r)].update(int(x) for x in lumis[mask])
    return {r: sorted(ls) for r, ls in run_lumis.items()}


def lumis_to_ranges(lumis: List[int]) -> List[List[int]]:
    """Collapse a sorted list of lumi-section numbers into contiguous [first, last] ranges."""
    if not lumis:
        return []
    ranges = []
    start = prev = lumis[0]
    for x in lumis[1:]:
        if x == prev + 1:
            prev = x
            continue
        ranges.append([start, prev])
        start = prev = x
    ranges.append([start, prev])
    return ranges


def write_brilcalc_json(run_lumis: Dict[int, List[int]], out_path: str) -> None:
    payload = {str(r): lumis_to_ranges(ls) for r, ls in sorted(run_lumis.items())}
    with open(out_path, "w") as f:
        json.dump(payload, f)


BRILWS_ENV = "/cvmfs/cms-bril.cern.ch/cms-lumi-pog/brilws-docker/brilws-env"


def run_brilcalc(json_path: str, normtag: str) -> Tuple[float, List[str]]:
    """Run brilcalc on json_path; return (totrecorded /fb, summary-block lines).

    brilcalc is a shell `alias` defined by sourcing brilws-env (it wraps a
    `singularity exec ...brilws-docker... brilcalc` call, and brilws-env also
    exports SINGULARITY_BINDPATH/APPTAINER_BINDPATH so /eos, /afs, /cvmfs are
    visible inside the container). Aliases don't propagate to subshells, so
    this sources brilws-env and sets the alias within the SAME script rather
    than relying on it already being present in the inherited environment.
    """
    quoted_normtag = shlex.quote(normtag)
    quoted_json = shlex.quote(json_path)
    # A single `-c` string doesn't reliably expand an alias defined by the
    # `source` on the same line (alias expansion happens at parse time, which
    # can run ahead of the source taking effect) — write a real script file
    # instead, which bash parses line-by-line, so the alias from `source` is
    # visible by the time the brilcalc line is reached.
    script_lines = [
        "#!/bin/bash",
        "shopt -s expand_aliases",
        f"source {shlex.quote(BRILWS_ENV)}",
        (
            f'brilcalc lumi -b "STABLE BEAMS" --byls --normtag {quoted_normtag} '
            f'-u /fb -i {quoted_json}'
        ),
    ]
    script_path = json_path + ".sh"
    with open(script_path, "w") as f:
        f.write("\n".join(script_lines) + "\n")
    try:
        result = subprocess.run(
            ["bash", script_path], capture_output=True, text=True,
        )
    finally:
        os.remove(script_path)
    full_output = result.stdout + result.stderr
    if result.returncode != 0:
        raise RuntimeError(f"brilcalc failed (exit {result.returncode}):\n{full_output}")

    lines = full_output.splitlines(keepends=True)
    # Keep only from "#Summary:" onward (the summary table + any
    # merge-warning lines after it) — drop the --byls per-lumisection table,
    # which is huge and not what we want saved.
    summary_start = next(
        (i for i, line in enumerate(lines) if line.startswith("#Summary:")), None,
    )
    kept_lines = lines[summary_start:] if summary_start is not None else lines

    # Summary line looks like: "| nfill | nrun | nls | ncms | totdelivered(/fb) | totrecorded(/fb) |"
    for line in reversed(kept_lines):
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 3 and parts[1].isdigit():
            return float(parts[-2]), kept_lines
    raise RuntimeError(f"Could not parse brilcalc summary:\n{''.join(kept_lines)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True,
                         help="Directory of merged *_EVENTSELECTION.root data files")
    parser.add_argument("--output-dir", default="outputs/lumi",
                         help="Where to write the per-stream JSON and the combined brilcalc_summary.txt")
    parser.add_argument("--normtag", default=NORMTAG, help="brilcalc normtag JSON path")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    by_stream = find_data_files(args.input_dir)
    if not by_stream:
        print(f"No EGamma/JetMET EVENTSELECTION.root files found under {args.input_dir}")
        sys.exit(1)

    per_stream_run_lumis: Dict[str, Dict[int, List[int]]] = {}
    for stream, files in by_stream.items():
        print(f"{stream}: {len(files)} file(s)")
        for f in files:
            print(f"  {os.path.basename(f)}")
        per_stream_run_lumis[stream] = unique_run_lumis(files)

    totals: Dict[str, float] = {}
    summary_path = os.path.join(args.output_dir, "brilcalc_summary.txt")
    with open(summary_path, "w") as summary_file:
        for stream, run_lumis in per_stream_run_lumis.items():
            json_path = os.path.join(args.output_dir, f"{stream}_processedLumis.json")
            write_brilcalc_json(run_lumis, json_path)
            print(f"\nRunning brilcalc for {stream}...")
            lumi, summary_lines = run_brilcalc(json_path, args.normtag)
            totals[stream] = lumi
            print(f"  {stream}: {lumi:.6f} /fb")

            summary_file.write(f"=== {stream} ===\n")
            summary_file.writelines(summary_lines)
            summary_file.write("\n")

    print("\n=== Summary ===")
    for stream, lumi in totals.items():
        print(f"{stream:10s} {lumi:.6f} /fb")


if __name__ == "__main__":
    main()
