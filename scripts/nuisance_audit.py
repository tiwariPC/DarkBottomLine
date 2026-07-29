#!/usr/bin/env python3
"""Compare a datacard's declared nuisances (lnN/shape) against the nuisances
that actually appear in Impacts output (impacts.json), per mass point.

rateParam lines are reported separately (not in [1..k] `kmax`, not expected
in impacts.json unless explicitly floated as a POI/nuisance by combineTool.py).

Usage:
    python3 scripts/nuisance_audit.py --combine-dir outputs/combine/2024/C \\
        [--mass-points MH3_600_MH4_150_Mchi_1 ...] [--csv report.csv]
"""

import argparse
import csv
import glob
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

NUISANCE_RE = re.compile(r"^(\S+)\s+(lnN|shape\S*)\s")
RATEPARAM_RE = re.compile(r"^(\S+)\s+rateParam\s")


def find_datacard(mass_point_dir: Path) -> Path:
    candidates = sorted(mass_point_dir.glob("datacard*.txt"))
    if not candidates:
        raise FileNotFoundError(f"No datacard*.txt found in {mass_point_dir}")
    return candidates[0]


def find_impacts_json(mass_point_dir: Path) -> Path:
    candidates = sorted(mass_point_dir.glob("impacts.json")) or sorted(mass_point_dir.glob("*impacts*.json"))
    if not candidates:
        raise FileNotFoundError(f"No impacts.json found in {mass_point_dir}")
    return candidates[0]


def parse_datacard_nuisances(datacard: Path) -> Tuple[List[str], List[str]]:
    """Return (nuisance_names, rateparam_names) in file order, deduplicated."""
    nuisances: List[str] = []
    rateparams: List[str] = []
    seen_n: Set[str] = set()
    seen_r: Set[str] = set()
    with open(datacard) as f:
        for line in f:
            m = NUISANCE_RE.match(line)
            if m:
                name = m.group(1)
                if name not in seen_n:
                    seen_n.add(name)
                    nuisances.append(name)
                continue
            m = RATEPARAM_RE.match(line)
            if m:
                name = m.group(1)
                if name not in seen_r:
                    seen_r.add(name)
                    rateparams.append(name)
    return nuisances, rateparams


def parse_impacts_names(impacts_json: Path) -> Set[str]:
    with open(impacts_json) as f:
        data = json.load(f)
    return {p["name"] for p in data.get("params", [])}


def audit_mass_point(mass_point_dir: Path) -> Dict:
    mass_point = mass_point_dir.name
    datacard = find_datacard(mass_point_dir)
    impacts_json = find_impacts_json(mass_point_dir)

    nuisances, rateparams = parse_datacard_nuisances(datacard)
    impact_names = parse_impacts_names(impacts_json)

    rows = []
    for name in nuisances:
        rows.append({
            "mass_point": mass_point,
            "name": name,
            "kind": "nuisance",
            "in_datacard": True,
            "in_impacts": name in impact_names,
        })
    for name in rateparams:
        rows.append({
            "mass_point": mass_point,
            "name": name,
            "kind": "rateParam",
            "in_datacard": True,
            "in_impacts": name in impact_names,
        })
    extra = impact_names - set(nuisances) - set(rateparams) - {"r"}
    for name in sorted(extra):
        rows.append({
            "mass_point": mass_point,
            "name": name,
            "kind": "impacts_only",
            "in_datacard": False,
            "in_impacts": True,
        })
    return {
        "mass_point": mass_point,
        "datacard": str(datacard),
        "impacts_json": str(impacts_json),
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--combine-dir", required=True,
                     help="Dir containing {mass_point}/datacard*.txt + impacts.json (e.g. outputs/combine/2024/C)")
    ap.add_argument("--mass-points", nargs="+", help="Restrict to specific mass points (default: all found)")
    ap.add_argument("--csv", help="Write full per-nuisance rows to this CSV path")
    args = ap.parse_args()

    combine_dir = Path(args.combine_dir)
    if args.mass_points:
        mp_dirs = [combine_dir / mp for mp in args.mass_points]
    else:
        mp_dirs = sorted(d for d in combine_dir.iterdir() if d.is_dir())

    all_results = []
    errors = []
    for mp_dir in mp_dirs:
        try:
            all_results.append(audit_mass_point(mp_dir))
        except FileNotFoundError as e:
            errors.append(f"{mp_dir.name}: {e}")

    if errors:
        print(f"Skipped {len(errors)} mass point(s):", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        print(file=sys.stderr)

    if not all_results:
        print("No mass points with both a datacard and impacts.json were found.", file=sys.stderr)
        sys.exit(1)

    # Summary table: one row per mass point
    header = f"{'mass_point':<32} {'nuisances':>10} {'in_impacts':>11} {'missing':>8} {'rateParams':>11} {'extra_in_impacts':>17}"
    print(header)
    print("-" * len(header))

    all_missing = []
    for result in all_results:
        rows = result["rows"]
        n_rows = [r for r in rows if r["kind"] == "nuisance"]
        rp_rows = [r for r in rows if r["kind"] == "rateParam"]
        extra_rows = [r for r in rows if r["kind"] == "impacts_only"]
        n_total = len(n_rows)
        n_in = sum(1 for r in n_rows if r["in_impacts"])
        n_missing = [r["name"] for r in n_rows if not r["in_impacts"]]
        print(f"{result['mass_point']:<32} {n_total:>10} {n_in:>11} {len(n_missing):>8} "
              f"{len(rp_rows):>11} {len(extra_rows):>17}")
        for name in n_missing:
            all_missing.append((result["mass_point"], name))

    print()
    if all_missing:
        print(f"MISSING nuisances (declared in datacard, absent from impacts.json) — {len(all_missing)} total:")
        print(f"{'mass_point':<32} {'nuisance':<30}")
        print("-" * 62)
        for mp, name in all_missing:
            print(f"{mp:<32} {name:<30}")
    else:
        print("No missing nuisances — every datacard nuisance appears in impacts.json for all mass points checked.")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["mass_point", "name", "kind", "in_datacard", "in_impacts"])
            writer.writeheader()
            for result in all_results:
                writer.writerows(result["rows"])
        print(f"\nFull per-nuisance report written to {args.csv}")


if __name__ == "__main__":
    main()
