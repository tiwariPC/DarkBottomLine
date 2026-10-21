#!/usr/bin/env python3
"""Switch the 2024 V+jets samples between pT-binned and inclusive.

    python3 scripts/switch_sample_mode.py status
    python3 scripts/switch_sample_mode.py inclusive [--dry-run]
    python3 scripts/switch_sample_mode.py binned    [--dry-run]

Moves sample lists in configs/sample_modes.yaml's `managed` families between
data/samplelist/<year>/ (active) and data/samplelist/backup/<year>/ (inactive), and
sets `sample_mode:` in configs/plotting.yaml. Refuses to move a sample that has jobs in the
HTCondor queue (the jobs read the sample list while they run).
"""
import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
CFG = REPO / "configs" / "sample_modes.yaml"
PLOT_CFG = REPO / "configs" / "plotting.yaml"


def short(name: str) -> str:
    return re.sub(r"_TuneCP5.*", "", name)


def tracked(path: Path) -> bool:
    return subprocess.run(["git", "ls-files", "--error-unmatch", str(path)], cwd=REPO,
                          capture_output=True).returncode == 0


def move(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if tracked(src):
        subprocess.run(["git", "mv", str(src), str(dst)], cwd=REPO, check=True)
    else:
        shutil.move(str(src), str(dst))


def queued_batches() -> set:
    try:
        out = subprocess.run(["condor_q", "-af", "JobBatchName"], capture_output=True, text=True,
                             timeout=120).stdout
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: could not query condor_q ({exc}); not checking for running jobs")
        return set()
    return {l.strip() for l in out.splitlines() if l.strip()}


def current_mode() -> str:
    m = re.search(r"^sample_mode:\s*(\w+)", PLOT_CFG.read_text(), flags=re.M)
    return m.group(1) if m else "binned"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["status", "binned", "inclusive"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-plotting-config", action="store_true", help="do not edit configs/plotting.yaml")
    a = ap.parse_args()

    cfg = yaml.safe_load(CFG.read_text())
    year = str(cfg["year"])
    active_dir = REPO / "data" / "samplelist" / year
    backup_dir = REPO / "data" / "samplelist" / "backup" / year
    managed = cfg["managed"]

    files = {}  # short name -> (path, is_active)
    for d, is_active in ((active_dir, True), (backup_dir, False)):
        for p in sorted(d.glob("*.txt")):
            if any(short(p.stem).startswith(m) for m in managed):
                files[short(p.stem)] = (p, is_active)

    if a.mode == "status":
        print(f"plotting.yaml sample_mode: {current_mode()}")
        for name in sorted(files):
            print(f"  {'active  ' if files[name][1] else 'inactive'}  {name}")
        for mode, spec in cfg["modes"].items():
            have = {n for n, (_, act) in files.items() if act}
            print(f"matches '{mode}' sample lists: {have == set(spec['active'])}")
        return 0

    want = set(cfg["modes"][a.mode]["active"])
    missing = want - set(files)
    if missing:
        print("ERROR: sample lists missing from both directories:", *sorted(missing), sep="\n  ")
        return 1

    plan = [(n, p, backup_dir / p.name if act else active_dir / p.name)
            for n, (p, act) in sorted(files.items())
            if (n in want) != act]
    busy = queued_batches()
    blocked = [n for n, p, _ in plan if any(short(p.stem) in b for b in busy)]
    if blocked:
        print("ERROR: these samples have jobs in the HTCondor queue, not moving them:", *blocked, sep="\n  ")
        return 1

    for n, src, dst in plan:
        print(f"{'would move' if a.dry_run else 'moving'}: {src.relative_to(REPO)} -> {dst.relative_to(REPO)}")
        if not a.dry_run:
            move(src, dst)
    if not plan:
        print(f"sample lists already in '{a.mode}' mode")
    if not a.no_plotting_config:
        txt = PLOT_CFG.read_text()
        new, n = re.subn(r"^sample_mode:.*$", f"sample_mode: {a.mode}", txt, count=1, flags=re.M)
        if n == 0:
            print("WARNING: no sample_mode: line in configs/plotting.yaml")
        elif not a.dry_run:
            PLOT_CFG.write_text(new)
        print(f"{'would set' if a.dry_run else 'set'} sample_mode: {a.mode} in configs/plotting.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
