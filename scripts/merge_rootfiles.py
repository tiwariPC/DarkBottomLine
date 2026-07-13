#!/usr/bin/env python3
"""Parallel ROOT merger using hadd, skipping zombie/corrupt files.

- Merge ROOT files in multiple input folders (samples) in parallel.
- Each sample's ROOT files (searched recursively) are merged into one output file.
- Per-file corruption checks run in isolated subprocesses (a corrupt file can
  segfault ROOT; the child dies, the parent just records a skip).
- hadd's own -n flag caps how many files hadd opens at once, so there's no
  practical limit on how many files a sample can have.
- If hadd still fails, the offending file(s) are identified from hadd's log,
  dropped, and the merge is retried (bounded attempts).
- Outputs on /eos/ are staged to a local/work directory first and moved into
  place, with an optional byte-budget gate to cap concurrent staging pressure.

Usage:
    # One sample folder -> one output file
    python3 merge_root_hadd.py -i /path/to/dataset_folder -o /path/to/output_dir

    # Multiple sample folders in parallel
    python3 merge_root_hadd.py -i /path/to/datasetA -i /path/to/datasetB -i /path/to/datasetC -o /path/to/output_dir -j 4

    # All subfolders under one parent, each treated as a sample
    python3 merge_root_hadd.py --input-root /path/to/parent_dir -o /path/to/output_dir -j 4

    # Dry run first (prints planned hadd commands, no merging)
    python3 merge_root_hadd.py -i /path/to/dataset_folder -o /path/to/output_dir --dry-run

    # Custom work-dir (staging + hadd logs) and staging cap for /eos/ outputs
    python3 merge_root_hadd.py -i /path/to/dataset_folder -o /eos/path/output_dir --work-dir /path/to/scratch --max-staging-gb 16
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple

_CHECK_SNIPPET = r"""
import sys
import ROOT
ROOT.gErrorIgnoreLevel = ROOT.kFatal
ROOT.gROOT.SetBatch(True)
ROOT.gSystem.ResetSignals()

path = sys.argv[1]
f = None

def fail(reason):
    print(reason)
    if f:
        try: f.Close()
        except: pass
    sys.exit(1)

try:
    f = ROOT.TFile.Open(path, "READ")
except Exception as exc:
    fail(f"open-exception: {exc}")
if not f or f.IsZombie():
    fail("open-failed-or-zombie")

try:
    if f.TestBit(ROOT.TFile.kRecovered):
        fail("recovered")
    if f.GetNkeys() == 0:
        fail("no-keys")
    for key in f.GetListOfKeys():
        obj = key.ReadObj()
        if isinstance(obj, ROOT.TTree):
            n = obj.GetEntries()
            if n > 0:
                if obj.GetEntry(0) <= 0 or obj.GetEntry(n - 1) <= 0:
                    fail(f"unreadable-entries:{obj.GetName()}")
        if obj:
            obj.Delete()
except SystemExit:
    raise
except Exception as exc:
    fail(f"read-exception: {exc}")
finally:
    if f:
        f.Close()
sys.exit(0)
"""

MAX_HADD_ATTEMPTS = 4


@dataclass
class MergeTask:
    folder: Path
    group_key: str
    files: List[str]
    output_root: Path


@dataclass
class MergeResult:
    task: MergeTask
    status: str  # "ok", "skip", "fail"
    message: str
    corrupt_entries: List[str]


class _ByteBudget:
    """Simple byte budget gate to cap concurrent staging pressure."""

    def __init__(self, total_bytes: int):
        if total_bytes <= 0:
            raise ValueError("total_bytes must be positive")
        self._total = total_bytes
        self._available = total_bytes
        self._cond = threading.Condition()

    def acquire(self, requested_bytes: int) -> int:
        need = max(1, min(requested_bytes, self._total))
        with self._cond:
            while self._available < need:
                self._cond.wait()
            self._available -= need
        return need

    def release(self, used_bytes: int) -> None:
        with self._cond:
            self._available = min(self._total, self._available + max(0, used_bytes))
            self._cond.notify_all()


def _check_hadd_exists() -> None:
    if shutil.which("hadd") is None:
        raise RuntimeError("Cannot find 'hadd' in PATH. Please load ROOT environment first.")


def _is_eos_path(path: Path) -> bool:
    return path.as_posix().startswith("/eos/")


def check_file(path: str, timeout: float) -> Tuple[str, Optional[str]]:
    """Probe one file in an isolated child process. Returns (path, reason|None)."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _CHECK_SNIPPET, path],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return path, f"timeout-{timeout:g}s"
    if proc.returncode == 0:
        return path, None
    reason = proc.stdout.strip().splitlines()
    if reason:
        return path, reason[0]
    if proc.returncode < 0:
        return path, f"crash-signal{-proc.returncode}"
    return path, f"failed-exit{proc.returncode}"


def find_offenders(hadd_log: Path, inputs: Sequence[str]) -> Set[str]:
    offenders: Set[str] = set()
    try:
        text = hadd_log.read_text(errors="replace")
    except OSError:
        return offenders
    input_set = set(inputs)
    for line in text.splitlines():
        if any(w in line for w in ("Error", "SysError", "Fatal")):
            for p in input_set:
                if p in line:
                    offenders.add(p)
    return offenders


def _resolve_input_dirs(input_dirs: Sequence[str], input_root: Optional[str]) -> List[Path]:
    if input_dirs:
        return [Path(p).expanduser().resolve() for p in input_dirs]

    if input_root is None:
        raise ValueError("Provide either --input-dir (one or more) or --input-root.")

    root = Path(input_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"input-root does not exist: {root}")

    return sorted(p for p in root.iterdir() if p.is_dir())


def _collect_tasks(input_dirs: Sequence[Path], output_dir: Path) -> List[MergeTask]:
    tasks: List[MergeTask] = []
    for in_dir in input_dirs:
        if not in_dir.is_dir():
            raise FileNotFoundError(f"Input directory does not exist: {in_dir}")

        files = sorted(str(p) for p in in_dir.rglob("*.root") if p.is_file())
        if not files:
            print(f"[warn] {in_dir.name}: no .root files, skipping")
            continue

        tasks.append(
            MergeTask(
                folder=in_dir,
                group_key=in_dir.name,
                files=files,
                output_root=output_dir / f"{in_dir.name}.root",
            )
        )
    return tasks


def _merge_one_task(
    task: MergeTask,
    dry_run: bool,
    batch: int,
    check_timeout: float,
    check_pool: ThreadPoolExecutor,
    work_dir: Path,
    staging_budget: Optional[_ByteBudget],
) -> MergeResult:
    out = task.output_root
    if not dry_run and out.exists() and out.stat().st_size > 0:
        return MergeResult(task, "skip", f"{task.group_key} (output exists)", [])

    results = list(check_pool.map(lambda p: check_file(p, check_timeout), task.files))
    good = [p for p, reason in results if reason is None]
    bad = [(p, reason) for p, reason in results if reason is not None]
    corrupt_entries = [f"BAD {p} : {reason}" for p, reason in bad]

    skip_note = f", skipped={len(bad)} bad file(s)" if bad else ""
    if not good:
        return MergeResult(task, "fail", f"{task.group_key}: all files corrupt", corrupt_entries)

    if dry_run:
        cmd = ["hadd", "-f", "-n", str(batch), str(out)] + good
        print("[DRY-RUN]", " ".join(cmd))
        return MergeResult(task, "ok", f"{task.group_key} -> {out} ({len(good)} files{skip_note}) [dry-run]", corrupt_entries)

    needs_staging = _is_eos_path(out)
    budget_used = 0
    temp_dir_cm = None
    if needs_staging:
        input_bytes = sum(Path(p).stat().st_size for p in good)
        if staging_budget is not None:
            budget_used = staging_budget.acquire(input_bytes)
        temp_dir_cm = tempfile.TemporaryDirectory(prefix="merge_hadd_", dir=str(work_dir))
        staging_dir = Path(temp_dir_cm.name)
        merge_target = staging_dir / out.name
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        merge_target = out

    try:
        tmp = merge_target.with_suffix(".root.tmp")
        hadd_log = work_dir / f"{task.group_key}.hadd.log"
        rc = 1
        for attempt in range(1, MAX_HADD_ATTEMPTS + 1):
            cmd = ["hadd", "-f", "-n", str(batch), str(tmp), *good]
            with open(hadd_log, "w") as log:
                rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
            if rc == 0:
                break
            offenders = find_offenders(hadd_log, good)
            if not offenders or len(offenders) == len(good):
                break
            for p in offenders:
                corrupt_entries.append(f"BAD {p} : hadd-error")
            good = [p for p in good if p not in offenders]

        if rc != 0 or not good:
            tmp.unlink(missing_ok=True)
            return MergeResult(task, "fail", f"{task.group_key}: hadd failed, see {hadd_log}", corrupt_entries)

        merge_target.parent.mkdir(parents=True, exist_ok=True)
        tmp.replace(merge_target)
        hadd_log.unlink(missing_ok=True)

        if needs_staging:
            out.parent.mkdir(parents=True, exist_ok=True)
            if merge_target.stat().st_dev == out.parent.stat().st_dev:
                merge_target.replace(out)
            else:
                shutil.copy2(merge_target, out)
                merge_target.unlink(missing_ok=True)

        return MergeResult(task, "ok", f"{task.group_key} -> {out} ({len(good)} files{skip_note})", corrupt_entries)
    finally:
        if temp_dir_cm is not None:
            temp_dir_cm.cleanup()
        if budget_used and staging_budget is not None:
            staging_budget.release(budget_used)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "-i", "--input-dir", action="append", default=[],
        help="Input directory containing ROOT files (searched recursively). Can be used multiple times.",
    )
    parser.add_argument(
        "--input-root", default=None,
        help="Parent directory containing multiple sample folders (used if --input-dir is omitted).",
    )
    parser.add_argument("-o", "--output-dir", required=True, help="Output directory for merged ROOT files.")
    parser.add_argument(
        "-j", "--workers", type=int, default=max(1, min(4, (os.cpu_count() or 1))),
        help="Samples merged in parallel.",
    )
    parser.add_argument(
        "--check-workers", type=int, default=8,
        help="Parallel per-file corruption-check subprocesses (default: 8).",
    )
    parser.add_argument(
        "--check-timeout", type=float, default=300.0,
        help="Timeout in seconds for each per-file corruption check (default: 300).",
    )
    parser.add_argument(
        "--batch", type=int, default=100,
        help="hadd -n value: max files hadd opens at once (default: 100). "
             "hadd still merges every good file in one call; this just bounds open file handles.",
    )
    parser.add_argument(
        "--work-dir", default=None,
        help="Directory used for temporary staging (EOS outputs) and hadd logs. "
             "Defaults to a merge_root_hadd subdirectory of the system temp directory.",
    )
    parser.add_argument(
        "--max-staging-gb", type=float, default=8.0,
        help="Cap total concurrent temporary staging size in GB for /eos/ outputs (default: 8). Set <=0 to disable.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print planned hadd commands.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    _check_hadd_exists()

    input_dirs = _resolve_input_dirs(args.input_dir, args.input_root)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = (
        Path(args.work_dir).expanduser().resolve()
        if args.work_dir
        else Path(tempfile.gettempdir()) / "merge_root_hadd"
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    staging_budget = _ByteBudget(int(args.max_staging_gb * 1024**3)) if args.max_staging_gb > 0 else None

    tasks = _collect_tasks(input_dirs, output_dir)
    if not tasks:
        print("No ROOT files found to merge.")
        return 0

    print(f"Planned merge tasks: {len(tasks)}")
    for t in tasks:
        print(f"  - {t.folder.name}/{t.group_key}: {len(t.files)} files")

    corrupt_log = Path.cwd() / "corrupt_files.log"
    all_corrupt_entries: List[str] = []
    n_ok = n_skip = n_fail = 0

    with ThreadPoolExecutor(max_workers=max(1, args.check_workers)) as check_pool, \
         ThreadPoolExecutor(max_workers=max(1, args.workers)) as sample_pool:
        future_map = {
            sample_pool.submit(
                _merge_one_task, task, args.dry_run, args.batch, args.check_timeout,
                check_pool, work_dir, staging_budget,
            ): task
            for task in tasks
        }
        for fut in as_completed(future_map):
            task = future_map[fut]
            try:
                result = fut.result()
            except Exception as exc:
                print(f"[FAIL] {task.folder.name}/{task.group_key}: {exc}")
                n_fail += 1
                continue

            all_corrupt_entries.extend(result.corrupt_entries)
            if result.status == "ok":
                print(f"[OK] {result.message}")
                n_ok += 1
            elif result.status == "skip":
                print(f"[skip] {result.message}")
                n_skip += 1
            else:
                print(f"[FAIL] {result.message}")
                n_fail += 1

    if all_corrupt_entries:
        corrupt_log.write_text("\n".join(all_corrupt_entries) + "\n")
    elif corrupt_log.exists():
        corrupt_log.unlink()

    print(f"\nSummary: {n_ok} merged, {n_skip} skipped, {n_fail} failed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())