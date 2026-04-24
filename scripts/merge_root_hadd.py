#!/usr/bin/env python3
"""Parallel ROOT merger using hadd.

Features:
- Merge ROOT files in multiple input folders in parallel.
- Group files by common prefix up to a marker token (default: NANOAODSIM).
- Write merged ROOT files to a configurable output directory.
- Export a sidecar JSON with summed numeric values from Metadata tree.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


UUID_SUFFIX_RE = re.compile(
    r"-(?:\d+)-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


@dataclass
class MergeTask:
    folder: Path
    group_key: str
    files: List[Path]
    input_bytes: int
    output_root: Path
    output_meta_json: Path


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


def _resolve_group_key(file_path: Path, marker: str) -> str:
    stem = file_path.stem
    idx = stem.find(marker)
    if idx >= 0:
        return stem[: idx + len(marker)]

    # Fallback: remove known chunk+uuid suffix, otherwise merge all unnamed chunks in folder.
    no_suffix = UUID_SUFFIX_RE.sub("", stem)
    return no_suffix if no_suffix != stem else "ALL_FILES"


def _collect_tasks(input_dirs: Sequence[Path], output_dir: Path, marker: str) -> List[MergeTask]:
    tasks: List[MergeTask] = []
    for in_dir in input_dirs:
        if not in_dir.is_dir():
            raise FileNotFoundError(f"Input directory does not exist: {in_dir}")

        root_files = sorted(p for p in in_dir.iterdir() if p.is_file() and p.suffix == ".root")
        if not root_files:
            continue

        groups: Dict[str, List[Path]] = {}
        for f in root_files:
            key = _resolve_group_key(f, marker)
            groups.setdefault(key, []).append(f)

        sub_out = output_dir / in_dir.name
        sub_out.mkdir(parents=True, exist_ok=True)
        for key, files in sorted(groups.items()):
            input_bytes = sum((f.stat().st_size for f in files), 0)
            out_root = sub_out / f"{key}.root"
            out_json = sub_out / f"{key}.metadata_sum.json"
            tasks.append(
                MergeTask(
                    folder=in_dir,
                    group_key=key,
                    files=sorted(files),
                    input_bytes=input_bytes,
                    output_root=out_root,
                    output_meta_json=out_json,
                )
            )
    return tasks


def _hadd_merge(task: MergeTask, dry_run: bool = False) -> None:
    cmd = ["hadd", "-f", "-k", str(task.output_root)] + [str(p) for p in task.files]
    if dry_run:
        print("[DRY-RUN]", " ".join(cmd))
        return

    task.output_root.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"hadd failed for {task.output_root.name} (folder={task.folder.name}):\n{proc.stdout}"
        )


def _probe_hadd_merge(files: Sequence[Path], workdir: Path, label: str) -> bool:
    probe_root = workdir / f"probe_{label}.root"
    probe_task = MergeTask(
        folder=workdir,
        group_key=label,
        files=list(files),
        input_bytes=sum((p.stat().st_size for p in files), 0),
        output_root=probe_root,
        output_meta_json=workdir / f"probe_{label}.json",
    )
    try:
        _hadd_merge(probe_task, dry_run=False)
        return True
    except Exception:
        return False
    finally:
        try:
            if probe_root.exists():
                probe_root.unlink()
        except Exception:
            pass
        try:
            probe_json = probe_task.output_meta_json
            if probe_json.exists():
                probe_json.unlink()
        except Exception:
            pass


def _find_mergeable_files(files: Sequence[Path], workdir: Path, label: str) -> Tuple[List[Path], List[Path]]:
    """Return (good_files, skipped_files) by isolating inputs that break hadd."""
    ordered = list(files)
    if not ordered:
        return [], []

    if _probe_hadd_merge(ordered, workdir, label):
        return ordered, []

    if len(ordered) == 1:
        return [], ordered

    mid = len(ordered) // 2
    left_good, left_bad = _find_mergeable_files(ordered[:mid], workdir, f"{label}_L")
    right_good, right_bad = _find_mergeable_files(ordered[mid:], workdir, f"{label}_R")
    return left_good + right_good, left_bad + right_bad


def _sum_metadata_to_json(task: MergeTask) -> Dict[str, float]:
    try:
        import uproot  # type: ignore
    except Exception:
        return {}

    sums: Dict[str, float] = {}
    files_seen = 0
    for file_path in task.files:
        try:
            with uproot.open(file_path) as f:
                if "Metadata" not in f:
                    continue
                tree = f["Metadata"]
                for branch_name in tree.keys():
                    arr = tree[branch_name].array(library="np")
                    if not hasattr(arr, "dtype"):
                        continue
                    if not getattr(arr.dtype, "kind", "") in {"i", "u", "f", "b"}:
                        continue
                    sums[branch_name] = sums.get(branch_name, 0.0) + float(arr.sum())
                files_seen += 1
        except Exception:
            continue

    payload = {
        "group": task.group_key,
        "input_folder": str(task.folder),
        "input_files": [str(p) for p in task.files],
        "metadata_files_seen": files_seen,
        "sums": sums,
    }
    task.output_meta_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return sums


def _is_eos_path(path: Path) -> bool:
    return path.as_posix().startswith("/eos/")


def _copy_artifacts(src_root: Path, dst_root: Path, src_json: Path, dst_json: Path) -> None:
    dst_root.parent.mkdir(parents=True, exist_ok=True)
    # Use rename on same filesystem to avoid a second full-size copy.
    if src_root.stat().st_dev == dst_root.parent.stat().st_dev:
        src_root.replace(dst_root)
    else:
        shutil.copy2(src_root, dst_root)
        src_root.unlink(missing_ok=True)
    if src_json.exists():
        dst_json.parent.mkdir(parents=True, exist_ok=True)
        if src_json.stat().st_dev == dst_json.parent.stat().st_dev:
            src_json.replace(dst_json)
        else:
            shutil.copy2(src_json, dst_json)
            src_json.unlink(missing_ok=True)


def _find_recoil_branch(branches: Sequence[str], preferred: str) -> str | None:
    # uproot keys may include titles; keep only raw branch names for matching.
    raw = [str(b) for b in branches]
    lower_map = {b.lower(): b for b in raw}

    pref = preferred.strip().lower()
    if pref in lower_map:
        return lower_map[pref]

    # Exact aliases commonly seen in analyses.
    for alias in ("recoil", "recoil_pt", "pfmet_pt", "met_pt"):
        if alias in lower_map:
            return lower_map[alias]

    # Fallback: first branch containing requested token or recoil token.
    for b in raw:
        bl = b.lower()
        if pref and pref in bl:
            return b
    for b in raw:
        if "recoil" in b.lower():
            return b
    return None


def _apply_recoil_filter(output_root: Path, recoil_min: float, recoil_branch: str) -> Tuple[int, int, str | None]:
    """Filter merged Events tree in-place: keep events with recoil >= recoil_min.

    Returns: (events_before, events_after, matched_branch_name)
    """
    try:
        import awkward as ak  # type: ignore
        import uproot  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Recoil filtering requested but required Python packages are missing (uproot, awkward)."
        ) from exc

    tmp_output = output_root.with_suffix(".tmp.root")
    events_before = 0
    events_after = 0
    used_branch: str | None = None

    with uproot.open(output_root) as f_in:
        if "Events" not in f_in:
            return 0, 0, None

        events_tree = f_in["Events"]
        used_branch = _find_recoil_branch(events_tree.keys(), recoil_branch)
        if used_branch is None:
            return events_tree.num_entries, events_tree.num_entries, None

        recoil_arr = events_tree[used_branch].array(library="ak")
        mask = ak.fill_none(recoil_arr >= float(recoil_min), False)
        events_before = int(events_tree.num_entries)
        events_after = int(ak.sum(mask))

        with uproot.recreate(tmp_output) as f_out:
            for key, classname in f_in.classnames().items():
                obj_name = str(key).split(";")[0]
                if not classname.startswith("TTree"):
                    continue

                tree = f_in[obj_name]
                arrays = tree.arrays(library="ak")
                if obj_name == "Events":
                    filtered = {name: arr[mask] for name, arr in arrays.items()}
                    f_out[obj_name] = filtered
                else:
                    f_out[obj_name] = arrays

    tmp_output.replace(output_root)
    return events_before, events_after, used_branch


def _run_one_task(
    task: MergeTask,
    dry_run: bool = False,
    recoil_min: float | None = None,
    recoil_branch: str = "recoil",
    work_dir: Path | None = None,
    staging_budget: _ByteBudget | None = None,
) -> str:
    needs_staging = _is_eos_path(task.output_root) or _is_eos_path(task.output_meta_json)
    working_task = task
    temp_dir_cm = None
    filter_note = ""
    skip_note = ""
    budget_used = 0

    if not dry_run:
        if needs_staging and staging_budget is not None:
            budget_used = staging_budget.acquire(task.input_bytes)
        staging_base = work_dir if work_dir is not None else Path("/eos/home-x/xdu/dbl_praveen/DarkBottomLine/.merge_root_hadd")
        staging_base.mkdir(parents=True, exist_ok=True)
        temp_dir_cm = tempfile.TemporaryDirectory(prefix="merge_root_hadd_", dir=str(staging_base))
        staging_dir = Path(temp_dir_cm.name)
        working_task = MergeTask(
            folder=task.folder,
            group_key=task.group_key,
            files=task.files,
            input_bytes=task.input_bytes,
            output_root=(staging_dir / task.output_root.name) if needs_staging else task.output_root,
            output_meta_json=(staging_dir / task.output_meta_json.name) if needs_staging else task.output_meta_json,
        )

    try:
        if not dry_run:
            good_files, skipped_files = _find_mergeable_files(working_task.files, working_task.output_root.parent, working_task.group_key)
            if not good_files:
                raise RuntimeError(f"All files were rejected while probing mergeability for {task.output_root.name}")
            if skipped_files:
                skip_note = f", skipped={len(skipped_files)} bad file(s)"
                for bad_file in skipped_files[:10]:
                    print(f"[SKIP] {task.folder.name}/{task.group_key}: {bad_file}")
                if len(skipped_files) > 10:
                    print(f"[SKIP] {task.folder.name}/{task.group_key}: ... and {len(skipped_files) - 10} more")
            working_task = MergeTask(
                folder=working_task.folder,
                group_key=working_task.group_key,
                files=good_files,
                input_bytes=sum((p.stat().st_size for p in good_files), 0),
                output_root=working_task.output_root,
                output_meta_json=working_task.output_meta_json,
            )

        _hadd_merge(working_task, dry_run=dry_run)
        if not dry_run and recoil_min is not None:
            before, after, used_branch = _apply_recoil_filter(working_task.output_root, recoil_min, recoil_branch)
            if used_branch is None:
                filter_note = f", recoil_filter=skipped(branch='{recoil_branch}' not found)"
            else:
                filter_note = f", recoil_filter={used_branch}>={recoil_min:g} ({before}->{after})"
        if not dry_run:
            _sum_metadata_to_json(working_task)
            if needs_staging:
                _copy_artifacts(
                    working_task.output_root,
                    task.output_root,
                    working_task.output_meta_json,
                    task.output_meta_json,
                )
    finally:
        if temp_dir_cm is not None:
            temp_dir_cm.cleanup()
        if budget_used and staging_budget is not None:
            staging_budget.release(budget_used)

    return f"{task.folder.name}/{task.group_key} -> {task.output_root} ({len(working_task.files)} files{skip_note}{filter_note})"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge ROOT files with hadd by grouping names that share a common prefix "
            "up to marker (default: NANOAODSIM)."
        )
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        action="append",
        default=[],
        help=(
            "Input directory containing ROOT files. Can be used multiple times. "
            "If omitted, all first-level subdirectories of --input-root are used."
        ),
    )
    parser.add_argument(
        "--input-root",
        default=None,
        help="Parent directory containing multiple dataset folders.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="Output directory for merged ROOT files.",
    )
    parser.add_argument(
        "--marker",
        default="NANOAODSIM",
        help="Grouping marker token in filename (default: NANOAODSIM).",
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 1))),
        help="Parallel workers across merge tasks.",
    )
    parser.add_argument(
        "--recoil-min",
        type=float,
        default=None,
        help="If set, keep only Events entries with recoil >= this value after merge.",
    )
    parser.add_argument(
        "--recoil-branch",
        default="recoil",
        help="Preferred branch name used for recoil filtering (default: recoil).",
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help=(
            "Directory used for temporary merge staging. Defaults to the EOS-backed "
            "/eos/home-x/xdu/dbl_praveen/DarkBottomLine/.merge_root_hadd directory."
        ),
    )
    parser.add_argument(
        "--max-staging-gb",
        type=float,
        default=8.0,
        help=(
            "Cap total concurrent temporary staging size in GB (default: 8). "
            "Set <=0 to disable this limiter."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print planned hadd commands.")
    return parser.parse_args()


def _resolve_input_dirs(input_dirs: Sequence[str], input_root: str | None) -> List[Path]:
    if input_dirs:
        return [Path(p).expanduser().resolve() for p in input_dirs]

    if input_root is None:
        raise ValueError("Provide either --input-dir (one or more) or --input-root.")

    root = Path(input_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"input-root does not exist: {root}")

    return sorted(p for p in root.iterdir() if p.is_dir())


def main() -> int:
    args = _parse_args()
    _check_hadd_exists()

    input_dirs = _resolve_input_dirs(args.input_dir, args.input_root)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(args.work_dir).expanduser().resolve() if args.work_dir else Path("/eos/home-x/xdu/dbl_praveen/DarkBottomLine/.merge_root_hadd")
    work_dir.mkdir(parents=True, exist_ok=True)
    staging_budget = None
    if args.max_staging_gb > 0:
        staging_budget = _ByteBudget(int(args.max_staging_gb * 1024**3))

    tasks = _collect_tasks(input_dirs, output_dir, args.marker)
    if not tasks:
        print("No ROOT files found to merge.")
        return 0

    print(f"Planned merge tasks: {len(tasks)}")
    for t in tasks:
        print(f"  - {t.folder.name}/{t.group_key}: {len(t.files)} files")
    if args.recoil_min is not None:
        print(
            f"Post-merge recoil filter enabled: {args.recoil_branch} >= {args.recoil_min:g}"
        )

    failures: List[str] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        future_map = {
            pool.submit(
                _run_one_task,
                task,
                args.dry_run,
                args.recoil_min,
                args.recoil_branch,
                work_dir,
                staging_budget,
            ): task
            for task in tasks
        }
        for fut in as_completed(future_map):
            task = future_map[fut]
            try:
                msg = fut.result()
                print(f"[OK] {msg}")
            except Exception as exc:
                failure = f"[FAIL] {task.folder.name}/{task.group_key}: {exc}"
                print(failure)
                failures.append(failure)

    if failures:
        print(f"\nCompleted with {len(failures)} failure(s).")
        return 2

    print("\nAll merge tasks completed successfully.")
    if not args.dry_run:
        print("Metadata sums were written to *.metadata_sum.json sidecar files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
