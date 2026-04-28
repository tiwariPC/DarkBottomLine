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


def _output_stem(group_key: str, marker: str, strip_marker: bool) -> str:
    """Return the output file stem for a group key.

    strip_marker=True: 'WtoLNu-2Jets_PTLNu-100to250_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_NANOAODSIM'
                     → 'WtoLNu-2Jets_PTLNu-100to250_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8'
    """
    if strip_marker and marker in group_key:
        idx = group_key.find(marker)
        stem = group_key[:idx].rstrip("_-")
        return stem if stem else group_key
    return group_key


def _collect_tasks(
    input_dirs: Sequence[Path],
    output_dir: Path,
    marker: str,
    file_filter: "set[Path] | None" = None,
    strip_marker: bool = True,
) -> List[MergeTask]:
    tasks: List[MergeTask] = []
    for in_dir in input_dirs:
        if not in_dir.is_dir():
            raise FileNotFoundError(f"Input directory does not exist: {in_dir}")

        root_files = sorted(p for p in in_dir.iterdir() if p.is_file() and p.suffix == ".root")
        if file_filter is not None:
            root_files = [p for p in root_files if p in file_filter]
        if not root_files:
            continue

        groups: Dict[str, List[Path]] = {}
        for f in root_files:
            key = _resolve_group_key(f, marker)
            groups.setdefault(key, []).append(f)

        output_dir.mkdir(parents=True, exist_ok=True)
        for key, files in sorted(groups.items()):
            input_bytes = sum((f.stat().st_size for f in files), 0)
            stem = _output_stem(key, marker, strip_marker)
            out_root = output_dir / f"{stem}.root"
            out_json = output_dir / f"{stem}.metadata_sum.json"
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


_HADD_CHUNK_SIZE = 500  # max files per hadd invocation to stay well under ARG_MAX


def _hadd_one_shot(output: Path, inputs: List[Path]) -> None:
    """Single hadd call — caller must ensure len(inputs) is safe."""
    cmd = ["hadd", "-f", "-k", "-O", str(output)] + [str(p) for p in inputs]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"hadd failed for {output.name}:\n{proc.stdout}")


def _hadd_merge(task: MergeTask, dry_run: bool = False, chunk_size: int = _HADD_CHUNK_SIZE) -> None:
    """Merge task.files into task.output_root using chunked tree-merge if needed.

    When len(files) > chunk_size, files are merged in batches into temporary
    intermediates, then those intermediates are merged into the final output.
    This avoids ARG_MAX overflows on EOS paths with >~500 files.
    """
    files = list(task.files)
    if dry_run:
        n_chunks = max(1, (len(files) + chunk_size - 1) // chunk_size)
        if n_chunks > 1:
            print(f"[DRY-RUN] chunked merge: {len(files)} files → {n_chunks} chunks → {task.output_root}")
        else:
            cmd = ["hadd", "-f", "-k", "-O", str(task.output_root)] + [str(p) for p in files]
            print("[DRY-RUN]", " ".join(cmd))
        return

    task.output_root.parent.mkdir(parents=True, exist_ok=True)

    if len(files) <= chunk_size:
        _hadd_one_shot(task.output_root, files)
        return

    # Tree merge: batch → intermediates → final
    chunks = [files[i : i + chunk_size] for i in range(0, len(files), chunk_size)]
    tmp_dir = task.output_root.parent / f"_chunks_{task.group_key}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    intermediates: List[Path] = []
    try:
        for idx, chunk in enumerate(chunks):
            inter = tmp_dir / f"chunk_{idx:04d}.root"
            _hadd_one_shot(inter, chunk)
            intermediates.append(inter)
        # Final merge of intermediates (always ≤ chunk_size since each original chunk
        # is one file now; if somehow intermediates > chunk_size, recurse once more)
        if len(intermediates) <= chunk_size:
            _hadd_one_shot(task.output_root, intermediates)
        else:
            # Second level (handles up to chunk_size² ≈ 250 000 files)
            second_chunks = [intermediates[i : i + chunk_size] for i in range(0, len(intermediates), chunk_size)]
            level2: List[Path] = []
            for idx2, schunk in enumerate(second_chunks):
                inter2 = tmp_dir / f"level2_{idx2:04d}.root"
                _hadd_one_shot(inter2, schunk)
                level2.append(inter2)
            _hadd_one_shot(task.output_root, level2)
    finally:
        import shutil as _shutil
        if tmp_dir.exists():
            _shutil.rmtree(tmp_dir, ignore_errors=True)


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


def _load_process_groups(path: str) -> Dict[str, List[str]]:
    """Load process group pattern mapping from a YAML or JSON file.

    Returns {group_name: [pattern, ...]} for all groups that have a 'patterns' key.
    Supports plotting.yaml format (process_groups section) and plain JSON dicts.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Process groups file not found: {path}")

    raw: dict
    if p.suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError("PyYAML required for --process-groups yaml: pip install pyyaml") from exc
        with p.open() as f:
            raw = yaml.safe_load(f)
        # Support plotting.yaml with top-level 'process_groups' section
        if "process_groups" in raw:
            raw = raw["process_groups"]
    else:
        with p.open() as f:
            raw = json.load(f)

    groups: Dict[str, List[str]] = {}
    for name, cfg in raw.items():
        if isinstance(cfg, dict):
            patterns = cfg.get("patterns") or cfg.get("files") or []
        elif isinstance(cfg, list):
            patterns = cfg
        else:
            continue
        if patterns:
            groups[name] = [str(pt) for pt in patterns]
    return groups


def _collect_process_group_tasks(
    merged_dir: Path,
    output_dir: Path,
    process_groups: Dict[str, List[str]],
) -> List[MergeTask]:
    """Scan merged_dir for .root files and group them by process_groups patterns.

    Each group produces one MergeTask whose output is output_dir/{group_name}.root.
    Files not matched by any group are reported as warnings.
    """
    all_roots = sorted(p for p in merged_dir.rglob("*.root") if p.is_file())
    if not all_roots:
        return []

    assigned: set[Path] = set()
    tasks: List[MergeTask] = []

    for group_name, patterns in process_groups.items():
        matched: List[Path] = []
        for p in all_roots:
            if p in assigned:
                continue
            if any(pat in p.name for pat in patterns):
                matched.append(p)
                assigned.add(p)
        if not matched:
            print(f"[WARN] process group '{group_name}': no files matched patterns {patterns}")
            continue
        input_bytes = sum(f.stat().st_size for f in matched)
        out_root = output_dir / f"{group_name}.root"
        out_json = output_dir / f"{group_name}.metadata_sum.json"
        tasks.append(MergeTask(
            folder=merged_dir,
            group_key=group_name,
            files=sorted(matched),
            input_bytes=input_bytes,
            output_root=out_root,
            output_meta_json=out_json,
        ))

    unmatched = [p for p in all_roots if p not in assigned]
    if unmatched:
        print(f"[WARN] {len(unmatched)} file(s) not assigned to any process group:")
        for u in unmatched[:10]:
            print(f"  {u.name}")
        if len(unmatched) > 10:
            print(f"  ... and {len(unmatched) - 10} more")

    return tasks


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
    parser.add_argument(
        "--process-groups",
        default=None,
        metavar="FILE",
        help=(
            "YAML or JSON file mapping process group names to filename patterns "
            "(e.g. configs/plotting.yaml). When provided, a second merge pass "
            "combines per-sample merged files into one file per process group. "
            "Accepts plotting.yaml directly (reads process_groups section)."
        ),
    )
    parser.add_argument(
        "--strip-marker",
        action="store_true",
        default=True,
        help=(
            "Strip the marker token (and everything after it) from the output filename, "
            "e.g. WtoLNu-2Jets_PTLNu-100to250_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_NANOAODSIM-abc.root "
            "→ WtoLNu-2Jets_PTLNu-100to250_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.root (default: true)."
        ),
    )
    parser.add_argument(
        "--no-strip-marker",
        dest="strip_marker",
        action="store_false",
        help="Keep the marker token in output filenames.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print planned hadd commands.")
    return parser.parse_args()


def _resolve_input_dirs(
    input_dirs: Sequence[str], input_root: str | None
) -> "tuple[List[Path], set[Path] | None]":
    """Return (dirs, file_filter).

    file_filter is None when all files in the dirs should be included.
    When --input-root is a glob of .root files, file_filter contains only
    the matched files so that _collect_tasks ignores siblings.
    """
    if input_dirs:
        return [Path(p).expanduser().resolve() for p in input_dirs], None

    if input_root is None:
        raise ValueError("Provide either --input-dir (one or more) or --input-root.")

    raw = str(input_root)
    if "*" in raw or "?" in raw:
        from glob import glob as _glob
        matches = sorted(_glob(os.path.expanduser(raw)))
        if not matches:
            raise FileNotFoundError(f"--input-root glob matched nothing: {raw}")
        files = [Path(m).resolve() for m in matches if Path(m).is_file() and Path(m).suffix == ".root"]
        dirs = sorted(set(Path(m).resolve() for m in matches if Path(m).is_dir()))
        if files:
            parent_dirs: List[Path] = list(dict.fromkeys(f.parent for f in files))
            return parent_dirs, set(files)
        return dirs, None

    root = Path(input_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"input-root does not exist: {root}")

    return sorted(p for p in root.iterdir() if p.is_dir()), None


def main() -> int:
    args = _parse_args()
    _check_hadd_exists()

    input_dirs, file_filter = _resolve_input_dirs(args.input_dir, args.input_root)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.work_dir:
        work_dir = Path(args.work_dir).expanduser().resolve()
    else:
        work_dir = Path.cwd() / ".merge_root_hadd"
    work_dir.mkdir(parents=True, exist_ok=True)
    staging_budget = None
    if args.max_staging_gb > 0:
        staging_budget = _ByteBudget(int(args.max_staging_gb * 1024**3))

    tasks = _collect_tasks(
        input_dirs, output_dir, args.marker,
        file_filter=file_filter, strip_marker=args.strip_marker,
    )
    if not tasks:
        print("No ROOT files found to merge.")
        return 0

    print(f"Planned merge tasks: {len(tasks)}")
    for t in tasks:
        print(f"  - {t.folder.name}/{t.group_key}: {len(t.files)} files → {t.output_root.name}")
    if args.recoil_min is not None:
        print(f"Post-merge recoil filter enabled: {args.recoil_branch} >= {args.recoil_min:g}")

    failures: List[str] = []

    def _run_tasks(task_list: List[MergeTask]) -> List[str]:
        errs: List[str] = []
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            future_map = {
                pool.submit(
                    _run_one_task, task, args.dry_run,
                    args.recoil_min, args.recoil_branch, work_dir, staging_budget,
                ): task
                for task in task_list
            }
            for fut in as_completed(future_map):
                task = future_map[fut]
                try:
                    print(f"[OK] {fut.result()}")
                except Exception as exc:
                    msg = f"[FAIL] {task.folder.name}/{task.group_key}: {exc}"
                    print(msg)
                    errs.append(msg)
        return errs

    failures = _run_tasks(tasks)
    if failures:
        print(f"\nPass 1 completed with {len(failures)} failure(s).")
        return 2

    # --- Pass 2: group merged per-sample files into process groups ---
    if args.process_groups:
        process_groups = _load_process_groups(args.process_groups)
        # Scan all output subdirs produced by pass 1
        pg_output_dir = output_dir / "process_groups"
        pg_output_dir.mkdir(parents=True, exist_ok=True)
        pg_tasks = _collect_process_group_tasks(output_dir, pg_output_dir, process_groups)
        if pg_tasks:
            print(f"\nProcess group merge tasks: {len(pg_tasks)}")
            for t in pg_tasks:
                print(f"  - {t.group_key}: {len(t.files)} files → {t.output_root.name}")
            failures = _run_tasks(pg_tasks)
            if failures:
                print(f"\nPass 2 completed with {len(failures)} failure(s).")
                return 2
        else:
            print("\n[WARN] --process-groups provided but no tasks created.")

    print("\nAll merge tasks completed successfully.")
    if not args.dry_run:
        print("Metadata sums were written to *.metadata_sum.json sidecar files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
