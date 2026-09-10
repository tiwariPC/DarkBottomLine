"""Dataset reading helpers for DNN training.

Purpose:
- Enumerate per-sample trees inside the ppbbchichi output ROOT file.
- Read requested branches into numpy arrays with basic validation.

This module intentionally stays small so both training and inference scripts can
reuse the same ROOT IO logic.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
from typing import Iterable
from pathlib import Path

import numpy as np
import uproot


def read_branch_as_array(
    tree,
    branch_name: str,
    max_events: int | None = None,
    entry_mask=None,
) -> np.ndarray:
    """Read one flat branch through aligned basket payloads.

    Some event-selection files have a corrupt TBranch ``fEntries`` value even
    though their basket entry offsets and payloads are valid. Uproot's regular
    ``array`` path can fail or stall on those files. This reader is DNN-local,
    read-only, and stops before the first malformed or non-contiguous basket.

    When *entry_mask* (a boolean array aligned to tree entries) is provided,
    only entries whose mask value is true are kept. This is used for random
    event subsampling without materializing the full branch array.
    """
    branch = tree[branch_name]
    tree_entries = int(tree.num_entries)
    target = tree_entries if max_events is None else min(tree_entries, int(max_events))
    if target <= 0:
        return np.empty(0, dtype="f8")

    if entry_mask is not None:
        mask = np.asarray(entry_mask, dtype=bool)
        parts = []
        expected = 0
        for ibasket in range(branch.num_baskets):
            start, stop = map(int, branch.basket_entry_start_stop(ibasket))
            if start >= target:
                break
            if start != expected:
                break
            stop = min(stop, target)
            try:
                values = np.asarray(
                    branch.basket(ibasket).array(branch.interpretation, library="np")
                )
            except Exception:
                break
            required = stop - start
            if len(values) < required:
                break
            parts.append(values[:required][mask[start:stop]])
            expected = stop

        if expected < target:
            logging.warning(
                "DNN basket read truncated %s/%s to %d/%d aligned entries",
                getattr(tree.file, "file_path", "<ROOT>"),
                branch_name,
                expected,
                target,
            )
        if not parts:
            return np.empty(0, dtype="f8")
        return np.concatenate(parts)

    out = None
    expected = 0
    for ibasket in range(branch.num_baskets):
        start, stop = map(int, branch.basket_entry_start_stop(ibasket))
        if start >= target:
            break
        if start != expected:
            break
        stop = min(stop, target)
        try:
            values = np.asarray(
                branch.basket(ibasket).array(branch.interpretation, library="np")
            )
        except Exception:
            break
        required = stop - start
        if len(values) < required:
            break
        if out is None:
            out = np.empty(target, dtype=values.dtype)
        out[start:stop] = values[:required]
        expected = stop

    if out is None:
        return np.empty(0, dtype="f8")
    if expected < target:
        logging.warning(
            "DNN basket read truncated %s/%s to %d/%d aligned entries",
            getattr(tree.file, "file_path", "<ROOT>"),
            branch_name,
            expected,
            target,
        )
    return out[:expected]


def read_tree_branches_as_arrays(
    tree,
    branches: list[str],
    max_events: int | None = None,
    entry_mask=None,
) -> dict[str, np.ndarray]:
    """Read flat branches and trim all arrays to their common aligned prefix."""
    arrays = {
        name: read_branch_as_array(tree, name, max_events=max_events, entry_mask=entry_mask)
        for name in branches
    }
    if not arrays:
        return {}
    common = min(len(values) for values in arrays.values())
    return {name: np.asarray(values)[:common] for name, values in arrays.items()}


def list_sample_region_trees(root_file: uproot.ReadOnlyFile, region: str) -> list[tuple[str, str]]:
    """Return list of (sample, tree_path) for trees ending with '/<region>'."""

    out: list[tuple[str, str]] = []
    classmap = root_file.classnames()
    suffix = f"/{region}"
    for k, cls in classmap.items():
        if cls != "TTree":
            continue
        path = k.split(";")[0]
        if not path.endswith(suffix):
            continue
        sample = path.split("/")[0]
        out.append((sample, path))
    out.sort(key=lambda x: (x[0], x[1]))
    return out


def read_tree_as_arrays(
    root_file: uproot.ReadOnlyFile,
    tree_path: str,
    branches: list[str],
    max_events: int | None,
) -> dict:
    tree = root_file[tree_path]
    available = set(map(str, tree.keys()))

    present = [b for b in branches if b in available]
    missing = [b for b in branches if b not in available]
    if missing:
        raise KeyError(f"Tree '{tree_path}' missing branches: {missing}")

    return read_tree_branches_as_arrays(tree, present, max_events=max_events)


def _parse_date_like_dir_name(name: str, *, today: dt.date | None = None) -> dt.date | None:
    """Parse common date-like folder names.

    Supported patterns:
    - YYYY-MM-DD, YYYY_M_D
    - YYYYMMDD
    - M-D, M_D (year inferred from current year)
    """
    if today is None:
        today = dt.date.today()

    m = re.fullmatch(r"(\d{4})[-_](\d{1,2})[-_](\d{1,2})", name)
    if m:
        y, mm, dd = map(int, m.groups())
        try:
            return dt.date(y, mm, dd)
        except ValueError:
            return None

    m = re.fullmatch(r"(\d{8})", name)
    if m:
        raw = m.group(1)
        y, mm, dd = int(raw[0:4]), int(raw[4:6]), int(raw[6:8])
        try:
            return dt.date(y, mm, dd)
        except ValueError:
            return None

    m = re.fullmatch(r"(\d{1,2})[-_](\d{1,2})", name)
    if m:
        mm, dd = map(int, m.groups())
        try:
            cand = dt.date(today.year, mm, dd)
        except ValueError:
            return None
        # If month-day is in the "future" relative to today, it likely belongs to previous year.
        if cand > today + dt.timedelta(days=1):
            try:
                cand = dt.date(today.year - 1, mm, dd)
            except ValueError:
                return None
        return cand

    return None


def summarize_year_subdirs(year_dir: str | Path) -> dict[str, list[str]]:
    """Return categorized subdirectories under a year folder."""
    p = Path(year_dir)
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"Year directory not found: {p}")

    subdirs = sorted([x.name for x in p.iterdir() if x.is_dir()])
    new_dirs = [s for s in subdirs if s.lower().startswith("new")]
    run_dirs = [s for s in subdirs if s.lower().startswith("run")]
    other_dirs = [s for s in subdirs if (s not in new_dirs and s not in run_dirs)]
    return {
        "all": subdirs,
        "new": new_dirs,
        "run": run_dirs,
        "other": other_dirs,
    }


def find_latest_year_dir(
    base_dir: str | Path,
    *,
    target_year: str | None = None,
    require_new_and_run: bool = False,
    required_signal_prefix: str | None = None,
) -> dict:
    """Find the latest date folder that contains a pure year directory.

    The search walks date-like folders from newest to oldest and returns the first
    matching pure-year directory (e.g. "2022").
    """
    base = Path(base_dir)
    if not base.exists() or not base.is_dir():
        raise FileNotFoundError(f"Base outputs directory not found: {base}")

    date_candidates: list[tuple[dt.date, Path]] = []
    for entry in base.iterdir():
        if not entry.is_dir():
            continue
        parsed = _parse_date_like_dir_name(entry.name)
        if parsed is not None:
            date_candidates.append((parsed, entry))

    if not date_candidates:
        raise FileNotFoundError(
            "No date-like folders found under base outputs directory. "
            "Expected names like 4-2, 2026-04-02, or 20260402."
        )

    date_candidates.sort(key=lambda x: x[0], reverse=True)

    year_re = re.compile(r"\d{4}")
    year_norm = str(target_year).strip() if target_year is not None else None

    sig_prefix = str(required_signal_prefix).strip().lower() if required_signal_prefix else None

    for parsed_date, date_dir in date_candidates:
        year_dirs = [x for x in date_dir.iterdir() if x.is_dir() and year_re.fullmatch(x.name)]
        year_dirs.sort(key=lambda z: z.name)
        if year_norm is not None:
            year_dirs = [y for y in year_dirs if y.name == year_norm]
        if not year_dirs:
            continue

        for yd in year_dirs:
            summary = summarize_year_subdirs(yd)
            if require_new_and_run and (len(summary["new"]) == 0 or len(summary["run"]) == 0):
                continue
            if sig_prefix and not any(s.lower().startswith(sig_prefix) for s in summary["all"]):
                continue
            return {
                "date": parsed_date.isoformat(),
                "date_dir": str(date_dir),
                "year": yd.name,
                "year_dir": str(yd),
                "summary": summary,
            }

    req_txt = " with both new*/run*" if require_new_and_run else ""
    yr_txt = f" and year={year_norm}" if year_norm is not None else ""
    sig_txt = f" and signal prefix={sig_prefix}" if sig_prefix else ""
    raise FileNotFoundError(
        f"No suitable date folder found{req_txt}{yr_txt}{sig_txt} under: {base}"
    )
