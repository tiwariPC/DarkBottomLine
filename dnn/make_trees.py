"""Convert per-sample event-selection ROOT outputs to ppbbchichi-trees format.

Input:  one or more flat ROOT files produced by `darkbottomline analyze --event-selection-only`
        Each file has a single `Events` TTree with DNN feature branches + full_event_weight.

Output: ppbbchichi-trees.root with structure:
          <sample_name>/preselection    <- TTree of all events
        plus per-event branches:
          signal  (int8)  1=signal 0=background
          isdata  (int8)  1=data   0=MC
          weight_preselection  (float64)  = full_event_weight (already normalized)

Signal/background detection (in order of priority):
  1. --label-csv  (columns: path, label)  label=1 signal, label=0 background
  2. --signal-prefix  e.g. "newdiboson" — samples whose filename starts with this are signal
  3. --signal-pattern  regex matched against filename
  4. built-in heuristic: keywords chichi/bbdm/dark/invisible/dm in filename → signal
"""
from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
import uproot


# ---------------------------------------------------------------------------
# Signal / data detection
# ---------------------------------------------------------------------------

_DEFAULT_SIGNAL_KEYWORDS = ("chichi", "bbdm", "bbchi", "dark", "invisible", "_dm_", "signal")
_DATA_PREFIXES = ("run20", "data")
# Real collision-data dataset naming (matches configs/plotting.yaml's data_groups
# patterns: JetMET/JetMET0/JetMET1/MET/EGamma/EGamma0/EGamma1, each "-Run...").
_DATA_SUBSTRINGS = ("jetmet-run", "jetmet0-run", "jetmet1-run", "met-run",
                    "egamma-run", "egamma0-run", "egamma1-run")


def _is_data(name: str) -> bool:
    n = os.path.basename(name).lower()
    return (any(n.startswith(p) for p in _DATA_PREFIXES)
            or any(s in n for s in _DATA_SUBSTRINGS)
            or "collisions" in n)


def _is_signal_heuristic(name: str, signal_patterns: tuple[str, ...], signal_prefix: Optional[str]) -> bool:
    n = os.path.basename(name).lower()
    if _is_data(n):
        return False
    if signal_prefix:
        return n.startswith(signal_prefix.lower())
    if signal_patterns:
        return any(re.search(p, n, re.IGNORECASE) for p in signal_patterns)
    return any(k in n for k in _DEFAULT_SIGNAL_KEYWORDS)


def _sample_name(path: str) -> str:
    """Derive stable sample name from file path.

    For EVENTSELECTION-style paths (.../SampleName_EVENTSELECTION/uuid.root)
    use the parent directory name. Otherwise use stem.
    """
    p = Path(path)
    if "EVENTSELECTION" in p.parent.name.upper():
        return p.parent.name
    stem = p.stem
    # strip _partN / _chunkN suffixes
    stem = re.sub(r"(_part\d+|_chunk\d+|_\d+of\d+)$", "", stem, flags=re.IGNORECASE)
    return stem


# ---------------------------------------------------------------------------
# Core converter
# ---------------------------------------------------------------------------

def convert_files(
    input_files: list[str],
    output_path: str,
    *,
    signal_patterns: Optional[list[str]] = None,
    signal_prefix: Optional[str] = None,
    label_csv: Optional[str] = None,
    weight_branch: str = "full_event_weight",
    region_name: str = "preselection",
    max_events_per_file: Optional[int] = None,
    verbose: bool = True,
) -> dict:
    """Convert flat per-sample ROOT files → ppbbchichi-trees.root.

    Returns summary dict: {sample_name: {"n_events": int, "signal": bool, "isdata": bool}}
    """
    # Build label map from CSV if provided
    label_map: dict[str, int] = {}
    if label_csv:
        with open(label_csv, "r", newline="") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                label_map[str(row["path"]).strip()] = int(row["label"])

    sig_patterns = tuple(signal_patterns) if signal_patterns else ()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}

    with uproot.recreate(output_path) as out_f:
        for fpath in input_files:
            sample = _sample_name(fpath)

            # Determine signal/data flags
            if label_map:
                # exact path match first, then basename match
                key = fpath if fpath in label_map else os.path.basename(fpath)
                if key not in label_map:
                    raise KeyError(
                        f"File '{fpath}' (basename '{os.path.basename(fpath)}') "
                        "not found in --label-csv. Add it or remove --label-csv."
                    )
                sig_flag = bool(label_map[key] == 1)
                data_flag = False
            else:
                data_flag = _is_data(fpath)
                sig_flag = False if data_flag else _is_signal_heuristic(fpath, sig_patterns, signal_prefix)

            if verbose:
                print(
                    f"[INFO] {sample}  signal={int(sig_flag)}  isdata={int(data_flag)}  "
                    f"file={os.path.basename(fpath)}"
                )

            # Read Events tree
            with uproot.open(fpath) as in_f:
                if "Events" not in in_f:
                    raise KeyError(f"No 'Events' tree in {fpath}. Run `analyze --event-selection-only` first.")
                tree = in_f["Events"]
                n_entries = int(tree.num_entries)
                if max_events_per_file is not None:
                    n_entries = min(n_entries, int(max_events_per_file))

                # Read all branches
                arrays = tree.arrays(entry_stop=n_entries, library="np")

            n = n_entries

            # Weight: use full_event_weight if present, else ones
            if weight_branch in arrays:
                w = np.asarray(arrays[weight_branch], dtype="f8")
                w = np.where(np.isfinite(w), w, 0.0)
            else:
                if verbose:
                    print(f"[WARN] '{weight_branch}' not found in {sample}, using weight=1")
                w = np.ones(n, dtype="f8")

            # Add flags + weight_region to output dict
            out_dict: dict[str, np.ndarray] = {}
            for k, v in arrays.items():
                arr = np.asarray(v)
                if arr.dtype == np.bool_:
                    arr = arr.astype("int8")
                elif arr.dtype.kind in ("i", "u"):
                    arr = arr.astype("int64")
                elif arr.dtype.kind == "f":
                    arr = arr.astype("float64")
                # skip jagged (variable-length) arrays — DNN features are all flat scalars
                if arr.ndim == 1 and len(arr) == n:
                    out_dict[k] = arr

            out_dict["signal"] = np.full(n, int(sig_flag), dtype="int8")
            out_dict["isdata"] = np.full(n, int(data_flag), dtype="int8")
            out_dict[f"weight_{region_name}"] = w

            # Write to sample/region tree
            tree_key = f"{sample}/{region_name}"
            out_f[tree_key] = out_dict

            summary[sample] = {
                "n_events": n,
                "signal": sig_flag,
                "isdata": data_flag,
                "weight_sum": float(np.sum(w)),
                "file": fpath,
            }
            if verbose:
                print(f"[OK]   {tree_key}  n={n}  weight_sum={float(np.sum(w)):.4g}")

    return summary
