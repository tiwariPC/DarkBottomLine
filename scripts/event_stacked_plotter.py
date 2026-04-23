#!/usr/bin/env python3
"""
Standalone stacked plotting tool for event-level output PKL/ROOT files.

Features:
- Reads event-level PKL format (keys: n_events/events/objects)
- Reads ROOT files (Events tree with branches, Metadata tree with n_events)
- Merges multiple PKL/ROOT files inside each input folder
- Supports multiple background folders and optional data folder
- Draws log-scale stacked background plot with data overlay
- Normalizes all histograms by n_events and luminosity
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import array
import json
import math
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches
import matplotlib.pyplot as plt
import matplotlib.ticker
import mplhep as hep
hep.style.use("CMS")
import numpy as np
import uproot

try:
    import ROOT
except Exception:
    ROOT = None


@dataclass
class SubSampleData:
    process: str
    n_events: int
    objects: Dict[str, Any]
    cross_section: Optional[float] = None  # pb
    cutflow_labels: Optional[List[str]] = None
    cutflow_values: Optional[np.ndarray] = None


@dataclass
class SampleData:
    name: str
    n_events: int
    objects: Dict[str, Any]
    luminosity: float = 1.0  # Luminosity normalization factor
    cross_section: Optional[float] = None  # Cross-section in pb; None means no cross-section weighting
    metadata_n_events_available: bool = True  # Whether n_events is from metadata (for ROOT inputs)
    sub_samples: Optional[List[SubSampleData]] = None  # Optional per-process components for mixed-pt samples
    cutflow_labels: Optional[List[str]] = None
    cutflow_values: Optional[np.ndarray] = None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _load_xsection_json(json_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Load cross-section data from JSON file.

    Expected format:
    {
        "ProcessName": [
            {"year": "2022", "process": "...", "xsection": 123.45, ...},
            ...
        ],
        ...
    }
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Warning: Could not load cross-section JSON {json_path}: {e}")
        return {}


def _extract_xsection_by_year(
    xsection_data: Dict[str, List[Dict[str, Any]]],
    sample_name: str,
    year: str
) -> Optional[float]:
    """Extract cross-section (in pb) for a given sample and year.

    Searches for matching process in xsection_data and returns cross-section in pb.
    Returns None if not found.
    """
    if not xsection_data:
        return None

    # Try to find the process category from sample name
    # Sample names are usually like: newDIBOSON_2022_EVENTSELECTION or newWtoLNU-2Jets_2022_EVENTSELECTION
    # Remove prefixes and suffixes
    sample_base = sample_name
    if sample_base.startswith("new"):
        sample_base = sample_base[3:]
    # Remove year suffix and _EVENTSELECTION
    sample_base = re.sub(r"_20\d{2}(EE)?.*$", "", sample_base)
    # Normalize case for matching (WtoLNu vs WtoLNU)
    sample_base_lower = sample_base.lower()

    alias_map = {
        "wtolnu-2jets": "WtoLNu-2Jets",
        "zto2nu-2jets": "Zto2Nu-2Jets",
        "dyto2l-2jets": "DYto2L-2Jets",
        "top": "Top",
        "diboson": "DIBOSON",
        "singletop": "SingleTop",
        "smhiggs": "SMHiggs",
    }
    mapped_category = alias_map.get(sample_base_lower, sample_base)

    # Search in xsection_data for matching process
    for process_category, entries in xsection_data.items():
        # Try case-insensitive match for process category
        if mapped_category.lower() == process_category.lower():
            for entry in entries:
                if entry.get("year") == year:
                    xsec = entry.get("xsection")
                    if xsec is not None:
                        return float(xsec)  # Return in pb
    return None


def _build_xsection_process_lookup(
    xsection_data: Dict[str, List[Dict[str, Any]]],
    year: str,
) -> Dict[str, float]:
    """Build a mapping from full_dataset process name to cross-section in pb for a given year."""
    process_to_xsec: Dict[str, float] = {}
    for entries in xsection_data.values():
        for entry in entries:
            if entry.get("year") != year:
                continue
            full_dataset = entry.get("full_dataset")
            xsec = entry.get("xsection")
            if not full_dataset or xsec is None:
                continue
            process_to_xsec[str(full_dataset)] = float(xsec)
    return process_to_xsec


def _normalize_xsection_name(name: str) -> str:
    value = Path(name).stem
    if value.startswith("new"):
        value = value[3:]
    value = re.sub(r"_20\d{2}(EE)?_EVENTSELECTION", "", value)
    value = re.sub(r"_20\d{2}(EE)?$", "", value)
    # DY JSON keys can include MLL tags while merged ROOT names may omit them.
    value = re.sub(r"_MLL-\d+", "", value, flags=re.IGNORECASE)
    value = value.replace("_EVENTSELECTION", "")
    value = value.replace("singletop", "singleTop")
    value = value.replace("smhiggs", "smHiggs")
    return value.lower()


def _match_root_file_to_process(root_file_name: str, process_to_xsec: Dict[str, float], _cache: Dict[str, str] = {}) -> Optional[Tuple[str, float]]:
    """Match ROOT file name to JSON full_dataset key using normalized longest-prefix match."""
    stem = _normalize_xsection_name(root_file_name)
    # Cache normalized keys to avoid repeated calls to _normalize_xsection_name per lookup
    cache_key = id(process_to_xsec)
    if cache_key not in _cache:
        _cache[cache_key] = {key: _normalize_xsection_name(key) for key in process_to_xsec}
    norm_keys = _cache[cache_key]
    best_key: Optional[str] = None
    best_len = -1
    for key, key_norm in norm_keys.items():
        if stem.startswith(key_norm) or key_norm.startswith(stem):
            if len(key_norm) > best_len:
                best_key = key
                best_len = len(key_norm)
    if best_key is None:
        return None
    return best_key, process_to_xsec[best_key]



def _merge_values(old_value: Any, new_value: Any) -> Any:
    if old_value is None:
        return new_value
    if new_value is None:
        return old_value

    if isinstance(old_value, dict) and isinstance(new_value, dict):
        merged = dict(old_value)
        for key, value in new_value.items():
            merged[key] = _merge_values(merged.get(key), value)
        return merged

    if isinstance(old_value, list) and isinstance(new_value, list):
        return old_value + new_value

    if _is_number(old_value) and _is_number(new_value):
        return old_value + new_value

    return copy.deepcopy(new_value)


def _load_single_pkl(path: Path) -> Dict[str, Any]:
    with open(path, "rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"PKL is not a dictionary: {path}")
    return data


def _load_single_root(path: Path, target_variables: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Load a single ROOT file and convert to pkl-like format."""
    with uproot.open(path) as f:
        if "Events" not in f:
            raise ValueError(f"No 'Events' tree found in ROOT file: {path}")

        tree = f["Events"]
        objects = {}

        selected_branches = list(tree.keys())
        if target_variables:
            target_set = set(target_variables)
            selected_branches = [
                branch_name
                for branch_name in tree.keys()
                if branch_name in target_set
                or any(var.startswith(f"{branch_name}_") for var in target_set)
            ]

        # Read selected branches as arrays
        for branch_name in selected_branches:
            try:
                arr = tree[branch_name].array(library="np")
                objects[branch_name] = arr.tolist() if hasattr(arr, 'tolist') else list(arr)
            except Exception as e:
                print(f"Warning: Could not read branch {branch_name}: {e}")

        # Get n_events from Metadata tree if available
        n_events = 0
        metadata_n_events_available = False
        if "Metadata" in f:
            meta_tree = f["Metadata"]
            if "n_events" in meta_tree.keys():
                n_events_arr = meta_tree["n_events"].array(library="np")
                # hadd-merged files can carry one n_events entry per source file.
                # Sum all entries so normalization uses the total generated event count.
                n_events = int(np.sum(n_events_arr)) if len(n_events_arr) > 0 else 0
                metadata_n_events_available = n_events > 0

        # Do not fallback to selected entries for MC normalization.
        # If metadata is missing, keep n_events=0 and let caller decide whether to fail.
        return {
            "n_events": n_events,
            "objects": objects,
            "metadata_n_events_available": metadata_n_events_available,
        }


def _clean_root_key(name: str) -> str:
    return str(name).split(";", 1)[0]


def _extract_cut_labels_from_tree(tree: Any, n_cuts: int) -> List[str]:
    candidate_branches = ["cf_labels", "cut_labels", "cutflow_labels", "cuts", "labels"]
    branch_map: Dict[str, str] = {}
    if hasattr(tree, "keys"):
        for key in tree.keys():
            branch_map[_clean_root_key(str(key))] = str(key)

    for branch in candidate_branches:
        real_key = branch_map.get(branch)
        if real_key is None:
            continue
        try:
            arr = tree[real_key].array(library="np")
        except Exception:
            continue
        labels: List[str] = []
        if isinstance(arr, np.ndarray):
            flat = arr.ravel()
            for item in flat:
                labels.append(str(item.decode("utf-8")) if isinstance(item, bytes) else str(item))
        else:
            for item in arr:
                labels.append(str(item.decode("utf-8")) if isinstance(item, bytes) else str(item))
        labels = [label.strip() for label in labels if label is not None and str(label).strip()]
        if labels:
            return labels[:n_cuts]
    return [f"Cut {i + 1}" for i in range(n_cuts)]


def _extract_cutflow_from_root_file(path: Path) -> Optional[Tuple[List[str], np.ndarray]]:
    try:
        with uproot.open(path) as f:
            # Fallback: cutflow stored in Metadata tree as cf_XX_<step> branches.
            if "Metadata" in f:
                meta = f["Metadata"]
                if hasattr(meta, "keys"):
                    key_map: Dict[str, str] = {}
                    for key in meta.keys():
                        key_map[_clean_root_key(str(key))] = str(key)

                    cf_items: List[Tuple[int, str, float]] = []
                    for clean_key, real_key in key_map.items():
                        if not clean_key.startswith("cf_") or clean_key.endswith("_frac"):
                            continue
                        m = re.match(r"^cf_(\d+)_(.+)$", clean_key)
                        if m is None:
                            continue
                        idx = int(m.group(1))
                        step_name = m.group(2).replace("_", " ").strip()
                        try:
                            arr = meta[real_key].array(library="np")
                            if len(arr) == 0:
                                value = 0.0
                            else:
                                value = float(arr[0])
                        except Exception:
                            continue
                        cf_items.append((idx, step_name, value))

                    if cf_items:
                        cf_items.sort(key=lambda item: item[0])
                        labels = [item[1] for item in cf_items]
                        values = np.asarray([item[2] for item in cf_items], dtype=float)
                        return labels, values

            if "metdata" not in f:
                return None
            metdata = f["metdata"]

            # Case 1: metdata is a TTree with branch "cf"
            if hasattr(metdata, "keys"):
                key_map: Dict[str, str] = {}
                for key in metdata.keys():
                    key_map[_clean_root_key(str(key))] = str(key)
                cf_key = key_map.get("cf")
                if cf_key is not None:
                    try:
                        arr = metdata[cf_key].array(library="np")
                        values = _flatten_numeric(arr)
                        if values.size > 0:
                            labels = _extract_cut_labels_from_tree(metdata, values.size)
                            if len(labels) < values.size:
                                labels.extend(f"Cut {i + 1}" for i in range(len(labels), values.size))
                            return labels[:values.size], values.astype(float)
                    except Exception:
                        pass

            # Case 2: metdata/cf is a histogram-like object
            try:
                cf_obj = f["metdata/cf"]
                if hasattr(cf_obj, "to_numpy"):
                    hist_data = cf_obj.to_numpy()
                    values = np.asarray(hist_data[0], dtype=float)
                    labels: List[str] = []
                    try:
                        axis = cf_obj.axis()
                        if hasattr(axis, "labels"):
                            axis_labels = axis.labels()
                            labels = [str(lbl).strip() for lbl in axis_labels]
                    except Exception:
                        labels = []
                    if not labels:
                        labels = [f"Cut {i + 1}" for i in range(values.size)]
                    return labels[:values.size], values
            except Exception:
                pass

    except Exception as e:
        print(f"Warning: Could not extract cutflow from {path.name}: {e}")
        return None

    return None


def _merge_cutflow_series(
    base_labels: Optional[List[str]],
    base_values: Optional[np.ndarray],
    new_labels: List[str],
    new_values: np.ndarray,
) -> Tuple[List[str], np.ndarray]:
    if base_labels is None or base_values is None or len(base_labels) == 0:
        return list(new_labels), np.asarray(new_values, dtype=float)

    merged_labels = list(base_labels)
    merged_values = np.asarray(base_values, dtype=float)
    index_map = {label: i for i, label in enumerate(merged_labels)}

    for i, label in enumerate(new_labels):
        if i >= len(new_values):
            break
        if label not in index_map:
            merged_labels.append(label)
            merged_values = np.append(merged_values, 0.0)
            index_map[label] = len(merged_labels) - 1
        merged_values[index_map[label]] += float(new_values[i])

    return merged_labels, merged_values


def _load_cutflow_from_root_folder(folder: Path) -> Tuple[Optional[List[str]], Optional[np.ndarray]]:
    root_files = sorted([p for p in folder.glob("*.root") if p.is_file()])
    if not root_files:
        return None, None

    merged_labels: Optional[List[str]] = None
    merged_values: Optional[np.ndarray] = None
    loaded_files = 0

    for root_path in root_files:
        parsed = _extract_cutflow_from_root_file(root_path)
        if parsed is None:
            continue
        labels, values = parsed
        merged_labels, merged_values = _merge_cutflow_series(merged_labels, merged_values, labels, values)
        loaded_files += 1

    if loaded_files == 0:
        return None, None

    return merged_labels, merged_values


def _should_skip_pkl_file(path: Path) -> bool:
    name = path.name
    return name.endswith(".awk_raw.pkl") or name.endswith("raw.pkl")


def _filter_objects_for_variables(objects: Dict[str, Any], target_variables: Optional[Sequence[str]]) -> Dict[str, Any]:
    if not target_variables:
        return objects
    target_set = set(target_variables)
    filtered: Dict[str, Any] = {}
    for key, value in objects.items():
        if key in target_set or any(var.startswith(f"{key}_") for var in target_set):
            filtered[key] = value
    return filtered


def merge_pkl_folder(folder: Path, target_variables: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Merge all .pkl files in a folder."""
    pkl_files = sorted([p for p in folder.glob("*.pkl") if p.is_file() and not _should_skip_pkl_file(p)])
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found in folder: {folder}")

    merged: Dict[str, Any] = {}
    total_files = len(pkl_files)
    loaded_files = 0
    skipped_files = 0
    print(f"Loading {total_files} PKL files from {folder.name}...")

    for idx, pkl_path in enumerate(pkl_files, 1):
        try:
            data = _load_single_pkl(pkl_path)
        except Exception as exc:
            skipped_files += 1
            print(f"Warning: Could not load PKL file {pkl_path.name}: {exc}")
            continue
        objects = data.get("objects", {})
        if not isinstance(objects, dict):
            objects = {}
        reduced_data = {
            "n_events": data.get("n_events", 0) or 0,
            "objects": objects,
        }
        merged = _merge_values(merged, reduced_data)
        loaded_files += 1

        # Progress indication every 50 files
        if idx % 50 == 0 or idx == total_files:
            print(f"  Progress: {idx}/{total_files} files loaded")

    if loaded_files == 0:
        raise ValueError(f"All PKL files failed to load in folder: {folder}")
    if skipped_files > 0:
        print(f"  Loaded {loaded_files}/{total_files} PKL files (skipped {skipped_files})")

    merged.setdefault("n_events", 0)
    merged.setdefault("objects", {})
    if not isinstance(merged["objects"], dict):
        raise ValueError(f"Merged objects is not a dictionary in folder: {folder}")

    return merged


def merge_root_folder(folder: Path, target_variables: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Merge all .root files in a folder."""
    root_files = sorted([p for p in folder.glob("*.root") if p.is_file()])
    if not root_files:
        raise FileNotFoundError(f"No .root files found in folder: {folder}")

    merged: Dict[str, Any] = {}
    total_files = len(root_files)
    loaded_files = 0
    skipped_files = 0
    print(f"Loading {total_files} ROOT files from {folder.name}...")

    for idx, root_path in enumerate(root_files, 1):
        try:
            data = _load_single_root(root_path, target_variables=target_variables)
            if not data.get("metadata_n_events_available", False):
                skipped_files += 1
                print(f"Warning: ROOT file {root_path.name} is missing Metadata.n_events; skipping file.")
                continue
            merged = _merge_values(merged, data)
            loaded_files += 1

            # Progress indication every 50 files
            if idx % 50 == 0 or idx == total_files:
                print(f"  Progress: {idx}/{total_files} files loaded")
        except Exception as e:
            print(f"Warning: Could not load ROOT file {root_path.name}: {e}")

    if loaded_files == 0:
        raise ValueError(f"All ROOT files failed to load in folder: {folder}")
    if skipped_files > 0:
        print(f"  Loaded {loaded_files}/{total_files} ROOT files (skipped {skipped_files})")

    merged.setdefault("n_events", 0)
    merged.setdefault("objects", {})
    merged["metadata_n_events_available"] = loaded_files > 0
    if not isinstance(merged["objects"], dict):
        raise ValueError(f"Merged objects is not a dictionary in folder: {folder}")

    return merged


def _flatten_numeric(values: Any) -> np.ndarray:
    # Fast path: already a numpy array of numeric dtype
    if isinstance(values, np.ndarray) and np.issubdtype(values.dtype, np.number):
        flat = values.ravel()
        return flat[np.isfinite(flat)].astype(float)

    result: List[float] = []

    def walk(item: Any) -> None:
        if item is None:
            return
        if isinstance(item, dict):
            return
        if isinstance(item, np.ndarray):
            if np.issubdtype(item.dtype, np.number):
                flat = item.ravel()
                result.extend(flat[np.isfinite(flat)].tolist())
            return
        if isinstance(item, (list, tuple)):
            for sub in item:
                walk(sub)
            return
        if _is_number(item):
            val = float(item)
            if math.isfinite(val):
                result.append(val)

    walk(values)
    if not result:
        return np.array([], dtype=float)
    return np.asarray(result, dtype=float)


def _extract_object_distributions(objects: Dict[str, Any]) -> Dict[str, np.ndarray]:
    distributions: Dict[str, np.ndarray] = {}

    for key, value in objects.items():
        if key.endswith("_mask"):
            continue

        if not isinstance(value, list):
            arr = _flatten_numeric(value)
            if arr.size > 0:
                distributions[key] = arr
            continue

        first_non_none = None
        for item in value:
            if item is not None:
                first_non_none = item
                break

        if first_non_none is None:
            continue

        if isinstance(first_non_none, dict):
            numeric_fields: set[str] = set()
            for item in value:
                if isinstance(item, dict):
                    for field, field_value in item.items():
                        if _is_number(field_value):
                            numeric_fields.add(field)
            for field in sorted(numeric_fields):
                flattened = _flatten_numeric([item.get(field) for item in value if isinstance(item, dict)])
                if flattened.size > 0:
                    distributions[f"{key}_{field}"] = flattened
            continue

        if isinstance(first_non_none, list):
            first_inner = None
            for inner in first_non_none:
                if inner is not None:
                    first_inner = inner
                    break

            if isinstance(first_inner, dict):
                numeric_fields: set[str] = set()
                for event_items in value:
                    if not isinstance(event_items, list):
                        continue
                    for item in event_items:
                        if isinstance(item, dict):
                            for field, field_value in item.items():
                                if _is_number(field_value):
                                    numeric_fields.add(field)

                for field in sorted(numeric_fields):
                    flattened_field: List[float] = []
                    for event_items in value:
                        if not isinstance(event_items, list):
                            continue
                        for item in event_items:
                            if isinstance(item, dict):
                                field_value = item.get(field)
                                if _is_number(field_value):
                                    fv = float(field_value)
                                    if math.isfinite(fv):
                                        flattened_field.append(fv)
                    if flattened_field:
                        distributions[f"{key}_{field}"] = np.asarray(flattened_field, dtype=float)
                continue

            flattened = _flatten_numeric(value)
            if flattened.size > 0:
                distributions[key] = flattened
            continue

        flattened = _flatten_numeric(value)
        if flattened.size > 0:
            distributions[key] = flattened

    return distributions


# Per-variable fixed bin edges (np.arrays built once at import).
# Values derived from StackPlotter_addMeanWeight.py makeplot() XMIN/XMAX/Rebin calls.
# Variable-bin arrays (CTS, MET/pT) override the uniform grid entirely.
_VARIABLE_BINS: Dict[str, np.ndarray] = {
    # ---- angular / delta-phi ----
    "METPhi":        np.linspace(-3.14, 3.14, 21),
    "PFMET_phi":     np.linspace(-3.14, 3.14, 21),
    "dPhi_jetMET":   np.linspace(0.0, 3.2, 17),
    "dPhiJet12":     np.linspace(-7.5, 7.5, 31),
    "DetaJet12":     np.linspace(-7.5, 7.5, 31),
    "dRJet12":       np.linspace(0.0, 7.0, 29),
    "Jet1Eta":       np.linspace(-2.5, 2.5, 21),
    "Jet1Phi":       np.linspace(-3.14, 3.14, 21),
    "Jet2Eta":       np.linspace(-2.5, 2.5, 21),
    "Jet2Phi":       np.linspace(-3.14, 3.14, 21),
    "bjet_eta":      np.linspace(-2.5, 2.5, 21),
    "bjet_phi":      np.linspace(-3.14, 3.14, 21),
    "jet_eta":       np.linspace(-2.5, 2.5, 21),
    "jet_phi":       np.linspace(-3.14, 3.14, 21),
    "electron_eta":  np.linspace(-2.5, 2.5, 21),
    "electron_phi":  np.linspace(-3.14, 3.14, 21),
    "muon_eta":      np.linspace(-2.5, 2.5, 21),
    "muon_phi":      np.linspace(-3.14, 3.14, 21),
    # ---- pT / energy  (variable bins, 250-1000 GeV) ----
    "MET":           np.array([250., 300., 400., 550., 1000.]),
    "PFMET_pt":      np.array([250., 300., 400., 550., 1000.]),
    "recoil":        np.array([250., 300., 400., 550., 1000.]),
    "Jet1Pt":        np.linspace(30., 800., 32),
    "Jet2Pt":        np.linspace(30., 800., 32),
    "bjet_pt":       np.linspace(30., 800., 32),
    "jet_pt":        np.linspace(30., 800., 32),
    "electron_pt":   np.linspace(20., 500., 25),
    "muon_pt":       np.linspace(20., 500., 25),
    "JetHT":         np.linspace(0., 2000., 41),
    # ---- invariant mass ----
    "M_Jet1Jet2":    np.linspace(0., 2000., 41),
    # ---- b-tagging / discriminators ----
    "Jet1deepCSV":   np.linspace(0., 1.2, 25),
    "Jet2deepCSV":   np.linspace(0., 1.2, 25),
    "jet_btag":      np.linspace(0., 1.2, 25),
    # ---- multiplicities (integer-like; half-integer edges) ----
    "n_jets":        np.arange(-0.5, 10.5, 1.0),
    "n_bjets":       np.arange(-0.5, 10.5, 1.0),
    "n_electrons":   np.arange(-0.5, 10.5, 1.0),
    "n_muons":       np.arange(-0.5, 10.5, 1.0),
    "n_taus":        np.arange(-0.5, 10.5, 1.0),
    # ---- CTS / BDT score (variable bins, 0-1) ----
    "costheta_star": np.array([0.0, 0.25, 0.50, 0.75, 1.0]),
    "PFMET_significance": np.linspace(0., 100., 26),
}

# Case-insensitive suffix patterns that map to the MET variable-bin edges.
_MET_SUFFIX_PATTERNS = ("met_pt", "recoil")


def _reference_bins_for_variable(variable: Optional[str]) -> Optional[np.ndarray]:
    if not variable:
        return None
    # Exact match first
    bins = _VARIABLE_BINS.get(variable)
    if bins is not None:
        return bins
    # Suffix / substring fallback for dynamically-named branches
    name = variable.lower()
    if "ctsvalue" in name:
        return _VARIABLE_BINS["costheta_star"]
    if any(name.endswith(p) or p in name for p in _MET_SUFFIX_PATTERNS):
        return _VARIABLE_BINS["MET"]
    return None


def _make_bins(all_values: Sequence[np.ndarray], n_bins: int, variable: Optional[str] = None) -> Optional[np.ndarray]:
    reference_bins = _reference_bins_for_variable(variable)
    if reference_bins is not None:
        return reference_bins

    valid_arrays = [arr[np.isfinite(arr)] for arr in all_values if arr.size > 0]
    if not valid_arrays:
        return None

    merged = np.concatenate(valid_arrays)
    if merged.size == 0:
        return None

    data_min = float(np.min(merged))
    data_max = float(np.max(merged))
    if not math.isfinite(data_min) or not math.isfinite(data_max):
        return None

    if abs(data_max - data_min) < 1e-12:
        width = max(1.0, abs(data_max) * 0.05)
        return np.linspace(data_min - width, data_max + width, n_bins + 1)

    is_integer_like = np.allclose(merged, np.round(merged), atol=1e-8)
    unique_count = len(np.unique(np.round(merged).astype(int))) if is_integer_like else 999
    if is_integer_like and unique_count <= 20:
        low = int(np.min(np.round(merged)))
        high = int(np.max(np.round(merged)))
        return np.arange(low - 0.5, high + 1.5, 1.0)

    return np.linspace(data_min, data_max, n_bins + 1)


def _histogram(values: np.ndarray, bins: np.ndarray, n_events: int, luminosity: float = 1.0, cross_section_pb: Optional[float] = None) -> np.ndarray:
    """Create histogram normalized by n_events and scaled by luminosity and cross-section.

    Normalization formula:
    - If cross_section_pb is provided: weight = (luminosity × cross_section_fb) / n_events
      where cross_section_fb = cross_section_pb × 1000 (convert pb to fb)
    - If cross_section_pb is None: weight = luminosity / n_events
    """
    if values.size == 0 or n_events <= 0:
        return np.zeros(len(bins) - 1, dtype=float)

    if cross_section_pb is not None:
        # Convert cross-section from pb to fb and apply
        cross_section_fb = cross_section_pb * 1000.0
        weight = (luminosity * cross_section_fb) / float(n_events)
    else:
        weight = luminosity / float(n_events)

    values = _clip_overflow(values, bins)
    weights = np.full(values.shape[0], weight, dtype=float)
    hist, _ = np.histogram(values, bins=bins, weights=weights)
    return hist


def _event_weight(n_events: int, luminosity: float = 1.0, cross_section_pb: Optional[float] = None) -> float:
    """Compute per-event weight with the same convention used by histogram normalization."""
    if n_events <= 0:
        return 0.0
    if cross_section_pb is not None:
        # Keep consistency with histogram code: convert pb to fb.
        return (luminosity * cross_section_pb * 1000.0) / float(n_events)
    return luminosity / float(n_events)


def _scale_cutflow(
    cutflow_values: np.ndarray,
    n_events: int,
    luminosity: float = 1.0,
    cross_section_pb: Optional[float] = None,
) -> np.ndarray:
    weight = _event_weight(n_events, luminosity=luminosity, cross_section_pb=cross_section_pb)
    return np.asarray(cutflow_values, dtype=float) * weight


def _clip_overflow(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Clip values outside [bins[0], bins[-1]) into the first/last bin.

    Mirrors ROOT TH1 overflow merging: entries above the last edge are pulled
    just inside it so np.histogram counts them in the last bin.
    """
    lo, hi = float(bins[0]), float(bins[-1])
    # np.histogram includes the right edge only for the last bin, so nudge
    # overflow to exactly the last left edge to land in the last bin.
    return np.clip(values, lo, np.nextafter(hi, -np.inf))


def _histogram_and_sumw2(
    values: np.ndarray,
    bins: np.ndarray,
    n_events: int,
    luminosity: float = 1.0,
    cross_section_pb: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create weighted histogram and per-bin sumw2 (ROOT-compatible stat unc).

    For a constant event weight w in a given sample, per-bin variance is n_bin * w^2,
    equivalent to ROOT TH1 Sumw2 behavior after scaling.
    Overflow/underflow are merged into the first/last bin (matches ROOT behaviour).
    """
    if n_events <= 0 or values.size == 0:
        zeros = np.zeros(len(bins) - 1, dtype=float)
        return zeros, zeros

    if cross_section_pb is not None:
        cross_section_fb = cross_section_pb * 1000.0
        weight = (luminosity * cross_section_fb) / float(n_events)
    else:
        weight = luminosity / float(n_events)

    values = _clip_overflow(values, bins)
    counts, _ = np.histogram(values, bins=bins)
    counts = counts.astype(float)
    hist = counts * weight
    sumw2 = counts * (weight ** 2)
    return hist, sumw2


def _histogram_counts(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Create raw-count histogram without normalization (for data).
    Overflow/underflow are merged into the first/last bin.
    """
    if values.size == 0:
        return np.zeros(len(bins) - 1, dtype=float)
    values = _clip_overflow(values, bins)
    hist, _ = np.histogram(values, bins=bins)
    return hist.astype(float)


def _simplify_sample_label(sample_name: str) -> str:
    label = sample_name
    if label.startswith("new"):
        label = label[3:]
    label = re.sub(r"_20\d{2}(EE)?_.*$", "", label)
    label = re.sub(r"_EVENTSELECTION$", "", label)
    # Normalize known aliases to canonical keys used in _PROCESS_CONFIG
    _alias: Dict[str, str] = {
        "singletop": "SingleTop",
        "smhiggs":   "SMHiggs",
        "wtolnu-2jets": "WtoLNu-2Jets",
        "wtolnu-4jets": "WtoLNu-2Jets",
        "zto2nu-2jets": "Zto2Nu-2Jets",
        "dyto2l-2jets": "DYto2L-2Jets",
        "top":       "Top",
        "diboson":   "DIBOSON",
    }
    return _alias.get(label.lower(), label)


# Single source of truth for all per-process metadata.
# Keys are canonical process names (same as _simplify_sample_label output).
_PROCESS_CONFIG: Dict[str, Dict[str, str]] = {
    "DYto2L-2Jets": {"color": "#3f90da", "label": r"$Z(\ell\ell)+$jets"},
    "Zto2Nu-2Jets": {"color": "#a96b59", "label": r"$Z(\nu\bar{\nu})+$jets"},
    "WtoLNu-2Jets": {"color": "#bd1f01", "label": r"W($\ell\nu$)+jets"},
    "Top":          {"color": "#e76300", "label": r"$t\bar{t}$"},
    "DIBOSON":      {"color": "#832db6", "label": r"WW/WZ/ZZ"},
    "SingleTop":    {"color": "#ffa90e", "label": r"Single $t$"},
    "SMHiggs":      {"color": "#b9ac70", "label": r"SMH"},
    "GJets":        {"color": "#94a4a2", "label": r"$\gamma$+jets"},
    "QCD":          {"color": "#717581", "label": r"QCD"},
}

# CMS 10-color accessible palette (M. Petroff, arXiv:2107.02270v2) — fallback for unknown processes
_PALETTE = ["#3f90da","#ffa90e","#bd1f01","#94a4a2","#832db6","#a96b59","#e76300","#b9ac70","#717581","#92dadd"]


def _get_background_color_map(sample_names: Sequence[str]) -> Dict[str, str]:
    """Assign fixed colors to known background processes; cycle palette for unknowns."""
    color_map: Dict[str, str] = {}
    palette_idx = len(_PROCESS_CONFIG)  # first N slots reserved for known processes
    for name in sorted({_simplify_sample_label(n) for n in sample_names}):
        cfg = _PROCESS_CONFIG.get(name)
        if cfg:
            color_map[name] = cfg["color"]
        else:
            color_map[name] = _PALETTE[palette_idx % len(_PALETTE)]
            palette_idx += 1
    return color_map


def _get_legend_label(canonical_name: str) -> str:
    cfg = _PROCESS_CONFIG.get(canonical_name)
    return cfg["label"] if cfg else canonical_name


def _variable_unit(variable: str) -> str:
    name = variable.lower()
    if name.endswith("_pt"):
        return "GeV"
    if name.endswith("_phi"):
        return "rad"
    return "arb. unit"


def _apply_variable_plot_filter(variable: str, values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    if variable.lower().endswith("met_pt"):
        return values[values >= 100.0]
    return values


_AXIS_LABELS: Dict[str, str] = {
    "Jet1Pt": r"Jet$_1$ $p_T$ [GeV]",
    "bjet_eta": r"b-jet $\eta$",
    "bjet_phi": r"b-jet $\phi$",
    "bjet_pt": r"b-jet $p_T$ [GeV]",
    "costheta_star": r"$\cos\theta^*$",
    "DetaJet12": r"$\Delta\eta$(Jet$_1$, Jet$_2$)",
    "dPhi_jetMET": r"$\Delta\phi$(jet, $p_T^{miss}$)",
    "dPhiJet12": r"$\Delta\phi$(Jet$_1$, Jet$_2$)",
    "dRJet12": r"$\Delta R$(Jet$_1$, Jet$_2$)",
    "electron_eta": r"Electron $\eta$",
    "electron_phi": r"Electron $\phi$",
    "electron_pt": r"Electron $p_T$ [GeV]",
    "eta_Jet1Jet2": r"$\eta$(Jet$_1$, Jet$_2$)",
    "event": r"Event",
    "jet_btag": r"Jet b-tag",
    "jet_eta": r"Jet $\eta$",
    "jet_phi": r"Jet $\phi$",
    "jet_pt": r"Jet $p_T$ [GeV]",
    "Jet1deepCSV": r"Jet$_1$ DeepCSV",
    "Jet1Eta": r"Jet$_1$ $\eta$",
    "Jet1Phi": r"Jet$_1$ $\phi$",
    "Jet2deepCSV": r"Jet$_2$ DeepCSV",
    "Jet2Eta": r"Jet$_2$ $\eta$",
    "Jet2Phi": r"Jet$_2$ $\phi$",
    "Jet2Pt": r"Jet$_2$ $p_T$ [GeV]",
    "JetHT": r"Jet H$_T$ [GeV]",
    "luminosityBlock": r"Luminosity Block",
    "M_Jet1Jet2": r"M(Jet$_1$, Jet$_2$) [GeV]",
    "MET": r"$p_T^{miss}$ [GeV]",
    "METPhi": r"$\phi(p_T^{miss})$",
    "muon_eta": r"Muon $\eta$",
    "muon_phi": r"Muon $\phi$",
    "muon_pt": r"Muon $p_T$ [GeV]",
    "n_bjets": r"Number of b-jets",
    "n_electrons": r"Number of electrons",
    "n_jets": r"Number of jets",
    "n_muons": r"Number of muons",
    "n_taus": r"Number of taus",
    "nbjet_eta": r"b-jet $\eta$ (count)",
    "nbjet_phi": r"b-jet $\phi$ (count)",
    "nelectron_eta": r"electron $\eta$ (count)",
    "nelectron_phi": r"electron $\phi$ (count)",
    "nfatjet_eta": r"Fatjet $\eta$ (count)",
    "nfatjet_mass": r"Fatjet mass [GeV] (count)",
    "nfatjet_phi": r"Fatjet $\phi$ (count)",
    "njet_btag": r"Jet b-tag (count)",
    "njet_eta": r"Jet $\eta$ (count)",
    "njet_phi": r"Jet $\phi$ (count)",
    "PFMET_phi": r"$\phi(p_T^{miss})$ (PF)",
    "PFMET_pt": r"$p_T^{miss}$ (PF) [GeV]",
    "PFMET_significance": r"$p_T^{miss}$ significance",
    "recoil": r"Hadronic Recoil [GeV]",
}


def _axis_label(variable: str) -> str:
    """Return the formatted x-axis label for a variable, falling back to the variable name."""
    return _AXIS_LABELS.get(variable, variable)


def _root_safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name)


def _root_hist_from_numpy(
    name: str,
    bins: np.ndarray,
    values: np.ndarray,
    errors: Optional[np.ndarray] = None,
    labels: Optional[Sequence[str]] = None,
):
    if ROOT is None:
        raise RuntimeError("ROOT is not available, cannot write ROOT output")

    hist = ROOT.TH1D(
        _root_safe_name(name),
        "",
        len(bins) - 1,
        array.array("d", [float(v) for v in bins]),
    )
    hist.SetDirectory(0)
    hist.Sumw2()
    hist.SetStats(0)
    for index, value in enumerate(values, start=1):
        hist.SetBinContent(index, float(value))
        if errors is not None:
            hist.SetBinError(index, float(errors[index - 1]))
        elif value >= 0:
            hist.SetBinError(index, math.sqrt(float(value)))
    if labels is not None:
        for index, label in enumerate(labels, start=1):
            hist.GetXaxis().SetBinLabel(index, str(label))
    return hist


def _save_stacked_root_plot(
    root_path: Path,
    plot_title: str,
    x_title: str,
    y_title: str,
    bins: np.ndarray,
    background_hists: List[Tuple[str, np.ndarray, np.ndarray]],
    data_hist: Optional[np.ndarray],
    luminosity: float,
    labels: Optional[Sequence[str]] = None,
) -> None:
    if ROOT is None:
        raise RuntimeError("ROOT is required for --save-root but is not available in this environment")

    root_file = ROOT.TFile.Open(str(root_path), "RECREATE")
    if root_file is None or root_file.IsZombie():
        raise OSError(f"Could not create ROOT file: {root_path}")

    try:
        ROOT.gStyle.SetOptStat(0)
        canvas = ROOT.TCanvas(_root_safe_name(f"c_{root_path.stem}"), plot_title, 1200, 900)
        canvas.cd()

        has_ratio = data_hist is not None
        if has_ratio:
            top_pad = ROOT.TPad("top_pad", "top_pad", 0.0, 0.30, 1.0, 1.0)
            bottom_pad = ROOT.TPad("bottom_pad", "bottom_pad", 0.0, 0.0, 1.0, 0.30)
            top_pad.SetBottomMargin(0.03)
            bottom_pad.SetTopMargin(0.03)
            bottom_pad.SetBottomMargin(0.33)
            bottom_pad.SetGridy(False)
            top_pad.Draw()
            bottom_pad.Draw()
        else:
            top_pad = canvas
            bottom_pad = None

        # Main stack
        top_pad.cd()
        stack = ROOT.THStack(_root_safe_name(f"stack_{root_path.stem}"), "")
        legend = ROOT.TLegend(0.63, 0.60, 0.92, 0.90)
        legend.SetBorderSize(0)
        legend.SetFillStyle(0)
        legend.SetTextFont(42)

        background_root_hists = []
        total_values = np.zeros(len(bins) - 1, dtype=float)
        total_sumw2 = np.zeros(len(bins) - 1, dtype=float)
        color_map = _get_background_color_map([label for label, _, _ in background_hists])

        for label, hist_values, hist_sumw2 in background_hists:
            total_values += hist_values
            total_sumw2 += hist_sumw2
            root_hist = _root_hist_from_numpy(
                f"h_{label}",
                bins,
                hist_values,
                errors=np.sqrt(hist_sumw2),
                labels=labels,
            )
            root_hist.SetFillColor(int(ROOT.TColor.GetColor(color_map.get(label, "#3f90da"))))
            root_hist.SetLineColor(ROOT.kBlack)
            root_hist.SetLineWidth(1)
            background_root_hists.append(root_hist)
            stack.Add(root_hist, "hist")
            legend.AddEntry(root_hist, _get_legend_label(label), "f")

        stack.SetTitle(f";{x_title};{y_title}")
        stack.Draw("hist")
        stack.GetYaxis().SetTitleOffset(1.2)
        stack.SetMaximum(max(float(np.max(total_values)) * 1.4 if total_values.size else 1.0, 1.0))

        total_band = _root_hist_from_numpy(
            f"band_{root_path.stem}",
            bins,
            total_values,
            errors=np.sqrt(total_sumw2),
            labels=labels,
        )
        total_band.SetFillColor(ROOT.kGray + 2)
        total_band.SetFillStyle(3004)
        total_band.SetLineColor(ROOT.kGray + 2)
        total_band.Draw("e2 same")
        legend.AddEntry(total_band, "Uncertainty", "f")

        data_graph = None
        if data_hist is not None:
            centers = 0.5 * (bins[:-1] + bins[1:])
            half_width = 0.5 * (bins[1:] - bins[:-1])
            data_graph = ROOT.TGraphErrors(int(len(data_hist)))
            data_graph.SetName(_root_safe_name(f"data_{root_path.stem}"))
            data_graph.SetMarkerStyle(20)
            data_graph.SetMarkerSize(1.0)
            data_graph.SetLineColor(ROOT.kBlack)
            data_graph.SetMarkerColor(ROOT.kBlack)
            for index, value in enumerate(data_hist):
                data_graph.SetPoint(index, float(centers[index]), float(value))
                data_graph.SetPointError(index, float(half_width[index]), math.sqrt(float(value)) if value > 0 else 0.0)
            data_graph.Draw("P same")
            legend.AddEntry(data_graph, "Data", "pe")

        legend.Draw()

        if has_ratio and bottom_pad is not None:
            bottom_pad.cd()
            ratio_hist = _root_hist_from_numpy(
                f"ratio_{root_path.stem}",
                bins,
                np.ones(len(bins) - 1, dtype=float),
                errors=np.divide(np.sqrt(total_sumw2), total_values, out=np.zeros_like(total_values), where=total_values > 0),
                labels=labels,
            )
            ratio_hist.SetTitle(f";{x_title};Data / MC")
            ratio_hist.SetMinimum(0.0)
            ratio_hist.SetMaximum(2.0)
            ratio_hist.SetFillColor(ROOT.kGray + 2)
            ratio_hist.SetFillStyle(3004)
            ratio_hist.SetLineColor(ROOT.kGray + 2)
            ratio_hist.Draw("e2")

            unity = ROOT.TLine(float(bins[0]), 1.0, float(bins[-1]), 1.0)
            unity.SetLineColor(ROOT.kBlack)
            unity.SetLineWidth(2)
            unity.Draw("same")

            if data_hist is not None:
                centers = 0.5 * (bins[:-1] + bins[1:])
                half_width = 0.5 * (bins[1:] - bins[:-1])
                ratio_graph = ROOT.TGraphErrors(int(len(data_hist)))
                ratio_graph.SetName(_root_safe_name(f"ratio_data_{root_path.stem}"))
                ratio_graph.SetMarkerStyle(20)
                ratio_graph.SetMarkerSize(1.0)
                ratio_graph.SetLineColor(ROOT.kBlack)
                ratio_graph.SetMarkerColor(ROOT.kBlack)
                for index, value in enumerate(data_hist):
                    pred = float(total_values[index])
                    ratio = float(value) / pred if pred > 0 else 0.0
                    ratio_err = math.sqrt(float(value)) / pred if pred > 0 and value > 0 else 0.0
                    ratio_graph.SetPoint(index, float(centers[index]), ratio)
                    ratio_graph.SetPointError(index, float(half_width[index]), ratio_err)
                ratio_graph.Draw("P same")

        canvas.Write()
        for hist in background_root_hists:
            hist.Write()
        total_band.Write()
        if data_graph is not None:
            data_graph.Write()
        root_file.Write()
    finally:
        root_file.Close()


def _plot_stacked_variable(
    variable: str,
    bins: np.ndarray,
    background_hists: List[Tuple[str, np.ndarray, np.ndarray]],
    data_hist: Optional[np.ndarray],
    output_dir: Path,
    luminosity: float = 1.0,
    color_map: Optional[Dict[str, str]] = None,
) -> Path:
    """Produce CMS-style stacked histogram matching StackPlotter_addMeanWeight.py."""
    background_hists = [(_simplify_sample_label(label), hist, sumw2) for label, hist, sumw2 in background_hists]
    background_hists = sorted(background_hists, key=lambda item: float(np.sum(item[1])))

    if color_map is None:
        color_map = _get_background_color_map([label for label, _, _ in background_hists])

    show_ratio = data_hist is not None

    if show_ratio:
        fig, (ax, ax_ratio) = plt.subplots(
            2, 1,
            figsize=(12, 12),
            sharex=True,
            gridspec_kw={"height_ratios": [3.0, 1.0], "hspace": 0.08},
        )
        fig.subplots_adjust(top=0.92, bottom=0.09, left=0.14, right=0.95)
    else:
        fig, ax = plt.subplots(figsize=(12, 10))
        ax_ratio = None
        fig.subplots_adjust(top=0.92, bottom=0.12, left=0.14, right=0.95)

    # =========================== Main Stack Plot ===========================
    cumulative = np.zeros(len(bins) - 1, dtype=float)
    cumulative_sq = np.zeros(len(bins) - 1, dtype=float)

    for label, hist_values, hist_sumw2 in background_hists:
        next_cumulative = cumulative + hist_values
        cumulative_sq += hist_sumw2
        color = color_map.get(label, "#3f90da")
        legend_label = _get_legend_label(label)
        ax.stairs(
            next_cumulative,
            bins,
            baseline=cumulative,
            fill=True,
            alpha=1.0,
            linewidth=0,
            color=color,
            label=legend_label,
        )
        cumulative = next_cumulative

    # MC stat uncertainty band — gray fill + diagonal hatch (matches reference)
    # drawn before data so data points sit on top
    mc_stat_err = np.sqrt(cumulative_sq)
    centers = 0.5 * (bins[:-1] + bins[1:])
    # Use step="post" on left bin edges so the band spans exactly each bin's
    # full width — critical for variable-bin plots where bin widths are unequal.
    # Append the last right edge value so the final bin is fully covered.
    unc_x = np.append(bins[:-1], bins[-1])
    unc_lo = np.append(cumulative - mc_stat_err, (cumulative - mc_stat_err)[-1])
    unc_hi = np.append(cumulative + mc_stat_err, (cumulative + mc_stat_err)[-1])
    ax.fill_between(
        unc_x,
        unc_lo,
        unc_hi,
        step="post",
        hatch="////",
        facecolor="#bbbbbb",
        edgecolor="#666666",
        linewidth=0.0,
        alpha=0.8,
        label="Uncertainty",
        zorder=5,
    )

    # Data: filled circle, x errors to bin edges, y errors Poisson
    if data_hist is not None:
        half_width = 0.5 * (bins[1:] - bins[:-1])
        mask = data_hist > 0
        if np.any(mask):
            ax.errorbar(
                centers[mask],
                data_hist[mask],
                xerr=half_width[mask],
                yerr=np.sqrt(data_hist[mask]),
                fmt="o",
                color="black",
                markerfacecolor="black",
                markeredgecolor="black",
                markersize=5.5,
                elinewidth=1.2,
                capsize=0,
                label="Data",
                zorder=10,
            )

    # Y axis: "Events / bin", scientific notation max 3 digits (×10³ style)
    ax.set_yscale("log")
    stacked_max = float(np.max(cumulative)) if cumulative.size else 0.0
    data_max = float(np.max(data_hist)) if data_hist is not None and data_hist.size else 0.0
    ymax = max(stacked_max, data_max, 1e-3) * 1000.0
    ax.set_ylim(0.1, ymax)

    # X range: trim to the extent of actual content so there is no empty gap at
    # either edge (e.g. Jet1Pt bins start at 30 but data only starts at ~100).
    # Use the stacked MC total (cumulative) as the reference — it covers all
    # backgrounds. Then also extend to cover any data points outside that range.
    nonzero_mc = np.where(cumulative > 0)[0]
    if nonzero_mc.size:
        x_lo = float(bins[nonzero_mc[0]])
        x_hi = float(bins[nonzero_mc[-1] + 1])
    else:
        x_lo, x_hi = float(bins[0]), float(bins[-1])
    if data_hist is not None:
        nonzero_data = np.where(data_hist > 0)[0]
        if nonzero_data.size:
            x_lo = min(x_lo, float(bins[nonzero_data[0]]))
            x_hi = max(x_hi, float(bins[nonzero_data[-1] + 1]))
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylabel("Events / bin", fontsize=22, labelpad=6)

    if show_ratio:
        ax.tick_params(axis="x", labelbottom=False)
    else:
        ax.set_xlabel(_axis_label(variable), fontsize=22)

    # X-axis ticks: equidistant round numbers for all plots via MaxNLocator.
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=8, steps=[1, 2, 5, 10]))
    ax.grid(False)

    # =========================== Ratio Plot (Data / SM) ===========================
    if show_ratio and ax_ratio is not None and data_hist is not None:
        pred = cumulative

        data_ratio = np.divide(
            data_hist,
            pred,
            out=np.full_like(data_hist, np.nan, dtype=float),
            where=pred > 0,
        )
        data_ratio_err = np.divide(
            np.sqrt(data_hist),
            pred,
            out=np.zeros_like(data_hist, dtype=float),
            where=pred > 0,
        )
        ratio_mask = np.isfinite(data_ratio)

        pred_rel_err = np.divide(
            mc_stat_err,
            pred,
            out=np.zeros_like(pred, dtype=float),
            where=pred > 0,
        )

        # unity line
        ax_ratio.axhline(1.0, color="black", linestyle="-", linewidth=1.2)

        # uncertainty band — same gray hatch style as main plot
        ratio_lo = np.append(1.0 - pred_rel_err, (1.0 - pred_rel_err)[-1])
        ratio_hi = np.append(1.0 + pred_rel_err, (1.0 + pred_rel_err)[-1])
        ax_ratio.fill_between(
            unc_x,
            ratio_lo,
            ratio_hi,
            step="post",
            hatch="////",
            facecolor="#bbbbbb",
            edgecolor="#3d3d3d",
            linewidth=0.0,
            alpha=0.8,
            zorder=5,
        )

        if np.any(ratio_mask):
            half_width = 0.5 * (bins[1:] - bins[:-1])
            ax_ratio.errorbar(
                centers[ratio_mask],
                data_ratio[ratio_mask],
                xerr=half_width[ratio_mask],
                yerr=data_ratio_err[ratio_mask],
                fmt="o",
                color="black",
                markerfacecolor="black",
                markeredgecolor="black",
                markersize=5.5,
                elinewidth=1.0,
                capsize=0,
                zorder=10,
            )

        ax_ratio.set_ylabel("Data / MC", fontsize=22, labelpad=6)
        ax_ratio.set_xlabel(_axis_label(variable), fontsize=22, labelpad=8)
        ax_ratio.set_ylim(0, 2.0)
        ax_ratio.set_xlim(x_lo, x_hi)
        ax_ratio.axhline(1.0, color="none")  # force ylim anchor
        ax_ratio.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=8, steps=[1, 2, 5, 10]))
        ax_ratio.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0, 0.5, 1.0, 1.5, 2.0]))
        ax_ratio.grid(False)

    # =========================== CMS Labels ===========================
    hep.cms.label(
        "Work in progress",
        data=data_hist is not None,
        lumi=round(luminosity, 2),
        com=13.6,
        loc=0,
        ax=ax,
    )

    # =========================== Legend ===========================
    # cmsLeg(0.42, 0.50, 0.92, 0.88, textSize=0.045, columns=2)
    # Data first, then backgrounds, then Uncertainty patch last
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        # reorder: Data first, Uncertainty last
        data_idx = next((i for i, l in enumerate(labels) if l == "Data"), None)
        unc_idx  = next((i for i, l in enumerate(labels) if l == "Uncertainty"), None)
        ordered_idx = (
            ([data_idx] if data_idx is not None else [])
            + [i for i in range(len(labels)) if i not in (data_idx, unc_idx)]
            + ([unc_idx] if unc_idx is not None else [])
        )
        handles = [handles[i] for i in ordered_idx]
        labels  = [labels[i]  for i in ordered_idx]
        # replace Uncertainty handle with explicit patch for correct legend swatch
        if unc_idx is not None:
            handles[-1] = matplotlib.patches.Patch(
                hatch="////", facecolor="#bbbbbb", edgecolor="#3d3d3d", linewidth=0.0,
                label="Uncertainty",
            )
        ax.legend(
            handles,
            labels,
            loc="upper right",
            bbox_to_anchor=(0.97, 0.97),
            ncol=2,
            frameon=False,
            borderaxespad=0.0,
            handlelength=1.5,
            columnspacing=1.0,
            handletextpad=0.5,
            fontsize=20,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"stacked_{variable}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _align_cutflow_to_labels(
    source_labels: Optional[Sequence[str]],
    source_values: Optional[np.ndarray],
    target_labels: Sequence[str],
) -> np.ndarray:
    aligned = np.zeros(len(target_labels), dtype=float)
    if source_labels is None or source_values is None:
        return aligned
    value_map = {
        str(source_labels[i]): float(source_values[i])
        for i in range(min(len(source_labels), len(source_values)))
    }
    for i, label in enumerate(target_labels):
        aligned[i] = value_map.get(str(label), 0.0)
    return aligned


def _plot_stacked_cutflow(
    cut_labels: Sequence[str],
    background_cutflows: List[Tuple[str, np.ndarray]],
    data_cutflow: Optional[np.ndarray],
    output_dir: Path,
    luminosity: float = 1.0,
    color_map: Optional[Dict[str, str]] = None,
    output_name: str = "stacked_cutflow.png",
    y_label: str = "Events",
    y_scale: Literal["linear", "log"] = "log",
) -> Path:
    background_cutflows = [(_simplify_sample_label(label), values) for label, values in background_cutflows]
    background_cutflows = sorted(background_cutflows, key=lambda item: float(np.sum(item[1])))

    if color_map is None:
        color_map = _get_background_color_map([label for label, _ in background_cutflows])

    fig, ax = plt.subplots(figsize=(14, 9))
    fig.subplots_adjust(top=0.9, bottom=0.25, left=0.12, right=0.95)

    x = np.arange(len(cut_labels), dtype=float)
    cumulative = np.zeros(len(cut_labels), dtype=float)

    for label, values in background_cutflows:
        color = color_map.get(label, "#3f90da")
        ax.bar(
            x,
            values,
            bottom=cumulative,
            width=0.8,
            color=color,
            edgecolor="none",
            label=_get_legend_label(label),
        )
        cumulative += values

    # Optional data overlay (restored): draw points with Poisson uncertainties.
    has_data_overlay = False
    if data_cutflow is not None:
        mask = np.asarray(data_cutflow > 0, dtype=bool)
        if np.any(mask):
            has_data_overlay = True
            ax.errorbar(
                x[mask],
                data_cutflow[mask],
                yerr=np.sqrt(data_cutflow[mask]),
                fmt="o",
                color="black",
                markerfacecolor="black",
                markeredgecolor="black",
                markersize=5.5,
                elinewidth=1.2,
                capsize=0,
                label="Data",
                zorder=10,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([str(label) for label in cut_labels], rotation=30, ha="right")
    ax.set_ylabel(y_label, fontsize=22, labelpad=6)
    ax.set_xlabel("Cut step", fontsize=22)
    ax.grid(False)
    ax.set_yscale(y_scale)

    if y_scale == "log":
        positive = cumulative[cumulative > 0]
        data_positive = np.array([], dtype=float)
        if data_cutflow is not None:
            data_positive = np.asarray(data_cutflow[data_cutflow > 0], dtype=float)
        if positive.size and data_positive.size:
            ymin = max(1e-3, min(float(np.min(positive)), float(np.min(data_positive))) * 0.5)
        elif positive.size:
            ymin = max(1e-3, float(np.min(positive)) * 0.5)
        elif data_positive.size:
            ymin = max(1e-3, float(np.min(data_positive)) * 0.5)
        else:
            ymin = 1e-3
        # Keep cutflow log-axis readable: always show down to at least 1e2.
        ymin = min(ymin, 1e2)
        ymax = max(
            1.0,
            float(np.max(cumulative)) if cumulative.size else 1.0,
            float(np.max(data_cutflow)) if data_cutflow is not None and data_cutflow.size else 1.0,
        )
        ax.set_ylim(ymin, ymax * 10.0)
    else:
        ymax = max(
            1.0,
            float(np.max(cumulative)) if cumulative.size else 1.0,
            float(np.max(data_cutflow)) if data_cutflow is not None and data_cutflow.size else 1.0,
        )
        ax.set_ylim(0.0, ymax * 1.25)

    hep.cms.label(
        "Work in progress",
        data=has_data_overlay,
        lumi=round(luminosity, 1),
        com=13.6,
        loc=0,
        ax=ax,
    )

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        data_idx = next((i for i, l in enumerate(labels) if l == "Data"), None)
        ordered_idx = ([data_idx] if data_idx is not None else []) + [i for i in range(len(labels)) if i != data_idx]
        handles = [handles[i] for i in ordered_idx]
        labels = [labels[i] for i in ordered_idx]
        ax.legend(
            handles,
            labels,
            loc="upper right",
            bbox_to_anchor=(0.97, 0.97),
            ncol=2,
            frameon=False,
            borderaxespad=0.0,
            handlelength=1.5,
            columnspacing=1.0,
            handletextpad=0.5,
            fontsize=16,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / output_name
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _load_sample_from_folder(
    folder: Path,
    luminosity: float = 1.0,
    file_type: Literal["auto", "pkl", "root"] = "auto",
    target_variables: Optional[Sequence[str]] = None,
    xsection_data: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    year: Optional[str] = None,
    is_data: bool = False,
    load_cutflow: bool = False,
) -> SampleData:
    """Load sample from folder with configurable file-type selection."""
    root_files = [p for p in folder.glob("*.root") if p.is_file()]
    pkl_files = [p for p in folder.glob("*.pkl") if p.is_file() and not _should_skip_pkl_file(p)]

    if file_type == "pkl":
        if not pkl_files:
            raise FileNotFoundError(f"No eligible .pkl files found in folder: {folder}")
        merged = merge_pkl_folder(folder, target_variables=target_variables)
    elif file_type == "root":
        if not root_files:
            raise FileNotFoundError(f"No .root files found in folder: {folder}")
        merged = merge_root_folder(folder, target_variables=target_variables)
    else:
        if root_files:
            merged = merge_root_folder(folder, target_variables=target_variables)
        elif pkl_files:
            merged = merge_pkl_folder(folder, target_variables=target_variables)
        else:
            raise FileNotFoundError(f"No .pkl or .root files found in folder: {folder}")

    n_events = int(merged.get("n_events", 0) or 0)
    objects = merged.get("objects", {})
    metadata_n_events_available = bool(merged.get("metadata_n_events_available", True))
    if not isinstance(objects, dict):
        raise ValueError(f"objects is not a dictionary for folder: {folder}")

    # Build per-process sub-samples for ROOT inputs so each pt bin uses its own cross-section.
    sub_samples: Optional[List[SubSampleData]] = None
    cutflow_labels: Optional[List[str]] = None
    cutflow_values: Optional[np.ndarray] = None
    if load_cutflow and root_files:
        cutflow_labels, cutflow_values = _load_cutflow_from_root_folder(folder)

    if root_files and xsection_data and year and not is_data:
        process_to_xsec = _build_xsection_process_lookup(xsection_data, year)
        per_process_merged: Dict[str, Dict[str, Any]] = {}
        unmatched_files: List[str] = []
        unmatched_merged: Dict[str, Any] = {
            "n_events": 0,
            "objects": {},
            "metadata_n_events_available": True,
            "cross_section": None,
            "cutflow_labels": None,
            "cutflow_values": None,
        }

        for root_file in sorted(root_files):
            try:
                loaded = _load_single_root(root_file, target_variables=target_variables)
            except Exception as exc:
                print(f"Warning: Could not load ROOT file {root_file.name}: {exc}")
                continue

            parsed_cutflow = _extract_cutflow_from_root_file(root_file)

            if not loaded.get("metadata_n_events_available", False):
                print(
                    f"Warning: ROOT file {root_file.name} is missing Metadata.n_events; skipping for xsection grouping."
                )
                continue

            match = _match_root_file_to_process(root_file.name, process_to_xsec)
            if match is None:
                unmatched_files.append(root_file.name)
                unmatched_merged["n_events"] = int(unmatched_merged.get("n_events", 0) or 0) + int(
                    loaded.get("n_events", 0) or 0
                )
                unmatched_merged["objects"] = _merge_values(unmatched_merged.get("objects", {}), loaded.get("objects", {}))
                if parsed_cutflow is not None:
                    cf_labels, cf_values = parsed_cutflow
                    merged_labels, merged_values = _merge_cutflow_series(
                        unmatched_merged.get("cutflow_labels"),
                        unmatched_merged.get("cutflow_values"),
                        cf_labels,
                        cf_values,
                    )
                    unmatched_merged["cutflow_labels"] = merged_labels
                    unmatched_merged["cutflow_values"] = merged_values
                continue
            process_key, xsec_pb = match
            group = per_process_merged.get(process_key)
            if group is None:
                group = {
                    "n_events": 0,
                    "objects": {},
                    "metadata_n_events_available": True,
                    "cross_section": xsec_pb,
                    "cutflow_labels": None,
                    "cutflow_values": None,
                }
                per_process_merged[process_key] = group
            group["n_events"] = int(group.get("n_events", 0) or 0) + int(loaded.get("n_events", 0) or 0)
            group["objects"] = _merge_values(group.get("objects", {}), loaded.get("objects", {}))
            if parsed_cutflow is not None:
                cf_labels, cf_values = parsed_cutflow
                merged_labels, merged_values = _merge_cutflow_series(
                    group.get("cutflow_labels"),
                    group.get("cutflow_values"),
                    cf_labels,
                    cf_values,
                )
                group["cutflow_labels"] = merged_labels
                group["cutflow_values"] = merged_values

        if unmatched_files:
            if per_process_merged:
                print(
                    f"Warning: Could not match {len(unmatched_files)} ROOT files in {folder.name} "
                    f"to JSON full_dataset keys; unmatched files will use fallback normalization. "
                    f"Example: {unmatched_files[0]}"
                )
            else:
                print(
                    f"Warning: No ROOT files in {folder.name} matched JSON full_dataset keys; "
                    "falling back to folder-level cross-section matching."
                )

        sub_samples = []
        print(f"Process-xsection mapping for {folder.name} ({year}):")
        for process_key in sorted(per_process_merged.keys()):
            group = per_process_merged[process_key]
            group_n_events = int(group.get("n_events", 0) or 0)
            group_xsec = float(group.get("cross_section", 0.0))
            print(f"  {process_key} -> xsec={group_xsec:.6g} pb, n_events={group_n_events}")
            if group_n_events <= 0:
                raise ValueError(
                    f"Invalid n_events={group_n_events} for sub-sample {process_key} in {folder.name}."
                )
            if not bool(group.get("metadata_n_events_available", True)):
                raise ValueError(
                    f"Sub-sample {process_key} in {folder.name} has ROOT file(s) without Metadata.n_events."
                )
            sub_samples.append(
                SubSampleData(
                    process=process_key,
                    n_events=group_n_events,
                    objects=group.get("objects", {}),
                    cross_section=group_xsec,
                    cutflow_labels=group.get("cutflow_labels"),
                    cutflow_values=group.get("cutflow_values"),
                )
            )
        unmatched_n_events = int(unmatched_merged.get("n_events", 0) or 0)
        if unmatched_n_events > 0:
            print(
                f"  __unmatched__ -> xsec=None (fallback), n_events={unmatched_n_events}"
            )
            sub_samples.append(
                SubSampleData(
                    process="__unmatched__",
                    n_events=unmatched_n_events,
                    objects=unmatched_merged.get("objects", {}),
                    cross_section=None,
                    cutflow_labels=unmatched_merged.get("cutflow_labels"),
                    cutflow_values=unmatched_merged.get("cutflow_values"),
                )
            )

    # Extract folder-level cross-section only when per-process matching is unavailable.
    # This avoids confusing logs for mixed-pt folders that already use sub-sample xsections.
    cross_section_pb = None
    if xsection_data and year and not sub_samples:
        cross_section_pb = _extract_xsection_by_year(xsection_data, folder.name, year)
        if cross_section_pb is not None:
            print(f"Found cross-section for {folder.name} ({year}): {cross_section_pb:.4f} pb")

    # Enforce strict MC normalization:
    # 1) n_events must come from metadata for ROOT-based samples
    # 2) when xsection mode is enabled, MC sample must have a matched cross-section
    if not is_data:
        if xsection_data and year and cross_section_pb is None and not sub_samples:
            raise ValueError(
                f"No cross-section found for MC sample {folder.name} in year {year}. "
                "Cannot apply strict MC normalization."
            )
        if n_events <= 0:
            raise ValueError(
                f"Invalid n_events={n_events} for MC sample {folder.name}. "
                "Expected positive total generated events from metadata."
            )

    return SampleData(
        name=folder.name,
        n_events=n_events,
        objects=objects,
        luminosity=luminosity,
        cross_section=cross_section_pb,
        metadata_n_events_available=metadata_n_events_available,
        sub_samples=sub_samples,
        cutflow_labels=cutflow_labels,
        cutflow_values=cutflow_values,
    )


def run_plotting(
    data_folder: Optional[Path],
    data_folders: Optional[Sequence[Path]],
    background_folders: Sequence[Path],
    output_dir: Path,
    variables: Optional[Sequence[str]] = None,
    n_bins: int = 40,
    max_variables: Optional[int] = None,
    luminosity: float = 1.0,
    file_type: Literal["auto", "pkl", "root"] = "auto",
    xsection_json_path: Optional[Path] = None,
    year: Optional[str] = None,
    draw_cutflow: bool = False,
    draw_cutflow_frac: bool = False,
    save_root: bool = False,
) -> List[Path]:
    if not background_folders:
        raise ValueError("At least one background folder is required")

    # Load cross-section data if provided
    xsection_data = None
    if xsection_json_path:
        xsection_data = _load_xsection_json(xsection_json_path)
        print(f"Loaded cross-section data from: {xsection_json_path}")

    # Cutflow-only mode: skip reading full event branches to keep ROOT I/O light.
    cutflow_only_mode = bool(draw_cutflow and max_variables == 0 and not variables)
    effective_variables = ["__no_variables__"] if cutflow_only_mode else variables

    def _load_bkg(folder: Path) -> SampleData:
        return _load_sample_from_folder(
            folder,
            luminosity,
            file_type=file_type,
            target_variables=effective_variables,
            xsection_data=xsection_data,
            year=year,
            is_data=False,
            load_cutflow=draw_cutflow,
        )

    # Load background folders in parallel (I/O-bound → threads)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        bkg_samples = list(executor.map(_load_bkg, background_folders))
    combined_data_folders: List[Path] = []
    if data_folders:
        combined_data_folders.extend(data_folders)
    if data_folder is not None:
        combined_data_folders.append(data_folder)

    data_sample: Optional[SampleData] = None
    if combined_data_folders:
        merged_data: Dict[str, Any] = {"n_events": 0, "objects": {}}
        for folder in combined_data_folders:
            loaded_data = _load_sample_from_folder(
                folder,
                luminosity,
                file_type=file_type,
                target_variables=effective_variables,
                xsection_data=xsection_data,
                year=year,
                is_data=True,
                load_cutflow=draw_cutflow,
            )
            merged_data["n_events"] = int(merged_data.get("n_events", 0) or 0) + int(loaded_data.n_events)
            merged_data["objects"] = _merge_values(merged_data.get("objects", {}), loaded_data.objects)

            if draw_cutflow and loaded_data.cutflow_labels is not None and loaded_data.cutflow_values is not None:
                existing_labels = merged_data.get("cutflow_labels")
                existing_values = merged_data.get("cutflow_values")
                merged_labels, merged_values = _merge_cutflow_series(
                    existing_labels,
                    existing_values,
                    loaded_data.cutflow_labels,
                    loaded_data.cutflow_values,
                )
                merged_data["cutflow_labels"] = merged_labels
                merged_data["cutflow_values"] = merged_values

        data_sample = SampleData(
            name="data",
            n_events=int(merged_data.get("n_events", 0) or 0),
            objects=merged_data.get("objects", {}),
            luminosity=luminosity,
            cutflow_labels=merged_data.get("cutflow_labels"),
            cutflow_values=merged_data.get("cutflow_values"),
        )

    bkg_dist_by_sample = {sample.name: _extract_object_distributions(sample.objects) for sample in bkg_samples}
    # Pre-compute per-sub-sample distributions once (avoids re-extracting inside the variable loop)
    sub_dist_by_sample: Dict[str, List[Dict[str, np.ndarray]]] = {}
    for sample in bkg_samples:
        if sample.sub_samples:
            sub_dist_by_sample[sample.name] = [
                _extract_object_distributions(sub.objects) for sub in sample.sub_samples
            ]
    data_dist = _extract_object_distributions(data_sample.objects) if data_sample else {}

    print("MC normalization summary:")
    for sample in bkg_samples:
        if sample.sub_samples:
            matched = sum(1 for sub in sample.sub_samples if sub.cross_section is not None)
            unmatched = len(sample.sub_samples) - matched
            print(
                f"  {sample.name}: n_events={sample.n_events}, lumi={sample.luminosity} fb^-1, "
                f"xsec=per-process ({matched} matched"
                + (f", {unmatched} fallback" if unmatched > 0 else "")
                + ")"
            )
        elif sample.cross_section is not None:
            event_weight = (sample.luminosity * sample.cross_section * 1000.0) / float(sample.n_events)
            print(
                f"  {sample.name}: n_events={sample.n_events}, lumi={sample.luminosity} fb^-1, "
                f"xsec={sample.cross_section:.6g} pb, event_weight={event_weight:.6g}"
            )
        else:
            event_weight = sample.luminosity / float(sample.n_events) if sample.n_events > 0 else 0.0
            print(
                f"  {sample.name}: n_events={sample.n_events}, lumi={sample.luminosity} fb^-1, "
                f"xsec=None, fallback_event_weight={event_weight:.6g}"
            )

    variable_names: set[str] = set()
    for dist in bkg_dist_by_sample.values():
        variable_names.update(dist.keys())
    variable_names.update(data_dist.keys())

    if variables:
        selected_vars = [v for v in variables if v in variable_names]
    else:
        selected_vars = sorted(variable_names)

    if max_variables is not None:
        selected_vars = selected_vars[:max_variables]

    color_map = _get_background_color_map([sample.name for sample in bkg_samples])

    created_files: List[Path] = []

    if draw_cutflow:
        merged_cut_labels: Optional[List[str]] = None
        merged_cut_values: Optional[np.ndarray] = None
        for sample in bkg_samples:
            if sample.cutflow_labels is None or sample.cutflow_values is None:
                continue
            merged_cut_labels, merged_cut_values = _merge_cutflow_series(
                merged_cut_labels,
                merged_cut_values,
                sample.cutflow_labels,
                sample.cutflow_values,
            )
        if merged_cut_labels is not None and merged_cut_values is not None:
            bkg_cutflows: List[Tuple[str, np.ndarray]] = []
            for sample in bkg_samples:
                if sample.sub_samples:
                    # Mixed-process sample: normalize each process cutflow separately.
                    normalized = np.zeros(len(merged_cut_labels), dtype=float)
                    for sub in sample.sub_samples:
                        sub_aligned = _align_cutflow_to_labels(sub.cutflow_labels, sub.cutflow_values, merged_cut_labels)
                        normalized += _scale_cutflow(
                            sub_aligned,
                            sub.n_events,
                            luminosity=sample.luminosity,
                            cross_section_pb=sub.cross_section,
                        )
                    bkg_cutflows.append((sample.name, normalized))
                else:
                    aligned = _align_cutflow_to_labels(sample.cutflow_labels, sample.cutflow_values, merged_cut_labels)
                    normalized = _scale_cutflow(
                        aligned,
                        sample.n_events,
                        luminosity=sample.luminosity,
                        cross_section_pb=sample.cross_section,
                    )
                    bkg_cutflows.append((sample.name, normalized))

            data_cutflow = None
            if data_sample is not None:
                # Data should remain as raw counts.
                data_cutflow = _align_cutflow_to_labels(data_sample.cutflow_labels, data_sample.cutflow_values, merged_cut_labels)

            cutflow_out = _plot_stacked_cutflow(
                merged_cut_labels,
                bkg_cutflows,
                data_cutflow,
                output_dir,
                luminosity,
                color_map=color_map,
            )
            created_files.append(cutflow_out)
            if save_root:
                cutflow_root_path = output_dir / "stacked_cutflow.root"
                _save_stacked_root_plot(
                    cutflow_root_path,
                    "stacked_cutflow",
                    "Cut step",
                    "Events",
                    np.arange(len(merged_cut_labels) + 1, dtype=float),
                    [(label, values, np.zeros_like(values, dtype=float)) for label, values in bkg_cutflows],
                    data_cutflow,
                    luminosity,
                    labels=merged_cut_labels,
                )

            if draw_cutflow_frac:
                bkg_cutflows_frac: List[Tuple[str, np.ndarray]] = []
                for sample_name, values in bkg_cutflows:
                    frac_values = np.zeros_like(values, dtype=float)
                    if values.size > 0 and values[0] > 0:
                        frac_values = values / float(values[0])
                    bkg_cutflows_frac.append((sample_name, frac_values))

                cutflow_frac_out = _plot_stacked_cutflow(
                    merged_cut_labels,
                    bkg_cutflows_frac,
                    data_cutflow=None,
                    output_dir=output_dir,
                    luminosity=luminosity,
                    color_map=color_map,
                    output_name="stacked_cutflow_frac.png",
                    y_label="Fraction",
                    y_scale="linear",
                )
                created_files.append(cutflow_frac_out)
                if save_root:
                    cutflow_frac_root_path = output_dir / "stacked_cutflow_frac.root"
                    _save_stacked_root_plot(
                        cutflow_frac_root_path,
                        "stacked_cutflow_frac",
                        "Cut step",
                        "Fraction",
                        np.arange(len(merged_cut_labels) + 1, dtype=float),
                        [(sample_name, values, np.zeros_like(values, dtype=float)) for sample_name, values in bkg_cutflows_frac],
                        None,
                        luminosity,
                        labels=merged_cut_labels,
                    )
        else:
            print("Warning: --draw-cutflow enabled but no metdata/cf or Metadata cf_* branches were found in the provided ROOT samples.")

    for variable in selected_vars:
        all_values = []
        for dist in bkg_dist_by_sample.values():
            if variable in dist:
                all_values.append(_apply_variable_plot_filter(variable, dist[variable]))
        if variable in data_dist:
            all_values.append(_apply_variable_plot_filter(variable, data_dist[variable]))

        bins = _make_bins(all_values, n_bins=n_bins, variable=variable)
        if bins is None or len(bins) < 2:
            continue

        bkg_hists: List[Tuple[str, np.ndarray, np.ndarray]] = []
        for sample in bkg_samples:
            if sample.sub_samples:
                # Mixed-pt folders are summed from per-process components, each with its own xsec.
                hist_values = np.zeros(len(bins) - 1, dtype=float)
                hist_sumw2 = np.zeros(len(bins) - 1, dtype=float)
                pre_sub_dists = sub_dist_by_sample.get(sample.name, [])
                for i, sub in enumerate(sample.sub_samples):
                    sub_dist = pre_sub_dists[i] if i < len(pre_sub_dists) else _extract_object_distributions(sub.objects)
                    values = sub_dist.get(variable, np.array([], dtype=float))
                    values = _apply_variable_plot_filter(variable, values)
                    sub_hist, sub_sumw2 = _histogram_and_sumw2(
                        values,
                        bins,
                        sub.n_events,
                        sample.luminosity,
                        cross_section_pb=sub.cross_section,
                    )
                    hist_values += sub_hist
                    hist_sumw2 += sub_sumw2
            else:
                sample_dist = bkg_dist_by_sample.get(sample.name, {})
                values = sample_dist.get(variable, np.array([], dtype=float))
                values = _apply_variable_plot_filter(variable, values)
                hist_values, hist_sumw2 = _histogram_and_sumw2(
                    values,
                    bins,
                    sample.n_events,
                    sample.luminosity,
                    cross_section_pb=sample.cross_section,
                )
            bkg_hists.append((sample.name, hist_values, hist_sumw2))

        if np.allclose(sum(hist for _, hist, _ in bkg_hists), 0.0):
            continue

        data_hist = None
        if data_sample is not None:
            values = data_dist.get(variable, np.array([], dtype=float))
            values = _apply_variable_plot_filter(variable, values)
            data_hist = _histogram_counts(values, bins)

        out_path = _plot_stacked_variable(
            variable,
            bins,
            bkg_hists,
            data_hist,
            output_dir,
            luminosity,
            color_map=color_map,
        )
        created_files.append(out_path)
        if save_root:
            root_path = output_dir / f"stacked_{variable}.root"
            _save_stacked_root_plot(
                root_path,
                f"stacked_{variable}",
                _axis_label(variable),
                "Events / bin",
                bins,
                bkg_hists,
                data_hist,
                luminosity,
            )

    return created_files


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create stacked plots from event-level PKL/ROOT folders."
    )
    parser.add_argument("--data-folder", type=Path, default=None, help="(Optional) Folder containing data PKL/ROOT file(s). If not provided, no data overlay will be shown.")
    parser.add_argument(
        "--data-folders",
        type=Path,
        nargs="+",
        default=None,
        help="(Optional) Multiple data folders to merge before plotting.",
    )
    parser.add_argument(
        "--background-folders",
        type=Path,
        nargs="+",
        required=True,
        help="One or more background folders. Each folder is merged into one stack component.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for output stacked plot PNG files")
    parser.add_argument("--variables", nargs="+", default=None, help="Optional list of variables to plot")
    parser.add_argument("--bins", type=int, default=40, help="Number of bins for continuous variables")
    parser.add_argument("--max-variables", type=int, default=None, help="Optional maximum number of variables to plot")
    parser.add_argument("--lumi", type=float, default=1.0, help="Luminosity in fb^-1 for normalization (default: 1.0)")
    parser.add_argument(
        "--file-type",
        choices=["auto", "pkl", "root"],
        default="auto",
        help="Input file type selection for each sample folder (default: auto).",
    )
    parser.add_argument(
        "--xsection-json",
        type=Path,
        default=None,
        help="(Optional) Path to JSON file containing cross-section data for samples.",
    )
    parser.add_argument(
        "--year",
        type=str,
        default=None,
        help="(Optional) Year to use for cross-section lookup (e.g., 2022, 2022EE, 2023).",
    )
    parser.add_argument(
        "--draw-cutflow",
        action="store_true",
        help="If set, also draw stacked cutflow from ROOT metdata/cf.",
    )
    parser.add_argument(
        "--draw-cutflow-frac",
        action="store_true",
        help="If set with --draw-cutflow, also draw stacked cutflow fraction plot (normalized by first cut per sample).",
    )
    parser.add_argument(
        "--save-root",
        action="store_true",
        help="If set, also save each plot as a ROOT file alongside the PNG output.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_paths = run_plotting(
        data_folder=args.data_folder,
        data_folders=args.data_folders,
        background_folders=args.background_folders,
        output_dir=args.output_dir,
        variables=args.variables,
        n_bins=args.bins,
        max_variables=args.max_variables,
        luminosity=args.lumi,
        file_type=args.file_type,
        xsection_json_path=args.xsection_json,
        year=args.year,
        draw_cutflow=args.draw_cutflow,
        draw_cutflow_frac=args.draw_cutflow_frac,
        save_root=args.save_root,
    )
    print(f"Created {len(output_paths)} stacked plot(s)")
    data_inputs: List[Path] = []
    if args.data_folders:
        data_inputs.extend(args.data_folders)
    if args.data_folder:
        data_inputs.append(args.data_folder)
    if data_inputs:
        if len(data_inputs) == 1:
            print(f"Data folder: {data_inputs[0]}")
        else:
            print("Data folders:")
            for folder in data_inputs:
                print(f"  - {folder}")
    else:
        print("No data folder provided - plotting backgrounds only")
    print(f"Luminosity: {args.lumi} fb^-1")
    print(f"File type: {args.file_type}")
    print(f"Draw cutflow: {args.draw_cutflow}")
    print(f"Draw cutflow frac: {args.draw_cutflow_frac}")
    print(f"Save ROOT: {args.save_root}")
    for out in output_paths:
        print(out)


if __name__ == "__main__":
    main()
