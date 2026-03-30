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
import copy
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
import matplotlib.pyplot as plt
import numpy as np
import uproot


@dataclass
class SubSampleData:
    process: str
    n_events: int
    objects: Dict[str, Any]
    cross_section: Optional[float] = None  # pb


@dataclass
class SampleData:
    name: str
    n_events: int
    objects: Dict[str, Any]
    luminosity: float = 1.0  # Luminosity normalization factor
    cross_section: Optional[float] = None  # Cross-section in pb; None means no cross-section weighting
    metadata_n_events_available: bool = True  # Whether n_events is from metadata (for ROOT inputs)
    sub_samples: Optional[List[SubSampleData]] = None  # Optional per-process components for mixed-pt samples


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
    import re as regex
    sample_base = regex.sub(r"_20\d{2}(EE)?.*$", "", sample_base)
    # Normalize case for matching (WtoLNu vs WtoLNU)
    sample_base_lower = sample_base.lower()
    
    # Search in xsection_data for matching process
    for process_category, entries in xsection_data.items():
        # Try case-insensitive match for process category
        if sample_base_lower == process_category.lower():
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


def _match_root_file_to_process(root_file_name: str, process_to_xsec: Dict[str, float]) -> Optional[Tuple[str, float]]:
    """Match ROOT file name to JSON full_dataset key using longest-prefix match."""
    stem = Path(root_file_name).stem
    best_key: Optional[str] = None
    for key in process_to_xsec.keys():
        if stem.startswith(key):
            if best_key is None or len(key) > len(best_key):
                best_key = key
    if best_key is None:
        return None
    return best_key, process_to_xsec[best_key]
    


def _merge_values(old_value: Any, new_value: Any) -> Any:
    if old_value is None:
        return copy.deepcopy(new_value)
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


def _load_single_root(path: Path) -> Dict[str, Any]:
    """Load a single ROOT file and convert to pkl-like format."""
    with uproot.open(path) as f:
        if "Events" not in f:
            raise ValueError(f"No 'Events' tree found in ROOT file: {path}")
        
        tree = f["Events"]
        objects = {}
        
        # Read all branches as arrays
        for branch_name in tree.keys():
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
                n_events = int(n_events_arr[0]) if len(n_events_arr) > 0 else 0
                metadata_n_events_available = n_events > 0

        # Do not fallback to selected entries for MC normalization.
        # If metadata is missing, keep n_events=0 and let caller decide whether to fail.
        return {
            "n_events": n_events,
            "objects": objects,
            "metadata_n_events_available": metadata_n_events_available,
        }


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


def merge_root_folder(folder: Path) -> Dict[str, Any]:
    """Merge all .root files in a folder."""
    root_files = sorted([p for p in folder.glob("*.root") if p.is_file()])
    if not root_files:
        raise FileNotFoundError(f"No .root files found in folder: {folder}")

    merged: Dict[str, Any] = {}
    all_files_have_metadata_n_events = True
    total_files = len(root_files)
    print(f"Loading {total_files} ROOT files from {folder.name}...")
    
    for idx, root_path in enumerate(root_files, 1):
        try:
            data = _load_single_root(root_path)
            if not data.get("metadata_n_events_available", False):
                all_files_have_metadata_n_events = False
            merged = _merge_values(merged, data)
            
            # Progress indication every 50 files
            if idx % 50 == 0 or idx == total_files:
                print(f"  Progress: {idx}/{total_files} files loaded")
        except Exception as e:
            print(f"Warning: Could not load ROOT file {root_path.name}: {e}")

    merged.setdefault("n_events", 0)
    merged.setdefault("objects", {})
    merged["metadata_n_events_available"] = all_files_have_metadata_n_events
    if not isinstance(merged["objects"], dict):
        raise ValueError(f"Merged objects is not a dictionary in folder: {folder}")

    return merged


def _flatten_numeric(values: Any) -> np.ndarray:
    result: List[float] = []

    def walk(item: Any) -> None:
        if item is None:
            return
        if isinstance(item, dict):
            return
        if isinstance(item, (list, tuple)):
            for sub in item:
                walk(sub)
            return
        if isinstance(item, np.ndarray):
            for sub in item.ravel().tolist():
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


def _make_bins(all_values: Sequence[np.ndarray], n_bins: int) -> Optional[np.ndarray]:
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
    
    weights = np.full(values.shape[0], weight, dtype=float)
    hist, _ = np.histogram(values, bins=bins, weights=weights)
    return hist


def _histogram_counts(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Create raw-count histogram without normalization (for data)."""
    if values.size == 0:
        return np.zeros(len(bins) - 1, dtype=float)
    hist, _ = np.histogram(values, bins=bins)
    return hist.astype(float)


def _simplify_sample_label(sample_name: str) -> str:
    label = sample_name
    if label.startswith("new"):
        label = label[3:]
    label = re.sub(r"_20\d{2}(EE)?_.*$", "", label)
    label = re.sub(r"_EVENTSELECTION$", "", label)
    return label


def _get_background_color_map(sample_names: Sequence[str]) -> Dict[str, str]:
    palette = {
        "DIBOSON": "#4E79A7",
        "DYto2L-2Jets": "#59A14F",
        "Top": "#E15759",
        "WtoLNU-2Jets": "#F28E2B",
        "Zto2Nu-2Jets": "#B07AA1",
        "QCD": "#9C755F",
        "GJets": "#76B7B2",
        "SMH": "#EDC948",
    }
    fallback_palette = [
        "#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#B07AA1",
        "#76B7B2", "#9C755F", "#EDC948", "#FF9DA7", "#BAB0AC",
    ]

    color_map: Dict[str, str] = {}
    fallback_idx = 0
    for raw_name in sorted({_simplify_sample_label(name) for name in sample_names}):
        if raw_name in palette:
            color_map[raw_name] = palette[raw_name]
        else:
            color_map[raw_name] = fallback_palette[fallback_idx % len(fallback_palette)]
            fallback_idx += 1
    return color_map


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


def _plot_stacked_variable(
    variable: str,
    bins: np.ndarray,
    background_hists: List[Tuple[str, np.ndarray]],
    data_hist: Optional[np.ndarray],
    output_dir: Path,
    luminosity: float = 1.0,
    color_map: Optional[Dict[str, str]] = None,
) -> Path:
    background_hists = [(_simplify_sample_label(label), hist) for label, hist in background_hists]
    background_hists = sorted(background_hists, key=lambda item: float(np.sum(item[1])))

    # CMS matplotlib style configuration
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['ytick.labelsize'] = 11
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['lines.linewidth'] = 1.5

    if color_map is None:
        color_map = _get_background_color_map([label for label, _ in background_hists])
    
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.subplots_adjust(top=0.91, bottom=0.12, left=0.11, right=0.97)
    
    cumulative = np.zeros(len(bins) - 1, dtype=float)
    for label, hist_values in background_hists:
        next_cumulative = cumulative + hist_values
        color = color_map.get(label, "#4E79A7")
        ax.stairs(next_cumulative, bins, baseline=cumulative, fill=True, alpha=0.8, 
             linewidth=0.5, color=color, label=label, edgecolor='white')
        cumulative = next_cumulative

    if data_hist is not None:
        centers = 0.5 * (bins[:-1] + bins[1:])
        half_width = 0.5 * (bins[1:] - bins[:-1])
        mask = data_hist > 0
        if np.any(mask):
            ax.errorbar(
                centers[mask],
                data_hist[mask],
                xerr=half_width[mask],
                yerr=np.sqrt(data_hist[mask]),
                fmt="k_",
                markersize=10,
                elinewidth=1.1,
                capsize=0,
                label="Data",
                zorder=10,
            )

    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-6)
    ax.set_xlabel(variable, fontsize=12, fontweight='normal')
    ax.set_ylabel("MC: Events × lumi / n_events, Data: Events", fontsize=12, fontweight='normal')

    # CMS header (outside plotting region)
    fig.text(0.11, 0.945, "CMS", fontsize=14, fontweight="bold", ha="left", va="top")
    fig.text(0.175, 0.945, "Simulation", fontsize=12, style="italic", ha="left", va="top")
    fig.text(0.97, 0.945, f"{luminosity:g} fb$^{{-1}}$", fontsize=11, ha="right", va="top")

    # Unit label at lower-right outside plotting region
    unit = _variable_unit(variable)
    fig.text(0.97, 0.035, unit, fontsize=10, ha="right", va="bottom")
    
    # Improved legend
    ax.legend(loc='upper right', frameon=True, fancybox=False, 
             shadow=False, fontsize=10, edgecolor='black', framealpha=0.95)
    
    # CMS-style grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, color='gray')
    ax.set_axisbelow(True)
    
    # Improve spine visibility
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.2)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"stacked_{variable}.png"
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
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
        merged = merge_root_folder(folder)
    else:
        if root_files:
            merged = merge_root_folder(folder)
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
    if root_files and xsection_data and year and not is_data:
        process_to_xsec = _build_xsection_process_lookup(xsection_data, year)
        per_process_merged: Dict[str, Dict[str, Any]] = {}
        unmatched_files: List[str] = []

        for root_file in sorted(root_files):
            match = _match_root_file_to_process(root_file.name, process_to_xsec)
            if match is None:
                unmatched_files.append(root_file.name)
                continue
            process_key, xsec_pb = match
            loaded = _load_single_root(root_file)
            group = per_process_merged.get(process_key)
            if group is None:
                group = {
                    "n_events": 0,
                    "objects": {},
                    "metadata_n_events_available": True,
                    "cross_section": xsec_pb,
                }
                per_process_merged[process_key] = group
            if not loaded.get("metadata_n_events_available", False):
                group["metadata_n_events_available"] = False
            group["n_events"] = int(group.get("n_events", 0) or 0) + int(loaded.get("n_events", 0) or 0)
            group["objects"] = _merge_values(group.get("objects", {}), loaded.get("objects", {}))

        if unmatched_files:
            raise ValueError(
                f"Could not match {len(unmatched_files)} ROOT files in {folder.name} to JSON full_dataset keys. "
                f"Example: {unmatched_files[0]}"
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
                )
            )
    
    # Extract cross-section if data available and year is provided
    cross_section_pb = None
    if xsection_data and year:
        cross_section_pb = _extract_xsection_by_year(xsection_data, folder.name, year)
        if cross_section_pb is not None:
            print(f"Found cross-section for {folder.name} ({year}): {cross_section_pb:.4f} pb")

    # Enforce strict MC normalization:
    # 1) n_events must come from metadata for ROOT-based samples
    # 2) when xsection mode is enabled, MC sample must have a matched cross-section
    if not is_data:
        if root_files and not metadata_n_events_available:
            raise ValueError(
                f"Sample {folder.name} is missing Metadata.n_events in one or more ROOT files. "
                "Refusing to normalize MC with selected-entry fallback."
            )
        if xsection_data and year and cross_section_pb is None:
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
    )


def run_plotting(
    data_folder: Optional[Path],
    background_folders: Sequence[Path],
    output_dir: Path,
    variables: Optional[Sequence[str]] = None,
    n_bins: int = 40,
    max_variables: Optional[int] = None,
    luminosity: float = 1.0,
    file_type: Literal["auto", "pkl", "root"] = "auto",
    xsection_json_path: Optional[Path] = None,
    year: Optional[str] = None,
) -> List[Path]:
    if not background_folders:
        raise ValueError("At least one background folder is required")

    # Load cross-section data if provided
    xsection_data = None
    if xsection_json_path:
        xsection_data = _load_xsection_json(xsection_json_path)
        print(f"Loaded cross-section data from: {xsection_json_path}")

    bkg_samples = [
        _load_sample_from_folder(
            folder,
            luminosity,
            file_type=file_type,
            target_variables=variables,
            xsection_data=xsection_data,
            year=year,
            is_data=False,
        )
        for folder in background_folders
    ]
    data_sample = (
        _load_sample_from_folder(
            data_folder,
            luminosity,
            file_type=file_type,
            target_variables=variables,
            xsection_data=xsection_data,
            year=year,
            is_data=True,
        )
        if data_folder
        else None
    )

    bkg_dist_by_sample = {sample.name: _extract_object_distributions(sample.objects) for sample in bkg_samples}
    data_dist = _extract_object_distributions(data_sample.objects) if data_sample else {}

    print("MC normalization summary:")
    for sample in bkg_samples:
        if sample.cross_section is not None:
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
    for variable in selected_vars:
        all_values = []
        for dist in bkg_dist_by_sample.values():
            if variable in dist:
                all_values.append(_apply_variable_plot_filter(variable, dist[variable]))
        if variable in data_dist:
            all_values.append(_apply_variable_plot_filter(variable, data_dist[variable]))

        bins = _make_bins(all_values, n_bins=n_bins)
        if bins is None or len(bins) < 2:
            continue

        bkg_hists: List[Tuple[str, np.ndarray]] = []
        for sample in bkg_samples:
            if sample.sub_samples:
                # Mixed-pt folders are summed from per-process components, each with its own xsec.
                hist_values = np.zeros(len(bins) - 1, dtype=float)
                for sub in sample.sub_samples:
                    sub_dist = _extract_object_distributions(sub.objects)
                    values = sub_dist.get(variable, np.array([], dtype=float))
                    values = _apply_variable_plot_filter(variable, values)
                    hist_values += _histogram(
                        values,
                        bins,
                        sub.n_events,
                        sample.luminosity,
                        cross_section_pb=sub.cross_section,
                    )
            else:
                sample_dist = bkg_dist_by_sample.get(sample.name, {})
                values = sample_dist.get(variable, np.array([], dtype=float))
                values = _apply_variable_plot_filter(variable, values)
                hist_values = _histogram(values, bins, sample.n_events, sample.luminosity, cross_section_pb=sample.cross_section)
            bkg_hists.append((sample.name, hist_values))

        if np.allclose(sum(hist for _, hist in bkg_hists), 0.0):
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

    return created_files


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create stacked plots from event-level PKL/ROOT folders."
    )
    parser.add_argument("--data-folder", type=Path, default=None, help="(Optional) Folder containing data PKL/ROOT file(s). If not provided, no data overlay will be shown.")
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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_paths = run_plotting(
        data_folder=args.data_folder,
        background_folders=args.background_folders,
        output_dir=args.output_dir,
        variables=args.variables,
        n_bins=args.bins,
        max_variables=args.max_variables,
        luminosity=args.lumi,
        file_type=args.file_type,
        xsection_json_path=args.xsection_json,
        year=args.year,
    )
    print(f"Created {len(output_paths)} stacked plot(s)")
    if args.data_folder:
        print(f"Data folder: {args.data_folder}")
    else:
        print("No data folder provided - plotting backgrounds only")
    print(f"Luminosity: {args.lumi} fb^-1")
    print(f"File type: {args.file_type}")
    for out in output_paths:
        print(out)


if __name__ == "__main__":
    main()