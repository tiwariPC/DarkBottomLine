#!/usr/bin/env python3
"""Physics event-weight computation using cross-section × luminosity.

Two-step process:
  1. Per-file scale factor:   scale = xsec × lumi / N_MC
  2. Per-event weight:        w = full_event_weight × scale

Signal xsec is looked up from a TXT table (MH3/MH4 mass points).
Background xsec comes from a JSON file.

Usage:
    from dnn.phys_weight import PhysWeightComputer, compute_event_weights

    pwc = PhysWeightComputer(
        bkg_xsec_json="xsection_results.json",
        signal_xsec_txt="bbdm_2hdma_typeII_xsec_table.txt",
        lumi=7.99,
    )
    scale = pwc.compute_scale("/path/to/file.root")
    # then in analyzer:  w_event = compute_event_weights(full_event_weight, scale)
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import uproot


def _strip_uuid(filename: str) -> str:
    """Remove .root extension and trailing NANOAODSIM* suffix."""
    base = os.path.basename(str(filename))
    base = re.sub(r"\.root$", "", base)
    return re.sub(r"-NANOAODSIM.*$", "", base, flags=re.IGNORECASE)


def _resolve_dataset_name(filepath: str) -> str:
    """Extract dataset name from a file path, handling EVENTSELECTION convention.

    For EVENTSELECTION-style paths (parent dir ends with ``_EVENTSELECTION``):
        ``.../TTto2L2Nu_..._EVENTSELECTION/uuid.root`` → ``TTto2L2Nu_...``

    For all other paths, falls back to ``_strip_uuid`` on the filename.
    """
    p = Path(filepath)
    parent = p.parent.name
    if re.search(r"_EVENTSELECTION$", parent, flags=re.IGNORECASE):
        return re.sub(r"_EVENTSELECTION$", "", parent, flags=re.IGNORECASE)
    return _strip_uuid(filepath)


class PhysWeightComputer:
    """Compute physics scale factors = xsec × lumi / N_MC."""

    def __init__(
        self,
        bkg_xsec_json: str | Path | None = None,
        signal_xsec_txt: str | Path | None = None,
        lumi: float = 7.99,
        year: str | None = None,
    ):
        self.lumi = float(lumi)
        self.year = str(year).strip() if year else None

        # --- Background xsec lookup ---
        # _bkg_map: (year_norm, ds_norm) → xsec
        self._bkg_map: dict[tuple[str, str], float] = {}
        self._bkg_map_no_year: dict[str, float] = {}
        self._bkg_process_map: dict[str, float] = {}
        if bkg_xsec_json and Path(bkg_xsec_json).exists():
            with open(bkg_xsec_json) as f:
                raw = json.load(f)
            for _category, entries in raw.items():
                for entry in entries:
                    ds = entry.get("full_dataset", "")
                    if not ds:
                        continue
                    xsec = float(entry.get("xsection", 0))
                    yr = str(entry.get("year", "")).strip()
                    ds_key = ds.lower().replace("-", "").replace("_", "")
                    yr_key = yr.lower()
                    key = (yr_key, ds_key)
                    if key not in self._bkg_map:
                        self._bkg_map[key] = xsec
                    if ds_key not in self._bkg_map_no_year:
                        self._bkg_map_no_year[ds_key] = xsec
                    proc = entry.get("process", "")
                    if proc:
                        pk = proc.lower().replace("-", "").replace("_", "")
                        if pk not in self._bkg_process_map:
                            self._bkg_process_map[pk] = xsec

        # --- Signal xsec lookup: (mh3, mh4) → xsec (pb) ---
        self._signal_xsec: dict[tuple[int, int], float] = {}
        if signal_xsec_txt and Path(signal_xsec_txt).exists():
            with open(signal_xsec_txt) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("sno"): continue
                    parts = line.split()
                    if len(parts) >= 7:
                        try:
                            mh3 = int(parts[3]); mh4 = int(parts[4])
                            self._signal_xsec[(mh3, mh4)] = float(parts[6])
                        except (ValueError, IndexError): continue

        self._cache: dict[str, float] = {}

    @staticmethod
    def is_signal_file(filepath: str) -> bool:
        p = Path(filepath)
        # For EVENTSELECTION-style paths, the parent dir contains the dataset name
        parent = p.parent.name.lower()
        if "2hdma" in parent or parent.startswith("run3summer"):
            return True
        # Fallback: check filename
        return "2hdma" in p.name.lower()

    def _read_total_events(self, filepath: str) -> float | None:
        try:
            with uproot.open(filepath) as f:
                if "total_events;1" in f:
                    vals = f["total_events;1"].values()
                    return float(vals[0]) if len(vals) else None
        except Exception: pass
        return None

    def _extract_mass_points(self, filepath: str) -> tuple[int, int] | None:
        m = re.search(r"MH3_(\d+).*?MH4_(\d+)", os.path.basename(str(filepath)), re.IGNORECASE)
        return (int(m.group(1)), int(m.group(2))) if m else None

    def _lookup_xsec(self, filepath: str) -> float:
        basename = os.path.basename(str(filepath))
        if self.is_signal_file(filepath):
            mp = self._extract_mass_points(filepath)
            if mp and mp in self._signal_xsec:
                return self._signal_xsec[mp]
            print(f"[WARN] phys_weight: signal {basename} mp={mp} not in xsec table, using 1.0 pb")
            return 1.0

        # Resolve dataset name (handles EVENTSELECTION parent-dir convention)
        ds_name = _resolve_dataset_name(filepath)
        ds_key = ds_name.lower().replace("-", "").replace("_", "")

        # First: try exact year + dataset match
        yr_key = (self.year or "").lower()
        if (yr_key, ds_key) in self._bkg_map:
            return self._bkg_map[(yr_key, ds_key)]

        # Second: try dataset-only match (any year)
        if ds_key in self._bkg_map_no_year:
            return self._bkg_map_no_year[ds_key]

        # Third: try stripping trailing version number
        ds_nov = re.sub(r"v\d+$", "", ds_key)
        if ds_nov in self._bkg_map_no_year:
            return self._bkg_map_no_year[ds_nov]

        # Fourth: try process-name substring match
        for proc_name, xsec in self._bkg_process_map.items():
            if proc_name and proc_name in ds_key:
                return xsec

        print(f"[WARN] phys_weight: no xsec for {ds_name} (year={self.year}), using 1.0 pb")
        return 1.0

    def compute_scale(self, filepath: str | Path) -> float:
        """Return scale factor = xsec × lumi / N_MC.

        Multiply by ``full_event_weight`` per event to get physics weight.
        Returns 1.0 if N_MC or xsec cannot be determined.
        """
        filepath = str(filepath)
        if filepath in self._cache:
            return self._cache[filepath]
        n_total = self._read_total_events(filepath)
        if n_total is None or n_total <= 0:
            print(f"[WARN] phys_weight: no total_events for {os.path.basename(filepath)}, scale=1.0")
            self._cache[filepath] = 1.0
            return 1.0
        xsec = self._lookup_xsec(filepath)
        scale = xsec * self.lumi / n_total
        self._cache[filepath] = float(scale)
        return float(scale)


def compute_event_weights(full_event_weight, scale_factor: float, clip_max: float = 100.0):
    """Compute per-event physics weights from generator weights and scale factor.

    Returns
    -------
    numpy.ndarray
        w = full_event_weight × scale_factor, clipped to [-clip_max, +clip_max].
        Negative weights are preserved (NLO cancellation).
    """
    import numpy as np
    w = np.asarray(full_event_weight, dtype="f8")
    w = np.where(np.isfinite(w), w, 0.0)
    w = w * float(scale_factor)
    w = np.clip(w, -float(clip_max), float(clip_max))
    return w
