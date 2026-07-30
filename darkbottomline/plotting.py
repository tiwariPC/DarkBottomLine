"""
Data/MC plotting module for DarkBottomLine framework.
"""

import copy
import math
import re
from collections import Counter
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for batch mode
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.patches
import numpy as np
import awkward as ak
import logging
import pickle
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from pathlib import Path
import yaml
import os
from datetime import datetime

try:
    import mplhep as hep
    _HAS_MPLHEP = True
except ImportError:
    _HAS_MPLHEP = False

from utils.plot_utils import (
    CMSPlotStyle, get_process_colors, get_process_labels,
    get_process_config, get_background_color_map, simplify_sample_label,
    _PALETTE,
)

# ---------------------------------------------------------------------------
# Sentinel & histogram utilities (used by create_stacked_plots)
# ---------------------------------------------------------------------------

from .objects import SENTINEL as _SENTINEL

# Pseudo-variables: plotted like a normal variable but sourced from a different
# branch and/or weighted by a different weight branch. Used for cross-check plots
# that reuse an existing quantity with an alternate event weight.
#   _VAR_VALUE_ALIAS   : plot-variable name -> branch name to read values from
#   _VAR_WEIGHT_OVERRIDE: plot-variable name -> weight branch to use (instead of
#                         full_event_weight / the active systematic weight)
# PV_npvsGood_noPU = PV_npvsGood weighted WITHOUT the pileup reweight (all other
# SFs kept), via the weight_noPileup branch. Data/MC pileup validation: reweighted
# MC (PV_npvsGood) and un-reweighted MC (PV_npvsGood_noPU) each compared to data.
_VAR_VALUE_ALIAS: Dict[str, str] = {
    "PV_npvsGood_noPU": "PV_npvsGood",
}
_VAR_WEIGHT_OVERRIDE: Dict[str, str] = {
    "PV_npvsGood_noPU": "weight_noPileup",
}


def _value_branch(var: str) -> str:
    """Branch to read plot values from (identity unless var is a pseudo-variable)."""
    return _VAR_VALUE_ALIAS.get(var, var)


def _bins_key(var: str) -> str:
    """Config key for binning (pseudo-variables share their source var's bins)."""
    return _VAR_VALUE_ALIAS.get(var, var)


def _weight_branch_for(var: str, default_weight: str) -> str:
    """Weight branch for this variable (override for pseudo-variables)."""
    return _VAR_WEIGHT_OVERRIDE.get(var, default_weight)


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)


def _flatten_numeric(values: Any) -> np.ndarray:
    if isinstance(values, np.ndarray) and np.issubdtype(values.dtype, np.number):
        flat = values.ravel()
        return flat[np.isfinite(flat)].astype(float)
    result: List[float] = []

    def walk(item: Any) -> None:
        if item is None or isinstance(item, dict):
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
    return np.asarray(result, dtype=float) if result else np.array([], dtype=float)


def _should_load_key(key: str, variables: Optional[List[str]]) -> bool:
    """Return True if key matches any variable in the whitelist (fuzzy match)."""
    if variables is None:
        return True
    key_l = key.lower()
    for var in variables:
        var_l = var.lower()
        if var_l in key_l or key_l in var_l:
            return True
    return False


# Some datasets were fully renamed between the 2022/2023 and 2024 campaigns
# (not a simple token swap). Map each old core name to the canonical (2024 /
# xsec-JSON) core so all years resolve to the same cross-section key.
# Longer / more specific keys first — applied by substring, first match wins.
_DATASET_ALIASES = [
    # SM Higgs
    ("ggZH_Hto2B_Zto2L_M-125",     "GluGluZH-Zto2L-Hto2B_Par-M-125"),
    ("ggZH_Hto2B_Zto2Nu_M-125",    "GluGluZH-Zto2Nu-Hto2B_Par-M-125"),
    ("WminusH_Hto2B_WtoLNu_M-125", "WminusH-WtoLNu-Hto2B_Par-M-125"),
    ("WplusH_Hto2B_WtoLNu_M-125",  "WplusH-WtoLNu-Hto2B_Par-M-125"),
    ("ZH_Hto2B_Zto2L_M-125",       "ZH-Zto2L-Hto2B_Par-M-125"),
    ("ZH_Hto2B_Zto2Nu_M-125",      "ZH-Zto2Nu-Hto2B_Par-M-125"),
    ("GluGluHto2B_M-125",          "GluGluH-Hto2B_Par-M-125"),
    ("VBFHto2B_M-125",             "VBFH-Hto2B_Par-M-125"),
    ("ttHto2B_M-125",              "TTH-Hto2B_Par-M-125"),
    # Single top t-channel: 2022/23 "TBbarQ_t-channel_4FS" (underscore, no toLNu)
    # → canonical 2024 "TBbarQtoLNu-t-channel-4FS".
    ("TBbarQ_t-channel_4FS", "TBbarQtoLNu-t-channel-4FS"),
    ("TbarBQ_t-channel_4FS", "TbarBQtoLNu-t-channel-4FS"),
]


def _clean_sample_name(name: str) -> str:
    """Canonicalize a dataset name to its cross-section-JSON core form.

    Collapses the year-specific naming variants (V+jets jet-bin token, DY
    dash/underscore, SM Higgs renames) and strips generator/tune/era suffixes so
    that the 2022/2022EE/2023 and 2024 forms of a process — and the xsec JSON
    ``full_dataset`` — all map to one identical string. Idempotent.
    """
    # Ordered: longer/more specific first to avoid partial false matches
    suffixes = [
        "_EVENTSELECTION", "_hadd",
        "_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
        "_TuneCP5_13p6TeV_madgraphMLM-pythia8",
        "_TuneCP5_13p6TeV_powhegMINLO-pythia8",
        "_TuneCP5_13p6TeV_powheg-minlo-pythia8",
        "_TuneCP5_13p6TeV_powheg-pythia8",
        "_TuneCP5_13TeV_amcatnloFXFX-pythia8",
        "_TuneCP5_13TeV_madgraphMLM-pythia8",
        "_TuneCP5_13TeV_powheg-pythia8",
        "_TuneCUETP8M1_13TeV_amcatnloFXFX-pythia8",
        "_TuneCUETP8M1_13TeV_madgraphMLM-pythia8",
        "_dipoleRecoilOn_TuneCP5_13p6TeV",
        "_TuneCP5_13p6TeV", "_TuneCP5_13TeV", "_TuneCUETP8M1_13TeV",
        "_13p6TeV", "_13TeV",
        "_nanoAOD", "_NANOAOD",
        "_Run3Summer22EE", "_Run3Summer22", "_Run3Summer23BPix",
        "_Run3Summer23", "_RunIII2024Summer24",
    ]
    result = name
    # 1. Old→canonical dataset renames (substring, most specific first)
    for old, canon in _DATASET_ALIASES:
        if old in result:
            result = result.replace(old, canon)
            break
    # 2. V+jets jet-bin token → canonical (no bin token). Handles the 2024
    #    "_Bin-2J-" prefix form and the 2022/2023 "_2J" infix/suffix form.
    result = result.replace("_Bin-2J-", "_").replace("_Bin-2J", "")
    result = result.replace("_2J_", "_")
    if result.endswith("_2J"):
        result = result[:-len("_2J")]
    # 3. DY dash/underscore: canonical JSON form uses MLL-50-PTLL (dash)
    result = result.replace("MLL-50_PTLL", "MLL-50-PTLL")
    # 4. Strip generator/tune/energy/era suffixes
    for suffix in suffixes:
        if suffix in result:
            result = result[:result.index(suffix)]
    return result


def _find_xsec(stem: str, cross_sections: Dict[str, float]) -> Optional[float]:
    """Look up cross-section for *stem*, using cleaned sample-name matching."""
    if not cross_sections:
        return None
    # 1. exact match
    xsec = cross_sections.get(stem)
    if xsec is not None:
        return xsec
    # 2. cleaned exact match
    clean = _clean_sample_name(stem)
    xsec = cross_sections.get(clean)
    if xsec is not None:
        return xsec
    # 3. cleaned substring match in either direction
    for k, v in cross_sections.items():
        k_clean = _clean_sample_name(k)
        if k_clean == clean or k_clean in clean or clean in k_clean:
            return v
    return None


def _extract_branches(objects: Dict[str, Any], variables: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
    """Flatten a nested objects dict from PKL/ROOT to {branch_name: flat_array}."""
    distributions: Dict[str, np.ndarray] = {}
    for key, value in objects.items():
        if key.endswith("_mask"):
            continue
        if not isinstance(value, list):
            if not _should_load_key(key, variables):
                continue
            arr = _flatten_numeric(value)
            if arr.size > 0:
                distributions[key] = arr
            continue
        first = next((v for v in value if v is not None), None)
        if first is None:
            continue
        if isinstance(first, dict):
            numeric_fields: set = set()
            for item in value:
                if isinstance(item, dict):
                    for f, fv in item.items():
                        if _is_number(fv):
                            numeric_fields.add(f)
            for field in sorted(numeric_fields):
                derived_key = f"{key}_{field}"
                if not _should_load_key(derived_key, variables):
                    continue
                flat = _flatten_numeric([item.get(field) for item in value if isinstance(item, dict)])
                if flat.size > 0:
                    distributions[derived_key] = flat
            continue
        if isinstance(first, list):
            inner = next((x for row in value if isinstance(row, list) for x in row if x is not None), None)
            if isinstance(inner, dict):
                numeric_fields = set()
                for row in value:
                    if not isinstance(row, list):
                        continue
                    for item in row:
                        if isinstance(item, dict):
                            for f, fv in item.items():
                                if _is_number(fv):
                                    numeric_fields.add(f)
                for field in sorted(numeric_fields):
                    derived_key = f"{key}_{field}"
                    if not _should_load_key(derived_key, variables):
                        continue
                    vals: List[float] = []
                    for row in value:
                        if not isinstance(row, list):
                            continue
                        for item in row:
                            if isinstance(item, dict):
                                fv = item.get(field)
                                if _is_number(fv):
                                    v2 = float(fv)
                                    if math.isfinite(v2):
                                        vals.append(v2)
                    if vals:
                        distributions[derived_key] = np.asarray(vals, dtype=float)
                continue
        if not _should_load_key(key, variables):
            continue
        flat = _flatten_numeric(value)
        if flat.size > 0:
            distributions[key] = flat
    return distributions


def _apply_variable_plot_filter(variable: str, values: np.ndarray,
                                return_mask: bool = False) -> np.ndarray:
    if values.size == 0:
        return np.ones(0, dtype=bool) if return_mask else values
    mask = values != _SENTINEL
    return mask if return_mask else values[mask]


_REGION_LABELS: Dict[str, str] = {
    # Signal regions
    "1b:SR":          "SR, 1b",
    "2b:SR":          "SR, 2b",
    # W CRs
    "1b:CR_Wmunu":    r"W($\mu\nu$) CR, 1b",
    "1b:CR_Wenu":     r"W(e$\nu$) CR, 1b",
    "2b:CR_Wmunu":    r"W($\mu\nu$) CR, 2b",
    "2b:CR_Wenu":     r"W(e$\nu$) CR, 2b",
    # Z CRs
    "1b:CR_Zmumu":    r"Z($\mu\mu$) CR, 1b",
    "1b:CR_Zee":      r"Z(ee) CR, 1b",
    "2b:CR_Zmumu":    r"Z($\mu\mu$) CR, 2b",
    "2b:CR_Zee":      r"Z(ee) CR, 2b",
    # Top CRs
    "2b:CR_Topmunu":  r"Top($\mu\nu$) CR, 2b",
    "2b:CR_Topenu":   r"Top(e$\nu$) CR, 2b",
}


def _pretty_region_label(region: str) -> str:
    """Return a human-readable region label for plot annotation."""
    return _REGION_LABELS.get(region, "")


def _make_bins(
    all_values: Sequence[np.ndarray],
    get_bins_fn,
    variable: Optional[str],
    n_bins_default: int = 40,
) -> Optional[np.ndarray]:
    ref = get_bins_fn(variable) if variable else None
    if ref is not None:
        return ref
    valid = [arr[np.isfinite(arr) & (arr != _SENTINEL)] for arr in all_values if arr.size > 0]
    if not valid:
        return None
    merged = np.concatenate(valid)
    if merged.size == 0:
        return None
    data_min, data_max = float(np.min(merged)), float(np.max(merged))
    if not (math.isfinite(data_min) and math.isfinite(data_max)):
        return None
    if abs(data_max - data_min) < 1e-12:
        w = max(1.0, abs(data_max) * 0.05)
        return np.linspace(data_min - w, data_max + w, n_bins_default + 1)
    is_int = np.allclose(merged, np.round(merged), atol=1e-8)
    if is_int and len(np.unique(np.round(merged).astype(int))) <= 20:
        lo, hi = int(np.min(np.round(merged))), int(np.max(np.round(merged)))
        return np.arange(lo - 0.5, hi + 1.5, 1.0)
    return np.linspace(data_min, data_max, n_bins_default + 1)


def _clip_overflow(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    lo, hi = float(bins[0]), float(bins[-1])
    return np.clip(values, lo, np.nextafter(hi, -np.inf))


def _histogram_and_sumw2(
    values: np.ndarray,
    bins: np.ndarray,
    weighted_total_events: int,
    luminosity: float = 1.0,
    cross_section_pb: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    zeros = np.zeros(len(bins) - 1, dtype=float)
    if weighted_total_events <= 0 or values.size == 0:
        return zeros, zeros
    if cross_section_pb is not None:
        weight = (luminosity * cross_section_pb * 1000.0) / float(weighted_total_events)
    else:
        weight = luminosity / float(weighted_total_events)
    values = _clip_overflow(values, bins)
    counts, _ = np.histogram(values, bins=bins)
    counts = counts.astype(float)
    return counts * weight, counts * (weight ** 2)


def _histogram_counts(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros(len(bins) - 1, dtype=float)
    values = _clip_overflow(values, bins)
    hist, _ = np.histogram(values, bins=bins)
    return hist.astype(float)


def _get_legend_label(name: str) -> str:
    cfg = get_process_config().get(simplify_sample_label(name))
    return cfg["label"] if cfg else name


def _plot_stacked_variable_worker(args: tuple) -> tuple:
    """Draw + save one (region, variable) stacked plot. Module-level so it's
    picklable as a multiprocessing worker target — each call's inputs are
    plain arrays/scalars already aggregated by the caller, independent of
    every other (region, variable) task."""
    (config, kwargs, region, var) = args
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    pm = PlotManager(config)
    files = pm._plot_stacked_variable(**kwargs)
    return region, var, files


def _plot_one_region_worker(args: tuple) -> tuple:
    """Produce all plots (individual-variable + grouped) for a single region.
    Module-level (picklable) multiprocessing worker target — reconstructs a
    fresh PlotManager from config in each process rather than sharing a live
    instance, since PlotManager mutates self mid-call (_regions_config_path
    etc.) and matplotlib figures aren't picklable across process boundaries."""
    (config, results, region, output_dir, show_data, version, formats, hist_scale) = args
    # Avoid N worker processes each spawning their own BLAS thread pool.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    pm = PlotManager(config)
    individual_plots = pm._create_individual_variable_plots(
        results, region, output_dir, show_data, version, formats,
        hist_scale=hist_scale,
    )
    grouped_plots = pm._create_region_plots_single(
        results, region, Path(output_dir), show_data, version, output_dir
    )
    return region, {**individual_plots, **grouped_plots}


class PlotManager:
    """
    Manager for creating data/MC plots with region comparisons.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize plot manager.

        Args:
            config: Plotting configuration dictionary
                - no_log_scale_vars: List of variables that should not use log scale
                - region_exclusions: Dict of region-specific variable exclusions
                  Format: {
                      "region_pattern": ["var1", "var2", ...],
                      "category_pattern": ["var3", "var4", ...]
                  }
                  Examples:
                  - "1b:SR": ["jet3_pt", "lep1_pt"] - excludes these from 1b SR
                  - "Top": ["z_mass", "z_pt"] - excludes these from all Top CRs
                  - "Wlnu": ["z_mass", "z_pt"] - excludes these from all W CRs
        """
        self.config = config or {}

        # Initialize plot style
        self.style = CMSPlotStyle()
        self.style.set_style()

        # Process colors and labels
        self.colors = get_process_colors()
        self.labels = get_process_labels()

        # Plot settings — all driven from plotting.yaml
        self.dpi              = self.config.get("dpi", 200)
        self.figsize          = tuple(self.config.get("figsize_ratio",    [12, 12]))
        self.figsize_no_ratio = tuple(self.config.get("figsize_no_ratio", [12, 10]))
        self.figsize_cutflow  = tuple(self.config.get("figsize_cutflow",  [12, 12]))
        self.subplots_top     = self.config.get("subplots_top",    0.92)
        self.subplots_bottom  = self.config.get("subplots_bottom", 0.09)
        self.subplots_left    = self.config.get("subplots_left",   0.14)
        self.subplots_right   = self.config.get("subplots_right",  0.95)
        self.subplots_hspace  = self.config.get("subplots_hspace", 0.08)
        self.main_height      = self.config.get("main_height",     3.0)
        self.ratio_height     = self.config.get("ratio_height",    1.0)
        self.fontsize_axis    = self.config.get("fontsize_axis",   22)
        self.fontsize_legend  = self.config.get("fontsize_legend", 20)
        self.fontsize_xtick_cutflow = self.config.get("fontsize_xtick_cutflow", 16)
        self.unc_facecolor    = self.config.get("uncertainty_facecolor", "#bbbbbb")
        self.unc_edgecolor    = self.config.get("uncertainty_edgecolor", "#666666")
        self.unc_hatch        = self.config.get("uncertainty_hatch",     "////")
        self.unc_alpha        = self.config.get("uncertainty_alpha",     0.8)
        self.unc_label        = self.config.get("uncertainty_label",     "Stat. unc.")
        self.cms_label        = self.config.get("cms_label",             "Work in progress")
        self.com_energy       = float(self.config.get("com_energy",      13.6))
        self.ratio_ylim       = tuple(self.config.get("ratio_ylim",      [0.0, 2.0]))
        self.data_markersize  = float(self.config.get("data_markersize", 5.5))
        self.data_elinewidth  = float(self.config.get("data_elinewidth", 1.2))
        self.data_color       = self.config.get("data_color",            "black")
        self.signal_linewidth = float(self.config.get("signal_linewidth", 2.0))
        self.signal_colors    = list(self.config.get("signal_colors",
                                    ["#000000", "#e31a1c", "#1f78b4", "#33a02c", "#ff7f00", "#6a3d9a"]))
        self.legend_ncol      = int(self.config.get("legend_ncol",       2))

        # Variables that should NOT use log scale (multiplicity plots, etc.)
        self.no_log_scale_vars = self.config.get("no_log_scale_vars", [
            'n_jets', 'n_bjets', 'n_muons', 'n_electrons', 'n_taus', 'n_leptons',
            'n_pv', 'pu_npv'
        ])

        # Per-variable x-axis title overrides — falls back to raw variable name.
        self.variable_labels: Dict[str, str] = dict(self.config.get("variable_labels", {}))

        # Region-specific exclusions — kept for backward-compat but no longer used by default path
        self.region_exclusions = self.config.get("region_exclusions", {})

        # Variables common to every region — prepended to each region's specific list.
        self.common_variables: List[str] = list(self.config.get("common_variables", []))
        self.systematic_variables: List[str] = list(self.config.get("systematic_variables", []))

        # Per-region variable additions (region-specific on top of common_variables).
        # Final plot list = common_variables + region_variables[region].
        # Private keys starting with "_" are ignored.
        raw_rv: Dict[str, Any] = self.config.get("region_variables", {})
        self.region_variables: Dict[str, List[str]] = {
            k: list(v) for k, v in raw_rv.items()
            if not k.startswith("_") and isinstance(v, list)
        }

        # Bin config from plotting.yaml
        self._variable_bins_cfg: Dict[str, Any] = self.config.get("variable_bins", {})
        self._met_suffix_patterns: List[str] = self.config.get("met_suffix_patterns", ["met_pt", "recoil"])
        self._n_bins_default: int = int(self.config.get("n_bins_default", 40))

        # Process groups from plotting.yaml.
        # Each group has: type (background|signal|data), patterns, color, label.
        # self.process_groups  -> {label: [patterns]}  for background groups
        # self.signal_groups   -> {label: [patterns]}  for signal groups
        # self.data_groups     -> {label: [patterns]}  for data groups
        raw_groups: Dict[str, Any] = self.config.get("process_groups", {})
        self.process_groups: Dict[str, List[str]] = {}
        self.signal_groups: Dict[str, List[str]] = {}
        self.data_groups: Dict[str, List[str]] = {}
        self._group_colors: Dict[str, str] = {}
        self._group_labels: Dict[str, str] = {}

        for label, grp in raw_groups.items():
            if not isinstance(grp, dict):
                grp = {"patterns": grp}
            grp_type = grp.get("type", "background")
            # support both "patterns" (new) and "files" (legacy) keys
            patterns: List[str] = grp.get("patterns") or grp.get("files") or []
            if grp.get("color"):
                self._group_colors[label] = grp["color"]
            if grp.get("label"):
                self._group_labels[label] = grp["label"]
            if grp_type == "signal":
                self.signal_groups[label] = patterns
            elif grp_type == "data":
                self.data_groups[label] = patterns
            else:
                self.process_groups[label] = patterns

        self.event_selection_variables: List[str] = self.config.get("event_selection_variables", [])

        logging.debug("Plot manager initialized")

    def _build_bins_from_config(self, variable: Optional[str]) -> Optional[np.ndarray]:
        """Resolve bin edges for *variable* from plotting.yaml variable_bins."""
        if not variable:
            return None
        spec = self._variable_bins_cfg.get(variable)
        if spec is not None:
            if "edges" in spec:
                return np.array(spec["edges"], dtype=float)
            return np.linspace(float(spec["low"]), float(spec["high"]), int(spec["n"]))
        # suffix fallback for MET-like branches
        name = variable.lower()
        if any(name.endswith(p) or p in name for p in self._met_suffix_patterns):
            met_spec = self._variable_bins_cfg.get("MET_pt") or self._variable_bins_cfg.get("PFMET_pt")
            if met_spec:
                return np.array(met_spec["edges"], dtype=float)
        if "ctsvalue" in name:
            cts_spec = self._variable_bins_cfg.get("costheta_star")
            if cts_spec:
                return np.array(cts_spec["edges"], dtype=float)
        if name.startswith("ml_score_mh3_"):
            ml_spec = self._variable_bins_cfg.get("ml_score")
            if ml_spec:
                if "edges" in ml_spec:
                    return np.array(ml_spec["edges"], dtype=float)
                return np.linspace(float(ml_spec["low"]), float(ml_spec["high"]), int(ml_spec["n"]))
        return None

    def create_stacked_plot_from_files(
        self,
        data_file: Optional[str],
        background_files: List[str],
        signal_file: Optional[str],
        output_path: str,
        variable: str = "met",
        region: Optional[str] = None,
        xlabel: str = "MET [GeV]",
        title_tag: str = "CMS Preliminary  (13.6 TeV, 2023)",
        bins: Optional[np.ndarray] = None,
        version: Optional[str] = None,
        formats: Optional[List[str]] = None,
    ) -> str:
        """
        Create a stacked Data/MC plot with ratio and uncertainty band from result files.

        Args:
            data_file: Path to data results pickle (may be None)
            background_files: List of background results pickles
            signal_file: Path to signal results pickle (may be None)
            output_path: Output file path (e.g. outputs/plots/stacked_met.pdf)
            variable: Variable key to plot (default: 'met')
            region: The analysis region to plot (default: '1b:SR')
            xlabel: X-axis label
            title_tag: CMS label text
            bins: Optional bin edges array (Note: rebinning is not yet supported for hist.Hist objects)

        Returns:
            Output file path
        """
        import pickle
        from matplotlib.gridspec import GridSpec
        try:
            import hist
            import mplhep as hep
        except ImportError:
            logging.error("The 'hist' and 'mplhep' libraries are required. Please install them.")
            return ""

        def load_hist_from_file(path: str, var: str, reg: Optional[str]) -> Optional[hist.Hist]:
            try:
                with open(path, 'rb') as f:
                    res = pickle.load(f)
                res = {k.strip(): v for k, v in res.items()} # Clean up keys by stripping whitespace

                hist_obj = None
                if reg: # If a region is specified, assume region-based analysis output
                    hist_obj = res.get('region_histograms', {}).get(reg, {}).get(var, None)
                    if hist_obj is None:
                        logging.warning(f"Could not find histogram '{var}' for region '{reg}' in 'region_histograms' of {path}")

                if hist_obj is None: # If not found in region_histograms or no region specified, look in top-level 'histograms'
                    hist_obj = res.get('histograms', {}).get(var, None)
                    if hist_obj is None:
                        logging.warning(f"Could not find histogram '{var}' in 'histograms' (top-level) of {path}")

                if hist_obj and isinstance(hist_obj, hist.Hist):
                    # Apply cut for MET plots if the variable is 'met' and histogram has a 'met' axis
                    if var == "met":
                        met_axis_name = None
                        for axis in hist_obj.axes:
                            if axis.name == "met":
                                met_axis_name = axis.name
                                break
                        if met_axis_name:
                            # Rebinning is not supported for hist.Hist objects if the binning is changed during slicing
                            # For simple cut, direct slicing is fine
                            try:
                                hist_obj = hist_obj[{met_axis_name: slice(hist.loc(150.0), None)}]
                            except Exception as e:
                                logging.warning(f"Failed to apply 150GeV cut on MET axis for variable '{var}': {e}")
                        else:
                            logging.warning(f"MET axis not found in histogram for variable '{var}'. Cannot apply 150GeV cut.")
                    return hist_obj
                else:
                    logging.warning(f"Histogram '{var}' (region: {reg}) not found or not a hist.Hist object in {path}.")
                    return None
            except Exception as e:
                logging.error(f"Error loading histogram from {path}: {e}")
                return None

        # Load data histogram
        data_hist = None
        if data_file:
            data_hist = load_hist_from_file(data_file, variable, region)

        # Load background histograms and group them by process
        bkg_hists_by_proc = {}
        for bkg_file in background_files:
            proc_name_from_file = Path(bkg_file).stem
            proc_name = proc_name_from_file.split('_')[0]

            h = load_hist_from_file(bkg_file, variable, region)
            if h:
                if proc_name in bkg_hists_by_proc:
                    bkg_hists_by_proc[proc_name] += h
                else:
                    bkg_hists_by_proc[proc_name] = h

        if not bkg_hists_by_proc:
            logging.error(f"No background histograms found for variable '{variable}' in region '{region}'.")
            return ""

        # --- Normalization ---
        data_integral = data_hist.sum().value if data_hist and data_hist.sum().value > 0 else 0.0
        mc_total_hist = sum(bkg_hists_by_proc.values())
        mc_integral = mc_total_hist.sum().value if mc_total_hist.sum().value > 0 else 0.0

        scale_factor = 1.0
        if mc_integral > 0 and data_integral > 0:
            scale_factor = data_integral / mc_integral

        logging.info(f"Data integral: {data_integral}, MC integral: {mc_integral}, Scale factor: {scale_factor}")

        # Apply scale factor
        bkg_hists_scaled = {proc: h * scale_factor for proc, h in bkg_hists_by_proc.items()}

        # --- Sorting for stacking ---
        # Sort backgrounds by their integral in ascending order
        sorted_procs = sorted(bkg_hists_scaled.keys(), key=lambda p: bkg_hists_scaled[p].sum().value)

        sorted_hists = [bkg_hists_scaled[p] for p in sorted_procs]
        sorted_labels = [self.labels.get(p, p) for p in sorted_procs]
        sorted_colors = [self.colors.get(p, '#a6cee3') for p in sorted_procs]

        mc_total_hist_scaled = sum(sorted_hists)

        # --- Plotting ---
        fig, (ax, rax) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": (3, 1)}, sharex=True)
        fig.subplots_adjust(hspace=0.06)

        # Plot stacked backgrounds
        hep.histplot(
            sorted_hists,
            ax=ax,
            stack=True,
            histtype='fill',
            label=sorted_labels,
            color=sorted_colors
        )

        # Plot data
        if data_hist and data_hist.sum().value > 0:
            hep.histplot(data_hist, ax=ax, histtype='errorbar', color='black', label='Data', yerr=True)

        # Plot total MC uncertainty
        if mc_total_hist_scaled.sum().value > 0:
            ax.stairs(
                values=mc_total_hist_scaled.values() + np.sqrt(mc_total_hist_scaled.variances()),
                baseline=mc_total_hist_scaled.values() - np.sqrt(mc_total_hist_scaled.variances()),
                edges=mc_total_hist_scaled.axes[0].edges,
                label='Stat. Unc.',
                hatch='///',
                facecolor='none',
                linewidth=0
            )

        ax.set_ylabel('Events/bin')
        ax.set_xlabel('') # Remove redundant x-label from top plot
        ax.set_yscale('log')
        ax.legend()

        # Ratio plot
        if data_hist and mc_total_hist_scaled.sum().value > 0:
            # Manually calculate ratio and errors
            data_vals = data_hist.values()
            data_vars = data_hist.variances()
            mc_vals = mc_total_hist_scaled.values()
            mc_vars = mc_total_hist_scaled.variances()

            # Avoid division by zero
            mc_vals_safe = np.where(mc_vals > 0, mc_vals, 1)

            ratio_vals = data_vals / mc_vals_safe

            # Error propagation for ratio: sqrt((err_data/mc)**2 + (data*err_mc/mc**2)**2)
            # Simplified: err_ratio = err_data / mc
            err_data_sq = data_vars / mc_vals_safe**2
            err_mc_sq = (data_vals**2 * mc_vars) / mc_vals_safe**4
            ratio_err = np.sqrt(err_data_sq + err_mc_sq)

            centers = data_hist.axes[0].centers

            rax.errorbar(centers, ratio_vals, yerr=ratio_err, fmt='o', color='black')

        rax.axhline(1, ls='--', color='gray')
        rax.set_ylabel('(Data-Pred)/Pred')
        rax.set_xlabel(xlabel)
        rax.set_ylim(0.5, 1.5)

        # CMS label
        hep.cms.label(ax=ax, data=True, year=title_tag.split(',')[1].strip() if ',' in title_tag else '2023', lumi=59.7)

        # Save
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Created stacked plot at {out_path}")
        return str(out_path)

    def create_event_level_variable_plots(self, results: Dict[str, Any], output_dir: str,
                                          show_data: bool, version: str) -> Dict[str, str]:
        """
        Create individual plots for each variable found in the top-level "histograms"
        key of the analysis results (event-level analysis output).

        Args:
            results: Analysis results dictionary (from DarkBottomLineProcessor)
            output_dir: Base output directory
            show_data: Whether to show data points
            version: Version string for multi-format output

        Returns:
            Dictionary of plot file paths
        """
        plot_files = {}

        event_histograms = results.get("histograms", {})
        if not event_histograms:
            logging.warning("No top-level 'histograms' found in the results for event-level plotting.")
            return plot_files

        logging.info("Creating event-level variable plots...")

        # Filter for key variables (MET, electron, muon pt and eta) based on configs
        # For now, let's assume all histograms in 'histograms' are desired.
        # A more sophisticated filtering could be added here if needed,
        # potentially by reading plotting.yaml or another config file.

        for var_name, hist_data in event_histograms.items():
            try:
                if hist_data is None:
                    continue

                # Create figure without ratio panel for single plots
                fig, ax_main = plt.subplots(1, 1, figsize=(10, 8))

                # Plot histogram on main axis
                # For event-level plots, we don't have a clear "region" concept,
                # so the ratio plot context is less direct. We'll plot a single histogram.
                self._plot_single_histogram(ax_main, hist_data, var_name, show_data)

                # Determine if this variable should use log scale
                use_log_scale = var_name not in self.no_log_scale_vars

                # Plot filename: just the variable name for event-level
                plot_filename = var_name

                # Save with log scale (default for most plots)
                if use_log_scale:
                    ax_main.set_yscale('log')
                    ax_main.set_ylim(bottom=0.1)
                    saved_files = self.save_plot_multi_format(
                        fig, plot_filename, "event_level", version, output_dir,
                        is_log=True, data_hists=None, mc_hists=None, signal_hists=None
                    )
                    # Also save linear version
                    ax_main.set_yscale('linear')
                    ax_main.set_ylim(bottom=0)
                    saved_files_linear = self.save_plot_multi_format(
                        fig, plot_filename, "event_level", version, output_dir,
                        is_log=False, data_hists=None, mc_hists=None, signal_hists=None
                    )
                else:
                    # Save linear version only (no log scale)
                    saved_files = self.save_plot_multi_format(
                        fig, plot_filename, "event_level", version, output_dir,
                        is_log=False, data_hists=None, mc_hists=None, signal_hists=None
                    )
                    saved_files_linear = {} # Placeholder for unused variable

                plt.close(fig)

                plot_files[var_name] = saved_files.get('png', '')

            except Exception as e:
                logging.warning(f"Failed to create event-level plot for {var_name}: {e}")

        logging.info(f"Created {len(plot_files)} event-level plots.")
        return plot_files

    def _parse_region_name(self, region: str) -> Dict[str, str]:
        """
        Parse region name into category and region directory name.

        Examples:
            "1b:SR" -> {"category": "1b", "region_dir": "SR"}
            "1b:CR_Wmunu" -> {"category": "1b", "region_dir": "Wlnu_mu"}
            "2b:CR_Topenu" -> {"category": "2b", "region_dir": "Top_el"}
            "1b:CR_Zmumu" -> {"category": "1b", "region_dir": "Zll_mu"}

        Args:
            region: Region name (e.g., "1b:SR", "2b:CR_Wmunu")

        Returns:
            Dictionary with category and region_dir
        """
        parts = region.split(":")
        if len(parts) != 2:
            return {"category": "event_selection", "region_dir": "event_selection"}

        category = parts[0]  # e.g., "1b" or "2b"
        region_part = parts[1]  # e.g., "SR" or "CR_Wmunu"

        # Clean up region directory name
        if region_part.startswith("CR_"):
            region_dir = region_part.replace("CR_", "")
        else:
            region_dir = region_part

        return {
            "category": category,
            "region_dir": region_dir
        }

    def save_plot_multi_format(
        self,
        fig: plt.Figure,
        hist_name: str,
        region: str,
        version: str,
        base_output_dir: str = "outputs",
        is_log: bool = False,
        data_hists: Optional[Dict[str, Any]] = None,
        mc_hists: Optional[Dict[str, Any]] = None,
        signal_hists: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Save plot in ALL formats (PNG, PDF, ROOT, TXT) automatically in batch mode.

        Directory structure:
        - {base_output_dir}/plots/{version}/png/{category}/{region_dir}/{hist_name}.png
        - {base_output_dir}/plots/{version}/pdf/{category}/{region_dir}/{hist_name}.pdf
        - {base_output_dir}/plots/{version}/root/{hist_name}.root (one file per variable, shared across regions)
        - {base_output_dir}/plots/{version}/text/{category}/{region_dir}/{hist_name}.txt (yields)

        Example:
            For region "1b:CR_Wmunu" and hist_name "met":
            - outputs/plots/20231105_1430/png/1b/Wlnu_mu/met.png
            - outputs/plots/20231105_1430/pdf/1b/Wlnu_mu/met.pdf
            - outputs/plots/20231105_1430/root/met.root
            - outputs/plots/20231105_1430/text/1b/Wlnu_mu/met.txt

        Args:
            fig: Matplotlib figure object
            hist_name: Name of the histogram
            region: Region name (e.g., "1b:SR", "2b:CR_Wmunu")
            version: Version string (e.g., "v20251029_HHMM")
            base_output_dir: Base output directory
            is_log: Whether plot is in log scale
            data_hists: Data histogram data for ROOT file
            mc_hists: MC histogram data for ROOT file
            signal_hists: Signal histogram data for ROOT file

        Returns:
            Dictionary with paths to saved files
        """
        saved_files = {}

        # Parse region name to get category and region directory name
        region_info = self._parse_region_name(region)
        category = region_info["category"]
        region_dir = region_info["region_dir"]

        # Create directory structure: plots/{version}/png/{category}/{region_dir}/
        if category == "event_selection":
            png_dir  = os.path.join(base_output_dir, "plots", version, "png",  "event_selection")
            pdf_dir  = os.path.join(base_output_dir, "plots", version, "pdf",  "event_selection")
            text_dir = os.path.join(base_output_dir, "plots", version, "text", "event_selection")
        else:
            region_label = f"{category}_{region_dir}"
            png_dir  = os.path.join(base_output_dir, "plots", version, "png",  "region_analysis", region_label)
            pdf_dir  = os.path.join(base_output_dir, "plots", version, "pdf",  "region_analysis", region_label)
            text_dir = os.path.join(base_output_dir, "plots", version, "text", "region_analysis", region_label)
        root_dir = os.path.join(base_output_dir, "plots", version, "root")

        # Create all directories
        for dir_path in [png_dir, pdf_dir, root_dir, text_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # File stem: region_hist_{category}_{region_dir}_{variable}{_log}
        log_suffix = "_log" if is_log else ""
        if category == "event_selection":
            file_stem = f"hist_event_selection_{hist_name}{log_suffix}"
        else:
            file_stem = f"hist_{category}_{region_dir}_{hist_name}{log_suffix}"

        # Save PNG
        png_path = os.path.join(png_dir, f"{file_stem}.png")
        fig.savefig(png_path, dpi=self.dpi, bbox_inches='tight')
        saved_files['png'] = png_path

        # Save PDF
        pdf_path = os.path.join(pdf_dir, f"{file_stem}.pdf")
        fig.savefig(pdf_path, bbox_inches='tight')
        saved_files['pdf'] = pdf_path

        # ROOT: written by _plot_stacked_variable directly via uproot
        saved_files['root'] = None

        return saved_files

    # -----------------------------------------------------------------------
    # Stacked-plot infrastructure (event-selection + region modes)
    # -----------------------------------------------------------------------

    def _load_folder(self, folder: str) -> Dict[str, Any]:
        """Load a single sample folder (ROOT or PKL files).

        Returns {"weighted_total_events": int, "objects": Dict[str, Any]}.
        """
        folder_path = Path(folder)
        if not folder_path.is_dir():
            # single file path
            if folder_path.suffix in (".root",):
                return self._load_root_files([folder_path])
            return self._load_pkl_files([folder_path])

        root_files = sorted(folder_path.glob("*.root"))
        pkl_files = sorted(p for p in folder_path.glob("*.pkl")
                           if not p.name.endswith((".awk_raw.pkl", "raw.pkl")))
        if root_files:
            return self._load_root_files(root_files)
        if pkl_files:
            return self._load_pkl_files(pkl_files)
        raise FileNotFoundError(f"No ROOT or PKL files in {folder}")

    def _load_pkl_files(self, paths: List[Path], keys_filter: Optional[List[str]] = None) -> Dict[str, Any]:
        merged: Dict[str, Any] = {"weighted_total_events": 0, "objects": {}}
        loaded_count = 0
        for p in paths:
            try:
                with open(p, "rb") as fh:
                    data = pickle.load(fh)
                if not isinstance(data, dict):
                    logging.warning("Skipping %s: top-level type is %s, expected dict", p.name, type(data).__name__)
                    continue
                wte = int(data.get("weighted_total_events", 0) or 0)
                merged["weighted_total_events"] += wte
                objs = data.get("objects", {})
                obj_keys = list(objs.keys()) if isinstance(objs, dict) else []
                if isinstance(objs, dict):
                    for k, v in objs.items():
                        if keys_filter is not None and not _should_load_key(k, keys_filter):
                            continue
                        if k in merged["objects"]:
                            if isinstance(merged["objects"][k], list) and isinstance(v, list):
                                merged["objects"][k] = merged["objects"][k] + v
                            elif isinstance(merged["objects"][k], np.ndarray) and isinstance(v, np.ndarray):
                                merged["objects"][k] = np.concatenate([merged["objects"][k], v])
                        else:
                            merged["objects"][k] = v
                loaded_count += 1
                kept_keys = [k for k in obj_keys if keys_filter is None or _should_load_key(k, keys_filter)]
                logging.info("Loaded PKL %s: wte=%d, objects=%d keys (kept=%d) %s",
                             p.name, wte, len(obj_keys), len(kept_keys),
                             f"({', '.join(kept_keys[:5])}{'...' if len(kept_keys) > 5 else ''})" if kept_keys else "()")
            except Exception as exc:
                logging.warning("Could not load %s: %s", p.name, exc)
        logging.info("PKL load complete: %d/%d files loaded, total wte=%d, total object keys=%d",
                     loaded_count, len(paths), merged["weighted_total_events"], len(merged["objects"]))
        return merged

    def _load_root_files(self, paths: List[Path], keys_filter: Optional[List[str]] = None) -> Dict[str, Any]:
        try:
            import uproot
        except ImportError:
            raise ImportError("uproot required for ROOT file loading")
        merged: Dict[str, Any] = {"weighted_total_events": 0, "objects": {}}
        loaded_count = 0
        for p in paths:
            try:
                with uproot.open(str(p)) as f:
                    if "Events" not in f:
                        logging.warning("Skipping ROOT %s: no 'Events' tree", p.name)
                        continue
                    tree = f["Events"]
                    objs: Dict[str, Any] = {}
                    skip_branches = 0
                    for branch in tree.keys():
                        if keys_filter is not None and not _should_load_key(str(branch), keys_filter):
                            skip_branches += 1
                            continue
                        try:
                            arr = tree[branch].array(library="np")
                            objs[branch] = arr.tolist() if hasattr(arr, "tolist") else list(arr)
                        except Exception:
                            skip_branches += 1
                    wte = 0
                    for key in ("weighted_total_events", "weighted_total_events;1"):
                        if key in f:
                            try:
                                wte = int(round(float(f[key].values()[0])))
                                break
                            except Exception:
                                pass
                    if wte == 0 and "Metadata" in f:
                        meta = f["Metadata"]
                        if "weighted_total_events" in meta.keys():
                            arr = meta["weighted_total_events"].array(library="np")
                            wte = int(np.sum(arr)) if len(arr) > 0 else 0
                    merged["weighted_total_events"] += wte
                    for k, v in objs.items():
                        if k in merged["objects"]:
                            if isinstance(merged["objects"][k], list) and isinstance(v, list):
                                merged["objects"][k] = merged["objects"][k] + v
                        else:
                            merged["objects"][k] = v
                    loaded_count += 1
                    branch_keys = list(objs.keys())
                    logging.info("Loaded ROOT %s: wte=%d, branches=%d (skipped=%d) keys=%s",
                                 p.name, wte, len(branch_keys), skip_branches,
                                 f"({', '.join(branch_keys[:5])}{'...' if len(branch_keys) > 5 else ''})" if branch_keys else "()")
            except Exception as exc:
                logging.warning("Could not load ROOT %s: %s", p.name, exc)
        logging.info("ROOT load complete: %d/%d files loaded, total wte=%d, total object keys=%d",
                     loaded_count, len(paths), merged["weighted_total_events"], len(merged["objects"]))
        return merged

    def _write_yield_table(
        self,
        stem: Path,
        variable: str,
        bins: np.ndarray,
        background_rows: List[Tuple[str, np.ndarray, np.ndarray]],
        data_ndarray: Optional[np.ndarray],
    ) -> None:
        n_bins = len(bins) - 1
        bin_labels = [f"[{bins[i]:.4g},{bins[i+1]:.4g})" for i in range(n_bins)]
        mc_total = np.zeros(n_bins, dtype=float)
        mc_total_sumw2 = np.zeros(n_bins, dtype=float)
        for _, hv, hs in background_rows:
            mc_total += hv
            mc_total_sumw2 += hs

        rows = list(background_rows)
        rows.append(("Total_Bkg", mc_total, mc_total_sumw2))
        if data_ndarray is not None:
            with np.errstate(invalid="ignore"):
                rows.append(("data_obs", data_ndarray, np.sqrt(np.maximum(data_ndarray, 0.0))))

        tex_path = stem.with_suffix(".tex")
        with open(tex_path, "w") as fh:
            col_spec = "l" + "c" * n_bins
            bin_hdr = " & ".join(f"\\textbf{{{b}}}" for b in bin_labels)
            fh.write("\\begin{table}[htbp]\n\\centering\n")
            _cap_var = variable.replace("_", r"\_")
            _lbl_var = variable.replace("_", "-")
            fh.write(f"\\caption{{Yield table for \\texttt{{{_cap_var}}}}}\n\\label{{tab:{_lbl_var}}}\n")
            fh.write(f"\\begin{{tabular}}{{{col_spec}}}\n\\toprule\n")
            fh.write(f"\\textbf{{Sample}} & {bin_hdr} \\\\ \\midrule\n")
            for label, vals, sumw2 in rows:
                if label in ("Total_Bkg", "data_obs"):
                    continue
                cells = " & ".join(f"${vals[i]:.2f} \\pm {np.sqrt(sumw2[i]):.2f}$" for i in range(n_bins))
                fh.write(f"{label} & {cells} \\\\\n")
            for label, vals, sumw2 in rows:
                if label != "Total_Bkg":
                    continue
                cells = " & ".join(f"${vals[i]:.2f} \\pm {np.sqrt(sumw2[i]):.2f}$" for i in range(n_bins))
                fh.write(f"\\midrule\nTotal Bkg & {cells} \\\\ \\midrule\n")
            for label, vals, sumw2 in rows:
                if label != "data_obs":
                    continue
                cells = " & ".join(f"${vals[i]:.2f} \\pm {np.sqrt(sumw2[i]):.2f}$" for i in range(n_bins))
                fh.write(f"Data obs & {cells} \\\\ \\bottomrule\n")
            fh.write("\\end{tabular}\n\\end{table}\n")

    def _plot_stacked_variable(
        self,
        variable: str,
        bins: np.ndarray,
        background_rows: List[Tuple[str, np.ndarray, np.ndarray]],
        data_ndarray: Optional[np.ndarray],
        output_dir: str,
        luminosity: float,
        year: str,
        region: str = "event_selection",
        version: str = "",
        save_root: bool = False,
        signal_rows: Optional[List[Tuple[str, np.ndarray]]] = None,
    ) -> List[str]:
        """Draw CMS-style stacked histogram with ratio panel and save in 5 formats.

        signal_rows: list of (label, hv) — pre-scaled per-masspoint histograms.
          PNG/PDF: first 3 drawn as dashed lines.
          ROOT: all written as individual TH1s.
        """
        if not _HAS_MPLHEP:
            logging.warning("mplhep not available — skipping stacked plot for %s", variable)
            return []

        total_mc_yield = float(np.sum(sum(h for _, h, _ in background_rows)))
        data_yield = float(np.sum(data_ndarray)) if data_ndarray is not None else None
        logging.debug("_plot_stacked_variable: %s/%s, rows=%d, bins=%d, total_mc=%.3f, data=%s",
                      region, variable, len(background_rows), len(bins) - 1,
                      total_mc_yield,
                      f"{data_yield:.1f}" if data_yield is not None else "None")

        # Sort backgrounds ascending by integral (smallest at bottom)
        rows = sorted(background_rows, key=lambda t: float(np.sum(t[1])))
        # Fall back to process-config color map for labels not in yaml group overrides
        color_map = get_background_color_map([r[0] for r in rows])

        show_ratio = data_ndarray is not None
        if show_ratio:
            fig, (ax, ax_ratio) = plt.subplots(
                2, 1, figsize=self.figsize, sharex=True,
                gridspec_kw={"height_ratios": [self.main_height, self.ratio_height],
                             "hspace": self.subplots_hspace},
            )
            fig.subplots_adjust(top=self.subplots_top, bottom=self.subplots_bottom,
                                left=self.subplots_left, right=self.subplots_right)
        else:
            fig, ax = plt.subplots(figsize=self.figsize_no_ratio)
            ax_ratio = None
            fig.subplots_adjust(top=self.subplots_top, bottom=0.12,
                                left=self.subplots_left, right=self.subplots_right)

        cumulative = np.zeros(len(bins) - 1, dtype=float)
        cumulative_sq = np.zeros(len(bins) - 1, dtype=float)
        for label, hist_values, hist_sumw2 in rows:
            next_cum = cumulative + hist_values
            cumulative_sq += hist_sumw2
            # yaml group color override → process config fallback → palette fallback
            color = (self._group_colors.get(label)
                     or color_map.get(simplify_sample_label(label), "#3f90da"))
            disp_label = (self._group_labels.get(label)
                          or _get_legend_label(label))
            ax.stairs(
                next_cum, bins, baseline=cumulative,
                fill=True, alpha=1.0, linewidth=0, color=color,
                label=disp_label,
            )
            cumulative = next_cum

        mc_stat_err = np.sqrt(cumulative_sq)
        centers = 0.5 * (bins[:-1] + bins[1:])
        unc_x = np.append(bins[:-1], bins[-1])
        unc_lo = np.append(cumulative - mc_stat_err, (cumulative - mc_stat_err)[-1])
        unc_hi = np.append(cumulative + mc_stat_err, (cumulative + mc_stat_err)[-1])
        ax.fill_between(
            unc_x, unc_lo, unc_hi, step="post",
            hatch=self.unc_hatch, facecolor=self.unc_facecolor, edgecolor=self.unc_edgecolor,
            linewidth=0.0, alpha=self.unc_alpha, zorder=5,
        )

        # Signal overlays — draw first 3 as dashed lines (PNG/PDF only; all go to ROOT)
        if signal_rows:
            for _si, (_slabel, _shv) in enumerate(signal_rows[:3]):
                _sc = self.signal_colors[_si % len(self.signal_colors)]
                ax.stairs(
                    _shv, bins, baseline=0,
                    fill=False, linewidth=self.signal_linewidth, linestyle="--", color=_sc,
                    label=_slabel, zorder=8,
                )

        if data_ndarray is not None:
            half_width = 0.5 * (bins[1:] - bins[:-1])
            mask = data_ndarray > 0
            if np.any(mask):
                ax.errorbar(
                    centers[mask], data_ndarray[mask],
                    xerr=half_width[mask], yerr=np.sqrt(data_ndarray[mask]),
                    fmt="o", color=self.data_color, markerfacecolor=self.data_color,
                    markeredgecolor=self.data_color, markersize=self.data_markersize,
                    elinewidth=self.data_elinewidth, capsize=0, label="Data", zorder=10,
                )

        use_log = variable not in self.no_log_scale_vars
        if use_log:
            ax.set_yscale("log")
            stacked_max = float(np.max(cumulative)) if cumulative.size else 0.0
            data_max = float(np.max(data_ndarray)) if data_ndarray is not None and data_ndarray.size else 0.0
            ax.set_ylim(0.1, max(stacked_max, data_max, 1e-3) * 1000.0)

        # Always use full bin range from config — never auto-trim to nonzero data range
        x_lo = float(bins[0])
        x_hi = float(bins[-1])
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylabel("Events / bin", fontsize=self.fontsize_axis, labelpad=6)
        if not show_ratio:
            ax.set_xlabel(self.variable_labels.get(variable, variable), fontsize=self.fontsize_axis)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=8, steps=[1, 2, 5, 10]))
        ax.grid(False)

        if show_ratio and ax_ratio is not None and data_ndarray is not None:
            pred = cumulative
            data_ratio = np.divide(
                data_ndarray, pred,
                out=np.full_like(data_ndarray, np.nan, dtype=float), where=pred > 0,
            )
            data_ratio_err = np.divide(
                np.sqrt(np.maximum(data_ndarray, 0.0)), pred,
                out=np.zeros_like(data_ndarray, dtype=float), where=pred > 0,
            )
            pred_rel_err = np.divide(
                mc_stat_err, pred,
                out=np.zeros_like(pred, dtype=float), where=pred > 0,
            )
            ratio_mask = np.isfinite(data_ratio)
            ax_ratio.axhline(1.0, color=self.data_color, linestyle="-", linewidth=1.2)
            ratio_lo = np.append(1.0 - pred_rel_err, (1.0 - pred_rel_err)[-1])
            ratio_hi = np.append(1.0 + pred_rel_err, (1.0 + pred_rel_err)[-1])
            ax_ratio.fill_between(
                unc_x, ratio_lo, ratio_hi, step="post",
                hatch=self.unc_hatch, facecolor=self.unc_facecolor, edgecolor=self.unc_edgecolor,
                linewidth=0.0, alpha=self.unc_alpha, label=self.unc_label, zorder=5,
            )
            if np.any(ratio_mask):
                half_width = 0.5 * (bins[1:] - bins[:-1])
                ax_ratio.errorbar(
                    centers[ratio_mask], data_ratio[ratio_mask],
                    xerr=half_width[ratio_mask], yerr=data_ratio_err[ratio_mask],
                    fmt="o", color=self.data_color, markerfacecolor=self.data_color,
                    markeredgecolor=self.data_color, markersize=self.data_markersize,
                    elinewidth=self.data_elinewidth, capsize=0, zorder=10,
                )
            ax_ratio.legend(loc="upper right", fontsize=self.fontsize_legend, frameon=False)
            ax_ratio.set_ylabel("Data / MC", fontsize=self.fontsize_axis, labelpad=6)
            ax_ratio.set_xlabel(self.variable_labels.get(variable, variable), fontsize=self.fontsize_axis, labelpad=8)
            ax_ratio.set_ylim(self.ratio_ylim[0], self.ratio_ylim[1])
            ax_ratio.set_xlim(x_lo, x_hi)
            ax_ratio.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=8, steps=[1, 2, 5, 10]))
            ax_ratio.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0, 0.5, 1.0, 1.5, 2.0]))
            ax_ratio.grid(False)

        hep.cms.label(
            llabel=self.cms_label,
            data=data_ndarray is not None,
            lumi=round(luminosity, 2),
            com=self.com_energy,
            loc=0,
            ax=ax,
        )

        # Region label text box (upper left, no border, bold)
        _region_pretty = _pretty_region_label(region)
        if _region_pretty:
            ax.text(
                0.05, 0.88, _region_pretty,
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=self.fontsize_legend,
                fontweight="bold",
                bbox=dict(boxstyle="square,pad=0.1", facecolor="none",
                          edgecolor="none", alpha=1.0),
            )

        handles, labels_leg = ax.get_legend_handles_labels()
        if handles:
            data_idx = next((i for i, l in enumerate(labels_leg) if l == "Data"), None)
            unc_idx = next((i for i, l in enumerate(labels_leg)
                            if l == self.unc_label or "unc" in l.lower()), None)
            ordered = (
                ([data_idx] if data_idx is not None else [])
                + [i for i in range(len(labels_leg)) if i not in (data_idx, unc_idx)]
                + ([unc_idx] if unc_idx is not None else [])
            )
            handles = [handles[i] for i in ordered]
            labels_leg = [labels_leg[i] for i in ordered]
            if unc_idx is not None:
                handles[-1] = matplotlib.patches.Patch(
                    hatch=self.unc_hatch, facecolor=self.unc_facecolor,
                    edgecolor=self.unc_edgecolor,
                    linewidth=0.0, label=self.unc_label,
                )
            ax.legend(
                handles, labels_leg,
                loc="upper right", bbox_to_anchor=(0.97, 0.97),
                ncol=self.legend_ncol, frameon=False, borderaxespad=0.0,
                handlelength=1.5, columnspacing=1.0,
                handletextpad=0.5, fontsize=self.fontsize_legend,
            )

        saved = self.save_plot_multi_format(
            fig, variable, region, version, output_dir,
            is_log=use_log,
        )
        plt.close(fig)

        # Write ROOT file via uproot with actual histogram arrays
        try:
            import uproot as _up
            region_info_r = self._parse_region_name(region)
            cat_r = region_info_r["category"]
            rdir_r = region_info_r["region_dir"]
            if cat_r == "event_selection":
                _root_stem = f"hist_event_selection_{variable}{'_log' if use_log else ''}"
            else:
                _root_stem = f"hist_{cat_r}_{rdir_r}_{variable}{'_log' if use_log else ''}"
            root_out = Path(output_dir) / "plots" / version / "root" / f"{_root_stem}.root"
            root_out.parent.mkdir(parents=True, exist_ok=True)
            with _up.recreate(str(root_out)) as rf:
                for label, hv, hs in rows:
                    rf[label] = (hv.astype(float), bins.astype(float))
                mc_total = sum(hv for _, hv, _ in rows)
                rf["TotalBkg"] = (mc_total.astype(float), bins.astype(float))
                if data_ndarray is not None:
                    rf["data_obs"] = (data_ndarray.astype(float), bins.astype(float))
                # All signal masspoints as individual TH1s (not stacked)
                if signal_rows:
                    for _slabel, _shv in signal_rows:
                        _safe_key = _slabel.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "_").replace("=", "_")
                        rf[f"sig_{_safe_key}"] = (_shv.astype(float), bins.astype(float))
            saved['root'] = str(root_out)
        except Exception as _re:
            logging.warning("ROOT write failed for %s/%s: %s", region, variable, _re)

        # Write yield tables alongside the PNG — consistent with save_plot_multi_format paths
        region_info = self._parse_region_name(region)
        category = region_info["category"]
        region_dir = region_info["region_dir"]
        _is_log_yield = saved.get("png", "").endswith("_log.png")
        _log_suf = "_log" if _is_log_yield else ""
        if category == "event_selection":
            text_dir = Path(output_dir) / "plots" / version / "text" / "event_selection"
            _yield_stem = text_dir / f"hist_event_selection_{variable}{_log_suf}"
        else:
            region_label = f"{category}_{region_dir}"
            text_dir = Path(output_dir) / "plots" / version / "text" / "region_analysis" / region_label
            _yield_stem = text_dir / f"hist_{category}_{region_dir}_{variable}{_log_suf}"
        text_dir.mkdir(parents=True, exist_ok=True)
        self._write_yield_table(_yield_stem, variable, bins, rows, data_ndarray)

        result_paths = [v for v in saved.values() if v]
        logging.debug("_plot_stacked_variable done: %s/%s -> %d files (%s)",
                      region, variable, len(result_paths),
                      ', '.join(Path(p).suffix for p in result_paths) if result_paths else "none")
        return result_paths

    def create_stacked_plots(
        self,
        mode: str,
        input_folder: str,
        process_groups: Dict[str, List[str]],
        output_dir: str,
        luminosity: float,
        year: str,
        version: str,
        signal_groups: Optional[Dict[str, List[str]]] = None,
        data_groups: Optional[Dict[str, List[str]]] = None,
        cross_sections: Optional[Dict[str, float]] = None,
        variables: Optional[List[str]] = None,
        regions: Optional[List[str]] = None,
        save_root: bool = False,
        regions_config: Optional[str] = None,
        weight_systematic: Optional[str] = None,
        show_data: bool = False,
        signal_scale: float = 1.0,
        make_syst_plots: bool = False,
        apply_dnn: bool = False,
        dnn_model: Optional[str] = None,
        dnn_config: Optional[str] = None,
        dnn_mass_scan: Optional[str] = None,
    ) -> List[str]:
        """Create stacked MC+data plots.

        input_folder:   single directory containing all ROOT/PKL files.
        process_groups: {label: [patterns]} for background stacks.
        signal_groups:  {label: [patterns]} for signal overlays (drawn as lines).
        data_groups:    {label: [patterns]} for data points (no xsec normalization).
        All groups resolved by substring pattern match against filenames in input_folder.
        mode: "event-selection" | "region" | "region-from-events"
        regions_config: path to regions.yaml (required for region-from-events mode)
        weight_systematic: weight branch suffix for syst variation, e.g. "weight_pileupUP"
                           (None = use full_event_weight nominal)
        """
        if mode not in ("event-selection", "region", "region-from-events"):
            raise ValueError(
                f"Unknown mode '{mode}'. Use 'event-selection', 'region', or 'region-from-events'."
            )

        self._regions_config_path = regions_config
        self._weight_systematic = weight_systematic
        cross_sections = cross_sections or {}
        signal_groups = signal_groups or {}
        data_groups = data_groups or {}
        created: List[str] = []

        if mode == "event-selection":
            created.extend(self._create_event_selection_plots(
                input_folder=input_folder,
                process_groups=process_groups,
                signal_groups=signal_groups,
                data_groups=data_groups,
                output_dir=output_dir,
                luminosity=luminosity,
                year=year,
                version=version,
                cross_sections=cross_sections,
                variables=variables,
                save_root=save_root,
            ))
        elif mode == "region":
            created.extend(self._create_region_stacked_plots(
                input_folder=input_folder,
                process_groups=process_groups,
                signal_groups=signal_groups,
                data_groups=data_groups,
                output_dir=output_dir,
                luminosity=luminosity,
                year=year,
                version=version,
                cross_sections=cross_sections,
                variables=variables,
                regions=regions,
                save_root=save_root,
            ))
        else:  # region-from-events
            regions_config = getattr(self, "_regions_config_path", None)
            created.extend(self._create_region_from_events_plots(
                input_folder=input_folder,
                process_groups=process_groups,
                signal_groups=signal_groups,
                data_groups=data_groups,
                output_dir=output_dir,
                luminosity=luminosity,
                year=year,
                version=version,
                cross_sections=cross_sections,
                variables=variables,
                regions=regions,
                regions_config=regions_config,
                weight_systematic=getattr(self, "_weight_systematic", None),
                show_data=show_data,
                signal_scale=signal_scale,
                make_syst_plots=make_syst_plots,
                apply_dnn=apply_dnn,
                dnn_model=dnn_model,
                dnn_config=dnn_config,
                dnn_mass_scan=dnn_mass_scan,
            ))
        return created

    def _load_one_file(self, path: Path, variables: Optional[List[str]] = None) -> Dict[str, Any]:
        """Load a single ROOT or PKL file. Returns {"weighted_total_events": int, "objects": dict}."""
        if path.suffix == ".root":
            return self._load_root_files([path], keys_filter=variables)
        return self._load_pkl_files([path], keys_filter=variables)

    @staticmethod
    def _normalize_cross_sections(cross_sections: Dict[str, float]) -> Dict[str, float]:
        """Convert nested xsection dict to flat format if needed.

        Nested format (from xsection_background.json):
            {"WtoLNu-2Jets": [{"year":..., "process":..., "xsection":..., "full_dataset":...}, ...]}
        Flat format (expected by lookup):
            {"WtoLNu-2Jets_PTLNu-40to100_...": 1598.0, ...}
        """
        if not cross_sections:
            return cross_sections
        _first_val = next(iter(cross_sections.values()), None)
        if not isinstance(_first_val, list):
            return cross_sections
        _flat_xs: Dict[str, float] = {}
        for _entries in cross_sections.values():
            if not isinstance(_entries, list):
                continue
            for _entry in _entries:
                if not isinstance(_entry, dict):
                    continue
                _xs = _entry.get("xsection")
                if _xs is None:
                    continue
                _xs_f = float(_xs)
                _fd = _entry.get("full_dataset")
                # full_dataset may be a single name (str) or a list of
                # per-era dataset-name variants — register every one so a
                # match on any variant resolves to this process's xsec.
                if isinstance(_fd, (list, tuple)):
                    for _name in _fd:
                        if _name:
                            _flat_xs[str(_name)] = _xs_f
                elif _fd:
                    _flat_xs[str(_fd)] = _xs_f
                _proc = _entry.get("process")
                if _proc:
                    _flat_xs[str(_proc)] = _xs_f
        logging.info("Normalized nested xsec dict: %d entries (flat)", len(_flat_xs))
        return _flat_xs

    def _resolve_group_files(self, input_folder: str, patterns: List[str]) -> List[Path]:
        """Find all ROOT/PKL files in input_folder whose name contains any of the patterns.

        Each pattern is a substring match against the filename stem (without extension).
        One file can only match one pattern; first pattern that matches wins.
        Files are returned in deterministic (sorted) order.
        """
        base = Path(input_folder)
        all_files = sorted(
            p for p in base.iterdir()
            if p.is_file() and p.suffix in (".root", ".pkl")
        )
        resolved: List[Path] = []
        matched_paths: set = set()
        # Match on canonicalized forms so a pattern written in canonical form
        # matches a file named in any year's convention (2J / Bin-2J / SMHiggs
        # rename). Data patterns (JetMET-Run, EGamma-Run) are untouched by
        # _clean_sample_name so they keep matching as before.
        canon_names = {p: _clean_sample_name(p.name) for p in all_files}
        for pattern in patterns:
            pat_canon = _clean_sample_name(pattern)
            pattern_hits = 0
            for p in all_files:
                if p in matched_paths:
                    continue
                if pat_canon in canon_names[p]:
                    resolved.append(p)
                    matched_paths.add(p)
                    pattern_hits += 1
            if pattern_hits:
                logging.info("Pattern '%s' matched %d file(s)", pattern, pattern_hits)
            else:
                logging.warning("Pattern '%s' matched 0 files in %s", pattern, input_folder)
        if not resolved:
            logging.warning("No files matched ANY patterns %s in %s (total files: %d)", patterns, input_folder, len(all_files))
        else:
            logging.info("Resolved %d file(s) for group patterns %s", len(resolved), patterns)
        return resolved

    def _load_group_entries(
        self,
        input_folder: str,
        groups: Dict[str, List[str]],
        cross_sections: Dict[str, float],
        variables: Optional[List[str]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Load files for each process group. Returns {label: [{"weighted_total_events", "branches", "xsec"}]}."""
        cross_sections = self._normalize_cross_sections(cross_sections)

        result: Dict[str, List[Dict[str, Any]]] = {}
        logging.info("=== Loading %d process groups from %s ===", len(groups), input_folder)
        for label, patterns in groups.items():
            paths = self._resolve_group_files(input_folder, patterns)
            entries = []
            loaded_paths = []
            skipped_paths = []
            for p in paths:
                _stem = p.stem.replace("_EVENTSELECTION", "")
                xsec = _find_xsec(p.stem, cross_sections)
                if xsec is None:
                    xsec = _find_xsec(_stem, cross_sections)
                if xsec is None:
                    xsec = _find_xsec(p.name, cross_sections)
                try:
                    loaded = self._load_one_file(p, variables)
                    branches = _extract_branches(loaded["objects"])
                    entries.append({
                        "weighted_total_events": loaded["weighted_total_events"],
                        "branches": branches,
                        "xsec": xsec,
                        "path": p,
                    })
                    loaded_paths.append(p.name)
                    logging.info("  [%s] loaded %s: wte=%d, branches=%d, xsec=%s",
                                 label, p.name, loaded["weighted_total_events"],
                                 len(branches), "%.4g pb" % xsec if xsec is not None else "None")
                except Exception as exc:
                    skipped_paths.append((p.name, str(exc)))
                    logging.warning("  [%s] skipping %s: %s", label, p.name, exc)
            if entries:
                result[label] = entries
                total_wte = sum(e["weighted_total_events"] for e in entries)
                logging.info("[%s] SUMMARY: %d/%d files loaded, total wte=%d, unique branches=%d",
                             label, len(entries), len(paths), total_wte,
                             len(set(k for e in entries for k in e["branches"].keys())))
            else:
                logging.warning("[%s] NO FILES LOADED — group will be missing from plots!", label)
            for name, err in skipped_paths:
                logging.debug("  [%s] skipped detail: %s -> %s", label, name, err)
        # Warn about files in the folder that were not claimed by any group
        base = Path(input_folder)
        all_files = sorted(p.name for p in base.iterdir() if p.is_file() and p.suffix in (".root", ".pkl"))
        all_patterns = [pat for pats in groups.values() for pat in pats]
        truly_unmatched = [f for f in all_files if not any(pat in f for pat in all_patterns)]
        if truly_unmatched:
            logging.warning("Files not matched by any process group (%d): %s",
                            len(truly_unmatched),
                            ', '.join(truly_unmatched[:10]) + ('...' if len(truly_unmatched) > 10 else ''))

        loaded_labels = list(result.keys())
        logging.info("=== Group loading complete: %d/%d groups with data (%s) ===",
                     len(loaded_labels), len(groups), ', '.join(loaded_labels) if loaded_labels else "NONE")
        return result

    def _create_event_selection_plots(
        self,
        input_folder: str,
        process_groups: Dict[str, List[str]],
        signal_groups: Dict[str, List[str]],
        data_groups: Dict[str, List[str]],
        output_dir: str,
        luminosity: float,
        year: str,
        version: str,
        cross_sections: Dict[str, float],
        variables: Optional[List[str]],
        save_root: bool,
    ) -> List[str]:
        logging.info("=== Creating event-selection plots ===")
        _load_vars = variables or self.event_selection_variables or None
        bkg_groups  = self._load_group_entries(input_folder, process_groups, cross_sections, _load_vars)
        sig_groups  = self._load_group_entries(input_folder, signal_groups,  cross_sections, _load_vars)
        dat_groups  = self._load_group_entries(input_folder, data_groups,    {}, _load_vars)  # no xsec for data

        logging.info("Loaded groups: bkg=%d, sig=%d, data=%d",
                     len(bkg_groups), len(sig_groups), len(dat_groups))

        # Merge all data entries into one flat branch dict (raw counts, no normalisation)
        data_branches: Optional[Dict[str, np.ndarray]] = None
        if dat_groups:
            merged: Dict[str, np.ndarray] = {}
            for entries in dat_groups.values():
                for e in entries:
                    for k, v in e["branches"].items():
                        merged[k] = np.concatenate([merged[k], v]) if k in merged else v
            data_branches = merged
            logging.info("Merged data branches: %d variables, total events=%s",
                         len(merged),
                         ', '.join(f"{k}={len(v)}" for k, v in list(merged.items())[:3]) + ('...' if len(merged) > 3 else ''))
        else:
            logging.warning("No data groups loaded — data points will not be shown")

        _skip_prefixes = ("weight_", "full_event_weight", "genWeight")
        # Priority: CLI --variables > plotting.yaml event_selection_variables > all branches
        all_vars: List[str] = (
            variables
            or self.event_selection_variables
            or sorted(set(
                k for entries in bkg_groups.values() for e in entries for k in e["branches"]
                if not any(k.startswith(p) for p in _skip_prefixes)
            ))
        )
        logging.info("Variables to plot: %d (%s)", len(all_vars),
                     ', '.join(all_vars[:5]) + ('...' if len(all_vars) > 5 else ''))

        # ---- event-selection cutflow ----
        evtsel_cutflow_per_proc: Dict[str, Dict[str, float]] = {}
        for proc_label, entries in bkg_groups.items():
            proc_cf: Dict[str, float] = {}
            for e in entries:
                cf = None
                try:
                    import uproot as _up
                    p = e.get("path")
                    if p and str(p).endswith(".root"):
                        with _up.open(str(p)) as _f:
                            for _key in ("cutflow", "cutflow;1"):
                                if _key in _f:
                                    _h = _f[_key]
                                    _labels = [str(b) for b in _h.axes[0]]
                                    _vals   = _h.values()
                                    cf = dict(zip(_labels, _vals))
                                    break
                except Exception:
                    pass
                if cf:
                    xsec = e.get("xsec")
                    wte  = e.get("weighted_total_events", 1.0)
                    scale = ((luminosity * xsec * 1000.0) / wte
                             if xsec is not None and wte > 0
                             else (luminosity / wte if wte > 0 else 1.0))
                    for cut, val in cf.items():
                        proc_cf[cut] = proc_cf.get(cut, 0.0) + val * scale
            if proc_cf:
                evtsel_cutflow_per_proc[proc_label] = proc_cf

        created: List[str] = []
        skipped_vars = []
        for var in all_vars:
            all_vals = [
                _apply_variable_plot_filter(var, e["branches"].get(var, np.array([])))
                for entries in bkg_groups.values() for e in entries
            ]
            if data_branches is not None:
                all_vals.append(_apply_variable_plot_filter(var, data_branches.get(var, np.array([]))))

            bins = _make_bins(all_vals, self._build_bins_from_config, var, self._n_bins_default)
            if bins is None or len(bins) < 2:
                total_events = sum(v.size for v in all_vals)
                logging.warning("Skipping %s: could not create bins (total events=%d, n_groups_with_data=%d)",
                                var, total_events, sum(1 for v in all_vals if v.size > 0))
                skipped_vars.append(var)
                continue

            bkg_rows: List[Tuple[str, np.ndarray, np.ndarray]] = []
            for proc_label, entries in bkg_groups.items():
                group_hv = np.zeros(len(bins) - 1, dtype=float)
                group_hs = np.zeros(len(bins) - 1, dtype=float)
                for e in entries:
                    vals = _apply_variable_plot_filter(var, e["branches"].get(var, np.array([])))
                    hv, hs = _histogram_and_sumw2(vals, bins, e["weighted_total_events"], luminosity, e["xsec"])
                    group_hv += hv
                    group_hs += hs
                bkg_rows.append((proc_label, group_hv, group_hs))

            total_mc = sum(h for _, h, _ in bkg_rows)
            if np.allclose(total_mc, 0.0):
                logging.warning("Skipping %s: total MC yield is zero (rows=%d)", var, len(bkg_rows))
                skipped_vars.append(var)
                continue

            data_hist: Optional[np.ndarray] = None
            data_count = 0
            if data_branches is not None:
                dv = _apply_variable_plot_filter(var, data_branches.get(var, np.array([])))
                data_hist = _histogram_counts(dv, bins)
                data_count = int(np.sum(data_hist)) if data_hist is not None else 0

            logging.info("Plotting %s: bins=%d, bkg_rows=%d, total_mc=%.3f, data_count=%d",
                         var, len(bins) - 1, len(bkg_rows), float(np.sum(total_mc)), data_count)

            files = self._plot_stacked_variable(
                variable=var, bins=bins,
                background_rows=bkg_rows, data_ndarray=data_hist,
                output_dir=output_dir, luminosity=luminosity, year=year,
                region="event_selection", version=version, save_root=save_root,
            )
            created.extend(files)
            logging.info("Created event-selection plot: %s -> %d files", var, len(files))
        if skipped_vars:
            logging.warning("Skipped %d/%d variables: %s", len(skipped_vars), len(all_vars), ', '.join(skipped_vars))

        # ---- data cutflow (raw counts, no normalisation) ----
        data_cutflow_arr: Optional[np.ndarray] = None
        _cf_labels_ref: Optional[List[str]] = None
        for label, entries in dat_groups.items():
            for e in entries:
                try:
                    import uproot as _up
                    p = e.get("path")
                    if p and str(p).endswith(".root"):
                        with _up.open(str(p)) as _f:
                            for _key in ("cutflow", "cutflow;1"):
                                if _key in _f:
                                    _h = _f[_key]
                                    _lbls = [str(b) for b in _h.axes[0]]
                                    _vals = _h.values().astype(float)
                                    if _cf_labels_ref is None:
                                        _cf_labels_ref = _lbls
                                    aligned = np.array([_vals[_lbls.index(l)] if l in _lbls else 0.0
                                                        for l in (_cf_labels_ref or _lbls)])
                                    data_cutflow_arr = aligned if data_cutflow_arr is None else data_cutflow_arr + aligned
                                    break
                except Exception:
                    pass

        # ---- event-selection cutflow plot ----
        if evtsel_cutflow_per_proc:
            cf_files = self._plot_cutflow(
                evtsel_cutflow_per_proc=evtsel_cutflow_per_proc,
                region_cutflow_per_proc={},
                region_name="event_selection",
                output_dir=output_dir,
                version=version,
                year=year,
                luminosity=luminosity,
                data_cutflow=data_cutflow_arr,
            )
            created.extend(cf_files)

        return created

    def _create_region_stacked_plots(
        self,
        input_folder: str,
        process_groups: Dict[str, List[str]],
        signal_groups: Dict[str, List[str]],
        data_groups: Dict[str, List[str]],
        output_dir: str,
        luminosity: float,
        year: str,
        version: str,
        cross_sections: Dict[str, float],
        variables: Optional[List[str]],
        regions: Optional[List[str]],
        save_root: bool,
    ) -> List[str]:
        logging.info("=== Creating region stacked plots ===")
        cross_sections = self._normalize_cross_sections(cross_sections)

        try:
            import hist as hist_lib
            _HAS_HIST = True
        except ImportError:
            _HAS_HIST = False
            logging.warning("'hist' library not available — histogram type detection limited")

        def _load_region_pkl(path: Path) -> Dict[str, Any]:
            with open(path, "rb") as fh:
                return pickle.load(fh)

        def _load_pkl_group(grp_patterns: List[str]) -> List[Dict[str, Any]]:
            entries = []
            for p in self._resolve_group_files(input_folder, grp_patterns):
                try:
                    entries.append(_load_region_pkl(p))
                except Exception as exc:
                    logging.warning("Skipping %s: %s", p, exc)
            return entries

        # Background PKLs: {proc_label: [{"data": pkl_dict, "xsec": float|None}]}
        bkg_groups: Dict[str, List[Dict[str, Any]]] = {}
        for proc_label, patterns in process_groups.items():
            paths = self._resolve_group_files(input_folder, patterns)
            entries = []
            for p in paths:
                _stem = p.stem.replace("_EVENTSELECTION", "")
                xsec = _find_xsec(p.stem, cross_sections)
                if xsec is None:
                    xsec = _find_xsec(_stem, cross_sections)
                if xsec is None:
                    xsec = _find_xsec(p.name, cross_sections)
                try:
                    data = _load_region_pkl(p)
                    entries.append({"data": data, "xsec": xsec})
                    regions_in_file = list(data.get("region_histograms", {}).keys())
                    logging.info("  [%s] loaded %s: xsec=%s, regions=%s",
                                 proc_label, p.name,
                                 "%.4g pb" % xsec if xsec is not None else "None",
                                 ', '.join(regions_in_file[:5]) + ('...' if len(regions_in_file) > 5 else ''))
                except Exception as exc:
                    logging.warning("[%s] skipping %s: %s", proc_label, p.name, exc)
            if entries:
                bkg_groups[proc_label] = entries
                all_regs = sorted({r for e in entries for r in e["data"].get("region_histograms", {}).keys()})
                logging.info("[%s] SUMMARY: %d files, regions=%s", proc_label, len(entries), ', '.join(all_regs))
            else:
                logging.warning("[%s] NO FILES LOADED — group will be missing!", proc_label)

        # Data PKLs: {data_label: {"pkls": [dict], "region_patterns": [str]}}
        # region_patterns from yaml data group "regions" key — if absent, matches all regions.
        raw_data_cfg: Dict[str, Any] = self.config.get("process_groups", {})
        data_loaded: Dict[str, Dict[str, Any]] = {}
        for label, patterns in data_groups.items():
            grp_cfg = raw_data_cfg.get(label, {})
            region_patterns: List[str] = grp_cfg.get("regions", []) if isinstance(grp_cfg, dict) else []
            pkls = _load_pkl_group(patterns)
            if pkls:
                data_loaded[label] = {"pkls": pkls, "region_patterns": region_patterns}
                logging.info("[Data:%s] loaded %d PKLs, region_patterns=%s",
                             label, len(pkls), region_patterns if region_patterns else "(all)")
            else:
                logging.warning("[Data:%s] NO DATA FILES LOADED", label)

        def _data_hist_for_region(region: str, var: str) -> Optional[np.ndarray]:
            """Sum histogram values from whichever data group(s) apply to this region."""
            total: Optional[np.ndarray] = None
            for label, info in data_loaded.items():
                rp = info["region_patterns"]
                # If no region_patterns → applies to all regions
                if rp and not any(pat in region for pat in rp):
                    continue
                for pkl in info["pkls"]:
                    rh = pkl.get("region_histograms", {}).get(region, {})
                    h = rh.get(var)
                    # Try common PKL→plotting name aliases
                    _aliases = {
                        'Recoil': 'recoil', 'MET_pt': 'met', 'MET_phi': 'met_phi',
                        'njets': 'n_jets', 'Jet1Pt': 'jet_pt', 'Jet1Eta': 'jet_eta',
                        'Jet1Phi': 'jet_phi', 'Jet2Pt': 'jet2_pt', 'Jet2Eta': 'jet2_eta',
                        'Jet2Phi': 'jet2_phi', 'dPhi_jetMET': 'min_dphi',
                        'dPhiJet12': 'dphi_jet12', 'dEtaJet12': 'deta_jet12',
                        'M_Jet1Jet2': 'm_jet1jet2', 'Jet1BTagScore': 'btag_deepjet',
                        'Jet2BTagScore': 'jet2_deepcsv',
                    }
                    if h is None:
                        _alias = _aliases.get(var)
                        if _alias:
                            h = rh.get(_alias)
                    if h is None:
                        continue
                    if _HAS_HIST and isinstance(h, hist_lib.Hist):
                        hv = h.values().astype(float)
                    elif isinstance(h, dict):
                        hv = np.array(h.get("values", []), dtype=float)
                    else:
                        continue
                    total = hv if total is None else total + hv
            return total

        all_regions: List[str] = regions or sorted({
            r
            for entries in bkg_groups.values()
            for e in entries
            for r in e["data"].get("region_histograms", {}).keys()
        })

        def _h_to_numpy(h: Any) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
            """Convert hist.Hist or dict → (edges, values, sumw2). Returns (None,None,None) on failure."""
            if _HAS_HIST and isinstance(h, hist_lib.Hist):
                edges = np.array(h.axes[0].edges)
                hv = h.values().astype(float)
                hvar = h.variances()
                hs = hvar.astype(float) if hvar is not None else np.zeros_like(hv)
                return edges, hv, hs
            if isinstance(h, dict):
                edges = np.array(h.get("bins", []))
                hv = np.array(h.get("values", []), dtype=float)
                hs = np.array(h.get("errors", np.zeros_like(hv)), dtype=float) ** 2
                return (edges if edges.size > 1 else None), hv, hs
            return None, None, None

        created: List[str] = []
        skipped_region_var = []
        plot_tasks: List[tuple] = []
        logging.info("Regions to process: %d (%s)", len(all_regions), ', '.join(all_regions[:5]) + ('...' if len(all_regions) > 5 else ''))
        for region in all_regions:
            all_vars_set: set = set()
            for entries in bkg_groups.values():
                for e in entries:
                    all_vars_set.update(e["data"].get("region_histograms", {}).get(region, {}).keys())
            candidate_vars = variables or sorted(all_vars_set)
            all_vars_for_region = self._get_allowed_variables_for_region(region, candidate_vars)
            logging.info("Region '%s': candidate_vars=%d, after filtering=%d",
                         region, len(candidate_vars), len(all_vars_for_region))

            for var in all_vars_for_region:
                bins_ref: Optional[np.ndarray] = self._build_bins_from_config(var)
                bkg_rows: List[Tuple[str, np.ndarray, np.ndarray]] = []
                procs_with_data = []

                for proc_label, entries in bkg_groups.items():
                    group_hv: Optional[np.ndarray] = None
                    group_hs: Optional[np.ndarray] = None
                    file_hits = 0
                    for e in entries:
                        rh = e["data"].get("region_histograms", {}).get(region, {})
                        h = rh.get(var)
                    # Try common PKL→plotting name aliases
                    _aliases = {
                        'Recoil': 'recoil', 'MET_pt': 'met', 'MET_phi': 'met_phi',
                        'njets': 'n_jets', 'Jet1Pt': 'jet_pt', 'Jet1Eta': 'jet_eta',
                        'Jet1Phi': 'jet_phi', 'Jet2Pt': 'jet2_pt', 'Jet2Eta': 'jet2_eta',
                        'Jet2Phi': 'jet2_phi', 'dPhi_jetMET': 'min_dphi',
                        'dPhiJet12': 'dphi_jet12', 'dEtaJet12': 'deta_jet12',
                        'M_Jet1Jet2': 'm_jet1jet2', 'Jet1BTagScore': 'btag_deepjet',
                        'Jet2BTagScore': 'jet2_deepcsv',
                    }
                    if h is None:
                        _alias = _aliases.get(var)
                        if _alias:
                            h = rh.get(_alias)
                        if h is None:
                            continue
                        wte = int(e["data"].get("metadata", {}).get("weighted_total_events", 0)
                                   or e["data"].get("weighted_total_events", 0) or 0)
                        edges, hv, hs = _h_to_numpy(h)
                        if hv is None or hv.size == 0:
                            logging.debug("  %s/%s/%s: histogram empty after conversion", region, proc_label, var)
                            continue
                        if edges is not None:
                            bins_ref = edges  # Use PKL pre-computed binning
                        scale = ((luminosity * e["xsec"] * 1000.0) / wte
                                 if e["xsec"] is not None and wte > 0
                                 else (luminosity / wte if wte > 0 else 1.0))
                        group_hv = hv * scale if group_hv is None else group_hv + hv * scale
                        group_hs = hs * scale**2 if group_hs is None else group_hs + hs * scale**2
                        file_hits += 1

                    if group_hv is not None and group_hv.size > 0:
                        bkg_rows.append((proc_label, group_hv, group_hs))
                        procs_with_data.append(proc_label)
                        logging.debug("  %s/%s: %d files, integral=%.4f", region, proc_label, file_hits, float(np.sum(group_hv)))

                total_mc = sum(h for _, h, _ in bkg_rows)
                if not bkg_rows or bins_ref is None:
                    logging.warning("Skipping %s/%s: no bkg_rows=%s, bins_ref=%s",
                                    region, var, not bkg_rows, bins_ref is None)
                    skipped_region_var.append((region, var, "no_data"))
                    continue
                if np.allclose(total_mc, 0.0):
                    logging.warning("Skipping %s/%s: total MC yield is zero (rows=%d, procs=%s)",
                                    region, var, len(bkg_rows), ', '.join(procs_with_data))
                    skipped_region_var.append((region, var, "zero_yield"))
                    continue

                data_hist = _data_hist_for_region(region, var)
                data_sum = float(np.sum(data_hist)) if data_hist is not None else 0
                logging.info("Queuing %s/%s: bins=%d, bkg_rows=%d, total_mc=%.3f, data_sum=%.1f, procs=%s",
                             region, var, len(bins_ref) - 1, len(bkg_rows),
                             float(np.sum(total_mc)), data_sum, ', '.join(procs_with_data))

                plot_kwargs = dict(
                    variable=var, bins=bins_ref,
                    background_rows=bkg_rows, data_ndarray=data_hist,
                    output_dir=output_dir, luminosity=luminosity, year=year,
                    region=region, version=version, save_root=save_root,
                )
                plot_tasks.append((self.config, plot_kwargs, region, var))

        # Each (region, var) draw+save is independent — parallelize across
        # CPU cores via multiprocessing (matplotlib draw + numpy hist math is
        # CPU-bound, so threading would just serialize on the GIL).
        num_workers = int(os.environ.get("PLOT_NUM_WORKERS", max(1, (os.cpu_count() or 1))))
        num_workers = max(1, min(num_workers, len(plot_tasks)))

        if num_workers > 1 and len(plot_tasks) > 1:
            import multiprocessing as mp

            logging.info(f"Plotting {len(plot_tasks)} region/variable combos with {num_workers} worker processes")
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=num_workers) as pool:
                task_results = pool.map(_plot_stacked_variable_worker, plot_tasks)
        else:
            task_results = [_plot_stacked_variable_worker(t) for t in plot_tasks]

        for region, var, files in task_results:
            created.extend(files)
            logging.info("Created region plot: %s / %s -> %d files", region, var, len(files))

        if skipped_region_var:
            logging.warning("Skipped %d region/var combos: %s",
                            len(skipped_region_var),
                            ', '.join(f"{r}/{v}({why})" for r, v, why in skipped_region_var[:10])
                            + ('...' if len(skipped_region_var) > 10 else ''))
        logging.info("Region stacked plots complete: %d plots created", len(created))
        return created

    def _create_region_from_events_plots(
        self,
        input_folder: str,
        process_groups: Dict[str, List[str]],
        signal_groups: Optional[Dict[str, List[str]]],
        data_groups: Dict[str, List[str]],
        output_dir: str,
        luminosity: float,
        year: str,
        version: str,
        cross_sections: Dict[str, float],
        variables: Optional[List[str]],
        regions: Optional[List[str]],
        regions_config: Optional[str],
        weight_systematic: Optional[str],
        show_data: bool = False,
        signal_scale: float = 1.0,
        make_syst_plots: bool = False,
        apply_dnn: bool = False,
        dnn_model: Optional[str] = None,
        dnn_config: Optional[str] = None,
        dnn_mass_scan: Optional[str] = None,
    ) -> List[str]:
        """Load event-selected ROOT/PKL files, apply region cuts in-memory, produce stacked region plots.

        For each process group:
          1. Load Events TTree + weighted_total_events from each matched file.
          2. Apply region masks via RegionManager.
          3. Fill per-variable histograms with full_event_weight (nominal) or weight_systematic.
          4. Scale by lumi * xsec * 1000 / wte and stack across groups.
        """
        cross_sections = self._normalize_cross_sections(cross_sections)

        import uproot as _uproot
        from .regions import RegionManager

        if not regions_config:
            raise ValueError(
                "--regions-config is required for region-from-events mode"
            )

        # Build RegionManager from yaml
        region_manager = RegionManager(regions_config)
        all_region_names: List[str] = list(region_manager.regions.keys())
        target_regions: List[str] = regions or all_region_names

        weight_branch = weight_systematic or "full_event_weight"

        # ---- DNN scoring setup (shared model instance across all files) ----
        _dnn_inference = None
        _dnn_mass_scan_resolved: Optional[List[Tuple[float, float]]] = None
        if apply_dnn and dnn_model:
            from .dnn_inference import DNNInference, _mass_branch_name, _resolve_mass_scan

            _dnn_inference = DNNInference(dnn_model, config_path=dnn_config)
            _dnn_mass_scan_resolved = _resolve_mass_scan(dnn_mass_scan, _dnn_inference)
            logging.info("DNN scoring enabled for region-from-events plots: model=%s, mass_scan=%s",
                        dnn_model, _dnn_mass_scan_resolved)

        def _score_tree_with_dnn(tree) -> Dict[str, np.ndarray]:
            """Score an uproot Events tree with the shared DNN model; returns {branch: scores}.

            Uses build_feature_frame_from_tree (via DNNInference.extract_features) so
            raw-branch aliasing (MET -> MET_pt, METPhi -> MET_phi, ...) and derived
            features (e.g. rJet1PtMET = Jet1Pt/MET) are resolved identically to
            training and the apply-dnn CLI command, instead of a flat name lookup.

            None mass_scan (default) -> single {"ml_score": ...} at the benchmark
            masspoint (masses=None lets DNNInference default to mass_grid[0] for
            parametric models; no-op for non-parametric models). Otherwise one
            {"ml_score_mh3_<a>_mh4_<b>": ...} per scanned point.
            """
            X = _dnn_inference.extract_features(tree)
            n = X.shape[0]
            if _dnn_mass_scan_resolved is None:
                scores = _dnn_inference.predict(X, None).ravel()
                return {"ml_score": scores.astype("f4")}
            out: Dict[str, np.ndarray] = {}
            for mh3, mh4 in _dnn_mass_scan_resolved:
                masses = np.tile(np.asarray([mh3, mh4], dtype="f8"), (n, 1))
                scores = _dnn_inference.predict(X, masses).ravel()
                out[_mass_branch_name("ml_score", mh3, mh4)] = scores.astype("f4")
            return out

        # ---- helpers ----

        def _load_cutflow_root(path: Path) -> Optional[Dict[str, float]]:
            """Load event-selection cutflow from EVENTSELECTION.root cutflow TH1."""
            try:
                with _uproot.open(str(path)) as f:
                    for key in ("cutflow", "cutflow;1"):
                        if key in f:
                            h = f[key]
                            labels = [str(b) for b in h.axes[0]]
                            vals   = h.values()
                            return dict(zip(labels, vals))
            except Exception:
                pass
            return None

        def _load_events_root(path: Path, variables: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
            """Return {"branches": {name: np.ndarray}, "wte": float} or None on failure.

            All branches are read individually; any branch that throws is skipped.
            After loading, branches with lengths different from the majority are
            dropped to ensure consistent lengths for ak.Array construction.
            """
            try:
                with _uproot.open(str(path)) as f:
                    # weighted_total_events stored as 1-bin TH1
                    wte = 0.0
                    for key in ("weighted_total_events", "weighted_total_events;1"):
                        if key in f:
                            try:
                                wte = float(f[key].values()[0])
                                break
                            except Exception:
                                pass
                    if "Events" not in f:
                        logging.debug("No Events tree in %s", path.name)
                        return None
                    tree = f["Events"]
                    branches: dict = {}
                    skipped_branches = []
                    _always_load_prefixes = (
                        "weight_", "full_event_weight", "pass_met_trigger",
                        "pass_ele_trigger", "GenModel_", "event", "run",
                        # object counts for region cuts
                        "n_bjets", "njets", "n_muons", "n_electrons", "n_taus",
                        "n_z_muons", "n_z_electrons",
                        # reco primary vertices (source for PV_npvsGood[_noPU] data/MC plots)
                        "PV_npvsGood", "PV_npvs",
                        # MET branches for MT, Mll, Recoil cuts
                        "MET_pt", "MET_phi", "PuppiMET_pt", "PuppiMET_phi",
                        "Recoil", "Recoil_JES", "Recoil_JER",
                        # lepton kinematics for MT / Z mass cuts
                        "muon_lep1_pt", "muon_lep1_phi", "muon_lep1_eta",
                        "muon_lep2_pt", "muon_lep2_phi", "muon_lep2_eta",
                        "electron_lep1_pt", "electron_lep1_phi", "electron_lep1_eta",
                        "electron_lep2_pt", "electron_lep2_phi", "electron_lep2_eta",
                        "mll", "Mll",
                    )
                    for bname in tree.keys():
                        _bname = str(bname)
                        _always = any(_bname.startswith(p) or _bname == p
                                      for p in _always_load_prefixes)
                        if variables is not None and not _always and not _should_load_key(_bname, variables):
                            continue
                        try:
                            branches[bname] = tree[bname].array(library="np")
                        except Exception as _berr:
                            skipped_branches.append((bname, str(_berr)))
                    # Enforce consistent length: drop branches whose length != the mode length
                    if branches:
                        _lens = [len(v) for v in branches.values()]
                        _mode_len = Counter(_lens).most_common(1)[0][0]
                        _bad = [k for k, v in branches.items() if len(v) != _mode_len]
                        for k in _bad:
                            skipped_branches.append((k, f"length {len(branches[k])} != mode {_mode_len}"))
                            del branches[k]
                        if _bad:
                            logging.warning(
                                "Dropped %d branches with non-modal length in %s (mode=%d): %s",
                                len(_bad), path.name, _mode_len,
                                ', '.join(_bad[:10]) + ('...' if len(_bad) > 10 else '')
                            )
                    if skipped_branches:
                        logging.debug(
                            "Skipped %d branches in %s: %s",
                            len(skipped_branches), path.name,
                            ', '.join(f"{n}({r[:40]})" for n, r in skipped_branches[:5])
                            + ('...' if len(skipped_branches) > 5 else '')
                        )
                    if _dnn_inference is not None and branches:
                        try:
                            branches.update(_score_tree_with_dnn(tree))
                        except Exception as _dnn_err:
                            logging.warning("DNN scoring failed for %s: %s", path.name, _dnn_err)
                    return {"branches": branches, "wte": wte}
            except Exception as exc:
                logging.warning("Could not load %s: %s", path.name, exc)
                return None

        def _load_events_pkl(path: Path, variables: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
            try:
                with open(path, "rb") as fh:
                    d = pickle.load(fh)
                branches = {}
                for k, v in d.get("branches", {}).items():
                    if variables is not None and not _should_load_key(k, variables):
                        continue
                    if isinstance(v, (np.ndarray, list)):
                        branches[k] = np.asarray(v)
                wte = float(d.get("weighted_total_events", 0.0))
                if _dnn_inference is not None and branches:
                    logging.warning(
                        "DNN scoring requested but %s is a PKL file (no Events tree available "
                        "for alias/derived-feature resolution) — ml_score not added", path.name
                    )
                return {"branches": branches, "wte": wte}
            except Exception as exc:
                logging.warning("Could not load %s: %s", path.name, exc)
                return None

        def _load_file(path: Path, variables: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
            if path.suffix == ".root":
                return _load_events_root(path, variables)
            return _load_events_pkl(path, variables)

        # ---- load per-group ----
        # {proc_label: [{"branches": dict, "wte": float, "xsec": float|None}]}
        bkg_entries: Dict[str, List[Dict[str, Any]]] = {}
        # Per-process event-selection cutflow: {proc_label: {cut_label: weighted_yield}}
        evtsel_cutflow_per_proc: Dict[str, Dict[str, float]] = {}
        logging.info("=== Loading background files for region-from-events ===")
        for proc_label, patterns in process_groups.items():
            paths = self._resolve_group_files(input_folder, patterns)
            entries = []
            proc_cf: Dict[str, float] = {}
            loaded_count = 0
            for p in paths:
                loaded = _load_file(p, variables)
                if loaded is None:
                    logging.warning("[%s] failed to load %s", proc_label, p.name)
                    continue
                _stem = p.stem.replace("_EVENTSELECTION", "")
                xsec = _find_xsec(p.stem, cross_sections)
                if xsec is None:
                    xsec = _find_xsec(_stem, cross_sections)
                if xsec is None:
                    xsec = _find_xsec(p.name, cross_sections)
                wte  = loaded["wte"]
                scale = ((luminosity * xsec * 1000.0) / wte
                         if xsec is not None and wte > 0
                         else (luminosity / wte if wte > 0 else 1.0))
                cf = _load_cutflow_root(p)
                if cf:
                    for cut, val in cf.items():
                        proc_cf[cut] = proc_cf.get(cut, 0.0) + val * scale
                entries.append({
                    "branches": loaded["branches"],
                    "wte": wte,
                    "xsec": xsec,
                })
                loaded_count += 1
                logging.info("[%s] loaded %s: wte=%.1f, xsec=%s, branches=%d",
                             proc_label, p.name, wte,
                             "%.4g pb" % xsec if xsec is not None else "None",
                             len(loaded["branches"]))
            if entries:
                bkg_entries[proc_label] = entries
                logging.info("[%s] SUMMARY: %d files loaded", proc_label, len(entries))
            else:
                logging.warning("[%s] NO FILES LOADED", proc_label)
            if proc_cf:
                evtsel_cutflow_per_proc[proc_label] = proc_cf

        logging.info("evtsel_cutflow_per_proc: %d processes, %s",
                     len(evtsel_cutflow_per_proc),
                     {k: len(v) for k, v in evtsel_cutflow_per_proc.items()})

        # Data groups (no xsec)
        logging.info("=== Loading data files for region-from-events ===")
        raw_data_cfg = self.config.get("process_groups", {})
        data_entries: Dict[str, Dict[str, Any]] = {}
        for label, patterns in data_groups.items():
            grp_cfg = raw_data_cfg.get(label, {})
            region_patterns: List[str] = grp_cfg.get("regions", []) if isinstance(grp_cfg, dict) else []
            paths = self._resolve_group_files(input_folder, patterns)
            loaded_list = [e for p in paths for e in [_load_file(p, variables)] if e is not None]
            if loaded_list:
                data_entries[label] = {"entries": loaded_list, "region_patterns": region_patterns}
                logging.info("[Data:%s] loaded %d files, region_patterns=%s",
                             label, len(loaded_list), region_patterns if region_patterns else "(all)")
            else:
                logging.warning("[Data:%s] NO FILES LOADED", label)

        if not bkg_entries:
            logging.warning("region-from-events: no background files loaded — nothing to plot")
            return []

        # ---- load signal files (GenModel multi-masspoint) ----
        # Signal files have GenModel_* branches. Detect them from signal_groups patterns.
        # Each GenModel_* branch is one masspoint; events with flag==1 belong to that masspoint.
        # {masspoint_label: {"hv": np.ndarray per var (built lazily), "wte": float, "xsec": float|None}}
        # We store raw events+mask per masspoint for per-variable histogramming later.
        # sig_file_entries: list of {"branches": dict, "wte": float, "xsec": float|None, "genmodel_cols": [str]}
        sig_file_entries: List[Dict[str, Any]] = []
        if signal_groups:
            logging.info("=== Loading signal files for region-from-events ===")
            for sig_label, sig_patterns in signal_groups.items():
                paths = self._resolve_group_files(input_folder, sig_patterns)
                for p in paths:
                    # Load all branches for signal (need GenModel_* regardless of variable whitelist)
                    loaded = _load_file(p, variables=None)
                    if loaded is None:
                        logging.warning("[Signal:%s] failed to load %s", sig_label, p.name)
                        continue
                    gm_cols = sorted(k for k in loaded["branches"] if k.startswith("GenModel_"))
                    if not gm_cols:
                        logging.info("[Signal:%s] %s has no GenModel_ branches — treated as single masspoint", sig_label, p.name)
                        gm_cols = []
                    _stem = p.stem.replace("_EVENTSELECTION", "")
                    xsec = _find_xsec(p.stem, cross_sections) or _find_xsec(_stem, cross_sections) or _find_xsec(p.name, cross_sections)
                    sig_file_entries.append({
                        "branches": loaded["branches"],
                        "wte": loaded["wte"],
                        "xsec": xsec,
                        "genmodel_cols": gm_cols,
                        "file_label": sig_label,
                    })
                    if gm_cols:
                        logging.info("[Signal:%s] loaded %s: wte=%.1f, masspoints=%d "
                                     "(per-masspoint xsec resolved individually below)",
                                     sig_label, p.name, loaded["wte"], len(gm_cols))
                    else:
                        logging.info("[Signal:%s] loaded %s: wte=%.1f, xsec=%s "
                                     "(single masspoint, file-level xsec)",
                                     sig_label, p.name, loaded["wte"],
                                     "%.4g pb" % xsec if xsec is not None else "None")
        logging.info("Signal files loaded: %d", len(sig_file_entries))

        # ---- per-region processing ----
        created: List[str] = []
        skipped_region_var = []
        events_plot_tasks: List[tuple] = []
        logging.info("Regions to process: %d (%s)", len(target_regions),
                     ', '.join(target_regions[:5]) + ('...' if len(target_regions) > 5 else ''))

        for region_name in target_regions:
            region_obj = region_manager.regions.get(region_name)
            if region_obj is None:
                logging.warning("Region %s not found in %s", region_name, regions_config)
                continue

            # Determine variable list from first available group
            if variables:
                candidate_vars = variables
            else:
                all_branches: set = set()
                for entries in bkg_entries.values():
                    for e in entries:
                        all_branches.update(e["branches"].keys())
                _weight_prefixes = ("weight_", "full_event_weight", "genWeight")
                candidate_vars = sorted(
                    b for b in all_branches
                    if not any(b.startswith(p) for p in _weight_prefixes)
                )
            # Include derived vars from the whitelist even if not stored as branches.
            # _get_allowed_variables_for_region intersects with candidate_vars, so
            # we union in the full whitelist first so derived vars survive intersection.
            _whitelist = (self.common_variables
                          + list(self.region_variables.get(region_name) or
                                 next((v for k, v in self.region_variables.items() if k in region_name), []) or []))
            if _dnn_mass_scan_resolved is not None:
                # Mass-scan mode: "ml_score" was never written as a branch (only
                # ml_score_mh3_<a>_mh4_<b> variants were) and regions.py's generic
                # fallback would otherwise silently return an all-zeros array for
                # it (treating it as a legitimate-but-missing derived variable) —
                # drop the literal whitelist entry so only the real scan branches
                # (already in candidate_vars) get expanded/plotted.
                _whitelist = [v for v in _whitelist if v != "ml_score"]
            candidate_vars = list(dict.fromkeys(list(candidate_vars) + _whitelist))
            var_list = self._get_allowed_variables_for_region(region_name, candidate_vars)
            logging.info("Region '%s': candidate_vars=%d, after filtering=%d",
                         region_name, len(candidate_vars), len(var_list))

            # Pre-compute region mask + events_ak once per entry (reused across all variables)
            import awkward as _ak

            def _build_entry_cache(entries_iter):
                cache: Dict[int, Dict] = {}
                for proc_label, entries in entries_iter:
                    for e in entries:
                        eid = id(e)
                        if eid in cache:
                            continue
                        br = e["branches"]
                        try:
                            ak_dict = {}
                            for k, v in br.items():
                                if not isinstance(v, np.ndarray) or v.ndim != 1:
                                    continue
                                if v.dtype == object:
                                    try:
                                        ak_dict[k] = _ak.Array(list(v))
                                    except Exception:
                                        pass
                                else:
                                    ak_dict[k] = v
                            ev_ak = _ak.Array(ak_dict)
                            msk = region_obj.apply_cuts(ev_ak, objects={})
                            cache[eid] = {"mask_np": np.asarray(msk, dtype=bool),
                                          "events_ak": ev_ak, "skip": False}
                        except Exception as _exc:
                            logging.warning("Region mask failed for %s / %s: %s",
                                            region_name, proc_label, _exc)
                            cache[eid] = {"skip": True}
                return cache

            _entry_cache: Dict[int, Dict] = _build_entry_cache(bkg_entries.items())
            # Also cache data entries
            _data_entry_cache: Dict[int, Dict] = _build_entry_cache(
                (label, info["entries"]) for label, info in data_entries.items()
            )

            for var in var_list:
                bins_ref: Optional[np.ndarray] = self._build_bins_from_config(var)
                bkg_rows: List[Tuple[str, np.ndarray, np.ndarray]] = []

                for proc_label, entries in bkg_entries.items():
                    group_hv: Optional[np.ndarray] = None
                    group_hs: Optional[np.ndarray] = None

                    for e in entries:
                        br = e["branches"]
                        wte = e["wte"]
                        xsec = e["xsec"]

                        _cached = _entry_cache.get(id(e), {"skip": True})
                        if _cached.get("skip"):
                            continue
                        mask_np = _cached["mask_np"]
                        events_ak = _cached["events_ak"]
                        vals_raw = br.get(_value_branch(var))
                        if vals_raw is None:
                            # Try computing derived variable (mt, z_mass, z_pt, etc.)
                            # that isn't stored as a flat branch but is derivable from
                            # the event fields via Region._get_variable_value.
                            _var_map = {
                                "mt":     "MT",
                                "z_mass": "Mll",
                                "z_pt":   "Zpt",
                                "mll":    "Mll",
                            }
                            _canon = _var_map.get(_value_branch(var), _value_branch(var))
                            try:
                                _derived = region_obj._get_variable_value(events_ak, objects={}, var=_canon)
                                if _derived is not None:
                                    if isinstance(_derived, np.ndarray):
                                        _arr = _derived.astype(float)
                                    else:
                                        _arr = np.asarray(_ak.to_numpy(
                                            _ak.fill_none(_ak.Array(_derived), _SENTINEL)
                                            if not isinstance(_derived, _ak.Array)
                                            else _ak.fill_none(_derived, _SENTINEL)
                                        ), dtype=float)
                                    if _arr.ndim == 1 and len(_arr) == len(mask_np):
                                        vals_raw = _arr
                            except Exception:
                                pass
                        if vals_raw is None:
                            continue
                        if isinstance(vals_raw, np.ndarray) and vals_raw.dtype == object:
                            continue  # jagged branch — not plottable as scalar histogram
                        try:
                            vals_masked = np.asarray(vals_raw[mask_np], dtype=float)
                        except (ValueError, TypeError):
                            continue
                        # sentinel filter: keep same mask for weights
                        sentinel_mask = _apply_variable_plot_filter(var, vals_masked, return_mask=True)
                        vals = vals_masked[sentinel_mask]
                        if vals.size == 0:
                            continue

                        w_arr = br.get(_weight_branch_for(var, weight_branch))
                        if w_arr is not None:
                            w = np.asarray(w_arr, dtype=float)[mask_np][sentinel_mask]
                        else:
                            w = np.ones(vals.size, dtype=float)

                        scale = ((luminosity * xsec * 1000.0) / wte
                                 if xsec is not None and wte > 0
                                 else (luminosity / wte if wte > 0 else 1.0))

                        if bins_ref is None:
                            all_vals_for_bins = []
                            for ee in bkg_entries.values():
                                for e2 in ee:
                                    v2 = e2["branches"].get(_value_branch(var), np.array([]))
                                    if isinstance(v2, np.ndarray) and v2.dtype != object:
                                        all_vals_for_bins.append(_apply_variable_plot_filter(_bins_key(var), v2))
                            bins_ref = _make_bins(all_vals_for_bins, self._build_bins_from_config,
                                                  _bins_key(var), self._n_bins_default)
                            if bins_ref is None or len(bins_ref) < 2:
                                break

                        vals_clipped = _clip_overflow(vals, bins_ref)
                        hv, _ = np.histogram(vals_clipped, bins=bins_ref, weights=w * scale)
                        hs, _ = np.histogram(vals_clipped, bins=bins_ref, weights=(w * scale) ** 2)

                        group_hv = hv if group_hv is None else group_hv + hv
                        group_hs = hs if group_hs is None else group_hs + hs

                    if group_hv is not None and group_hv.size > 0:
                        bkg_rows.append((proc_label, group_hv, group_hs))
                        logging.debug("  %s/%s/%s: integral=%.4f", region_name, proc_label, var, float(np.sum(group_hv)))

                total_mc = sum(h for _, h, _ in bkg_rows)
                if bins_ref is None:
                    logging.warning("Skipping %s/%s: bins_ref is None (no valid data for binning)", region_name, var)
                    skipped_region_var.append((region_name, var, "no_bins"))
                    continue
                if np.allclose(total_mc, 0.0):
                    logging.warning("Skipping %s/%s: total MC yield is zero (rows=%d)", region_name, var, len(bkg_rows))
                    skipped_region_var.append((region_name, var, "zero_yield"))
                    continue

                # Ensure every process group has a row (zero if no events pass) — always plot all groups
                loaded_labels = {label for label, _, _ in bkg_rows}
                for proc_label in bkg_entries:
                    if proc_label not in loaded_labels:
                        bkg_rows.append((proc_label,
                                         np.zeros(len(bins_ref) - 1),
                                         np.zeros(len(bins_ref) - 1)))

                # Data: CRs show real data; SR blinded by default (shows bkg-sum as pseudo-data)
                data_hist: Optional[np.ndarray] = None
                _is_sr = region_name.endswith(":SR")

                if _is_sr and not show_data:
                    # Blinded SR: use total bkg-sum as pseudo-data points
                    bkg_total = sum(h for _, h, _ in bkg_rows)
                    data_hist = bkg_total
                    logging.debug("  %s/%s: blinded SR, using bkg-sum as pseudo-data (sum=%.3f)", region_name, var, float(np.sum(bkg_total)))
                else:
                    for label, info in data_entries.items():
                        rp = info["region_patterns"]
                        if rp and not any(pat in region_name for pat in rp):
                            logging.debug("  %s/%s: data group %s skipped (region_patterns mismatch)", region_name, var, label)
                            continue
                        for e in info["entries"]:
                            br = e["branches"]
                            _dcached = _data_entry_cache.get(id(e), {"skip": True})
                            if _dcached.get("skip"):
                                continue
                            mask_d_np = _dcached["mask_np"]
                            _devents_ak = _dcached["events_ak"]
                            _dvals_raw = br.get(_value_branch(var))
                            if _dvals_raw is None:
                                _var_map = {"mt": "MT", "z_mass": "Mll", "z_pt": "Zpt", "mll": "Mll"}
                                _canon = _var_map.get(_value_branch(var), _value_branch(var))
                                try:
                                    _derived = region_obj._get_variable_value(_devents_ak, objects={}, var=_canon)
                                    if _derived is not None:
                                        _dvals_raw = np.asarray(_ak.to_numpy(
                                            _ak.fill_none(_derived if isinstance(_derived, _ak.Array)
                                                          else _ak.Array(_derived), _SENTINEL)
                                        ), dtype=float)
                                        if _dvals_raw.ndim != 1 or len(_dvals_raw) != len(mask_d_np):
                                            _dvals_raw = None
                                except Exception:
                                    pass
                            if _dvals_raw is None or (isinstance(_dvals_raw, np.ndarray) and _dvals_raw.dtype == object):
                                continue
                            dv = _apply_variable_plot_filter(var, np.asarray(_dvals_raw, dtype=float)[mask_d_np])
                            if dv.size == 0:
                                continue
                            dh, _ = np.histogram(_clip_overflow(dv, bins_ref), bins=bins_ref)
                            data_hist = dh if data_hist is None else data_hist + dh
                    if data_hist is not None:
                        logging.debug("  %s/%s: data sum=%.1f", region_name, var, float(np.sum(data_hist)))

                # ---- signal histograms per masspoint ----
                # SR only — signal must never be drawn in CR plots.
                sig_rows_for_plot: List[Tuple[str, np.ndarray]] = []
                if _is_sr and sig_file_entries and bins_ref is not None:
                    for _sfe in sig_file_entries:
                        _sbr  = _sfe["branches"]
                        _swte = _sfe["wte"]
                        _sxsec = _sfe["xsec"]
                        _gm_cols = _sfe["genmodel_cols"]

                        # Build ak.Array + region mask for this signal file (cache per file×region)
                        _sfeid = id(_sfe)
                        if _sfeid not in _entry_cache:
                            try:
                                _sak_dict = {}
                                for _k, _v in _sbr.items():
                                    if not isinstance(_v, np.ndarray) or _v.ndim != 1:
                                        continue
                                    if _v.dtype == object:
                                        try:
                                            _sak_dict[_k] = _ak.Array(list(_v))
                                        except Exception:
                                            pass
                                    else:
                                        _sak_dict[_k] = _v
                                _sev_ak = _ak.Array(_sak_dict)
                                _smsk = region_obj.apply_cuts(_sev_ak, objects={})
                                _entry_cache[_sfeid] = {"mask_np": np.asarray(_smsk, dtype=bool),
                                                        "events_ak": _sev_ak, "skip": False}
                            except Exception as _se:
                                logging.warning("Signal region mask failed for %s: %s", region_name, _se)
                                _entry_cache[_sfeid] = {"skip": True}

                        _scached = _entry_cache.get(_sfeid, {"skip": True})
                        if _scached.get("skip"):
                            continue
                        _smask_np = _scached["mask_np"]

                        _svals_raw = _sbr.get(var)
                        if _svals_raw is None or not isinstance(_svals_raw, np.ndarray) or _svals_raw.dtype == object:
                            continue

                        _sscale_base = ((luminosity * _sxsec * 1000.0) / _swte
                                        if _sxsec is not None and _swte > 0
                                        else (luminosity / _swte if _swte > 0 else 1.0))

                        if _gm_cols:
                            # One histogram per GenModel masspoint
                            for _gmc in _gm_cols:
                                _gm_arr = _sbr.get(_gmc)
                                if _gm_arr is None:
                                    continue
                                # mask: passed region AND this masspoint flag == 1
                                _mp_mask = _smask_np & (_gm_arr.astype(bool))
                                _svals = _apply_variable_plot_filter(
                                    var, np.asarray(_svals_raw, dtype=float)[_mp_mask])
                                if _svals.size == 0:
                                    continue
                                # xsec lookup by masspoint label (strip GenModel_ prefix)
                                _mp_label = _gmc[len("GenModel_"):]
                                _mp_xsec = _find_xsec(_mp_label, cross_sections)
                                _mp_scale = ((luminosity * _mp_xsec * 1000.0) / _swte
                                             if _mp_xsec is not None and _swte > 0
                                             else _sscale_base)
                                _sw = np.ones(_svals.size, dtype=float)
                                _sw_arr = _sbr.get(weight_branch)
                                if _sw_arr is not None:
                                    _sw = np.asarray(_sw_arr, dtype=float)[_mp_mask][
                                        _apply_variable_plot_filter(var, np.asarray(_svals_raw, dtype=float)[_mp_mask], return_mask=True)
                                    ]
                                _shv, _ = np.histogram(_clip_overflow(_svals, bins_ref),
                                                       bins=bins_ref, weights=_sw * _mp_scale)
                                # Format label with LaTeX: MH3→m_A, MH4→m_a, Mchi→m_χ
                                _tex_map = {"MH3": r"$m_A$", "MH4": r"$m_a$", "Mchi": r"$m_\chi$"}
                                _parts = _mp_label.split("_")
                                _pairs = []
                                _i = 0
                                while _i < len(_parts) - 1:
                                    _key = _parts[_i]
                                    _val = _parts[_i + 1]
                                    if _key in _tex_map:
                                        _pairs.append(f"{_tex_map[_key]}={_val}")
                                        _i += 2
                                    else:
                                        # multi-word key e.g. "MH3" split across underscores — skip
                                        _i += 1
                                _pretty = " ".join(_pairs) if _pairs else _mp_label
                                _scale_prefix = f"×{signal_scale:g} " if signal_scale != 1.0 else ""
                                sig_rows_for_plot.append((f"{_scale_prefix}{_pretty}", _shv * signal_scale))
                        else:
                            # Single-masspoint signal file
                            _svals = _apply_variable_plot_filter(
                                var, np.asarray(_svals_raw, dtype=float)[_smask_np])
                            if _svals.size > 0:
                                _shv, _ = np.histogram(_clip_overflow(_svals, bins_ref),
                                                       bins=bins_ref, weights=np.ones(_svals.size) * _sscale_base)
                                _scale_suffix = f" ×{signal_scale:g}" if signal_scale != 1.0 else ""
                                sig_rows_for_plot.append((f"{_sfe['file_label']}{_scale_suffix}", _shv * signal_scale))

                syst_label = f" [{weight_systematic}]" if weight_systematic else ""
                logging.info("Plotting %s/%s%s: bins=%d, bkg_rows=%d, total_mc=%.3f, data=%s, signal_masspoints=%d",
                             region_name, var, syst_label, len(bins_ref) - 1, len(bkg_rows),
                             float(np.sum(total_mc)),
                             "blinded" if (_is_sr and not show_data) else (f"sum={float(np.sum(data_hist)):.1f}" if data_hist is not None else "none"),
                             len(sig_rows_for_plot))

                plot_kwargs = dict(
                    variable=var, bins=bins_ref,
                    background_rows=bkg_rows, data_ndarray=data_hist,
                    output_dir=output_dir, luminosity=luminosity, year=year,
                    region=region_name, version=version, save_root=False,
                    signal_rows=sig_rows_for_plot if sig_rows_for_plot else None,
                )
                events_plot_tasks.append((self.config, plot_kwargs, region_name, f"{var}{syst_label}"))

        # ---- systematic ROOT files (weight systs + kinematic JES/JER) ----
        # Nominal (region, var) draw+save calls are independent of each other —
        # aggregation above needed the live _entry_cache/awkward arrays (main
        # process only), but by this point each task carries just plain
        # arrays/scalars, so dispatch the draw+save step across CPU cores.
        num_workers = int(os.environ.get("PLOT_NUM_WORKERS", max(1, (os.cpu_count() or 1))))
        num_workers = max(1, min(num_workers, len(events_plot_tasks)))

        if num_workers > 1 and len(events_plot_tasks) > 1:
            import multiprocessing as mp

            logging.info(f"Plotting {len(events_plot_tasks)} region/variable combos with {num_workers} worker processes")
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=num_workers) as pool:
                events_task_results = pool.map(_plot_stacked_variable_worker, events_plot_tasks)
        else:
            events_task_results = [_plot_stacked_variable_worker(t) for t in events_plot_tasks]

        for region_name, var_label, files in events_task_results:
            created.extend(files)
            logging.info("Created region-from-events plot: %s / %s -> %d files",
                         region_name, var_label, len(files))

        if self.systematic_variables and not weight_systematic:
            try:
                import uproot as _up_syst
                # Weight systematics: same events/mask, different weight branch
                _weight_systs = [
                    "weight_pileupUP", "weight_pileupDOWN",
                    "weight_btagUP", "weight_btagDOWN",
                    "weight_muonUP", "weight_muonDOWN",
                    "weight_electronUP", "weight_electronDOWN",
                    "weight_electronHLTUP", "weight_electronHLTDOWN",
                    "weight_metHLTUP", "weight_metHLTDOWN",
                    "weight_JECUP", "weight_JECDOWN",
                    "weight_pdfUP", "weight_pdfDOWN",
                    "weight_scaleUP", "weight_scaleDOWN",
                ]
                # Kinematic systematics: shifted recoil branches (JES/JER)
                _kin_systs = {
                    "JESUP":   "Recoil_JESUp",
                    "JESDOWN": "Recoil_JESDown",
                    "JERUP":   "Recoil_JERUp",
                    "JERDOWN": "Recoil_JERDown",
                }
                for region_name in target_regions:
                    region_obj = region_manager.regions.get(region_name)
                    if region_obj is None:
                        continue
                    region_info_s = self._parse_region_name(region_name)
                    cat_s = region_info_s["category"]
                    rdir_s = region_info_s["region_dir"]
                    root_dir = Path(output_dir) / "plots" / version / "root"
                    root_dir.mkdir(parents=True, exist_ok=True)
                    _all_region_branches: set = set()
                    for entries in bkg_entries.values():
                        for e in entries:
                            _all_region_branches.update(e["branches"].keys())
                    syst_vars = [v for v in self.systematic_variables if v in _all_region_branches]

                    # Build fresh region masks for this region (don't rely on _entry_cache
                    # which only covers the last region from the nominal variable loop)
                    _syst_cache: Dict[int, Dict] = _build_entry_cache(bkg_entries.items())

                    for svar in syst_vars:
                        # Determine bins from nominal (already computed above — recompute here)
                        all_vals_syst: List[np.ndarray] = []
                        for proc_label, entries in bkg_entries.items():
                            for e in entries:
                                cached = _syst_cache.get(id(e), {"skip": True})
                                if cached.get("skip"):
                                    continue
                                br = e["branches"]
                                raw_v = br.get(svar)
                                if raw_v is not None and isinstance(raw_v, np.ndarray) and raw_v.ndim == 1:
                                    mask_np = cached.get("mask_np")
                                    if mask_np is not None:
                                        v_f = _apply_variable_plot_filter(svar, np.asarray(raw_v, dtype=float)[mask_np])
                                        if v_f.size > 0:
                                            all_vals_syst.append(v_f)
                        bins_syst = _make_bins(all_vals_syst, self._build_bins_from_config, svar, self._n_bins_default)
                        if bins_syst is None or len(bins_syst) < 2:
                            continue

                        # --- Weight systematics ---
                        for wsyst in _weight_systs:
                            syst_rows: List = []
                            any_branch = False
                            for proc_label, entries in bkg_entries.items():
                                group_hv = None
                                for e in entries:
                                    cached = _syst_cache.get(id(e), {"skip": True})
                                    if cached.get("skip"):
                                        continue
                                    br = e["branches"]
                                    w_arr = br.get(wsyst)  # varied weight
                                    if w_arr is None:
                                        continue
                                    any_branch = True
                                    scale = ((luminosity * e["xsec"] * 1000.0) / e["wte"]
                                             if e["xsec"] is not None and e["wte"] > 0 else 1.0)
                                    mask_np = cached.get("mask_np")
                                    if mask_np is None:
                                        continue
                                    raw_v = br.get(svar)
                                    if raw_v is None or not isinstance(raw_v, np.ndarray) or raw_v.ndim != 1:
                                        continue
                                    vals_masked = np.asarray(raw_v, dtype=float)[mask_np]
                                    sent_mask = _apply_variable_plot_filter(svar, vals_masked, return_mask=True)
                                    vals = vals_masked[sent_mask]
                                    w = np.asarray(w_arr, dtype=float)[mask_np][sent_mask]
                                    hv, _ = np.histogram(_clip_overflow(vals, bins_syst), bins=bins_syst, weights=w * scale)
                                    group_hv = hv if group_hv is None else group_hv + hv
                                if group_hv is not None:
                                    syst_rows.append((proc_label, group_hv))
                            if not any_branch or not syst_rows:
                                continue
                            total_bkg = sum(h for _, h in syst_rows)
                            root_stem = f"hist_{cat_s}_{rdir_s}_{svar}_{wsyst}_log"
                            rpath = root_dir / f"{root_stem}.root"
                            with _up_syst.recreate(str(rpath)) as rf:
                                for label, hv in syst_rows:
                                    rf[label] = (hv.astype(float), bins_syst.astype(float))
                                rf["TotalBkg"] = (total_bkg.astype(float), bins_syst.astype(float))
                            created.append(str(rpath))
                            logging.debug("Syst ROOT: %s", rpath.name)

                        # --- Kinematic systematics (JES/JER shifted Recoil) ---
                        if svar == "Recoil":
                            for syst_name, kin_branch in _kin_systs.items():
                                syst_rows_k: List = []
                                any_kin = False
                                for proc_label, entries in bkg_entries.items():
                                    group_hv = None
                                    for e in entries:
                                        cached = _syst_cache.get(id(e), {"skip": True})
                                        if cached.get("skip"):
                                            continue
                                        br = e["branches"]
                                        kin_v = br.get(kin_branch)
                                        if kin_v is None:
                                            continue
                                        any_kin = True
                                        scale = ((luminosity * e["xsec"] * 1000.0) / e["wte"]
                                                 if e["xsec"] is not None and e["wte"] > 0 else 1.0)
                                        mask_np = cached.get("mask_np")
                                        if mask_np is None:
                                            continue
                                        w_arr = br.get("full_event_weight")
                                        w = np.asarray(w_arr, dtype=float)[mask_np] if w_arr is not None else np.ones(int(mask_np.sum()))
                                        # Re-apply recoil threshold with shifted value
                                        kin_vals = np.asarray(kin_v, dtype=float)[mask_np]
                                        recoil_min = region_obj.config.get("event_selection", {}).get("recoil_min", 0.0) if hasattr(region_obj, 'config') else 0.0
                                        kin_mask = kin_vals > recoil_min
                                        vals = _apply_variable_plot_filter(svar, kin_vals[kin_mask])
                                        w = w[kin_mask]
                                        hv, _ = np.histogram(_clip_overflow(vals, bins_syst), bins=bins_syst, weights=w * scale)
                                        group_hv = hv if group_hv is None else group_hv + hv
                                    if group_hv is not None:
                                        syst_rows_k.append((proc_label, group_hv))
                                if not any_kin or not syst_rows_k:
                                    continue
                                total_bkg_k = sum(h for _, h in syst_rows_k)
                                root_stem_k = f"hist_{cat_s}_{rdir_s}_{svar}_{syst_name}_log"
                                rpath_k = root_dir / f"{root_stem_k}.root"
                                with _up_syst.recreate(str(rpath_k)) as rf:
                                    for label, hv in syst_rows_k:
                                        rf[label] = (hv.astype(float), bins_syst.astype(float))
                                    rf["TotalBkg"] = (total_bkg_k.astype(float), bins_syst.astype(float))
                                created.append(str(rpath_k))
                                logging.debug("Syst ROOT (kin): %s", rpath_k.name)

            except Exception as _syst_exc:
                logging.error("Systematic ROOT file generation failed: %s", _syst_exc, exc_info=True)

        # ---- systematic comparison plots (central + up + down per uncertainty) ----
        if make_syst_plots and self.systematic_variables and not weight_systematic:
            try:
                created.extend(self._plot_systematics(
                    output_dir=output_dir, version=version, luminosity=luminosity,
                    target_regions=target_regions,
                ))
            except Exception as _sp_exc:
                logging.error("Systematic plots failed: %s", _sp_exc, exc_info=True)

        # ---- per-region cutflow plots ----
        if evtsel_cutflow_per_proc:
            import awkward as _ak
            for region_name in target_regions:
                region_obj = region_manager.regions.get(region_name)
                if region_obj is None:
                    continue
                # Per-process sequential region-cut yields: {proc: {cut: yield}}
                region_cf_per_proc: Dict[str, Dict[str, float]] = {}
                for proc_label, entries in bkg_entries.items():
                    proc_region_cf: Dict[str, float] = {}
                    for e in entries:
                        br, wte, xsec = e["branches"], e["wte"], e["xsec"]
                        scale = ((luminosity * xsec * 1000.0) / wte
                                 if xsec is not None and wte > 0
                                 else (luminosity / wte if wte > 0 else 1.0))
                        try:
                            ak_dict = {}
                            for k, v in br.items():
                                if not isinstance(v, np.ndarray) or v.ndim != 1:
                                    continue
                                if v.dtype == object:
                                    try:
                                        ak_dict[k] = _ak.Array(list(v))
                                    except Exception:
                                        pass
                                else:
                                    ak_dict[k] = v
                            events_ak = _ak.Array(ak_dict)
                            w_arr = br.get(weight_branch)
                            w = np.asarray(w_arr, dtype=float) if w_arr is not None else np.ones(len(events_ak))
                            per_cut = region_obj.apply_cuts_with_yields(events_ak, objects={}, weight=w * scale)
                            for cut_label, val in per_cut.items():
                                proc_region_cf[cut_label] = proc_region_cf.get(cut_label, 0.0) + val
                        except Exception:
                            pass
                    if proc_region_cf:
                        region_cf_per_proc[proc_label] = proc_region_cf
                # ---- data cutflow for this region (raw counts, no normalisation) ----
                # SR blinded unless --show-data: use bkg-sum pseudo-data instead.
                data_cf_arr: Optional[np.ndarray] = None
                _is_sr_cf = region_name.endswith(":SR")
                _all_evtsel_labels = list(next(iter(evtsel_cutflow_per_proc.values()), {}).keys()) if evtsel_cutflow_per_proc else []
                _all_region_labels = list(next(iter(region_cf_per_proc.values()), {}).keys()) if region_cf_per_proc else []
                _all_cf_labels = _all_evtsel_labels + _all_region_labels
                if _is_sr_cf and not show_data:
                    # Pseudo-data = MC total
                    if _all_cf_labels and (evtsel_cutflow_per_proc or region_cf_per_proc):
                        _bkg_total = np.zeros(len(_all_cf_labels), dtype=float)
                        for _proc_cf in {**evtsel_cutflow_per_proc, **region_cf_per_proc}.values():
                            pass  # already merged into evtsel/region dicts separately below
                        for _proc, _pcf in evtsel_cutflow_per_proc.items():
                            for _i, _lbl in enumerate(_all_evtsel_labels):
                                _bkg_total[_i] += _pcf.get(_lbl, 0.0)
                        for _proc, _pcf in region_cf_per_proc.items():
                            for _i, _lbl in enumerate(_all_region_labels):
                                _bkg_total[len(_all_evtsel_labels) + _i] += _pcf.get(_lbl, 0.0)
                        data_cf_arr = _bkg_total
                elif _all_cf_labels and data_entries:
                    for _dlabel, _dinfo in data_entries.items():
                        _drp = _dinfo["region_patterns"]
                        if _drp and not any(pat in region_name for pat in _drp):
                            continue
                        for _de, _dpath in zip(_dinfo["entries"], self._resolve_group_files(input_folder, data_groups.get(_dlabel, []))):
                            _dbr = _de["branches"]
                            try:
                                _ak_dict_d: Dict[str, Any] = {}
                                for k, v in _dbr.items():
                                    if not isinstance(v, np.ndarray) or v.ndim != 1:
                                        continue
                                    if v.dtype == object:
                                        try:
                                            _ak_dict_d[k] = _ak.Array(list(v))
                                        except Exception:
                                            pass
                                    else:
                                        _ak_dict_d[k] = v
                                _devents = _ak.Array(_ak_dict_d)
                                _dw = np.ones(len(_devents), dtype=float)
                                _d_evtsel_cf: Dict[str, float] = _load_cutflow_root(_dpath) or {}
                                _d_region_cf = region_obj.apply_cuts_with_yields(_devents, objects={}, weight=_dw)
                                _dcf_aligned = np.array(
                                    [_d_evtsel_cf.get(l, 0.0) for l in _all_evtsel_labels] +
                                    [_d_region_cf.get(l, 0.0) for l in _all_region_labels],
                                    dtype=float,
                                )
                                data_cf_arr = _dcf_aligned if data_cf_arr is None else data_cf_arr + _dcf_aligned
                            except Exception:
                                pass

                if region_cf_per_proc or evtsel_cutflow_per_proc:
                    logging.debug("Plotting cutflow for %s (evtsel=%d procs, region=%d procs)",
                                 region_name, len(evtsel_cutflow_per_proc), len(region_cf_per_proc))
                    try:
                        cf_files = self._plot_cutflow(
                            evtsel_cutflow_per_proc=evtsel_cutflow_per_proc,
                            region_cutflow_per_proc=region_cf_per_proc,
                            region_name=region_name,
                            output_dir=output_dir,
                            version=version,
                            year=year,
                            luminosity=luminosity,
                            data_cutflow=data_cf_arr,
                        )
                        created.extend(cf_files)
                    except Exception as _cf_exc:
                        logging.error("Cutflow plot failed for %s: %s",
                                      region_name, _cf_exc, exc_info=True)

        if skipped_region_var:
            logging.warning("Skipped %d region/var combos in region-from-events: %s",
                            len(skipped_region_var),
                            ', '.join(f"{r}/{v}({why})" for r, v, why in skipped_region_var[:10])
                            + ('...' if len(skipped_region_var) > 10 else ''))
        logging.info("Region-from-events plots complete: %d plots created", len(created))
        return created

    def _plot_systematics(
        self,
        output_dir: str,
        version: str,
        luminosity: float,
        target_regions: List[str],
    ) -> List[str]:
        """For each region × syst variable × UP/DOWN pair: plot central + up + down TotalBkg.

        Reads already-written syst ROOT files. Output same dirs as normal plots.
        Each plot: main panel (3 lines: central/up/down) + ratio panel (up/down / central).
        """
        import uproot as _up
        import re as _re

        root_dir = Path(output_dir) / "plots" / version / "root"
        created: List[str] = []

        if not root_dir.exists():
            return created

        # Discover all syst ROOT files: hist_{cat}_{rdir}_{svar}_{wsyst}_log.root
        _pat = _re.compile(r"hist_(.+?)_(.+?)_(.+?)_(weight_\w+|JES\w+|JER\w+)_log\.root$")
        syst_files: Dict[str, Dict] = {}  # key=(region_stem, svar, syst_base) → {up: path, down: path}
        for rf in root_dir.glob("*.root"):
            m = _pat.match(rf.name)
            if not m:
                continue
            cat, rdir, svar, wsyst = m.group(1), m.group(2), m.group(3), m.group(4)
            region_stem = f"{cat}_{rdir}"
            # Identify UP/DOWN
            if wsyst.endswith("UP") or wsyst.endswith("Up"):
                syst_base = wsyst[:-2] if wsyst.endswith("UP") else wsyst[:-2]
                direction = "up"
            elif wsyst.endswith("DOWN") or wsyst.endswith("Down"):
                syst_base = wsyst[:-4] if wsyst.endswith("DOWN") else wsyst[:-4]
                direction = "down"
            else:
                continue
            key = (region_stem, svar, syst_base)
            syst_files.setdefault(key, {})[direction] = rf

        # Nominal files: hist_{cat}_{rdir}_{svar}_log.root
        _nom_pat = _re.compile(r"hist_(.+?)_(.+?)_(.+?)_log\.root$")
        nom_files: Dict[Tuple, Path] = {}
        for rf in root_dir.glob("*.root"):
            m = _nom_pat.match(rf.name)
            if not m:
                continue
            cat, rdir, svar = m.group(1), m.group(2), m.group(3)
            # Skip syst files (svar would contain 'weight_' etc.)
            if any(x in svar for x in ("weight_", "JES", "JER", "JESUP", "JESDOWN", "JERUP", "JERDOWN")):
                continue
            nom_files[(f"{cat}_{rdir}", svar)] = rf

        if not _HAS_MPLHEP:
            logging.warning("mplhep not available — skipping systematic plots")
            return created

        import matplotlib.ticker as _ticker

        for (region_stem, svar, syst_base), directions in syst_files.items():
            if "up" not in directions or "down" not in directions:
                continue  # need both
            nom_path = nom_files.get((region_stem, svar))
            if nom_path is None:
                continue

            # Parse region for output dir
            region_info = self._parse_region_name(region_stem.replace("_", ":", 1))
            cat = region_info["category"]
            rdir = region_info["region_dir"]

            try:
                with _up.open(str(nom_path)) as f_nom, \
                     _up.open(str(directions["up"])) as f_up, \
                     _up.open(str(directions["down"])) as f_dn:
                    if "TotalBkg" not in f_nom or "TotalBkg" not in f_up or "TotalBkg" not in f_dn:
                        continue
                    h_nom = f_nom["TotalBkg"]
                    h_up  = f_up["TotalBkg"]
                    h_dn  = f_dn["TotalBkg"]
                    bins = np.array(h_nom.axes[0].edges())
                    v_nom = h_nom.values().astype(float)
                    v_up  = h_up.values().astype(float)
                    v_dn  = h_dn.values().astype(float)
            except Exception as _e:
                logging.warning("Syst plot load failed %s/%s: %s", region_stem, syst_base, _e)
                continue

            if v_nom.sum() == 0:
                continue

            # Pretty syst name
            _syst_pretty = syst_base.replace("weight_", "").replace("_", " ").strip()

            # ---- figure ----
            fig, (ax, ax_r) = plt.subplots(
                2, 1, figsize=self.figsize,
                gridspec_kw={"height_ratios": [self.main_height, self.ratio_height], "hspace": self.subplots_hspace},
            )
            fig.subplots_adjust(top=self.subplots_top, bottom=self.subplots_bottom,
                                left=self.subplots_left, right=self.subplots_right)

            # All three as lines (no fill)
            _c0 = self.signal_colors[0] if self.signal_colors else "#000000"
            _c1 = self.signal_colors[1] if len(self.signal_colors) > 1 else "#e31a1c"
            _c2 = self.signal_colors[2] if len(self.signal_colors) > 2 else "#1f78b4"
            ax.stairs(v_nom, bins, fill=False, linewidth=self.signal_linewidth, linestyle="-",  color=_c0, label="Central")
            ax.stairs(v_up,  bins, fill=False, linewidth=self.signal_linewidth, linestyle="--", color=_c1, label=f"{_syst_pretty} UP")
            ax.stairs(v_dn,  bins, fill=False, linewidth=self.signal_linewidth, linestyle=":",  color=_c2, label=f"{_syst_pretty} DOWN")

            ax.set_yscale("log")
            _ymax = max(v_nom.max(), v_up.max(), v_dn.max())
            ax.set_ylim(0.1, _ymax * 1000.0)
            ax.set_ylabel("Events / bin", fontsize=self.fontsize_axis, labelpad=6)
            ax.grid(False)
            _region_pretty = _pretty_region_label(region_stem.replace("_", ":", 1))
            if _region_pretty:
                ax.text(0.05, 0.88, _region_pretty, transform=ax.transAxes,
                        ha="left", va="top", fontsize=self.fontsize_legend, fontweight="bold",
                        bbox=dict(boxstyle="square,pad=0.1", facecolor="none", edgecolor="none"))
            hep.cms.label(llabel=self.cms_label, data=False, lumi=round(luminosity, 2),
                          com=self.com_energy, loc=0, ax=ax)
            ax.legend(loc="upper right", fontsize=self.fontsize_legend, frameon=False, ncol=1)

            # Ratio: up/central, down/central
            safe_nom = np.where(v_nom > 0, v_nom, np.nan)
            r_up = v_up / safe_nom
            r_dn = v_dn / safe_nom
            ax_r.axhline(1.0, color=self.data_color, linewidth=1.2)
            _sc_up = self.signal_colors[1] if len(self.signal_colors) > 1 else "#e31a1c"
            _sc_dn = self.signal_colors[2] if len(self.signal_colors) > 2 else "#1f78b4"
            ax_r.stairs(r_up, bins, fill=False, linewidth=self.signal_linewidth, linestyle="--", color=_sc_up)
            ax_r.stairs(r_dn, bins, fill=False, linewidth=self.signal_linewidth, linestyle=":",  color=_sc_dn)
            ax_r.set_ylim(0.5, 1.5)
            ax_r.set_ylabel("Var / Nom", fontsize=self.fontsize_axis, labelpad=6)
            ax_r.set_xlabel(self.variable_labels.get(svar, svar), fontsize=self.fontsize_axis, labelpad=8)
            ax_r.yaxis.set_major_locator(_ticker.FixedLocator([0.5, 0.75, 1.0, 1.25, 1.5]))
            ax_r.grid(False)
            _x_lim_lo = float(bins[np.where(v_nom > 0)[0][0]]) if np.any(v_nom > 0) else float(bins[0])
            _x_lim_hi = float(bins[np.where(v_nom > 0)[0][-1] + 1]) if np.any(v_nom > 0) else float(bins[-1])
            ax.set_xlim(_x_lim_lo, _x_lim_hi)
            ax_r.set_xlim(_x_lim_lo, _x_lim_hi)

            # Save PNG + PDF in same dirs as normal region plots
            _syst_stem = f"{cat}_{rdir}_{svar}_{syst_base}"
            for fmt in ("png", "pdf"):
                _out_dir = Path(output_dir) / "plots" / version / fmt / "region_analysis" / f"{cat}_{rdir}"
                _out_dir.mkdir(parents=True, exist_ok=True)
                _out = _out_dir / f"{_syst_stem}.{fmt}"
                fig.savefig(str(_out), dpi=self.dpi, bbox_inches="tight")
                created.append(str(_out))
            plt.close(fig)

        logging.info("Systematic plots: %d files written", len(created))
        return created

    @staticmethod
    def _root_to_mpl_label(label: str) -> str:
        """Convert ROOT LaTeX (N_{#mu}, #tau Veto, p_{T}^{Jet1}) to matplotlib math."""
        import re as _re
        _greek = {
            '#mu': r'\mu', '#tau': r'\tau', '#gamma': r'\gamma',
            '#nu': r'\nu', '#phi': r'\phi', '#eta': r'\eta',
            '#Delta': r'\Delta', '#alpha': r'\alpha', '#beta': r'\beta',
        }
        # Also handle plain label aliases
        _aliases = {
            'OR Trigger':  r'OR Trigger',
            'MET Trigger': r'MET Trigger',
            'ELE Trigger': r'EGamma Trigger',
            'Noise filters': 'Noise filters',
        }
        if label in _aliases:
            return _aliases[label]
        s = label
        for root_sym, mpl_sym in _greek.items():
            s = s.replace(root_sym, mpl_sym)
        # Wrap math tokens: any token containing ^, _, or \ in $...$
        # Split on spaces preserving non-math words
        parts = s.split(' ')
        result = []
        for p in parts:
            if any(c in p for c in ('^', '_', '\\')):
                result.append(f'${p}$')
            else:
                result.append(p)
        return ' '.join(result)

    def _plot_cutflow(
        self,
        evtsel_cutflow_per_proc: Dict[str, Dict[str, float]],
        region_cutflow_per_proc: Dict[str, Dict[str, float]],
        region_name: str,
        output_dir: str,
        version: str,
        year: str,
        luminosity: float,
        data_cutflow: Optional[np.ndarray] = None,
    ) -> List[str]:
        """
        Cutflow plot matching event_stacked_plotter.py aesthetics:
        stacked bars per background process (evtsel steps) + ratio panel.
        Region cuts appended as additional bars after event-selection steps.
        """
        import matplotlib.ticker as _ticker
        import mplhep as hep

        # ---- unified cut label list ----
        _all_evtsel = list(next(iter(evtsel_cutflow_per_proc.values()), {}).keys()) if evtsel_cutflow_per_proc else []
        _all_region = list(next(iter(region_cutflow_per_proc.values()), {}).keys()) if region_cutflow_per_proc else []
        all_labels = _all_evtsel + _all_region
        n_evtsel   = len(_all_evtsel)

        if not all_labels:
            logging.warning("_plot_cutflow: no cut labels — skipping")
            return []

        x = np.arange(len(all_labels), dtype=float)

        # ---- color map ----
        _PALETTE = ["#3f90da","#ffa90e","#bd1f01","#94a4a2","#832db6",
                    "#a96b59","#e76300","#b9ac70","#717581","#92dadd"]
        all_procs = list(evtsel_cutflow_per_proc.keys())
        color_map: Dict[str, str] = {}
        if hasattr(self, '_group_colors'):
            color_map = dict(self._group_colors)
        for i, proc in enumerate(all_procs):
            if proc not in color_map:
                color_map[proc] = _PALETTE[i % len(_PALETTE)]

        # ---- per-process arrays aligned to all_labels ----
        def _align(proc_cf: Dict[str, float], labels: List[str]) -> np.ndarray:
            return np.array([proc_cf.get(lbl, 0.0) for lbl in labels], dtype=float)

        total_vals = np.zeros(len(all_labels), dtype=float)
        proc_arrays: List[tuple] = []
        for proc in all_procs:
            arr = np.concatenate([
                _align(evtsel_cutflow_per_proc.get(proc, {}), _all_evtsel),
                _align(region_cutflow_per_proc.get(proc, {}), _all_region),
            ])
            proc_arrays.append((proc, arr))
            total_vals += arr

        # sort smallest→largest (same as event_stacked_plotter)
        proc_arrays.sort(key=lambda t: float(np.sum(t[1])))

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(max(self.figsize_cutflow[0], len(all_labels) * 0.85),
                           self.figsize_cutflow[1]),
            gridspec_kw={"height_ratios": [self.main_height, self.ratio_height],
                         "hspace": self.subplots_hspace},
            sharex=True,
        )
        fig.subplots_adjust(top=self.subplots_top, bottom=self.subplots_bottom,
                            left=self.subplots_left, right=self.subplots_right)

        # ---- top panel: stacked bars per process, no gaps ----
        cumulative = np.zeros(len(all_labels), dtype=float)
        for proc, arr in proc_arrays:
            color = color_map.get(proc, "#3f90da")
            label = self._group_labels.get(proc, proc) if hasattr(self, '_group_labels') else proc
            ax1.bar(x, arr, bottom=cumulative, width=1.0,
                    color=color, edgecolor="none", label=label)
            cumulative += arr

        # dashed line separating evtsel from region cuts
        if n_evtsel > 0 and _all_region:
            ax1.axvline(n_evtsel - 0.5, color="black", linestyle="--",
                        linewidth=1.0, alpha=0.5)

        # ---- data overlay (raw counts, same style as main stacked plots) ----
        has_data = False
        _dc = np.zeros(len(all_labels), dtype=float)
        if data_cutflow is not None:
            _raw = np.asarray(data_cutflow, dtype=float)
            n = min(len(_raw), len(all_labels))
            _dc[:n] = _raw[:n]
            mask = _dc > 0
            if np.any(mask):
                has_data = True
                ax1.errorbar(
                    x[mask], _dc[mask],
                    xerr=np.full(mask.sum(), 0.4),
                    yerr=np.sqrt(_dc[mask]),
                    fmt="o", color=self.data_color,
                    markerfacecolor=self.data_color, markeredgecolor=self.data_color,
                    markersize=self.data_markersize, elinewidth=self.data_elinewidth, capsize=0,
                    label="Data", zorder=10,
                )

        ax1.set_ylabel("Events", fontsize=self.fontsize_axis, labelpad=6)
        ax1.set_yscale("log")
        pos = cumulative[cumulative > 0]
        ymin = max(1e-3, min(float(np.min(pos)), 1e2)) if pos.size else 1e-3
        ymax = max(1.0, float(np.max(cumulative)),
                   float(np.max(_dc)) if data_cutflow is not None else 1.0)
        ax1.set_ylim(ymin, 10 ** (np.ceil(np.log10(ymax)) + 2))
        ax1.grid(False)

        hep.cms.label(llabel=self.cms_label, data=has_data,
                      lumi=round(luminosity, 2), com=self.com_energy, loc=0, ax=ax1)

        _region_pretty = _pretty_region_label(region_name)
        if _region_pretty:
            ax1.text(
                0.05, 0.88, _region_pretty,
                transform=ax1.transAxes,
                ha="left", va="top",
                fontsize=20,
                fontweight="bold",
                bbox=dict(boxstyle="square,pad=0.1", facecolor="none",
                          edgecolor="none", alpha=1.0),
            )

        handles, leg_labels = ax1.get_legend_handles_labels()
        if handles:
            ax1.legend(handles, leg_labels, loc="upper right",
                       bbox_to_anchor=(0.97, 0.97), ncol=self.legend_ncol, frameon=False,
                       handlelength=1.5, columnspacing=1.0,
                       handletextpad=0.5, fontsize=self.fontsize_legend)

        # ---- bottom panel: Data / MC ----
        mc_total = cumulative
        pred_rel_err = np.where(mc_total > 0, np.sqrt(mc_total) / mc_total, 0.0)
        x_edges = np.append(x - 0.5, x[-1] + 0.5)
        ratio_lo = np.append(1.0 - pred_rel_err, (1.0 - pred_rel_err)[-1])
        ratio_hi = np.append(1.0 + pred_rel_err, (1.0 + pred_rel_err)[-1])
        ax2.fill_between(x_edges, ratio_lo, ratio_hi, step="post",
                         facecolor="#bbbbbb", edgecolor="none", alpha=0.6, label="MC stat.")
        if has_data and data_cutflow is not None:
            ratio = np.divide(_dc, mc_total, out=np.full_like(mc_total, np.nan), where=mc_total > 0)
            ratio_err = np.where(_dc > 0, np.sqrt(_dc) / np.where(mc_total > 0, mc_total, 1.0), 0.0)
            ratio_mask = np.isfinite(ratio)
            if np.any(ratio_mask):
                ax2.errorbar(
                    x[ratio_mask], ratio[ratio_mask],
                    xerr=np.full(ratio_mask.sum(), 0.4),
                    yerr=ratio_err[ratio_mask],
                    fmt="o", color=self.data_color,
                    markerfacecolor=self.data_color, markeredgecolor=self.data_color,
                    markersize=self.data_markersize, elinewidth=self.data_elinewidth, capsize=0, zorder=10,
                )
        else:
            ax2.axhline(1.0, color="#999999", linestyle="--", linewidth=1.0)
        ax2.axhline(1.0, color=self.data_color, linestyle="-", linewidth=1.2)
        if n_evtsel > 0 and _all_region:
            ax2.axvline(n_evtsel - 0.5, color="black", linestyle="--",
                        linewidth=1.0, alpha=0.5)
        ax2.set_ylim(0.0, 2.0)
        ax2.set_ylabel("Data / MC", fontsize=self.fontsize_axis, labelpad=6)
        ax2.yaxis.set_major_locator(_ticker.FixedLocator([0, 0.5, 1.0, 1.5, 2.0]))
        ax2.grid(False)

        ax2.set_xticks(x)
        ax2.set_xticklabels(
            [self._root_to_mpl_label(str(lbl)) for lbl in all_labels],
            rotation=35, ha="right", fontsize=self.fontsize_xtick_cutflow,
        )
        ax2.set_xlabel("", fontsize=self.fontsize_axis)
        ax1.set_xlim(-0.5, len(all_labels) - 0.5)
        ax2.set_xlim(-0.5, len(all_labels) - 0.5)

        # ---- output dirs ----
        region_info  = self._parse_region_name(region_name)
        category     = region_info["category"]
        region_dir   = region_info["region_dir"]
        is_evtsel    = (category == "event_selection")
        region_label = "event_selection" if is_evtsel else f"{category}_{region_dir}"
        plot_subdir  = "event_selection" if is_evtsel else f"region_analysis/{region_label}"

        saved = []
        for fmt in ("png", "pdf"):
            out_dir = Path(output_dir) / "plots" / version / fmt / plot_subdir
            out_dir.mkdir(parents=True, exist_ok=True)
            p = out_dir / f"cutflow_{region_label}.{fmt}"
            fig.savefig(str(p), dpi=self.dpi, bbox_inches="tight")
            saved.append(str(p))
        plt.close(fig)

        # TXT + TEX tables
        step_eff = np.ones(len(total_vals), dtype=float)
        for i in range(1, len(total_vals)):
            step_eff[i] = total_vals[i] / total_vals[i - 1] if total_vals[i - 1] > 0 else 0.0
        txt_dir = Path(output_dir) / "plots" / version / "text" / plot_subdir
        txt_dir.mkdir(parents=True, exist_ok=True)

        def _root_to_latex(label: str) -> str:
            """Convert ROOT label (N_{#mu}, #tau Veto) to proper LaTeX."""
            _greek = {
                '#mu': r'\mu', '#tau': r'\tau', '#gamma': r'\gamma',
                '#nu': r'\nu', '#phi': r'\phi', '#eta': r'\eta',
                '#Delta': r'\Delta', '#Sigma': r'\Sigma',
            }
            s = label
            for root_sym, tex_sym in _greek.items():
                s = s.replace(root_sym, tex_sym)
            # wrap substrings containing ^, _, or \ in $...$
            import re as _re
            def _wrap(m):
                tok = m.group(0)
                return f'${tok}$' if any(c in tok for c in ('^', '_', '\\')) else tok
            return _re.sub(r'\S+', _wrap, s)

        tex_path = txt_dir / f"cutflow_{region_label}.tex"
        with open(tex_path, "w") as fh:
            fh.write("\\begin{table}[htbp]\n\\centering\n")
            fh.write(f"\\caption{{Cutflow: {region_name.replace('_', ' ')}}}\n")
            fh.write(f"\\label{{tab:cutflow_{region_label}}}\n")
            fh.write("\\begin{tabular}{lrr}\n\\toprule\n")
            fh.write("\\textbf{Step} & \\textbf{Yield} & \\textbf{Step eff.} \\\\ \\midrule\n")
            for i, lbl in enumerate(all_labels):
                tex_lbl = _root_to_latex(lbl)
                fh.write(f"{tex_lbl} & ${total_vals[i]:.2f}$ & ${step_eff[i]:.4f}$ \\\\\n")
            fh.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
        saved.append(str(tex_path))

        # ROOT TH1D
        try:
            import uproot as _up
            root_dir = Path(output_dir) / "plots" / version / "root"
            root_dir.mkdir(parents=True, exist_ok=True)
            root_path = root_dir / f"cutflow_{region_label}.root"
            edges = np.arange(len(all_labels) + 1, dtype=float)
            with _up.recreate(str(root_path)) as rf:
                rf[f"cutflow_{region_label}"] = (total_vals, edges)
            saved.append(str(root_path))
        except Exception as _exc:
            logging.warning("Could not save cutflow ROOT for %s: %s", region_name, _exc)

        logging.info("Cutflow for %s saved (%d files)", region_name, len(saved))
        return saved

    def _create_th1f_from_hist(self, hist_data: Any, name: str, title: str) -> Any:
        """
        Create ROOT TH1F from histogram data.

        Args:
            hist_data: Histogram data (could be hist.Hist or dict)
            name: Histogram name
            title: Histogram title

        Returns:
            ROOT TH1F object
        """
        try:
            import ROOT

            if hasattr(hist_data, 'values') and hasattr(hist_data, 'axes'):
                # hist.Hist object
                values = hist_data.values()
                edges = hist_data.axes[0].edges
                nbins = len(edges) - 1

                th1f = ROOT.TH1F(name, title, nbins, edges[0], edges[-1])

                for i in range(nbins):
                    th1f.SetBinContent(i + 1, values[i])

            elif isinstance(hist_data, dict) and 'values' in hist_data:
                # Fallback histogram dict
                values = hist_data['values']
                bins = hist_data['bins']
                nbins = len(bins) - 1

                th1f = ROOT.TH1F(name, title, nbins, bins[0], bins[-1])

                for i in range(min(nbins, len(values))):
                    th1f.SetBinContent(i + 1, values[i])
            else:
                # Create empty histogram
                th1f = ROOT.TH1F(name, title, 50, 0, 100)

            return th1f

        except ImportError:
            return None

    def _save_yield_text_file(
        self,
        txt_path: str,
        hist_name: str,
        data_hists: Optional[Dict[str, Any]],
        mc_hists: Optional[Dict[str, Any]],
        signal_hists: Optional[Dict[str, Any]]
    ):
        """
        Save yield information to text file.

        Args:
            txt_path: Path to text file
            hist_name: Histogram name
            data_hists: Data histogram data
            mc_hists: MC histogram data
            signal_hists: Signal histogram data
        """
        with open(txt_path, 'w') as f:
            f.write(f"# Yields for {hist_name}\n")
            f.write("# Process Yield±Error\n")

            # Data yield
            if data_hists and hist_name in data_hists:
                data_yield = self._get_histogram_yield(data_hists[hist_name])
                f.write(f"data_obs {data_yield:.2f}±{data_yield*0.1:.2f}\n")

            # MC yields
            if mc_hists:
                for process_name, process_hists in mc_hists.items():
                    if hist_name in process_hists:
                        yield_val = self._get_histogram_yield(process_hists[hist_name])
                        f.write(f"{process_name} {yield_val:.2f}±{yield_val*0.1:.2f}\n")

            # Signal yield
            if signal_hists and hist_name in signal_hists:
                signal_yield = self._get_histogram_yield(signal_hists[hist_name])
                f.write(f"signal {signal_yield:.2f}±{signal_yield*0.1:.2f}\n")

    def _get_histogram_yield(self, hist_data: Any) -> float:
        """
        Get total yield from histogram data.

        Args:
            hist_data: Histogram data

        Returns:
            Total yield
        """
        if hasattr(hist_data, 'values'):
            # hist.Hist object
            return float(np.sum(hist_data.values()))
        elif isinstance(hist_data, dict) and 'values' in hist_data:
            # Fallback histogram dict
            return float(np.sum(hist_data['values']))
        else:
            return 0.0

    def create_region_plots(self, results: Dict[str, Any], output_dir: str,
                          show_data: bool = True, regions: Optional[List[str]] = None,
                          version: Optional[str] = None, formats: Optional[List[str]] = None,
                          luminosity: float = 1.0,
                          cross_sections: Optional[Dict[str, float]] = None) -> Dict[str, str]:
        """
        Create plots for all regions.

        Args:
            results: Analysis results dictionary
            output_dir: Output directory for plots
            show_data: Whether to show data points
            regions: List of regions to plot (None for all)
            version: Version string for multi-format output
            formats: List of output formats
            luminosity: Integrated luminosity in fb-1
            cross_sections: Dict mapping process name to xsec in pb (optional)

        Returns:
            Dictionary of plot file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        plot_files = {}

        # Generate version string if not provided
        if not version:
            version = datetime.now().strftime("%Y%m%d_%H%M")

        # Get regions to plot
        if regions is None:
            regions = list(results.get("region_histograms", {}).keys())

        # Compute per-histogram scale: lumi * xsec * 1000 / weighted_total_events
        # xsec is None for single-process pkls without --xsection-json
        weighted_total_events = float(results.get("metadata", {}).get("weighted_total_events", 0.0))
        process_name = str(results.get("metadata", {}).get("process", ""))
        xsec = (cross_sections or {}).get(process_name)
        if weighted_total_events > 0:
            if xsec is not None:
                hist_scale = (luminosity * xsec * 1000.0) / weighted_total_events
            else:
                hist_scale = luminosity / weighted_total_events
            logging.info(
                f"Region histogram scale: {hist_scale:.4g} "
                f"(lumi={luminosity}, xsec={xsec}, wte={weighted_total_events:.0f})"
            )
        else:
            hist_scale = 1.0
            logging.warning("weighted_total_events=0 in metadata — histograms shown unnormalised")

        # Each region's plots (individual-variable + grouped) are independent
        # of every other region's — read-only config, own histogram slice,
        # own output files. Parallelize across CPU cores via multiprocessing
        # (not threading: matplotlib draw + hist fill are CPU-bound, GIL would
        # serialize them under threads). Override with PLOT_NUM_WORKERS env.
        num_workers = int(os.environ.get("PLOT_NUM_WORKERS", max(1, (os.cpu_count() or 1))))
        num_workers = max(1, min(num_workers, len(regions)))

        tasks = [
            (self.config, results, region, output_dir, show_data, version, formats, hist_scale)
            for region in regions
        ]

        if num_workers > 1 and len(regions) > 1:
            import multiprocessing as mp

            logging.info(f"Creating region plots with {num_workers} worker processes")
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=num_workers) as pool:
                task_results = pool.map(_plot_one_region_worker, tasks)
        else:
            task_results = [_plot_one_region_worker(t) for t in tasks]

        for region, all_region_plots in task_results:
            plot_files[region] = all_region_plots

        return plot_files

    def _get_excluded_variables_for_region(self, region: str) -> List[str]:
        """
        Get list of variables to exclude for a given region based on category and region type.

        Based on StackPlotter logic:
        - 1b SR: exclude jet3 variables and all lepton variables
        - 2b SR: exclude lepton variables (but include jet3)
        - Control Regions: include lepton variables, exclude jet3 for 1b regions

        Also applies configurable exclusions from self.region_exclusions.

        Args:
            region: Region name (e.g., "1b:SR", "2b:CR_Topmunu")

        Returns:
            List of variable name patterns to exclude
        """
        region_info = self._parse_region_name(region)
        category = region_info["category"]
        region_type = region_info["region_dir"]
        is_sr = "SR" in region_type
        is_cr = "CR" in region

        excluded = []

        # Jet3 variables to exclude (names must match keys in histograms.py define_histograms)
        jet3_vars = [
            'jet3_pt', 'jet3_eta', 'jet3_phi',
            'm_jet1jet3', 'isjet2_eta_match',
        ]

        # Lepton variables excluded from SRs (must match histogram keys in histograms.py)
        lepton_vars = [
            'muon_pt', 'muon_eta',
            'electron_pt', 'electron_eta',
            'lep1_pt', 'lep1_phi',
            'lep2_pt', 'lep2_phi',
            'dphi_lep1_met',
            'w_mass', 'w_pt', 'z_mass', 'z_pt', 'mll', 'mt',
            'n_muons', 'n_electrons',
            'dr_muon_jet', 'dr_electron_jet',
        ]

        if is_sr:
            # Signal regions: exclude all lepton variables (including z_mass and z_pt)
            excluded.extend(lepton_vars)

            if category == "1b":
                # 1b SR: also exclude jet3 variables
                excluded.extend(jet3_vars)
            # 2b SR: keep jet3 variables (don't exclude them)
        elif is_cr:
            # Control regions: include lepton variables (don't exclude them from base list)
            # But exclude jet3 for 1b regions
            if category == "1b":
                excluded.extend(jet3_vars)
            # 2b CRs: keep jet3 variables (e.g., Top CR may have >3 jets)

            # Note: z_mass and z_pt will be excluded from Top/W CRs via configurable exclusions below

        # Apply configurable exclusions from config
        # Check for exact region match first (e.g., "1b:SR")
        if region in self.region_exclusions:
            excluded.extend(self.region_exclusions[region])

        # Check for region pattern matches (e.g., "Top", "Wlnu", "1b:SR")
        for pattern, vars_to_exclude in self.region_exclusions.items():
            # Skip if it's an exact match (already handled above)
            if pattern == region:
                continue

            # Check if pattern matches region name or region_type
            if pattern in region or pattern in region_type:
                excluded.extend(vars_to_exclude)

        # Check for category-specific exclusions (e.g., "1b", "2b")
        category_pattern = f"{category}:"
        if category_pattern in self.region_exclusions:
            excluded.extend(self.region_exclusions[category_pattern])

        # Remove duplicates while preserving order
        seen = set()
        excluded_unique = []
        for var in excluded:
            if var not in seen:
                seen.add(var)
                excluded_unique.append(var)

        return excluded_unique

    def _get_allowed_variables_for_region(self, region: str, all_vars: List[str]) -> List[str]:
        """Return ordered variable list for *region*: common_variables + region-specific additions.

        common_variables always prepended. region_variables provides per-region extras
        (exact key matched first, then substring). Only vars present in all_vars kept.
        When neither common_variables nor region_variables are configured, returns all_vars.
        """
        region_extras: Optional[List[str]] = self.region_variables.get(region)
        if region_extras is None:
            for key, wl in self.region_variables.items():
                if key in region:
                    region_extras = wl
                    break

        if self.common_variables or region_extras is not None:
            combined = list(self.common_variables) + list(region_extras or [])
            # deduplicate preserving order
            seen: set = set()
            unique: List[str] = []
            for v in combined:
                if v not in seen:
                    seen.add(v)
                    unique.append(v)
            present = set(all_vars)
            # Alias mapping: plotting.yaml name → PKL name
            _valias = {
                'Recoil':'recoil','MET_pt':'met','MET_phi':'met_phi',
                'njets':'n_jets','Jet1Pt':'jet_pt','Jet1Eta':'jet_eta',
                'Jet1Phi':'jet_phi','Jet2Pt':'jet2_pt','Jet2Eta':'jet2_eta',
                'Jet2Phi':'jet2_phi','dPhi_jetMET':'min_dphi',
                'dPhiJet12':'dphi_jet12','dEtaJet12':'deta_jet12',
                'M_Jet1Jet2':'m_jet1jet2','Jet1BTagScore':'btag_deepjet',
                'Jet2BTagScore':'jet2_deepcsv','pT_Jet1Jet2':'pt_jet1jet2',
                'ratioPtJet21':'ratio_pt_jet21','dRJet12':'dr_jet12',
                'b_flavor_count':'btag_hf',
            }
            out: List[str] = []
            for v in unique:
                if v in present:
                    out.append(v)
                elif v == "ml_score":
                    out.extend(sorted(x for x in all_vars if x.startswith("ml_score_mh3_")))
                else:
                    alias = _valias.get(v)
                    if alias and alias in present:
                        out.append(v)  # keep the plotting name, but match alias
            return out

        return list(all_vars)

    def _create_individual_variable_plots(self, results: Dict[str, Any], region: str,
                                          output_dir: str, show_data: bool,
                                          version: str, formats: Optional[List[str]] = None,
                                          hist_scale: float = 1.0) -> Dict[str, str]:
        """
        Create individual plots for each variable in a region.

        Args:
            results: Analysis results dictionary
            region: Region name
            output_dir: Base output directory
            show_data: Whether to show data points
            version: Version string
            formats: List of output formats

        Returns:
            Dictionary of plot file paths
        """
        plot_files = {}

        # Get region histograms
        region_histograms = results.get("region_histograms", {}).get(region, {})
        if not region_histograms:
            logging.warning(f"No histograms found for region {region}")
            return plot_files

        _internal_patterns = ['_dnn_score', '_region_variables']
        candidate_vars = [
            v for v in region_histograms.keys()
            if not any(pat in v for pat in _internal_patterns)
        ]
        variables_to_plot = self._get_allowed_variables_for_region(region, candidate_vars)

        # Parse region once — used for all variables
        region_info = self._parse_region_name(region)
        category = region_info["category"]
        region_dir = region_info["region_dir"]

        # Create one plot per variable
        for var_name in variables_to_plot:
            try:
                hist_data = region_histograms.get(var_name)
                if hist_data is None:
                    continue

                # --- Extract arrays from histogram object ---
                if hasattr(hist_data, 'values') and hasattr(hist_data, 'axes'):
                    _hv = np.asarray(hist_data.values(), dtype=float)
                    _edges = np.asarray(hist_data.axes[0].edges, dtype=float)
                    _hvar = hist_data.variances()
                    _hs = np.asarray(_hvar, dtype=float) if _hvar is not None else np.zeros_like(_hv)
                elif isinstance(hist_data, dict):
                    _hv = np.asarray(hist_data.get('values', []), dtype=float)
                    _edges = np.asarray(hist_data.get('bins', []), dtype=float)
                    _errs = hist_data.get('errors', None)
                    _hs = (np.asarray(_errs, dtype=float) ** 2
                           if _errs is not None else np.zeros_like(_hv))
                else:
                    logging.warning("Unknown histogram format for %s in %s", var_name, region)
                    continue

                if _hv.size == 0 or _edges.size < 2:
                    continue

                # Apply lumi×xsec/wte scale
                _hv_scaled = _hv * hist_scale
                _hs_scaled = _hs * (hist_scale ** 2)

                # Create figure with ratio panel
                fig, (ax_main, ax_ratio) = plt.subplots(
                    2, 1, figsize=(10, 10),
                    gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05}
                )

                # Plot histogram on main axis (apply lumi×xsec/wte scale)
                self._plot_single_histogram(ax_main, hist_data, var_name, show_data, scale=hist_scale)

                # Ratio panel (placeholder for now)
                ax_ratio.set_xlabel(self._get_variable_label(var_name))
                ax_ratio.set_ylabel("Data/MC")
                ax_ratio.grid(True, alpha=0.3)

                # Determine if this variable should use log scale
                use_log_scale = var_name not in self.no_log_scale_vars

                # Create filename: {category}_{region}_{variable_name}
                plot_filename = f"{category}_{region_dir}_{var_name}"

                # Save with log scale (default for most plots)
                if use_log_scale:
                    ax_main.set_yscale('log')
                    ax_main.set_ylim(bottom=0.1)
                    saved_files = self.save_plot_multi_format(
                        fig, plot_filename, region, version, output_dir,
                        is_log=True, data_hists=None, mc_hists=None, signal_hists=None
                    )
                    # Also save linear version
                    ax_main.set_yscale('linear')
                    ax_main.set_ylim(bottom=0)
                    saved_files_linear = self.save_plot_multi_format(
                        fig, plot_filename, region, version, output_dir,
                        is_log=False, data_hists=None, mc_hists=None, signal_hists=None
                    )
                else:
                    # Save linear version only (no log scale)
                    saved_files = self.save_plot_multi_format(
                        fig, plot_filename, region, version, output_dir,
                        is_log=False, data_hists=None, mc_hists=None, signal_hists=None
                    )
                    saved_files_linear = {}

                plt.close(fig)

                # --- TXT + TEX yield tables ---
                text_dir = (Path(output_dir) / "plots" / version / "text"
                            / category / region_dir)
                text_dir.mkdir(parents=True, exist_ok=True)
                self._write_yield_table(
                    text_dir / plot_filename,
                    var_name,
                    _edges,
                    [("MC", _hv_scaled, _hs_scaled)],
                    data_ndarray=None,
                )

                # --- ROOT output (uproot flat TH1D, one per variable) ---
                root_dir = Path(output_dir) / "plots" / version / "root"
                root_dir.mkdir(parents=True, exist_ok=True)
                root_path = root_dir / f"{plot_filename}.root"
                try:
                    import uproot
                    with uproot.recreate(str(root_path)) as rf:
                        rf["mc"] = (_hv_scaled, _edges)
                except Exception as _root_exc:
                    logging.warning("ROOT write failed for %s: %s", plot_filename, _root_exc)

                plot_files[var_name] = saved_files.get('png', '')

            except Exception as e:
                logging.warning(f"Failed to create plot for {var_name} in {region}: {e}")

        return plot_files

    def _plot_single_histogram(self, ax, hist_data: Any, var_name: str, show_data: bool,
                               scale: float = 1.0):
        """Plot a single histogram on the given axis.

        scale = lumi * xsec * 1000 / weighted_total_events — applied after MC-SF weights
        already baked into the histogram at fill time.
        """
        # Extract values and edges from histogram
        if hasattr(hist_data, 'values') and hasattr(hist_data, 'axes'):
            values = hist_data.values()
            edges = hist_data.axes[0].edges
            errors = np.sqrt(hist_data.variances()) if hasattr(hist_data, 'variances') else None
        elif isinstance(hist_data, dict):
            values = np.array(hist_data.get('values', []))
            edges = np.array(hist_data.get('bins', []))
            errors = np.array(hist_data.get('errors', [])) if 'errors' in hist_data else None
        else:
            logging.warning(f"Unknown histogram format for {var_name}")
            return

        # Apply lumi×xsec/wte normalisation (scale=1.0 → no change)
        values = values * scale
        if errors is not None:
            errors = errors * scale

        # Plot histogram
        centers = (edges[:-1] + edges[1:]) / 2
        widths = edges[1:] - edges[:-1]

        ax.bar(centers, values, width=widths, alpha=0.7, label='MC')
        if errors is not None:
            ax.errorbar(centers, values, yerr=errors, fmt='none', color='black', alpha=0.5)

        ax.set_ylabel("Events")
        ax.set_title(self._get_variable_label(var_name))
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _get_variable_label(self, var_name: str) -> str:
        """Get a formatted label for a variable name."""
        if var_name in self.variable_labels:
            return self.variable_labels[var_name]
        if var_name.startswith("ml_score_mh3_"):
            m = re.match(r"ml_score_mh3_([\d.p]+)_mh4_([\d.p]+)$", var_name)
            if m:
                mh3 = m.group(1).replace("p", ".")
                mh4 = m.group(2).replace("p", ".")
                base = self.variable_labels.get("ml_score", "DNN score")
                return f"{base} (MH3={mh3}, MH4={mh4})"
        return var_name.replace('_', ' ').title()

    def _create_region_plots_single(self, results: Dict[str, Any], region: str,
                                  output_path: Path, show_data: bool,
                                  version: Optional[str] = None, output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Create plots for a single region (grouped plots).

        Args:
            results: Analysis results dictionary
            region: Region name
            output_path: Output directory (for fallback)
            show_data: Whether to show data points
            version: Version string for new directory structure
            output_dir: Base output directory for new structure

        Returns:
            Dictionary of plot file paths
        """
        plot_files = {}

        # Get region histograms
        region_histograms = results.get("region_histograms", {}).get(region, {})
        if not region_histograms:
            logging.warning(f"No histograms found for region {region}")
            return plot_files

        # Create different types of plots
        plot_types = [
            "kinematic_distributions",
            "multiplicity_distributions",
            "dnn_scores",
            "region_comparison"
        ]

        for plot_type in plot_types:
            try:
                plot_file = self._create_plot_type(
                    region_histograms, region, plot_type, output_path, show_data, version, output_dir
                )
                if plot_file:
                    plot_files[plot_type] = plot_file
            except Exception as e:
                logging.warning(f"Failed to create {plot_type} plot for {region}: {e}")

        return plot_files

    def _create_plot_type(self, histograms: Dict[str, Any], region: str,
                         plot_type: str, output_path: Path, show_data: bool,
                         version: Optional[str] = None, output_dir: Optional[str] = None) -> Optional[str]:
        """
        Create a specific type of plot.

        Args:
            histograms: Histogram dictionary
            region: Region name
            plot_type: Type of plot to create
            output_path: Output directory (for fallback)
            show_data: Whether to show data points
            version: Version string for new directory structure
            output_dir: Base output directory for new structure

        Returns:
            Plot file path or None
        """
        if plot_type == "kinematic_distributions":
            return self._create_kinematic_plots(histograms, region, output_path, show_data, version, output_dir)
        elif plot_type == "multiplicity_distributions":
            return self._create_multiplicity_plots(histograms, region, output_path, show_data, version, output_dir)
        elif plot_type == "dnn_scores":
            return self._create_dnn_plots(histograms, region, output_path, show_data, version, output_dir)
        elif plot_type == "region_comparison":
            return self._create_region_comparison_plots(histograms, region, output_path, show_data, version, output_dir)
        else:
            return None

    def _create_kinematic_plots(self, histograms: Dict[str, Any], region: str,
                               output_path: Path, show_data: bool,
                               version: Optional[str] = None, output_dir: Optional[str] = None) -> str:
        """Create kinematic distribution plots."""
        _allowed = set(self._get_allowed_variables_for_region(region, list(histograms.keys())))

        # Determine which plots to include based on region
        plot_vars = []

        # Plot MET
        if 'met' in _allowed:
            plot_vars.append(('met', "MET [GeV]", "Missing Transverse Energy"))

        # Plot jet pT (jet1)
        if 'jet1_pt' in _allowed:
            plot_vars.append(('jet1_pt', "Jet1 pT [GeV]", "Leading Jet pT"))
        elif 'jet_pt' in _allowed:
            plot_vars.append(('jet_pt', "Jet pT [GeV]", "Jet Transverse Momentum"))

        # Plot jet eta
        if 'jet1_eta' in _allowed:
            plot_vars.append(('jet1_eta', "Jet1 η", "Leading Jet Pseudorapidity"))
        elif 'jet_eta' in _allowed:
            plot_vars.append(('jet_eta', "Jet η", "Jet Pseudorapidity"))

        # Plot b-tag score
        if 'btag_deepjet' in _allowed:
            plot_vars.append(('btag_deepjet', "DeepJet Score", "B-tagging Discriminant"))

        # Plot jet2 pT
        if 'jet2_pt' in _allowed:
            plot_vars.append(('jet2_pt', "Jet2 pT [GeV]", "Subleading Jet pT"))

        # Plot jet3 pT
        if 'jet3_pt' in _allowed:
            plot_vars.append(('jet3_pt', "Jet3 pT [GeV]", "Third Jet pT"))

        # Create figure with appropriate number of subplots
        n_plots = len(plot_vars)
        if n_plots == 0:
            logging.warning(f"No kinematic plots to create for region {region}")
            return ""

        # Arrange plots in a grid (2 rows x 2 cols = 4 subplots, or adjust)
        n_cols = 2
        n_rows = (n_plots + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 7.5 * n_rows))
        if n_rows > 1:
            axes = axes.flatten()
        elif n_rows == 1:
            axes = [axes] if n_cols == 1 else axes.flatten()
        else:
            axes = []

        # Plot each variable
        for i, (var_name, xlabel, title) in enumerate(plot_vars):
            if i < len(axes):
                self._plot_histogram(axes[i], histograms[var_name], xlabel, "Events",
                                   title, show_data)

        # Hide unused subplots
        for i in range(len(plot_vars), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        # Use new directory structure if version and output_dir are provided
        if version and output_dir:
            # Parse region to get category and region_dir for filename
            region_info = self._parse_region_name(region)
            category = region_info["category"]
            region_dir = region_info["region_dir"]
            plot_filename = f"{category}_{region_dir}_kinematic_distributions"

            # Determine if log scale should be used (multi-panel plots usually don't use log)
            use_log_scale = False

            saved_files = self.save_plot_multi_format(
                fig, plot_filename, region, version, output_dir,
                is_log=use_log_scale, data_hists=None, mc_hists=None, signal_hists=None
            )
            plt.close()
            return saved_files.get('png', '')
        else:
            # Fallback to old method
            plot_file = output_path / f"{region}_kinematic_distributions.{self.format}"
            plt.savefig(plot_file, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            return str(plot_file)

    def _create_multiplicity_plots(self, histograms: Dict[str, Any], region: str,
                                  output_path: Path, show_data: bool,
                                  version: Optional[str] = None, output_dir: Optional[str] = None) -> str:
        """Create multiplicity distribution plots."""
        _allowed = set(self._get_allowed_variables_for_region(region, list(histograms.keys())))

        # Determine which plots to include based on region
        plot_indices = []
        plot_vars = []

        # Plot jet multiplicity
        if 'n_jets' in _allowed:
            plot_indices.append(0)
            plot_vars.append(('n_jets', "Number of Jets", "Jet Multiplicity"))

        # Plot b-jet multiplicity
        if 'n_bjets' in _allowed:
            plot_indices.append(1)
            plot_vars.append(('n_bjets', "Number of B-jets", "B-jet Multiplicity"))

        # Plot muon multiplicity
        if 'n_muons' in _allowed:
            plot_indices.append(2)
            plot_vars.append(('n_muons', "Number of Muons", "Muon Multiplicity"))

        # Plot electron multiplicity
        if 'n_electrons' in _allowed:
            plot_indices.append(3)
            plot_vars.append(('n_electrons', "Number of Electrons", "Electron Multiplicity"))

        # Plot tau multiplicity
        if 'n_taus' in _allowed:
            plot_indices.append(4)
            plot_vars.append(('n_taus', "Number of Taus", "Tau Multiplicity"))

        # Plot lepton multiplicity
        if 'n_leptons' in _allowed:
            plot_indices.append(5)
            plot_vars.append(('n_leptons', "Number of Leptons", "Lepton Multiplicity"))

        # Create figure with appropriate number of subplots
        n_plots = len(plot_vars)
        if n_plots == 0:
            logging.warning(f"No multiplicity plots to create for region {region}")
            return ""

        # Arrange plots in a grid (2 rows x 3 cols = 6 subplots max)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        # Plot each variable
        for i, (var_name, xlabel, title) in enumerate(plot_vars):
            if i < len(axes):
                self._plot_histogram(axes[i], histograms[var_name], xlabel, "Events",
                                   title, show_data, plot_type="bar")

        # Hide unused subplots
        for i in range(len(plot_vars), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        # Use new directory structure if version and output_dir are provided
        if version and output_dir:
            # Parse region to get category and region_dir for filename
            region_info = self._parse_region_name(region)
            category = region_info["category"]
            region_dir = region_info["region_dir"]
            plot_filename = f"{category}_{region_dir}_multiplicity_distributions"

            # Multiplicity plots don't use log scale
            use_log_scale = False

            saved_files = self.save_plot_multi_format(
                fig, plot_filename, region, version, output_dir,
                is_log=use_log_scale, data_hists=None, mc_hists=None, signal_hists=None
            )
            plt.close()
            return saved_files.get('png', '')
        else:
            # Fallback to old method
            plot_file = output_path / f"{region}_multiplicity_distributions.{self.format}"
            plt.savefig(plot_file, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            return str(plot_file)

    def _create_dnn_plots(self, histograms: Dict[str, Any], region: str,
                         output_path: Path, show_data: bool,
                         version: Optional[str] = None, output_dir: Optional[str] = None) -> str:
        """Create DNN score plots."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Plot DNN score distribution
        if f'{region}_dnn_score' in histograms:
            self._plot_histogram(axes[0], histograms[f'{region}_dnn_score'], "DNN Score", "Events",
                               "DNN Score Distribution", show_data)

        # Plot DNN score vs MET
        if 'met' in histograms and f'{region}_dnn_score' in histograms:
            self._plot_2d_histogram(axes[1], histograms['met'], histograms[f'{region}_dnn_score'],
                                  "MET [GeV]", "DNN Score", "DNN Score vs MET")

        plt.tight_layout()

        # Use new directory structure if version and output_dir are provided
        if version and output_dir:
            # Parse region to get category and region_dir for filename
            region_info = self._parse_region_name(region)
            category = region_info["category"]
            region_dir = region_info["region_dir"]
            plot_filename = f"{category}_{region_dir}_dnn_scores"

            # DNN plots can use log scale
            use_log_scale = True

            saved_files = self.save_plot_multi_format(
                fig, plot_filename, region, version, output_dir,
                is_log=use_log_scale, data_hists=None, mc_hists=None, signal_hists=None
            )
            # Also save linear version
            axes[0].set_yscale('linear')
            axes[0].set_ylim(bottom=0)
            saved_files_linear = self.save_plot_multi_format(
                fig, plot_filename, region, version, output_dir,
                is_log=False, data_hists=None, mc_hists=None, signal_hists=None
            )
            plt.close()
            return saved_files.get('png', '')
        else:
            # Fallback to old method
            plot_file = output_path / f"{region}_dnn_scores.{self.format}"
            plt.savefig(plot_file, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            return str(plot_file)

    def _create_region_comparison_plots(self, histograms: Dict[str, Any], region: str,
                                      output_path: Path, show_data: bool,
                                      version: Optional[str] = None, output_dir: Optional[str] = None) -> str:
        """Create region comparison plots."""
        # This would compare the current region with other regions
        # For now, create a simple summary plot

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Plot MET for region comparison
        if 'met' in histograms:
            self._plot_histogram(ax, histograms['met'], "MET [GeV]", "Events",
                               f"MET Distribution - {region}", show_data)

        plt.tight_layout()

        # Use new directory structure if version and output_dir are provided
        if version and output_dir:
            # Parse region to get category and region_dir for filename
            region_info = self._parse_region_name(region)
            category = region_info["category"]
            region_dir = region_info["region_dir"]
            plot_filename = f"{category}_{region_dir}_region_comparison"

            # Region comparison plots can use log scale
            use_log_scale = True

            saved_files = self.save_plot_multi_format(
                fig, plot_filename, region, version, output_dir,
                is_log=use_log_scale, data_hists=None, mc_hists=None, signal_hists=None
            )
            # Also save linear version
            ax.set_yscale('linear')
            ax.set_ylim(bottom=0)
            saved_files_linear = self.save_plot_multi_format(
                fig, plot_filename, region, version, output_dir,
                is_log=False, data_hists=None, mc_hists=None, signal_hists=None
            )
            plt.close()
            return saved_files.get('png', '')
        else:
            # Fallback to old method
            plot_file = output_path / f"{region}_region_comparison.{self.format}"
            plt.savefig(plot_file, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            return str(plot_file)

    def _plot_histogram(self, ax, histogram, xlabel: str, ylabel: str, title: str,
                       show_data: bool, plot_type: str = "line") -> None:
        """
        Plot a single histogram.

        Args:
            ax: Matplotlib axis
            histogram: Histogram data
            xlabel: X-axis label
            ylabel: Y-axis label
            title: Plot title
            show_data: Whether to show data points
            plot_type: Type of plot (line, bar, etc.)
        """
        if hasattr(histogram, 'values'):
            # Hist library histogram
            values = histogram.values()
            edges = histogram.axes[0].edges
            centers = histogram.axes[0].centers

            if plot_type == "bar":
                ax.bar(centers, values, width=np.diff(edges), alpha=0.7, color='blue')
            else:
                ax.hist(centers, bins=edges, weights=values, alpha=0.7, color='blue')
        else:
            # Fallback histogram
            values = histogram.get('values', [])
            weights = histogram.get('weights', [])
            bins = histogram.get('bins', [])

            if values and len(values) > 0:
                if plot_type == "bar":
                    ax.bar(bins[:-1], values, width=np.diff(bins), alpha=0.7, color='blue')
                else:
                    ax.hist(values, bins=bins, weights=weights, alpha=0.7, color='blue')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        # Add data points if requested
        if show_data and hasattr(histogram, 'values'):
            # Add data points on top of histogram
            centers = histogram.axes[0].centers
            values = histogram.values()
            ax.errorbar(centers, values, yerr=np.sqrt(values), fmt='o', color='black',
                       markersize=4, capsize=2, label='Data')
            ax.legend()

    def _plot_2d_histogram(self, ax, histogram_x, histogram_y, xlabel: str, ylabel: str, title: str) -> None:
        """
        Plot a 2D histogram.

        Args:
            ax: Matplotlib axis
            histogram_x: X-axis histogram
            histogram_y: Y-axis histogram
            xlabel: X-axis label
            ylabel: Y-axis label
            title: Plot title
        """
        # This is a simplified 2D plot
        # In a real implementation, you would need proper 2D histogram data

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        # Placeholder: create a simple 2D plot
        x = np.linspace(0, 500, 50)
        y = np.linspace(0, 1, 50)
        X, Y = np.meshgrid(x, y)
        Z = np.exp(-(X-250)**2/10000) * np.exp(-(Y-0.5)**2/0.1)

        im = ax.contourf(X, Y, Z, levels=20, cmap='viridis')
        plt.colorbar(im, ax=ax)

    def _combine_histograms(self, hist1, hist2):
        """
        Combine two histograms.

        Args:
            hist1: First histogram
            hist2: Second histogram

        Returns:
            Combined histogram
        """
        # Simple combination - in real implementation, you would need proper histogram arithmetic
        return hist1

    def create_cutflow_plot(self, cutflow: Dict[str, Any], output_path: Path) -> str:
        """
        Create cutflow plot.

        Args:
            cutflow: Cutflow data
            output_path: Output directory

        Returns:
            Plot file path
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Extract cutflow data
        cuts = list(cutflow.keys())
        values = list(cutflow.values())

        # Create bar plot
        bars = ax.bar(range(len(cuts)), values, color='skyblue', alpha=0.7)

        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                   f'{value}', ha='center', va='bottom')

        ax.set_xlabel('Selection Cuts')
        ax.set_ylabel('Number of Events')
        ax.set_title('Event Selection Cutflow')
        ax.set_xticks(range(len(cuts)))
        ax.set_xticklabels(cuts, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_file = output_path / "cutflow.pdf"
        plt.savefig(plot_file, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        return str(plot_file)

    def create_region_summary_plot(self, results: Dict[str, Any],
                                  version: Optional[str] = None,
                                  output_dir: Optional[str] = None) -> str:
        """
        Create region summary plot and save to outputs/plots/{version}/.

        Args:
            results: Analysis results
            version: Version string (required for new structure)
            output_dir: Base output directory (default: "outputs")

        Returns:
            Plot file path
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Region event counts
        region_counts = {}
        for region, region_data in results.get("regions", {}).items():
            region_counts[region] = region_data.get("n_events", 0)

        axes[0, 0].bar(region_counts.keys(), region_counts.values(), color='lightblue', alpha=0.7)
        axes[0, 0].set_xlabel('Region')
        axes[0, 0].set_ylabel('Number of Events')
        axes[0, 0].set_title('Events per Region')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Plot 2: Region efficiency
        total_events = results.get("metadata", {}).get("n_events_processed", 0)
        if total_events > 0:
            efficiencies = {region: count/total_events for region, count in region_counts.items()}
            axes[0, 1].bar(efficiencies.keys(), efficiencies.values(), color='lightgreen', alpha=0.7)
            axes[0, 1].set_xlabel('Region')
            axes[0, 1].set_ylabel('Efficiency')
            axes[0, 1].set_title('Region Efficiency')
            axes[0, 1].tick_params(axis='x', rotation=45)

        # Plot 3: Region overlap (placeholder)
        axes[1, 0].text(0.5, 0.5, 'Region Overlap\n(Placeholder)', ha='center', va='center',
                        transform=axes[1, 0].transAxes, fontsize=14)
        axes[1, 0].set_title('Region Overlap')

        # Plot 4: Summary statistics
        summary_text = f"Total Events: {total_events}\n"
        summary_text += f"Regions: {len(region_counts)}\n"
        summary_text += f"Total Selected: {sum(region_counts.values())}\n"

        axes[1, 1].text(0.1, 0.5, summary_text, ha='left', va='center',
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].set_title('Summary Statistics')

        plt.tight_layout()

        # Save to outputs/plots/{version}/region_summary.pdf
        if version and output_dir:
            summary_dir = os.path.join(output_dir, "plots", version)
            os.makedirs(summary_dir, exist_ok=True)

            # Save all formats (PNG, PDF)
            summary_path_png = os.path.join(summary_dir, "region_summary.png")
            summary_path_pdf = os.path.join(summary_dir, "region_summary.pdf")

            fig.savefig(summary_path_png, dpi=self.dpi, bbox_inches='tight')
            fig.savefig(summary_path_pdf, bbox_inches='tight')

            plt.close()
            logging.info(f"Saved region summary plot to {summary_dir}/")
            return summary_path_pdf
        else:
            # Fallback
            output_path = Path(output_dir or "outputs")
            plot_file = output_path / "region_summary.pdf"
            plt.savefig(plot_file, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            return str(plot_file)

    def create_all_plots(self, results: Dict[str, Any], output_dir: str,
                        show_data: bool = True, regions: Optional[List[str]] = None,
                        version: Optional[str] = None, formats: Optional[List[str]] = None,
                        luminosity: float = 1.0,
                        cross_sections: Optional[Dict[str, float]] = None) -> Dict[str, str]:
        """
        Create all types of plots.

        Args:
            results: Analysis results
            output_dir: Output directory
            show_data: Whether to show data points
            regions: List of regions to plot
            version: Version string for multi-format output
            formats: List of output formats
            luminosity: Integrated luminosity in fb-1 for histogram normalisation
            cross_sections: Dict mapping process name to xsec in pb (optional)

        Returns:
            Dictionary of all plot file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        all_plots = {}

        # Generate version string if not provided (format: YYYYMMDD_HHMM)
        if not version:
            version = datetime.now().strftime("%Y%m%d_%H%M")

        # Create region plots
        region_plots = self.create_region_plots(
            results, output_dir, show_data, regions, version, formats,
            luminosity=luminosity, cross_sections=cross_sections,
        )
        all_plots.update(region_plots)

        # Create cutflow plot — combine event-selection preselection steps with region counts
        combined_cutflow = {}
        if "event_selection_cutflow" in results and results["event_selection_cutflow"]:
            combined_cutflow.update(results["event_selection_cutflow"])
        if "region_cutflow" in results:
            rc = results["region_cutflow"]
            for region_name, rc_data in rc.get("regions", {}).items():
                combined_cutflow[region_name] = int(rc_data.get("n_events", 0))
        if combined_cutflow:
            cutflow_plot = self.create_cutflow_plot(combined_cutflow, output_path)
            all_plots["cutflow"] = cutflow_plot
        elif "cutflow" in results:
            cutflow_plot = self.create_cutflow_plot(results["cutflow"], output_path)
            all_plots["cutflow"] = cutflow_plot

        # Create region summary plot (save directly to version directory)
        region_summary_plot = self.create_region_summary_plot(results, version, output_dir)
        all_plots["region_summary"] = region_summary_plot

        logging.info(f"Created {len(all_plots)} plots in {output_dir}")

        return all_plots
