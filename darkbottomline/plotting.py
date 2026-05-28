"""
Data/MC plotting module for DarkBottomLine framework.
"""

import copy
import math
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

_SENTINEL = -9.0


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


def _extract_branches(objects: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Flatten a nested objects dict from PKL/ROOT to {branch_name: flat_array}."""
    distributions: Dict[str, np.ndarray] = {}
    for key, value in objects.items():
        if key.endswith("_mask"):
            continue
        if not isinstance(value, list):
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
                flat = _flatten_numeric([item.get(field) for item in value if isinstance(item, dict)])
                if flat.size > 0:
                    distributions[f"{key}_{field}"] = flat
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
                        distributions[f"{key}_{field}"] = np.asarray(vals, dtype=float)
                continue
        flat = _flatten_numeric(value)
        if flat.size > 0:
            distributions[key] = flat
    return distributions


def _apply_variable_plot_filter(variable: str, values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    values = values[values != _SENTINEL]
    if variable.lower().endswith("met_pt"):
        return values[values >= 100.0]
    return values


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

        # Plot settings
        self.figsize = self.config.get("figsize", (12, 8))
        self.dpi = self.config.get("dpi", 300)
        self.format = self.config.get("format", "pdf")

        # Variables that should NOT use log scale (multiplicity plots, etc.)
        self.no_log_scale_vars = self.config.get("no_log_scale_vars", [
            'n_jets', 'n_bjets', 'n_muons', 'n_electrons', 'n_taus', 'n_leptons',
            'n_pv', 'pu_npv'
        ])

        # Region-specific exclusions — loaded entirely from plotting.yaml (no hardcoded defaults here)
        self.region_exclusions = self.config.get("region_exclusions", {})

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

        logging.info("Plot manager initialized")

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
            met_spec = self._variable_bins_cfg.get("PFMET_pt")
            if met_spec:
                return np.array(met_spec["edges"], dtype=float)
        if "ctsvalue" in name:
            cts_spec = self._variable_bins_cfg.get("costheta_star")
            if cts_spec:
                return np.array(cts_spec["edges"], dtype=float)
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
            "1b:CR_Wlnu_mu" -> {"category": "1b", "region_dir": "Wlnu_mu"}
            "2b:CR_Top_el" -> {"category": "2b", "region_dir": "Top_el"}
            "1b:CR_Zll_mu" -> {"category": "1b", "region_dir": "Zll_mu"}

        Args:
            region: Region name (e.g., "1b:SR", "2b:CR_Wlnu_mu")

        Returns:
            Dictionary with category and region_dir
        """
        parts = region.split(":")
        if len(parts) != 2:
            # Fallback: use full region name
            return {"category": "unknown", "region_dir": region}

        category = parts[0]  # e.g., "1b" or "2b"
        region_part = parts[1]  # e.g., "SR" or "CR_Wlnu_mu"

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
            For region "1b:CR_Wlnu_mu" and hist_name "met":
            - outputs/plots/20231105_1430/png/1b/Wlnu_mu/met.png
            - outputs/plots/20231105_1430/pdf/1b/Wlnu_mu/met.pdf
            - outputs/plots/20231105_1430/root/met.root
            - outputs/plots/20231105_1430/text/1b/Wlnu_mu/met.txt

        Args:
            fig: Matplotlib figure object
            hist_name: Name of the histogram
            region: Region name (e.g., "1b:SR", "2b:CR_Wlnu_mu")
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
        png_dir = os.path.join(base_output_dir, "plots", version, "png", category, region_dir)
        pdf_dir = os.path.join(base_output_dir, "plots", version, "pdf", category, region_dir)
        root_dir = os.path.join(base_output_dir, "plots", version, "root")
        text_dir = os.path.join(base_output_dir, "plots", version, "text", category, region_dir)

        # Create all directories
        for dir_path in [png_dir, pdf_dir, root_dir, text_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # Determine file suffix for log plots
        suffix = "_log" if is_log else ""

        # Save PNG
        png_path = os.path.join(png_dir, f"{hist_name}{suffix}.png")
        fig.savefig(png_path, dpi=self.dpi, bbox_inches='tight')
        saved_files['png'] = png_path

        # Save PDF
        pdf_path = os.path.join(pdf_dir, f"{hist_name}{suffix}.pdf")
        fig.savefig(pdf_path, bbox_inches='tight')
        saved_files['pdf'] = pdf_path

        # Save ROOT file (if ROOT is available)
        # ROOT files: one file per variable (hist_name contains category_region_variable)
        # Each file contains histograms from this specific region
        try:
            import ROOT
            # hist_name already contains category_region_variable format
            root_path = os.path.join(root_dir, f"{hist_name}{suffix}.root")
            os.makedirs(root_dir, exist_ok=True)
            root_file = ROOT.TFile(root_path, "RECREATE")

            # Create TH1F for each process
            # Note: hist_name already includes category and region, so we use it directly
            if data_hists and hist_name in data_hists:
                data_hist = self._create_th1f_from_hist(data_hists[hist_name], "data_obs", f"Data_{category}_{region_dir}")
                data_hist.Write()

            if mc_hists:
                for process_name, process_hists in mc_hists.items():
                    if hist_name in process_hists:
                        th1f = self._create_th1f_from_hist(process_hists[hist_name], process_name, f"{process_name}_{category}_{region_dir}")
                        th1f.Write()

            if signal_hists and hist_name in signal_hists:
                signal_hist = self._create_th1f_from_hist(signal_hists[hist_name], "signal", f"Signal_{category}_{region_dir}")
                signal_hist.Write()

            root_file.Close()
            saved_files['root'] = root_path

        except ImportError:
            logging.warning("ROOT not available, skipping ROOT file creation")
            saved_files['root'] = None

        # Save yield text file
        txt_path = os.path.join(text_dir, f"{hist_name}{suffix}.txt")
        self._save_yield_text_file(txt_path, hist_name, data_hists, mc_hists, signal_hists)
        saved_files['txt'] = txt_path

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

    def _load_pkl_files(self, paths: List[Path]) -> Dict[str, Any]:
        merged: Dict[str, Any] = {"weighted_total_events": 0, "objects": {}}
        for p in paths:
            try:
                with open(p, "rb") as fh:
                    data = pickle.load(fh)
                if not isinstance(data, dict):
                    continue
                wte = int(data.get("weighted_total_events", 0) or 0)
                merged["weighted_total_events"] += wte
                objs = data.get("objects", {})
                if isinstance(objs, dict):
                    for k, v in objs.items():
                        if k in merged["objects"]:
                            if isinstance(merged["objects"][k], list) and isinstance(v, list):
                                merged["objects"][k] = merged["objects"][k] + v
                            elif isinstance(merged["objects"][k], np.ndarray) and isinstance(v, np.ndarray):
                                merged["objects"][k] = np.concatenate([merged["objects"][k], v])
                        else:
                            merged["objects"][k] = v
            except Exception as exc:
                logging.warning("Could not load %s: %s", p.name, exc)
        return merged

    def _load_root_files(self, paths: List[Path]) -> Dict[str, Any]:
        try:
            import uproot
        except ImportError:
            raise ImportError("uproot required for ROOT file loading")
        merged: Dict[str, Any] = {"weighted_total_events": 0, "objects": {}}
        for p in paths:
            try:
                with uproot.open(str(p)) as f:
                    if "Events" not in f:
                        continue
                    tree = f["Events"]
                    objs: Dict[str, Any] = {}
                    for branch in tree.keys():
                        try:
                            arr = tree[branch].array(library="np")
                            objs[branch] = arr.tolist() if hasattr(arr, "tolist") else list(arr)
                        except Exception:
                            pass
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
            except Exception as exc:
                logging.warning("Could not load ROOT %s: %s", p.name, exc)
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
            rows.append(("data_obs", data_ndarray, np.sqrt(data_ndarray)))

        col_w = 20
        txt_path = stem.with_suffix(".txt")
        with open(txt_path, "w") as fh:
            header = f"{'Sample':<24}" + "".join(f"{b:>{col_w}}" for b in bin_labels)
            fh.write(header + "\n" + "-" * len(header) + "\n")
            for label, vals, sumw2 in rows:
                cells = "".join(
                    f"{f'{vals[i]:.2f}±{np.sqrt(sumw2[i]):.2f}':>{col_w}}" for i in range(n_bins)
                )
                fh.write(f"{label:<24}{cells}\n")

        tex_path = stem.with_suffix(".tex")
        with open(tex_path, "w") as fh:
            col_spec = "l" + "c" * n_bins
            bin_hdr = " & ".join(f"\\textbf{{{b}}}" for b in bin_labels)
            fh.write("\\begin{table}[htbp]\n\\centering\n")
            fh.write(f"\\caption{{Yield table for {variable}}}\n\\label{{tab:{variable}}}\n")
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
    ) -> List[str]:
        """Draw CMS-style stacked histogram with ratio panel and save in 5 formats."""
        if not _HAS_MPLHEP:
            logging.warning("mplhep not available — skipping stacked plot for %s", variable)
            return []

        # Sort backgrounds ascending by integral (smallest at bottom)
        rows = sorted(background_rows, key=lambda t: float(np.sum(t[1])))
        # Fall back to process-config color map for labels not in yaml group overrides
        color_map = get_background_color_map([r[0] for r in rows])

        show_ratio = data_ndarray is not None
        if show_ratio:
            fig, (ax, ax_ratio) = plt.subplots(
                2, 1, figsize=(12, 12), sharex=True,
                gridspec_kw={"height_ratios": [3.0, 1.0], "hspace": 0.08},
            )
            fig.subplots_adjust(top=0.92, bottom=0.09, left=0.14, right=0.95)
        else:
            fig, ax = plt.subplots(figsize=(12, 10))
            ax_ratio = None
            fig.subplots_adjust(top=0.92, bottom=0.12, left=0.14, right=0.95)

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
            hatch="////", facecolor="#bbbbbb", edgecolor="#666666",
            linewidth=0.0, alpha=0.8, zorder=5,
        )

        if data_ndarray is not None:
            half_width = 0.5 * (bins[1:] - bins[:-1])
            mask = data_ndarray > 0
            if np.any(mask):
                ax.errorbar(
                    centers[mask], data_ndarray[mask],
                    xerr=half_width[mask], yerr=np.sqrt(data_ndarray[mask]),
                    fmt="o", color="black", markerfacecolor="black",
                    markeredgecolor="black", markersize=5.5,
                    elinewidth=1.2, capsize=0, label="Data", zorder=10,
                )

        use_log = variable not in self.no_log_scale_vars
        if use_log:
            ax.set_yscale("log")
            stacked_max = float(np.max(cumulative)) if cumulative.size else 0.0
            data_max = float(np.max(data_ndarray)) if data_ndarray is not None and data_ndarray.size else 0.0
            ax.set_ylim(0.1, max(stacked_max, data_max, 1e-3) * 1000.0)

        nonzero_mc = np.where(cumulative > 0)[0]
        x_lo = float(bins[nonzero_mc[0]]) if nonzero_mc.size else float(bins[0])
        x_hi = float(bins[nonzero_mc[-1] + 1]) if nonzero_mc.size else float(bins[-1])
        if data_ndarray is not None:
            nz_d = np.where(data_ndarray > 0)[0]
            if nz_d.size:
                x_lo = min(x_lo, float(bins[nz_d[0]]))
                x_hi = max(x_hi, float(bins[nz_d[-1] + 1]))
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylabel("Events / bin", fontsize=22, labelpad=6)
        if not show_ratio:
            ax.set_xlabel(variable, fontsize=22)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=8, steps=[1, 2, 5, 10]))
        ax.grid(False)

        if show_ratio and ax_ratio is not None and data_ndarray is not None:
            pred = cumulative
            data_ratio = np.divide(
                data_ndarray, pred,
                out=np.full_like(data_ndarray, np.nan, dtype=float), where=pred > 0,
            )
            data_ratio_err = np.divide(
                np.sqrt(data_ndarray), pred,
                out=np.zeros_like(data_ndarray, dtype=float), where=pred > 0,
            )
            pred_rel_err = np.divide(
                mc_stat_err, pred,
                out=np.zeros_like(pred, dtype=float), where=pred > 0,
            )
            ratio_mask = np.isfinite(data_ratio)
            ax_ratio.axhline(1.0, color="black", linestyle="-", linewidth=1.2)
            ratio_lo = np.append(1.0 - pred_rel_err, (1.0 - pred_rel_err)[-1])
            ratio_hi = np.append(1.0 + pred_rel_err, (1.0 + pred_rel_err)[-1])
            ax_ratio.fill_between(
                unc_x, ratio_lo, ratio_hi, step="post",
                hatch="////", facecolor="#bbbbbb", edgecolor="#3d3d3d",
                linewidth=0.0, alpha=0.8, label="Stat. unc.", zorder=5,
            )
            if np.any(ratio_mask):
                half_width = 0.5 * (bins[1:] - bins[:-1])
                ax_ratio.errorbar(
                    centers[ratio_mask], data_ratio[ratio_mask],
                    xerr=half_width[ratio_mask], yerr=data_ratio_err[ratio_mask],
                    fmt="o", color="black", markerfacecolor="black",
                    markeredgecolor="black", markersize=5.5,
                    elinewidth=1.0, capsize=0, zorder=10,
                )
            ax_ratio.legend(loc="upper right", fontsize=20, frameon=False)
            ax_ratio.set_ylabel("Data / MC", fontsize=22, labelpad=6)
            ax_ratio.set_xlabel(variable, fontsize=22, labelpad=8)
            ax_ratio.set_ylim(0, 2.0)
            ax_ratio.set_xlim(x_lo, x_hi)
            ax_ratio.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=8, steps=[1, 2, 5, 10]))
            ax_ratio.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0, 0.5, 1.0, 1.5, 2.0]))
            ax_ratio.grid(False)

        hep.cms.label(
            "Work in progress",
            data=data_ndarray is not None,
            lumi=round(luminosity, 2),
            com=13.6,
            loc=0,
            ax=ax,
        )

        handles, labels_leg = ax.get_legend_handles_labels()
        if handles:
            data_idx = next((i for i, l in enumerate(labels_leg) if l == "Data"), None)
            unc_idx = next((i for i, l in enumerate(labels_leg) if "Uncertainty" in l or "unc" in l.lower()), None)
            ordered = (
                ([data_idx] if data_idx is not None else [])
                + [i for i in range(len(labels_leg)) if i not in (data_idx, unc_idx)]
                + ([unc_idx] if unc_idx is not None else [])
            )
            handles = [handles[i] for i in ordered]
            labels_leg = [labels_leg[i] for i in ordered]
            if unc_idx is not None:
                handles[-1] = matplotlib.patches.Patch(
                    hatch="////", facecolor="#bbbbbb", edgecolor="#3d3d3d",
                    linewidth=0.0, label="Uncertainty",
                )
            ax.legend(
                handles, labels_leg,
                loc="upper right", bbox_to_anchor=(0.97, 0.97),
                ncol=2, frameon=False, borderaxespad=0.0,
                handlelength=1.5, columnspacing=1.0,
                handletextpad=0.5, fontsize=20,
            )

        saved = self.save_plot_multi_format(
            fig, variable, region, version, output_dir,
            is_log=use_log,
        )
        plt.close(fig)

        # Write yield tables alongside the PNG
        region_info = self._parse_region_name(region)
        category = region_info["category"]
        region_dir = region_info["region_dir"]
        text_dir = Path(output_dir) / "plots" / version / "text" / category / region_dir
        text_dir.mkdir(parents=True, exist_ok=True)
        self._write_yield_table(text_dir / variable, variable, bins, rows, data_ndarray)

        return [v for v in saved.values() if v]

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
            ))
        return created

    def _load_one_file(self, path: Path) -> Dict[str, Any]:
        """Load a single ROOT or PKL file. Returns {"weighted_total_events": int, "objects": dict}."""
        if path.suffix == ".root":
            return self._load_root_files([path])
        return self._load_pkl_files([path])

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
        for pattern in patterns:
            for p in all_files:
                if p in matched_paths:
                    continue
                if pattern in p.name:
                    resolved.append(p)
                    matched_paths.add(p)
        if not resolved:
            logging.warning("No files matched patterns %s in %s", patterns, input_folder)
        return resolved

    def _load_group_entries(
        self,
        input_folder: str,
        groups: Dict[str, List[str]],
        cross_sections: Dict[str, float],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Load files for each process group. Returns {label: [{"weighted_total_events", "branches", "xsec"}]}."""
        result: Dict[str, List[Dict[str, Any]]] = {}
        for label, patterns in groups.items():
            paths = self._resolve_group_files(input_folder, patterns)
            entries = []
            for p in paths:
                xsec = cross_sections.get(p.stem) or cross_sections.get(p.name)
                try:
                    loaded = self._load_one_file(p)
                    entries.append({
                        "weighted_total_events": loaded["weighted_total_events"],
                        "branches": _extract_branches(loaded["objects"]),
                        "xsec": xsec,
                    })
                except Exception as exc:
                    logging.warning("Skipping %s: %s", p, exc)
            if entries:
                result[label] = entries
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
        bkg_groups  = self._load_group_entries(input_folder, process_groups, cross_sections)
        sig_groups  = self._load_group_entries(input_folder, signal_groups,  cross_sections)
        dat_groups  = self._load_group_entries(input_folder, data_groups,    {})  # no xsec for data

        # Merge all data entries into one flat branch dict (raw counts, no normalisation)
        data_branches: Optional[Dict[str, np.ndarray]] = None
        if dat_groups:
            merged: Dict[str, np.ndarray] = {}
            for entries in dat_groups.values():
                for e in entries:
                    for k, v in e["branches"].items():
                        merged[k] = np.concatenate([merged[k], v]) if k in merged else v
            data_branches = merged

        all_vars: List[str] = variables or sorted(
            {k for entries in bkg_groups.values() for e in entries for k in e["branches"]}
        )
        created: List[str] = []
        for var in all_vars:
            all_vals = [
                _apply_variable_plot_filter(var, e["branches"].get(var, np.array([])))
                for entries in bkg_groups.values() for e in entries
            ]
            if data_branches is not None:
                all_vals.append(_apply_variable_plot_filter(var, data_branches.get(var, np.array([]))))

            bins = _make_bins(all_vals, self._build_bins_from_config, var, self._n_bins_default)
            if bins is None or len(bins) < 2:
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

            if np.allclose(sum(h for _, h, _ in bkg_rows), 0.0):
                continue

            data_hist: Optional[np.ndarray] = None
            if data_branches is not None:
                dv = _apply_variable_plot_filter(var, data_branches.get(var, np.array([])))
                data_hist = _histogram_counts(dv, bins)

            files = self._plot_stacked_variable(
                variable=var, bins=bins,
                background_rows=bkg_rows, data_ndarray=data_hist,
                output_dir=output_dir, luminosity=luminosity, year=year,
                region="event_selection", version=version, save_root=save_root,
            )
            created.extend(files)
            logging.info("Created event-selection plot: %s (%d files)", var, len(files))

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
        try:
            import hist as hist_lib
            _HAS_HIST = True
        except ImportError:
            _HAS_HIST = False

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
                xsec = cross_sections.get(p.stem) or cross_sections.get(p.name)
                try:
                    entries.append({"data": _load_region_pkl(p), "xsec": xsec})
                except Exception as exc:
                    logging.warning("Skipping %s: %s", p, exc)
            if entries:
                bkg_groups[proc_label] = entries

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
        for region in all_regions:
            excl = self._get_excluded_variables_for_region(region)
            all_vars_set: set = set()
            for entries in bkg_groups.values():
                for e in entries:
                    all_vars_set.update(e["data"].get("region_histograms", {}).get(region, {}).keys())
            all_vars_for_region = [v for v in (variables or sorted(all_vars_set)) if v not in excl]

            for var in all_vars_for_region:
                bins_ref: Optional[np.ndarray] = self._build_bins_from_config(var)
                bkg_rows: List[Tuple[str, np.ndarray, np.ndarray]] = []

                for proc_label, entries in bkg_groups.items():
                    group_hv: Optional[np.ndarray] = None
                    group_hs: Optional[np.ndarray] = None
                    for e in entries:
                        rh = e["data"].get("region_histograms", {}).get(region, {})
                        h = rh.get(var)
                        if h is None:
                            continue
                        wte = int(e["data"].get("metadata", {}).get("weighted_total_events", 0)
                                   or e["data"].get("weighted_total_events", 0) or 0)
                        edges, hv, hs = _h_to_numpy(h)
                        if hv is None or hv.size == 0:
                            continue
                        if bins_ref is None and edges is not None:
                            bins_ref = edges
                        scale = ((luminosity * e["xsec"] * 1000.0) / wte
                                 if e["xsec"] is not None and wte > 0
                                 else (luminosity / wte if wte > 0 else 1.0))
                        group_hv = hv * scale if group_hv is None else group_hv + hv * scale
                        group_hs = hs * scale**2 if group_hs is None else group_hs + hs * scale**2

                    if group_hv is not None and group_hv.size > 0:
                        bkg_rows.append((proc_label, group_hv, group_hs))

                if not bkg_rows or bins_ref is None:
                    continue
                if np.allclose(sum(h for _, h, _ in bkg_rows), 0.0):
                    continue

                data_hist = _data_hist_for_region(region, var)

                files = self._plot_stacked_variable(
                    variable=var, bins=bins_ref,
                    background_rows=bkg_rows, data_ndarray=data_hist,
                    output_dir=output_dir, luminosity=luminosity, year=year,
                    region=region, version=version, save_root=save_root,
                )
                created.extend(files)
                logging.info("Created region plot: %s / %s (%d files)", region, var, len(files))

        return created

    def _create_region_from_events_plots(
        self,
        input_folder: str,
        process_groups: Dict[str, List[str]],
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
    ) -> List[str]:
        """Load event-selected ROOT/PKL files, apply region cuts in-memory, produce stacked region plots.

        For each process group:
          1. Load Events TTree + weighted_total_events from each matched file.
          2. Apply region masks via RegionManager.
          3. Fill per-variable histograms with full_event_weight (nominal) or weight_systematic.
          4. Scale by lumi * xsec * 1000 / wte and stack across groups.
        """
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

        # ---- helpers ----

        def _load_events_root(path: Path) -> Optional[Dict[str, Any]]:
            """Return {"branches": {name: np.ndarray}, "wte": float} or None on failure."""
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
                        logging.warning("No Events tree in %s", path.name)
                        return None
                    branches = f["Events"].arrays(library="np")
                    return {"branches": dict(branches), "wte": wte}
            except Exception as exc:
                logging.warning("Could not load %s: %s", path.name, exc)
                return None

        def _load_events_pkl(path: Path) -> Optional[Dict[str, Any]]:
            try:
                with open(path, "rb") as fh:
                    d = pickle.load(fh)
                branches = {k: np.asarray(v) for k, v in d.get("branches", {}).items()
                            if isinstance(v, (np.ndarray, list))}
                wte = float(d.get("weighted_total_events", 0.0))
                return {"branches": branches, "wte": wte}
            except Exception as exc:
                logging.warning("Could not load %s: %s", path.name, exc)
                return None

        def _load_file(path: Path) -> Optional[Dict[str, Any]]:
            if path.suffix == ".root":
                return _load_events_root(path)
            return _load_events_pkl(path)

        # ---- load per-group ----
        # {proc_label: [{"branches": dict, "wte": float, "xsec": float|None}]}
        bkg_entries: Dict[str, List[Dict[str, Any]]] = {}
        for proc_label, patterns in process_groups.items():
            paths = self._resolve_group_files(input_folder, patterns)
            entries = []
            for p in paths:
                loaded = _load_file(p)
                if loaded is None:
                    continue
                xsec = cross_sections.get(p.stem) or cross_sections.get(p.name)
                entries.append({
                    "branches": loaded["branches"],
                    "wte": loaded["wte"],
                    "xsec": xsec,
                })
            if entries:
                bkg_entries[proc_label] = entries

        # Data groups (no xsec)
        raw_data_cfg = self.config.get("process_groups", {})
        data_entries: Dict[str, Dict[str, Any]] = {}
        for label, patterns in data_groups.items():
            grp_cfg = raw_data_cfg.get(label, {})
            region_patterns: List[str] = grp_cfg.get("regions", []) if isinstance(grp_cfg, dict) else []
            paths = self._resolve_group_files(input_folder, patterns)
            loaded_list = [e for p in paths for e in [_load_file(p)] if e is not None]
            if loaded_list:
                data_entries[label] = {"entries": loaded_list, "region_patterns": region_patterns}

        if not bkg_entries:
            logging.warning("region-from-events: no background files loaded — nothing to plot")
            return []

        # ---- per-region processing ----
        created: List[str] = []

        for region_name in target_regions:
            region_obj = region_manager.regions.get(region_name)
            if region_obj is None:
                logging.warning("Region %s not found in %s", region_name, regions_config)
                continue

            excl = self._get_excluded_variables_for_region(region_name)

            # Determine variable list from first available group
            if variables:
                var_list = [v for v in variables if v not in excl]
            else:
                all_branches: set = set()
                for entries in bkg_entries.values():
                    for e in entries:
                        all_branches.update(e["branches"].keys())
                # Exclude weight branches and internal branches from plot list
                _weight_prefixes = ("weight_", "full_event_weight", "genWeight")
                var_list = sorted(
                    b for b in all_branches
                    if not any(b.startswith(p) for p in _weight_prefixes)
                    and b not in excl
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

                        # Apply region mask using RegionManager
                        try:
                            import awkward as _ak
                            # Build minimal ak.Array from flat branches for region cuts
                            events_ak = _ak.Array({k: v for k, v in br.items()
                                                   if isinstance(v, np.ndarray) and v.ndim == 1})
                            mask = region_obj.apply_cuts(events_ak, objects={})
                        except Exception as _mask_exc:
                            logging.warning(
                                "Region mask failed for %s / %s: %s", region_name, proc_label, _mask_exc
                            )
                            continue

                        mask_np = np.asarray(mask, dtype=bool)
                        vals_raw = br.get(var)
                        if vals_raw is None:
                            continue
                        vals = _apply_variable_plot_filter(var, vals_raw[mask_np])
                        if vals.size == 0:
                            continue

                        w_arr = br.get(weight_branch)
                        if w_arr is not None:
                            w = np.asarray(w_arr, dtype=float)[mask_np]
                        else:
                            w = np.ones(mask_np.sum(), dtype=float)

                        scale = ((luminosity * xsec * 1000.0) / wte
                                 if xsec is not None and wte > 0
                                 else (luminosity / wte if wte > 0 else 1.0))

                        if bins_ref is None:
                            all_vals_for_bins = [
                                _apply_variable_plot_filter(var, e2["branches"].get(var, np.array([])))
                                for ee in bkg_entries.values() for e2 in ee
                            ]
                            bins_ref = _make_bins(all_vals_for_bins, self._build_bins_from_config,
                                                  var, self._n_bins_default)
                            if bins_ref is None or len(bins_ref) < 2:
                                break

                        hv, _ = np.histogram(vals, bins=bins_ref, weights=w * scale)
                        hs, _ = np.histogram(vals, bins=bins_ref, weights=(w * scale) ** 2)

                        group_hv = hv if group_hv is None else group_hv + hv
                        group_hs = hs if group_hs is None else group_hs + hs

                    if group_hv is not None and group_hv.size > 0:
                        bkg_rows.append((proc_label, group_hv, group_hs))

                if not bkg_rows or bins_ref is None:
                    continue
                if np.allclose(sum(h for _, h, _ in bkg_rows), 0.0):
                    continue

                # Data histogram (no xsec, no region mask applied — data already region-selected)
                data_hist: Optional[np.ndarray] = None
                for label, info in data_entries.items():
                    rp = info["region_patterns"]
                    if rp and not any(pat in region_name for pat in rp):
                        continue
                    for e in info["entries"]:
                        import awkward as _ak
                        br = e["branches"]
                        try:
                            events_ak = _ak.Array({k: v for k, v in br.items()
                                                   if isinstance(v, np.ndarray) and v.ndim == 1})
                            mask_d = region_obj.apply_cuts(events_ak, objects={})
                            mask_d_np = np.asarray(mask_d, dtype=bool)
                        except Exception:
                            mask_d_np = np.ones(len(next(iter(br.values()))), dtype=bool)
                        dv = _apply_variable_plot_filter(var, br.get(var, np.array([]))[mask_d_np])
                        if dv.size == 0:
                            continue
                        dh, _ = np.histogram(dv, bins=bins_ref)
                        data_hist = dh if data_hist is None else data_hist + dh

                syst_label = f" [{weight_systematic}]" if weight_systematic else ""
                files = self._plot_stacked_variable(
                    variable=var, bins=bins_ref,
                    background_rows=bkg_rows, data_ndarray=data_hist,
                    output_dir=output_dir, luminosity=luminosity, year=year,
                    region=region_name, version=version, save_root=False,
                )
                created.extend(files)
                logging.info(
                    "region-from-events: %s / %s%s (%d files)",
                    region_name, var, syst_label, len(files),
                )

        return created

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

        for region in regions:
            logging.info(f"Creating plots for region {region}")

            # Create individual variable plots - one plot per variable
            individual_plots = self._create_individual_variable_plots(
                results, region, output_dir, show_data, version, formats,
                hist_scale=hist_scale,
            )

            # Also create grouped plots (kinematic, multiplicity, dnn, region_comparison)
            output_path = Path(output_dir)
            grouped_plots = self._create_region_plots_single(
                results, region, output_path, show_data, version, output_dir
            )

            # Combine both types of plots
            all_region_plots = {**individual_plots, **grouped_plots}
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
            region: Region name (e.g., "1b:SR", "2b:CR_Top_mu")

        Returns:
            List of variable name patterns to exclude
        """
        region_info = self._parse_region_name(region)
        category = region_info["category"]
        region_type = region_info["region_dir"]
        is_sr = "SR" in region_type
        is_cr = "CR" in region or "Top" in region_type or "Wlnu" in region_type or "Zll" in region_type

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

        # Get excluded variables for this region
        excluded_vars = self._get_excluded_variables_for_region(region)

        # Get list of variables to plot (exclude internal/system variables and region-specific exclusions)
        variables_to_plot = []
        exclude_patterns = ['_dnn_score', '_region_variables']
        for var_name in region_histograms.keys():
            # Check if variable should be excluded
            should_exclude = False

            # Check against exclude patterns
            if any(exclude in var_name for exclude in exclude_patterns):
                should_exclude = True

            # Check against region-specific exclusions
            if any(excluded_var in var_name for excluded_var in excluded_vars):
                should_exclude = True

            if not should_exclude:
                variables_to_plot.append(var_name)

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
        # Simple mapping - can be expanded
        labels = {
            'met': 'MET [GeV]',
            'jet_pt': 'Jet pT [GeV]',
            'jet_eta': 'Jet η',
            'n_jets': 'Number of Jets',
            'n_bjets': 'Number of B-jets',
            'n_muons': 'Number of Muons',
            'n_electrons': 'Number of Electrons',
            'electron_pt': 'Electron pT [GeV]',
            'electron_eta': 'Electron η',
            'muon_pt': 'Muon pT [GeV]',
            'muon_eta': 'Muon η',
        }
        return labels.get(var_name, var_name.replace('_', ' ').title())

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
        # Get excluded variables for this region
        excluded_vars = self._get_excluded_variables_for_region(region)

        # Determine which plots to include based on region
        plot_vars = []

        # Plot MET
        if 'met' in histograms:
            plot_vars.append(('met', "MET [GeV]", "Missing Transverse Energy"))

        # Plot jet pT (jet1)
        if 'jet1_pt' in histograms:
            plot_vars.append(('jet1_pt', "Jet1 pT [GeV]", "Leading Jet pT"))
        elif 'jet_pt' in histograms:
            plot_vars.append(('jet_pt', "Jet pT [GeV]", "Jet Transverse Momentum"))

        # Plot jet eta
        if 'jet1_eta' in histograms:
            plot_vars.append(('jet1_eta', "Jet1 η", "Leading Jet Pseudorapidity"))
        elif 'jet_eta' in histograms:
            plot_vars.append(('jet_eta', "Jet η", "Jet Pseudorapidity"))

        # Plot b-tag score
        if 'btag_deepjet' in histograms:
            plot_vars.append(('btag_deepjet', "DeepJet Score", "B-tagging Discriminant"))

        # Plot jet2 pT (if available and not excluded)
        if 'jet2_pt' in histograms and not any('jet2' in var for var in excluded_vars):
            plot_vars.append(('jet2_pt', "Jet2 pT [GeV]", "Subleading Jet pT"))

        # Plot jet3 pT (only for 2b regions, not for 1b)
        if 'jet3_pt' in histograms and not any('jet3' in var for var in excluded_vars):
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
        # Get excluded variables for this region
        excluded_vars = self._get_excluded_variables_for_region(region)

        # Determine which plots to include based on region
        plot_indices = []
        plot_vars = []

        # Plot jet multiplicity
        if 'n_jets' in histograms:
            plot_indices.append(0)
            plot_vars.append(('n_jets', "Number of Jets", "Jet Multiplicity"))

        # Plot b-jet multiplicity
        if 'n_bjets' in histograms:
            plot_indices.append(1)
            plot_vars.append(('n_bjets', "Number of B-jets", "B-jet Multiplicity"))

        # Plot muon multiplicity (only for CRs)
        if 'n_muons' in histograms and not any('n_muons' in var for var in excluded_vars):
            plot_indices.append(2)
            plot_vars.append(('n_muons', "Number of Muons", "Muon Multiplicity"))

        # Plot electron multiplicity (only for CRs)
        if 'n_electrons' in histograms and not any('n_electrons' in var for var in excluded_vars):
            plot_indices.append(3)
            plot_vars.append(('n_electrons', "Number of Electrons", "Electron Multiplicity"))

        # Plot tau multiplicity
        if 'n_taus' in histograms:
            plot_indices.append(4)
            plot_vars.append(('n_taus', "Number of Taus", "Tau Multiplicity"))

        # Plot lepton multiplicity (only for CRs)
        if 'n_leptons' in histograms and not any('n_leptons' in var for var in excluded_vars):
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
