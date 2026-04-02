"""
Data/MC plotting module for DarkBottomLine framework.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for batch mode
import matplotlib.pyplot as plt
import numpy as np
import awkward as ak
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import yaml
import os
from datetime import datetime

from .utils.plot_utils import CMSPlotStyle, get_process_colors, get_process_labels


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

        # Region-specific exclusions from config
        # Default exclusions: z_mass and z_pt from Top and W CRs
        default_exclusions = {
            "Top": ["z_mass", "z_pt"],
            "Wlnu": ["z_mass", "z_pt"],
            "Wenu": ["z_mass", "z_pt"],
            "Wmunu": ["z_mass", "z_pt"],
        }
        self.region_exclusions = self.config.get("region_exclusions", {})
        # Merge with defaults (user config overrides defaults)
        for key, value in default_exclusions.items():
            if key not in self.region_exclusions:
                self.region_exclusions[key] = value
            else:
                # Merge lists, avoiding duplicates
                self.region_exclusions[key] = list(set(self.region_exclusions[key] + value))

        logging.info("Plot manager initialized")

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
                          version: Optional[str] = None, formats: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Create plots for all regions.

        Args:
            results: Analysis results dictionary
            output_dir: Output directory for plots
            show_data: Whether to show data points
            regions: List of regions to plot (None for all)
            version: Version string for multi-format output
            formats: List of output formats

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

        for region in regions:
            logging.info(f"Creating plots for region {region}")

            # Create individual variable plots - one plot per variable
            individual_plots = self._create_individual_variable_plots(
                results, region, output_dir, show_data, version, formats
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

        # Jet3 variables to exclude
        jet3_vars = [
            'jet3_pt', 'jet3_eta', 'jet3_phi', 'jet3_btag',
            'm_jet1jet3', 'deta_jet13', 'dphi_jet13', 'isjet2EtaMatch'
        ]

        # Lepton variables to exclude (z_mass and z_pt excluded from SRs, and separately from Top/W CRs)
        lepton_vars = [
            'muon_pt', 'muon_eta', 'muon_phi', 'muon_iso',
            'electron_pt', 'electron_eta', 'electron_phi', 'electron_iso',
            'lep1_pt', 'lep1_eta', 'lep1_phi', 'lep1_iso',
            'lep2_pt', 'lep2_eta', 'lep2_phi', 'lep2_iso',
            'dphi_lep1_met', 'dphi_lep2_met',
            'w_mass', 'w_pt', 'z_mass', 'z_pt', 'mll', 'mt',  # Include z_mass and z_pt for SR exclusion
            'n_muons', 'n_electrons', 'n_leptons',  # Multiplicity for SR
            'dr_muon_jet', 'dr_electron_jet'
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
                                          version: str, formats: Optional[List[str]] = None) -> Dict[str, str]:
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

        # Create one plot per variable
        for var_name in variables_to_plot:
            try:
                hist_data = region_histograms.get(var_name)
                if hist_data is None:
                    continue

                # Create figure with ratio panel
                fig, (ax_main, ax_ratio) = plt.subplots(
                    2, 1, figsize=(10, 10),
                    gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05}
                )

                # Plot histogram on main axis
                self._plot_single_histogram(ax_main, hist_data, var_name, show_data)

                # Ratio panel (placeholder for now)
                ax_ratio.set_xlabel(self._get_variable_label(var_name))
                ax_ratio.set_ylabel("Data/MC")
                ax_ratio.grid(True, alpha=0.3)

                # Determine if this variable should use log scale
                use_log_scale = var_name not in self.no_log_scale_vars

                # Parse region to get category and region_dir for filename
                region_info = self._parse_region_name(region)
                category = region_info["category"]
                region_dir = region_info["region_dir"]

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

                plot_files[var_name] = saved_files.get('png', '')

            except Exception as e:
                logging.warning(f"Failed to create plot for {var_name} in {region}: {e}")

        return plot_files

    def _plot_single_histogram(self, ax, hist_data: Any, var_name: str, show_data: bool):
        """Plot a single histogram on the given axis."""
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
                        version: Optional[str] = None, formats: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Create all types of plots.

        Args:
            results: Analysis results
            output_dir: Output directory
            show_data: Whether to show data points
            regions: List of regions to plot
            version: Version string for multi-format output
            formats: List of output formats

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
        region_plots = self.create_region_plots(results, output_dir, show_data, regions, version, formats)
        all_plots.update(region_plots)

        # Create cutflow plot
        if "cutflow" in results:
            cutflow_plot = self.create_cutflow_plot(results["cutflow"], output_path)
            all_plots["cutflow"] = cutflow_plot

        # Create region summary plot (save directly to version directory)
        region_summary_plot = self.create_region_summary_plot(results, version, output_dir)
        all_plots["region_summary"] = region_summary_plot

        logging.info(f"Created {len(all_plots)} plots in {output_dir}")

        return all_plots
