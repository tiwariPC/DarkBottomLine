"""
Post-fit diagnostic plotting for DarkBottomLine framework.
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import uproot
# import ROOT  # Optional: ROOT not available

from utils.plot_utils import CMSPlotStyle, add_cms_label, create_subplot_grid


class DiagnosticPlotter:
    """
    Creates post-fit diagnostic plots from Combine results.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize diagnostic plotter.

        Args:
            config: Plotting configuration dictionary
        """
        self.config = config or {}

        # Initialize plot style
        self.style = CMSPlotStyle()
        self.style.set_style()

        # Plot settings
        self.figsize = self.config.get("figsize", (12, 8))
        self.dpi = self.config.get("dpi", 300)
        self.format = self.config.get("format", "pdf")

        logging.info("DiagnosticPlotter initialized")

    def plot_impacts(self, input_file: str, output_dir: str,
                    max_parameters: int = 20) -> str:
        """
        Create impact plot from fitDiagnostics.

        Args:
            input_file: Path to fitDiagnostics.root or impacts.json
            output_dir: Output directory
            max_parameters: Maximum number of parameters to show

        Returns:
            Path to impact plot file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load impacts data
        impacts_data = self._load_impacts_data(input_file)

        # Create impact plot
        fig, ax = plt.subplots(1, 1, figsize=(10, max(6, len(impacts_data) * 0.3)))

        # Sort parameters by impact magnitude
        sorted_params = sorted(impacts_data.items(), key=lambda x: abs(x[1]), reverse=True)
        sorted_params = sorted_params[:max_parameters]

        # Extract data
        param_names = [param[0] for param in sorted_params]
        impact_values = [param[1] for param in sorted_params]

        # Create horizontal bar plot
        y_pos = np.arange(len(param_names))
        colors = ['red' if x < 0 else 'blue' for x in impact_values]

        bars = ax.barh(y_pos, impact_values, color=colors, alpha=0.7)

        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, impact_values)):
            ax.text(value + (0.01 if value >= 0 else -0.01), bar.get_y() + bar.get_height()/2,
                   f'{value:.3f}', ha='left' if value >= 0 else 'right', va='center')

        # Customize plot
        ax.set_yticks(y_pos)
        ax.set_yticklabels(param_names)
        ax.set_xlabel('Impact on Signal Strength')
        ax.set_title('Nuisance Parameter Impacts')
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)

        # Add CMS label
        add_cms_label(ax, year="2023", lumi=35.9)

        plt.tight_layout()

        # Save plot
        plot_file = output_path / "impacts.pdf"
        plt.savefig(plot_file, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        logging.info(f"Impact plot saved to {plot_file}")
        return str(plot_file)

    def plot_pulls(self, input_file: str, output_dir: str,
                  max_parameters: int = 30) -> str:
        """
        Create pull plot from fitDiagnostics.

        Args:
            input_file: Path to fitDiagnostics.root
            output_dir: Output directory
            max_parameters: Maximum number of parameters to show

        Returns:
            Path to pull plot file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load pulls data
        pulls_data = self._load_pulls_data(input_file)

        # Create pull plot
        fig, ax = plt.subplots(1, 1, figsize=(10, max(6, len(pulls_data) * 0.3)))

        # Sort parameters by pull magnitude
        sorted_params = sorted(pulls_data.items(), key=lambda x: abs(x[1]), reverse=True)
        sorted_params = sorted_params[:max_parameters]

        # Extract data
        param_names = [param[0] for param in sorted_params]
        pull_values = [param[1] for param in sorted_params]

        # Create horizontal bar plot
        y_pos = np.arange(len(param_names))
        colors = ['red' if abs(x) > 2 else 'orange' if abs(x) > 1 else 'green' for x in pull_values]

        bars = ax.barh(y_pos, pull_values, color=colors, alpha=0.7)

        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, pull_values)):
            ax.text(value + (0.1 if value >= 0 else -0.1), bar.get_y() + bar.get_height()/2,
                   f'{value:.2f}', ha='left' if value >= 0 else 'right', va='center')

        # Add pull bands
        ax.axvline(x=1, color='red', linestyle='--', alpha=0.7, label='±1σ')
        ax.axvline(x=-1, color='red', linestyle='--', alpha=0.7)
        ax.axvline(x=2, color='red', linestyle=':', alpha=0.7, label='±2σ')
        ax.axvline(x=-2, color='red', linestyle=':', alpha=0.7)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)

        # Customize plot
        ax.set_yticks(y_pos)
        ax.set_yticklabels(param_names)
        ax.set_xlabel('Pull (σ)')
        ax.set_title('Nuisance Parameter Pulls')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add CMS label
        add_cms_label(ax, year="2023", lumi=35.9)

        plt.tight_layout()

        # Save plot
        plot_file = output_path / "pulls.pdf"
        plt.savefig(plot_file, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        logging.info(f"Pull plot saved to {plot_file}")
        return str(plot_file)

    def plot_gof(self, input_file: str, output_dir: str) -> str:
        """
        Create goodness-of-fit plot.

        Args:
            input_file: Path to gof.root
            output_dir: Output directory

        Returns:
            Path to GOF plot file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load GOF data
        gof_data = self._load_gof_data(input_file)

        # Create GOF plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Plot toy distribution
        toys = gof_data.get('toys', [])
        observed = gof_data.get('observed', 0)
        p_value = gof_data.get('p_value', 0.5)

        if toys:
            ax.hist(toys, bins=50, alpha=0.7, color='blue', label='Toy Distribution', density=True)
            ax.axvline(x=observed, color='red', linestyle='-', linewidth=2, label=f'Observed (p={p_value:.3f})')

        # Customize plot
        ax.set_xlabel('Test Statistic')
        ax.set_ylabel('Density')
        ax.set_title('Goodness of Fit')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add CMS label
        add_cms_label(ax, year="2023", lumi=35.9)

        plt.tight_layout()

        # Save plot
        plot_file = output_path / "gof.pdf"
        plt.savefig(plot_file, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        logging.info(f"GOF plot saved to {plot_file}")
        return str(plot_file)

    def plot_prepost_fit(self, input_file: str, output_dir: str,
                        region: str = "SR") -> str:
        """
        Create pre/post-fit comparison plot.

        Args:
            input_file: Path to fitDiagnostics.root
            output_dir: Output directory
            region: Analysis region

        Returns:
            Path to pre/post-fit plot file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load pre/post-fit data
        prepost_data = self._load_prepost_fit_data(input_file, region)

        # Create pre/post-fit plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        # Pre-fit plot
        self._plot_fit_comparison(ax1, prepost_data['pre_fit'], 'Pre-fit')

        # Post-fit plot
        self._plot_fit_comparison(ax2, prepost_data['post_fit'], 'Post-fit')

        # Add CMS label
        add_cms_label(ax1, year="2023", lumi=35.9)

        plt.tight_layout()

        # Save plot
        plot_file = output_path / f"prepost_fit_{region}.pdf"
        plt.savefig(plot_file, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        logging.info(f"Pre/post-fit plot saved to {plot_file}")
        return str(plot_file)

    def _plot_fit_comparison(self, ax, data: Dict[str, Any], title: str):
        """Plot fit comparison (pre or post-fit)."""
        # Placeholder implementation
        x = np.linspace(0, 500, 50)

        # Plot background processes
        processes = ['ttbar', 'wjets', 'zjets', 'qcd', 'st', 'diboson']
        colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown']

        bottom = np.zeros_like(x)
        for process, color in zip(processes, colors):
            y = np.random.poisson(10, len(x))  # Placeholder data
            ax.fill_between(x, bottom, bottom + y, alpha=0.7, color=color, label=process)
            bottom += y

        # Plot data
        data_y = np.random.poisson(50, len(x))  # Placeholder data
        ax.errorbar(x, data_y, yerr=np.sqrt(data_y), fmt='ko', markersize=4, label='Data')

        ax.set_ylabel('Events')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _load_impacts_data(self, input_file: str) -> Dict[str, float]:
        """Load impacts data from file."""
        input_path = Path(input_file)

        if input_path.suffix == '.json':
            # Load from JSON file
            with open(input_file, 'r') as f:
                data = json.load(f)
            return data.get('impacts', {})
        else:
            # Load from ROOT file (placeholder)
            return {
                'lumi': 0.1,
                'btagSF': 0.05,
                'JES': 0.03,
                'JER': 0.02,
                'MET_scale': 0.01,
                'pileup': 0.02
            }

    def _load_pulls_data(self, input_file: str) -> Dict[str, float]:
        """Load pulls data from fitDiagnostics.root."""
        # Placeholder implementation
        return {
            'lumi': 0.5,
            'btagSF': -0.8,
            'JES': 1.2,
            'JER': -0.3,
            'MET_scale': 0.1,
            'pileup': -0.5,
            'muonSF': 0.2,
            'electronSF': -0.1
        }

    def _load_gof_data(self, input_file: str) -> Dict[str, Any]:
        """Load GOF data from gof.root."""
        # Placeholder implementation
        return {
            'toys': np.random.chisquare(10, 1000),
            'observed': 12.5,
            'p_value': 0.3
        }

    def _load_prepost_fit_data(self, input_file: str, region: str) -> Dict[str, Any]:
        """Load pre/post-fit data from fitDiagnostics.root."""
        # Placeholder implementation
        return {
            'pre_fit': {
                'processes': ['ttbar', 'wjets', 'zjets', 'qcd', 'st', 'diboson'],
                'rates': [50, 30, 20, 100, 15, 5]
            },
            'post_fit': {
                'processes': ['ttbar', 'wjets', 'zjets', 'qcd', 'st', 'diboson'],
                'rates': [45, 32, 18, 95, 16, 4]
            }
        }

    def create_all_diagnostics(self, input_files: Dict[str, str], output_dir: str) -> Dict[str, str]:
        """
        Create all diagnostic plots.

        Args:
            input_files: Dictionary of input file paths
            output_dir: Output directory

        Returns:
            Dictionary of plot file paths
        """
        plot_files = {}

        # Impact plot
        if 'fit_diagnostics' in input_files:
            impact_plot = self.plot_impacts(input_files['fit_diagnostics'], output_dir)
            plot_files['impacts'] = impact_plot

        # Pull plot
        if 'fit_diagnostics' in input_files:
            pull_plot = self.plot_pulls(input_files['fit_diagnostics'], output_dir)
            plot_files['pulls'] = pull_plot

        # GOF plot
        if 'gof' in input_files:
            gof_plot = self.plot_gof(input_files['gof'], output_dir)
            plot_files['gof'] = gof_plot

        # Pre/post-fit plot
        if 'fit_diagnostics' in input_files:
            prepost_plot = self.plot_prepost_fit(input_files['fit_diagnostics'], output_dir)
            plot_files['prepost_fit'] = prepost_plot

        logging.info(f"Created {len(plot_files)} diagnostic plots in {output_dir}")

        return plot_files
