#!/usr/bin/env python3
"""Plot normalized stacked histograms from merged ROOT files.

This script is intended for directories where each ROOT file corresponds to one
process/sample, such as the output of merge_root_hadd.py.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import uproot

from event_stacked_plotter import (
    _apply_variable_plot_filter,
    _build_xsection_process_lookup,
    _extract_object_distributions,
    _extract_xsection_by_year,
    _axis_label,
    _get_background_color_map,
    _histogram_and_sumw2,
    _load_single_root,
    _load_xsection_json,
    _make_bins,
    _match_root_file_to_process,
    _plot_stacked_variable,
    _simplify_sample_label,
    _event_normalisation,
)


@dataclass
class RootSample:
    name: str
    path: Path
    weighted_total_events: int
    objects: Dict[str, object]
    cross_section_pb: Optional[float] = None


def _load_sample(
    root_path: Path,
    xsection_data: Optional[Dict[str, List[Dict[str, object]]]] = None,
    year: Optional[str] = None,
    process_to_xsec: Optional[Dict[str, float]] = None,
) -> Optional[RootSample]:
    try:
        data = _load_single_root(root_path)
    except Exception as exc:
        print(f"Warning: could not load {root_path.name}: {exc}")
        return None

    weighted_total_events = int(data.get("weighted_total_events", 0) or 0)
    if weighted_total_events <= 0:
        print(f"Warning: skipping {root_path.name} because weighted_total_events is missing or invalid")
        return None

    cross_section_pb = None
    if xsection_data and year and process_to_xsec is not None:
        match = _match_root_file_to_process(root_path.name, process_to_xsec)
        if match is not None:
            cross_section_pb = float(match[1])
        else:
            cross_section_pb = _extract_xsection_by_year(xsection_data, root_path.stem, year)

    return RootSample(
        name=_simplify_sample_label(root_path.stem),
        path=root_path,
        weighted_total_events=weighted_total_events,
        objects=dict(data.get("objects", {})),
        cross_section_pb=cross_section_pb,
    )


def _collect_samples(
    input_root: Path,
    xsection_data: Optional[Dict[str, List[Dict[str, object]]]] = None,
    year: Optional[str] = None,
) -> List[RootSample]:
    root_files = sorted(p for p in input_root.glob("*.root") if p.is_file())
    if not root_files:
        raise FileNotFoundError(f"No ROOT files found in: {input_root}")

    process_to_xsec = _build_xsection_process_lookup(xsection_data, year) if xsection_data and year else None
    samples: List[RootSample] = []

    def _load(path: Path) -> Optional[RootSample]:
        return _load_sample(path, xsection_data=xsection_data, year=year, process_to_xsec=process_to_xsec)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for sample in executor.map(_load, root_files):
            if sample is not None:
                samples.append(sample)

    if not samples:
        raise ValueError(f"No valid ROOT samples could be loaded from: {input_root}")
    return samples


def _available_variables(samples: Sequence[RootSample]) -> List[str]:
    variables: set[str] = set()
    for sample in samples:
        variables.update(_extract_object_distributions(sample.objects).keys())
    return sorted(variables)


def _plot_all_variables(
    samples: Sequence[RootSample],
    output_dir: Path,
    variables: Optional[Sequence[str]] = None,
    n_bins: int = 40,
    max_variables: Optional[int] = None,
    luminosity: float = 1.0,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    distributions = {
        sample.name: _extract_object_distributions(sample.objects)
        for sample in samples
    }
    sample_names = [sample.name for sample in samples]
    color_map = _get_background_color_map(sample_names)

    if variables:
        selected_variables = list(variables)
    else:
        selected_variables = _available_variables(samples)

    if max_variables is not None:
        selected_variables = selected_variables[:max_variables]

    created_files: List[Path] = []

    for variable in selected_variables:
        all_values: List[np.ndarray] = []
        for sample in samples:
            values = distributions.get(sample.name, {}).get(variable)
            if values is None:
                continue
            filtered = _apply_variable_plot_filter(variable, values)
            if filtered.size > 0:
                all_values.append(filtered)

        bins = _make_bins(all_values, n_bins=n_bins, variable=variable)
        if bins is None or len(bins) < 2:
            continue

        background_hists: List[Tuple[str, np.ndarray, np.ndarray]] = []
        for sample in samples:
            values = distributions.get(sample.name, {}).get(variable, np.array([], dtype=float))
            values = _apply_variable_plot_filter(variable, values)
            hist_values, hist_sumw2 = _histogram_and_sumw2(
                values,
                bins,
                sample.weighted_total_events,
                luminosity=luminosity,
                cross_section_pb=sample.cross_section_pb,
            )
            background_hists.append((sample.name, hist_values, hist_sumw2))

        if all(np.allclose(hist, 0.0) for _, hist, _ in background_hists):
            continue

        created_files.extend(
            _plot_stacked_variable(
                variable,
                bins,
                background_hists,
                data_ndarray=None,
                output_dir=output_dir,
                luminosity=luminosity,
                color_map=color_map,
            )
        )

    return created_files


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot stacked, normalized distributions from merged ROOT files."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Directory containing one merged ROOT file per sample.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for the produced PNG plots.",
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        default=None,
        help="Optional list of variables to plot. If omitted, all common numeric branches are used.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=40,
        help="Number of bins for continuous variables.",
    )
    parser.add_argument(
        "--max-variables",
        type=int,
        default=None,
        help="Optional maximum number of variables to plot.",
    )
    parser.add_argument(
        "--lumi",
        type=float,
        default=1.0,
        help="Luminosity in fb^-1 used for normalization.",
    )
    parser.add_argument(
        "--xsection-json",
        type=Path,
        default=None,
        help="Optional JSON file mapping samples to cross-sections in pb.",
    )
    parser.add_argument(
        "--year",
        type=str,
        default=None,
        help="Optional year key used together with --xsection-json.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser()

    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root directory does not exist: {input_root}")

    xsection_data = None
    if args.xsection_json is not None:
        xsection_data = _load_xsection_json(args.xsection_json.expanduser().resolve())
        print(f"Loaded cross-section JSON: {args.xsection_json}")
    process_to_xsec = _build_xsection_process_lookup(xsection_data, args.year) if xsection_data and args.year else None

    samples = _collect_samples(input_root, xsection_data=xsection_data, year=args.year)
    print(f"Loaded {len(samples)} ROOT sample(s) from {input_root}")
    for sample in samples:
        xsec_note = f", xsec={sample.cross_section_pb:.6g} pb" if sample.cross_section_pb is not None else ""
        print(f"  - {sample.name}: weighted_total_events={sample.weighted_total_events}{xsec_note}")

    created = _plot_all_variables(
        samples,
        output_dir=output_dir,
        variables=args.variables,
        n_bins=args.bins,
        max_variables=args.max_variables,
        luminosity=args.lumi,
    )

    print(f"Created {len(created)} plot(s) in {output_dir}")
    for path in created:
        print(path)
    return 0


def _safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "plot"


def _is_data_file(path: Path, data_prefix: str) -> bool:
    return path.stem.startswith(data_prefix)


def _split_region_and_variable(hist_key: str) -> Tuple[str, str]:
    parts = hist_key.split("_")
    if len(parts) >= 4 and ":CR" in hist_key:
        return "_".join(parts[:3]), "_".join(parts[3:])
    return hist_key, hist_key


_VARIABLE_LABEL_ALIASES = {
    "met": "MET",
    "met_phi": "METPhi",
    "pfmet_pt": "PFMET_pt",
    "pfmet_phi": "PFMET_phi",
    "pfmet_significance": "PFMET_significance",
    "recoil": "recoil",
    "jet1_pt": "Jet1Pt",
    "jet1_eta": "Jet1Eta",
    "jet1_phi": "Jet1Phi",
    "jet1_deepcsv": "Jet1deepCSV",
    "jet2_pt": "Jet2Pt",
    "jet2_eta": "Jet2Eta",
    "jet2_phi": "Jet2Phi",
    "jet2_deepcsv": "Jet2deepCSV",
    "jet3_pt": "jet3_pt",
    "jet3_eta": "jet3_eta",
    "jet3_phi": "jet3_phi",
    "n_bjets": "n_bjets",
    "n_jets": "n_jets",
    "n_muons": "n_muons",
    "n_electrons": "n_electrons",
    "n_taus": "n_taus",
    "muon_pt": "muon_pt",
    "muon_eta": "muon_eta",
    "electron_pt": "electron_pt",
    "electron_eta": "electron_eta",
    "dr_muon_jet": "dr_muon_jet",
    "dr_electron_jet": "dr_electron_jet",
    "btag_deepjet": "jet_btag",
    "dnn_score": "dnn_score",
}


def _hist_axis_label(variable: str) -> str:
    canonical = _VARIABLE_LABEL_ALIASES.get(variable.lower(), variable)
    label = _axis_label(canonical)
    if label == canonical:
        return variable.replace("_", " ")
    return label


def _collect_root_files(input_root: Path, data_prefix: str) -> Tuple[List[Path], List[Path]]:
    root_files = sorted(p for p in input_root.glob("*.root") if p.is_file())
    background_files = [p for p in root_files if not _is_data_file(p, data_prefix)]
    data_files = [p for p in root_files if _is_data_file(p, data_prefix)]
    return background_files, data_files


def _collect_cr_hist_keys(files: Sequence[Path], region_pattern: str) -> List[str]:
    keys: set[str] = set()
    for path in files:
        with uproot.open(path) as root_file:
            for raw_key in root_file.keys():
                key = str(raw_key).split(";", 1)[0]
                if region_pattern and region_pattern not in key:
                    continue
                if ":CR" not in key:
                    continue
                keys.add(key)
    return sorted(keys)


def _load_histogram(path: Path, hist_key: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    with uproot.open(path) as root_file:
        if hist_key not in root_file:
            return None
        hist = root_file[hist_key]
        values = np.asarray(hist.values(), dtype=float)
        try:
            variances = np.asarray(hist.variances(), dtype=float)
        except Exception:
            variances = np.zeros_like(values)
        try:
            edges = np.asarray(hist.axes[0].edges(), dtype=float)
        except Exception as exc:
            raise RuntimeError(f"Could not read bin edges for {hist_key} in {path.name}") from exc

    if variances.shape != values.shape:
        variances = np.zeros_like(values)
    return values, variances, edges


def _get_weighted_total_events_from_root(path: Path) -> int:
    with uproot.open(path) as f:
        # Flat TH1 at root level: prefer 'h_n_events_input' first, then processed/weighted keys
        for _key in (
            "h_n_events_input",
            "h_n_events_input;1",
            "h_n_events_processed",
            "h_n_events_processed;1",
            "weighted_total_events",
            "weighted_total_events;1",
        ):
            if _key in f:
                try:
                    return int(round(float(f[_key].values()[0])))
                except Exception:
                    pass

        # Metadata tree fallback: prefer Metadata.h_n_events_input, then
        # Metadata.h_n_events_processed, then Metadata.weighted_total_events.
        if "Metadata" in f:
            meta = f["Metadata"]
            if hasattr(meta, "keys"):
                meta_keys = [str(k).split(";", 1)[0] for k in meta.keys()]

                if "h_n_events_input" in meta_keys:
                    try:
                        node = meta["h_n_events_input"]
                        try:
                            arr = node.array(library="np")
                            return int(np.sum(arr)) if len(arr) > 0 else 0
                        except Exception:
                            try:
                                vals = node.values()
                                if vals is not None:
                                    return int(np.sum(vals))
                            except Exception:
                                pass
                    except Exception:
                        pass

                if "h_n_events_processed" in meta_keys:
                    try:
                        node = meta["h_n_events_processed"]
                        try:
                            arr = node.array(library="np")
                            return int(np.sum(arr)) if len(arr) > 0 else 0
                        except Exception:
                            try:
                                vals = node.values()
                                if vals is not None:
                                    return int(np.sum(vals))
                            except Exception:
                                pass
                    except Exception:
                        pass

                if "weighted_total_events" in meta_keys:
                    try:
                        arr = meta["weighted_total_events"].array(library="np")
                        return int(np.sum(arr)) if len(arr) > 0 else 0
                    except Exception:
                        pass
    return 0


def _combine_histograms(
    histograms: Sequence[Tuple[str, np.ndarray, np.ndarray]],
    edges: np.ndarray,
    output_path: Path,
    title: str,
    x_label: str,
    y_label: str = "Events / bin",
    data_histogram: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sorted_backgrounds = sorted(histograms, key=lambda item: float(np.sum(item[1])))
    colors = _get_background_color_map([label for label, _, _ in sorted_backgrounds])

    show_ratio = data_histogram is not None
    if show_ratio:
        fig, (ax, ax_ratio) = plt.subplots(
            2,
            1,
            figsize=(12, 12),
            sharex=True,
            gridspec_kw={"height_ratios": [3.0, 1.0], "hspace": 0.08},
        )
        fig.subplots_adjust(top=0.92, bottom=0.09, left=0.14, right=0.95)
    else:
        fig, ax = plt.subplots(figsize=(12, 10))
        ax_ratio = None
        fig.subplots_adjust(top=0.92, bottom=0.12, left=0.14, right=0.95)

    cumulative = np.zeros(len(edges) - 1, dtype=float)
    cumulative_var = np.zeros(len(edges) - 1, dtype=float)
    for label, values, variances in sorted_backgrounds:
        next_cumulative = cumulative + values
        cumulative_var += variances
        ax.stairs(
            next_cumulative,
            edges,
            baseline=cumulative,
            fill=True,
            alpha=1.0,
            linewidth=0,
            color=colors.get(label, "#3f90da"),
            label=_simplify_sample_label(label),
        )
        cumulative = next_cumulative

    total_unc = np.sqrt(cumulative_var)
    unc_x = np.append(edges[:-1], edges[-1])
    unc_lo = np.append(cumulative - total_unc, (cumulative - total_unc)[-1])
    unc_hi = np.append(cumulative + total_unc, (cumulative + total_unc)[-1])
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

    data_values = None
    data_errors = None
    if data_histogram is not None:
        data_values, data_errors = data_histogram
        centers = 0.5 * (edges[:-1] + edges[1:])
        half_width = 0.5 * (edges[1:] - edges[:-1])
        mask = data_values > 0
        if np.any(mask):
            ax.errorbar(
                centers[mask],
                data_values[mask],
                xerr=half_width[mask],
                yerr=data_errors[mask],
                fmt="o",
                color="black",
                markerfacecolor="black",
                markeredgecolor="black",
                markersize=5.0,
                elinewidth=1.0,
                capsize=0,
                label="Data",
                zorder=10,
            )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Data has no positive values, and therefore cannot be log-scaled.",
        )
        ax.set_yscale("log")
    mc_max = float(np.max(cumulative)) if cumulative.size else 0.0
    data_max = float(np.max(data_values)) if data_values is not None and data_values.size else 0.0
    ax.set_ylim(0.1, max(mc_max, data_max, 1e-3) * 5.0)

    nonzero_mc = np.where(cumulative > 0)[0]
    if nonzero_mc.size:
        x_lo = float(edges[nonzero_mc[0]])
        x_hi = float(edges[nonzero_mc[-1] + 1])
    else:
        x_lo, x_hi = float(edges[0]), float(edges[-1])
    if data_values is not None:
        nonzero_data = np.where(data_values > 0)[0]
        if nonzero_data.size:
            x_lo = min(x_lo, float(edges[nonzero_data[0]]))
            x_hi = max(x_hi, float(edges[nonzero_data[-1] + 1]))

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylabel(y_label, fontsize=20, labelpad=8)
    ax.text(
        0.02,
        0.04,
        title,
        transform=ax.transAxes,
        fontsize=16,
        ha="left",
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 2.0},
        zorder=20,
    )
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=8, steps=[1, 2, 5, 10]))
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10.0))
    ax.grid(False)

    if show_ratio and ax_ratio is not None and data_values is not None:
        pred = cumulative
        pred_rel_err = np.divide(total_unc, pred, out=np.zeros_like(pred), where=pred > 0)
        ratio = np.divide(data_values, pred, out=np.full_like(data_values, np.nan, dtype=float), where=pred > 0)
        ratio_err = np.divide(data_errors, pred, out=np.zeros_like(data_errors), where=pred > 0)
        ax_ratio.axhline(1.0, color="black", linestyle="-", linewidth=1.2)
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
            label="Stat. unc.",
            zorder=5,
        )

        mask = np.isfinite(ratio)
        if np.any(mask):
            centers = 0.5 * (edges[:-1] + edges[1:])
            half_width = 0.5 * (edges[1:] - edges[:-1])
            ax_ratio.errorbar(
                centers[mask],
                ratio[mask],
                xerr=half_width[mask],
                yerr=ratio_err[mask],
                fmt="o",
                color="black",
                markerfacecolor="black",
                markeredgecolor="black",
                markersize=5.0,
                elinewidth=1.0,
                capsize=0,
                zorder=10,
            )

        ax_ratio.set_ylabel("Data / MC", fontsize=18, labelpad=6)
        ax_ratio.set_xlabel(x_label, fontsize=18, labelpad=6)
        ax_ratio.set_ylim(0.0, 2.0)
        ax_ratio.set_xlim(x_lo, x_hi)
        ax_ratio.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0, 0.5, 1.0, 1.5, 2.0]))
        ax_ratio.grid(False)

    hep.cms.label("Work in progress", data=data_values is not None, com=13.6, loc=0, ax=ax)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        data_idx = next((i for i, lbl in enumerate(labels) if lbl == "Data"), None)
        unc_idx = next((i for i, lbl in enumerate(labels) if lbl == "Uncertainty"), None)
        ordered_idx = (
            ([data_idx] if data_idx is not None else [])
            + [i for i in range(len(labels)) if i not in (data_idx, unc_idx)]
            + ([unc_idx] if unc_idx is not None else [])
        )
        handles = [handles[i] for i in ordered_idx]
        labels = [labels[i] for i in ordered_idx]
        if unc_idx is not None:
            handles[-1] = mpatches.Patch(
                hatch="////",
                facecolor="#bbbbbb",
                edgecolor="#3d3d3d",
                linewidth=0.0,
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
            fontsize=18,
        )

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot control-region histograms from merged ROOT files with JetMET data overlay."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Directory containing merged ROOT files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for the produced plots.",
    )
    parser.add_argument(
        "--data-prefix",
        type=str,
        default="JetMET",
        help="File name prefix used to identify data files (default: JetMET).",
    )
    parser.add_argument(
        "--region-pattern",
        type=str,
        default=":CR",
        help="Substring used to keep only control-region histograms (default: :CR).",
    )
    parser.add_argument(
        "--max-plots",
        type=int,
        default=None,
        help="Optional maximum number of histograms to plot.",
    )
    parser.add_argument(
        "--lumi",
        type=float,
        default=1.0,
        help="Luminosity in fb^-1 used to scale MC histograms (default: 1.0).",
    )
    parser.add_argument(
        "--xsection-json",
        type=Path,
        default=None,
        help="Optional JSON file mapping samples to cross-sections in pb.",
    )
    parser.add_argument(
        "--year",
        type=str,
        default=None,
        help="Optional year key used together with --xsection-json.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root directory does not exist: {input_root}")

    background_files, data_files = _collect_root_files(input_root, args.data_prefix)
    if not background_files:
        raise FileNotFoundError(f"No background ROOT files found in: {input_root}")
    if not data_files:
        print(f"Warning: no data files matched prefix '{args.data_prefix}' in {input_root}")

    print(f"Background files: {len(background_files)}")
    print(f"Data files: {len(data_files)}")

    xsection_data = None
    if args.xsection_json is not None:
        xsection_data = _load_xsection_json(args.xsection_json.expanduser().resolve())
        print(f"Loaded cross-section JSON: {args.xsection_json}")
    process_to_xsec = _build_xsection_process_lookup(xsection_data, args.year) if xsection_data and args.year else None

    hist_keys = _collect_cr_hist_keys(background_files + data_files, args.region_pattern)
    if args.max_plots is not None:
        hist_keys = hist_keys[: max(0, args.max_plots)]

    if not hist_keys:
        print("No CR histograms found to plot.")
        return 0

    print(f"Found {len(hist_keys)} CR histogram(s)")
    created = 0
    for hist_key in hist_keys:
        region, variable = _split_region_and_variable(hist_key)
        region_dir = output_dir / _safe_component(region)

        # Aggregate backgrounds by category and apply per-sample normalization when possible.
        edges: Optional[np.ndarray] = None
        category_acc: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for path in background_files:
            loaded = _load_histogram(path, hist_key)
            if loaded is None:
                continue
            values, variances, loaded_edges = loaded
            if edges is None:
                edges = loaded_edges
            elif not np.allclose(edges, loaded_edges):
                raise ValueError(f"Bin edges do not match for {hist_key} in {path.name}")

            # Determine per-file weighted_total_events and cross-section if available
            weighted_total = _get_weighted_total_events_from_root(path)
            cross_section_pb = None
            if process_to_xsec is not None:
                match = _match_root_file_to_process(path.name, process_to_xsec)
                if match is not None:
                    cross_section_pb = float(match[1])
                else:
                    cross_section_pb = _extract_xsection_by_year(xsection_data, path.stem, args.year) if xsection_data and args.year else None

            if weighted_total > 0:
                scale = _event_normalisation(weighted_total, luminosity=args.lumi, cross_section_pb=cross_section_pb)
            else:
                # If a file has no valid weighted_total_events but contributes non-zero
                # histogram entries for this control region, this is a normalization
                # problem that must be fixed upstream. Fail fast so the user can
                # inspect the offending ROOT file rather than silently producing
                # incorrect plots.
                if np.any(values > 0):
                    raise RuntimeError(
                        f"Normalization error: file {path.name} has no valid weighted_total_events but contains non-zero entries for {hist_key}.\n"
                        "Please inspect the ROOT file's Metadata or weighted_total_events TH1."
                    )
                scale = 1.0

            scaled_values = values * scale
            scaled_variances = variances * (scale ** 2)

            # Map sample to one of the seven canonical process keys used in _PROCESS_CONFIG.
            name_low = path.stem.lower()
            # Prefer simple substring matches; if none found, try one-shot process_to_xsec match
            def one_shot_map(n: str) -> Optional[str]:
                if "diboson" in n or "ww" in n or "wz" in n or "zz" in n:
                    return "DIBOSON"
                if "dyto" in n or "drell" in n or re.search(r"\bdy\b", n):
                    return "DYto2L-2Jets"
                if "zto2nu" in n or "z2nu" in n or "zto2n" in n:
                    return "Zto2Nu-2Jets"
                if "wtolnu" in n or "wto" in n or "wlnu" in n:
                    return "WtoLNu-2Jets"
                if "singletop" in n or ("single" in n and "top" in n):
                    return "SingleTop"
                if "ttbar" in n or ("top" in n and "singletop" not in n):
                    return "Top"
                if "smhiggs" in n or "higgs" in n or "smh" in n:
                    return "SMHiggs"
                return None

            category = one_shot_map(name_low)
            if category is None and process_to_xsec is not None:
                m = _match_root_file_to_process(path.name, process_to_xsec)
                if m is not None and isinstance(m[0], str):
                    matched_key = str(m[0])
                    # Try to map the matched JSON full_dataset key back to the
                    # JSON top-level category (e.g. 'Top', 'SingleTop', 'SMHiggs', ...)
                    found_topkey = None
                    if xsection_data:
                        for topkey, entries in xsection_data.items():
                            for ent in entries:
                                if ent.get("full_dataset") == matched_key:
                                    found_topkey = topkey
                                    break
                            if found_topkey:
                                break

                    def _map_topkey_to_canonical(key: str) -> Optional[str]:
                        if key is None:
                            return None
                        kl = key.lower()
                        if "singletop" in kl or kl.startswith("singletop"):
                            return "SingleTop"
                        if "top" in kl and "singletop" not in kl:
                            return "Top"
                        if "smhiggs" in kl or "smh" in kl or "higgs" in kl:
                            return "SMHiggs"
                        if "diboson" in kl or "ww" in kl or "wz" in kl or "zz" in kl:
                            return "DIBOSON"
                        if "zto2nu" in kl or "z2nu" in kl or "zto2n" in kl:
                            return "Zto2Nu-2Jets"
                        if "wtolnu" in kl or "wto" in kl or "wlnu" in kl:
                            return "WtoLNu-2Jets"
                        if "dyto" in kl or "drell" in kl or re.search(r"\bdy\b", kl):
                            return "DYto2L-2Jets"
                        return None

                    if found_topkey:
                        category = _map_topkey_to_canonical(found_topkey) or None
                    else:
                        # fallback to matching on the matched_key string itself
                        category = one_shot_map(matched_key.lower())
            if category is None:
                category = "DYto2L-2Jets"

            if category in category_acc:
                acc_vals, acc_vars = category_acc[category]
                category_acc[category] = (acc_vals + scaled_values, acc_vars + scaled_variances)
            else:
                category_acc[category] = (np.array(scaled_values, dtype=float), np.array(scaled_variances, dtype=float))

        if not category_acc or edges is None:
            continue

        # Build final list of background histograms from aggregated categories
        background_histograms: List[Tuple[str, np.ndarray, np.ndarray]] = []
        for cat, (vals, vars_) in sorted(category_acc.items(), key=lambda kv: float(np.sum(kv[1][0]))):
            background_histograms.append((cat, vals, vars_))

        # Print which MC categories will be drawn for this histogram (helps debug missing categories)
        drawn_cats = [cat for cat, _v in category_acc.items() if np.any(category_acc[cat][0] > 0)]
        print(f"Drawing categories for {hist_key}: {sorted(drawn_cats)}")

        if not background_histograms or edges is None:
            continue

        data_values: Optional[np.ndarray] = None
        for path in data_files:
            loaded = _load_histogram(path, hist_key)
            if loaded is None:
                continue
            values, _, loaded_edges = loaded
            if not np.allclose(edges, loaded_edges):
                raise ValueError(f"Bin edges do not match for {hist_key} in {path.name}")
            data_values = values if data_values is None else data_values + values

        data_histogram = None
        if data_values is not None and np.any(data_values > 0):
            data_histogram = (data_values, np.sqrt(np.maximum(data_values, 0.0)))

        output_path = region_dir / f"{_safe_component(variable)}.png"
        _combine_histograms(
            background_histograms,
            edges,
            output_path,
            title=region,
            x_label=_hist_axis_label(variable),
            data_histogram=data_histogram,
        )
        created += 1
        print(f"[OK] {hist_key} -> {output_path}")

    print(f"Created {created} plot(s) in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
