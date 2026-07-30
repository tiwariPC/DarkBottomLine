"""
Command-line interface for DarkBottomLine framework.
"""

import argparse
import logging
import sys
import yaml
import numpy as np
import uproot
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

try:
    from darkbottomline._version import __version__ as _fw_version
except Exception:
    _fw_version = None

def _default_version() -> str:
    """YYYYMMDD_<sha7> — today's date + commit SHA from framework version."""
    from datetime import datetime
    today = datetime.now().strftime("%Y%m%d")
    if _fw_version and "+" in _fw_version:
        sha = _fw_version.split("+", 1)[1]
        return f"{today}_{sha}"
    return today

from .processor import DarkBottomLineProcessor
from .analyzer import DarkBottomLineAnalyzer
from .dnn_trainer import DNNTrainer
from .dnn_inference import DNNInference, _parse_masspoint_label, _mass_branch_name, _resolve_mass_scan
from .plotting import PlotManager
from .regions import RegionManager
from utils.chunk_optimizer import (
    optimize_chunk_size_for_files,
    parse_chunk_size_arg,
)

# Try to import Coffea for chunk-size support
try:
    from coffea import processor
    from coffea.processor import Runner, FuturesExecutor
    from coffea.nanoevents import BaseSchema
    try:
        from dask.distributed import Client
        from coffea.processor import DaskExecutor
        DASK_AVAILABLE = True
    except ImportError:
        DASK_AVAILABLE = False
    COFFEA_AVAILABLE = True
except ImportError:
    COFFEA_AVAILABLE = False
    DASK_AVAILABLE = False


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _get_input_files(input_list: List[str]) -> List[str]:
    """
    Expand input list from a .txt file if provided.
    """
    if len(input_list) == 1 and input_list[0].endswith(".txt"):
        logging.info(f"Reading input files from {input_list[0]}")
        with open(input_list[0], 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    return input_list


def _is_signal_sample(input_list: List[str]) -> bool:
    """
    Detect fastsim signal samples (BBDM-2HDMa) by filename, so trigger
    requirements can be bypassed (fastsim has no HLT branches).
    Matches the "2hdma" substring convention used by dnn/phys_weight.py
    and configs/plotting.yaml's Signal_2HDMa pattern.
    """
    return any("2hdma" in entry.lower() for entry in input_list)


def run_analysis(args):
    """Run basic analysis."""
    logging.info("Running basic analysis...")

    # Load configuration
    config = load_config(args.config)
    if args.data:
        config.setdefault("data", {})["is_data"] = True
    config.setdefault("data", {})["is_signal"] = _is_signal_sample(args.input)

    # Initialize processor
    processor = DarkBottomLineProcessor(config)

    # Load events from ROOT file
    try:
        import uproot
        import awkward as ak

        input_files = _get_input_files(args.input)
        logging.info(f"Loading events from {str(input_files)} files")

        if args.max_events is not None and args.max_events < 0:
            args.max_events = None

        events = uproot.concatenate([f"{path}:Events" for path in input_files])

        # Limit events if specified
        if args.max_events and args.max_events > 0 and len(events) > args.max_events:
            events = events[:args.max_events]
            logging.info(f"Limited to {args.max_events} events")

        logging.info(f"Loaded {len(events)} events")

        # Process events (optionally save event-level selection)
        results = processor.process(events, event_selection_output=args.event_selection_output)

        # Save results
        import pickle
        import os

        # Create output directory if it doesn't exist
        outdir = os.path.dirname(args.output)
        if outdir:
            os.makedirs(outdir, exist_ok=True)

        with open(args.output, 'wb') as f:
            pickle.dump(results, f)

        logging.info(f"Results saved to {args.output}")

    except Exception as e:
        logging.error(f"Error processing events: {e}")
        raise

    logging.info("Basic analysis completed!")


def _merge_pickle_outputs(files: List[str], output_path: str):
    """Merge multiple pickle files containing coffea accumulators."""
    if not files:
        logging.warning("No files to merge.")
        return

    logging.info(f"Merging {len(files)} files into {output_path}")

    try:
        import pickle

        # Load the first file to initialize the merged accumulator
        with open(files[0], 'rb') as f:
            merged_accumulator = pickle.load(f)

        # Loop over the rest of the files and add them to the merged accumulator
        for file_path in files[1:]:
            with open(file_path, 'rb') as f:
                accumulator = pickle.load(f)
            # The loaded objects are coffea accumulators, so they support the `add` operation.
            if isinstance(merged_accumulator, dict) and isinstance(accumulator, dict):
                # Custom merging for dictionaries of histograms
                for key, value in accumulator.items():
                    if key in merged_accumulator and hasattr(merged_accumulator[key], 'add'):
                        merged_accumulator[key].add(value)
                    else:
                        merged_accumulator[key] = value
            elif hasattr(merged_accumulator, 'add'):
                 merged_accumulator.add(accumulator)
            else:
                raise TypeError(f"Unsupported accumulator type for merging: {type(merged_accumulator)}")


        # Save the merged accumulator
        with open(output_path, 'wb') as f:
            pickle.dump(merged_accumulator, f)

        logging.info(f"Successfully merged results to {output_path}")

    except Exception as e:
        logging.error(f"Error merging files: {e}")
        raise
    finally:
        # Clean up temporary files
        import os
        for file_path in files:
            try:
                os.remove(file_path)
                logging.debug(f"Removed temporary file: {file_path}")
            except OSError as e:
                logging.error(f"Error removing temporary file {file_path}: {e}")


def _add_dnn_scores_to_events(events, model_path: str, config_path: Optional[str],
                              score_branch: str = "ml_score",
                              objects: dict = None, config: dict = None):
    """Score all events with trained DNN; return ak.Array with ml_score field added.

    When *objects* and *config* are provided, uses the standard variable
    pipeline (compute_event_variables) so that the DNN sees the same
    feature values as the region-analysis and plotting code.
    """
    import awkward as ak
    from dnn.common import sanitize_feature_frame
    import pandas as _pd

    inference = DNNInference(model_path, config_path=config_path)
    features = inference.features
    n = len(events)

    if objects is not None and config is not None:
        df = _build_dnn_feature_matrix_from_events(events, objects, config, features)
    else:
        # Legacy: direct field lookup
        X_parts = {}
        for feat in features:
            if feat in events.fields:
                X_parts[feat] = np.asarray(ak.to_numpy(events[feat]), dtype="f8")
            else:
                X_parts[feat] = np.full(n, -9999.0, dtype="f8")
        df = _pd.DataFrame(X_parts)
        df = sanitize_feature_frame(df)

    X = df.to_numpy(dtype="f8")
    # None -> DNNInference defaults to its checkpoint's benchmark masspoint
    # for parametric models; ignored for non-parametric models.
    scores = inference.predict(X, None).ravel().astype("float32")

    # Append ml_score to events ak.Array
    events_with_score = ak.with_field(events, ak.Array(scores), score_branch)
    logging.info("DNN scoring complete: n=%d score_branch=%s", n, score_branch)
    return events_with_score


def _build_dnn_feature_matrix_from_events(
    events: "ak.Array",
    objects: dict,
    config: dict,
    features: list,
) -> "pd.DataFrame":
    """Build DNN feature matrix using compute_event_variables.

    Uses the same variable computation pipeline as the EVENTSELECTION output
    (variables.py), ensuring DNN training/inference sees exactly the same
    features as the plotting and region-analysis code. Feature names must
    match compute_event_variables() output keys exactly (configs/dnn.yaml
    features: is the source of truth) — no aliasing.
    """
    import pandas as _pd
    from .variables import compute_event_variables
    from dnn.common import sanitize_feature_frame

    n = len(events)

    # Compute all flat scalar variables via the standard pipeline
    all_vars = compute_event_variables(events, objects, config)

    X_dict: dict = {}
    for feat in features:
        arr = all_vars.get(feat)
        if arr is not None:
            X_dict[feat] = np.asarray(arr, dtype="f8").ravel()
        else:
            X_dict[feat] = np.full(n, -9999.0, dtype="f8")

    df = _pd.DataFrame(X_dict)
    df = sanitize_feature_frame(df)
    return df


def _train_dnn_on_events(events, train_dnn_config: str, dnn_outdir: str, args,
                         objects: dict = None, config: dict = None,
                         y_train: np.ndarray = None,
                         dnn_plotdir: str = None) -> Tuple[np.ndarray, str]:
    """Train DNN on selected events (in-memory); return (scores, model_path).

    Scores array aligns 1:1 with events — same length, float32.
    Model artifacts (dnn_model.pt, scaler.json, features.json, train_metrics.json)
    written to dnn_outdir (default: data/dnn).
    Training plots written to dnn_plotdir (default: outputs/dnn).

    When *y_train* is provided, the internal per-file label-building and
    data-exclusion logic is skipped entirely — the caller guarantees that
    *events* contains only MC events with correct labels.
    """
    import awkward as ak
    import pandas as pd
    from dnn.common import sanitize_feature_frame
    from dnn.make_trees import _is_data, _is_signal_heuristic

    n = len(events)
    sig_patterns = tuple(getattr(args, "signal_pattern", None) or ())
    sig_prefix = getattr(args, "signal_prefix", None)
    label_csv_path = getattr(args, "label_csv", None)

    # Build label array.  When *y_train* is provided externally (integrated
    # path with correct per-file→per-event alignment), skip the internal
    # per-file label-building and data-exclusion logic entirely.
    if y_train is not None:
        y = np.asarray(y_train, dtype="i4")
        keep = np.ones(len(y), dtype=bool)  # caller already excluded data
        n_data_skipped = 0
        if np.unique(y).size < 2:
            logging.warning("Only one class present — skipping DNN training, scores set to 0.5")
            return np.full(n, 0.5, dtype="float32"), ""
    else:
        input_files = _get_input_files(args.input)
        import os as _os

        # Phase 1: collect per-file info (entry counts, labels, data flags)
        file_entries: list = []
        if label_csv_path:
            import csv as _csv
            label_map = {}
            with open(label_csv_path, "r", newline="") as fp:
                for row in _csv.DictReader(fp):
                    label_map[str(row["path"]).strip()] = int(row["label"])
            for fpath in input_files:
                key = fpath if fpath in label_map else _os.path.basename(fpath)
                lbl = label_map.get(key, 0)
                is_data = _is_data(fpath)
                with uproot.open(fpath) as f:
                    cnt = int(f["Events"].num_entries)
                file_entries.append((cnt, lbl, is_data))
        else:
            for fpath in input_files:
                is_data = _is_data(fpath)
                if is_data:
                    sig = -1  # sentinel: will be excluded
                else:
                    sig = 1 if _is_signal_heuristic(fpath, sig_patterns, sig_prefix) else 0
                with uproot.open(fpath) as f:
                    cnt = int(f["Events"].num_entries)
                file_entries.append((cnt, sig, is_data))

        # Phase 2: build per-event y and a keep-mask (True = keep for training)
        y_parts = []
        keep_parts = []
        n_data_skipped = 0
        for cnt, sig, is_data in file_entries:
            if is_data:
                y_parts.append(np.full(cnt, -1, dtype="i4"))
                keep_parts.append(np.zeros(cnt, dtype=bool))
                n_data_skipped += cnt
            else:
                y_parts.append(np.full(cnt, sig, dtype="i4"))
                keep_parts.append(np.ones(cnt, dtype=bool))
        y_full = np.concatenate(y_parts)[:n]
        keep = np.concatenate(keep_parts)[:n]

        if n_data_skipped > 0:
            logging.info("Excluded %d data events from DNN training (files: %s)",
                         n_data_skipped,
                         [f for f in input_files if _is_data(f)])

        # Filter to non-data events only
        y = y_full[keep]
        if np.unique(y).size < 2:
            logging.warning("Only one class present after excluding data — skipping DNN training, scores set to 0.5")
            return np.full(n, 0.5, dtype="float32"), ""

    # Build full feature matrix from all events (data + MC)
    n_keep = int(keep.sum())
    _dnn_yaml = load_config(train_dnn_config)
    feat_list = _dnn_yaml.get("features")
    if not feat_list:
        raise ValueError(
            f"'{train_dnn_config}' has no features: list — required for DNN training."
        )

    if objects is not None and config is not None:
        # Use standard variable-computation pipeline (compute_event_variables)
        X_full_df = _build_dnn_feature_matrix_from_events(
            events, objects, config, feat_list
        )
        X_dict_full = {c: X_full_df[c].to_numpy(dtype="f8") for c in X_full_df.columns}
    else:
        # Fallback: direct field lookup
        X_dict_full = {}
        for feat in feat_list:
            if feat in events.fields:
                X_dict_full[feat] = np.asarray(ak.to_numpy(events[feat]), dtype="f8")
            else:
                X_dict_full[feat] = np.full(n, -9999.0, dtype="f8")

    # Training subset: non-data events only
    X_dict_train = {feat: arr[keep] for feat, arr in X_dict_full.items()}
    X_df = pd.DataFrame(X_dict_train)
    X_df = sanitize_feature_frame(X_df)

    # Weights, filtered to non-data events
    if "full_event_weight" in events.fields:
        w_full = np.asarray(ak.to_numpy(events["full_event_weight"]), dtype="f8")
        w_full = np.where(np.isfinite(w_full), np.maximum(w_full, 0.0), 0.0)
        w = w_full[keep]
    else:
        w = np.ones(n_keep, dtype="f8")

    _plot_dir = dnn_plotdir if dnn_plotdir is not None else "outputs/dnn"
    trainer = DNNTrainer(train_dnn_config)
    metrics = trainer.train_from_arrays(
        X=X_df, y=y, w=w,
        feature_sources={},
        outdir=dnn_outdir,
        plot_dir=_plot_dir,
    )
    model_path = str(Path(dnn_outdir) / "dnn_model.pt")
    logging.info("DNN trained — AUC(val)=%.4f  model=%s", metrics.get("auc_val", float("nan")), model_path)

    # Score ALL events (including data) with the just-trained model
    # X_full is the feature matrix for every event; data events are scored but not trained on
    X_full = pd.DataFrame(X_dict_full)
    X_full = sanitize_feature_frame(X_full)
    # This inline (per-analyze) training path has no GenModel mass source to
    # train on, so it always trains/scores non-parametrically regardless of
    # dnn.yaml's parametric_input — mirrors train_from_arrays(mass=None) above.
    scores = trainer.predict(X_full.to_numpy(dtype="f8"), None).ravel().astype("float32")
    return scores, model_path


def _plot_dnn_score_only(scores: np.ndarray, plot_dir: str) -> None:
    """Write a simple DNN score distribution plot when --dnn-only is set."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path(plot_dir).mkdir(parents=True, exist_ok=True)
    s = np.clip(scores, 0.0, 1.0)
    plt.figure(figsize=(7, 5))
    plt.hist(s, bins=50, range=(0, 1), color="#3f90da", edgecolor="black", linewidth=0.6)
    plt.xlabel("DNN score (ml_score)")
    plt.ylabel("Events")
    plt.title("DNN score distribution (all passing events)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    out = Path(plot_dir) / "dnn_score_distribution.png"
    plt.savefig(out, dpi=300)
    plt.close()
    logging.info("DNN score plot saved to %s", out)


def _run_analyzer_from_eventselection(args):
    """
    Path 2b: load per-sample EVENTSELECTION.root files, run region analysis, save PKL output.

    Reads all ROOT files matching process_groups patterns from --input (folder or list of files),
    runs region cuts via RegionManager, fills histograms, and saves a merged PKL identical in
    structure to the output of Path 1 analyze.
    """
    import json
    import os
    import uproot
    import pickle

    config = load_config(args.config)
    if args.data:
        config.setdefault("data", {})["is_data"] = True
    is_data = config.get("data", {}).get("is_data", False)

    cross_sections: Dict[str, float] = {}
    if getattr(args, "xsection_json", None):
        with open(args.xsection_json) as f:
            _raw_xsec = json.load(f)
        # Flatten the nested {group: [{full_dataset, xsection, ...}]} JSON to a
        # {full_dataset: xsec} dict so per-file lookup below works regardless of
        # year (canonicalized via _find_xsec).
        from darkbottomline.plotting import PlotManager
        cross_sections = PlotManager._normalize_cross_sections(_raw_xsec)

    dnn_model  = getattr(args, "dnn_model",  None)
    dnn_config = getattr(args, "dnn_config", None)

    # Build the DNN model once (checkpoint + scaler load) instead of per-file —
    # process_from_eventselection is called once per input file below.
    dnn_inference = None
    dnn_mass_scan = None
    if dnn_model:
        from darkbottomline.dnn_inference import DNNInference
        dnn_inference = DNNInference(dnn_model, config_path=dnn_config)
        dnn_mass_scan = _resolve_mass_scan(getattr(args, "dnn_mass_scan", None), dnn_inference)

    raw_inputs = _get_input_files(args.input)
    # Expand any directories to their ROOT files
    input_files = []
    for p in raw_inputs:
        if os.path.isdir(p):
            input_files.extend(sorted(str(f) for f in Path(p).iterdir()
                                      if f.suffix in (".root", ".pkl")))
        else:
            input_files.append(p)

    if args.output:
        outdir = os.path.dirname(args.output)
        if outdir:
            os.makedirs(outdir, exist_ok=True)

    merged_result: Optional[Dict] = None
    analyzer = DarkBottomLineAnalyzer(config, args.regions_config)

    from darkbottomline.plotting import _find_xsec
    from dnn.make_trees import _is_data as _fname_is_data, _is_signal_heuristic
    n_bkg_scored = n_sig_scored = n_data_scored = 0
    for file_path in input_files:
        stem = Path(file_path).stem
        # Canonicalized lookup so 2022/23 (_2J) and 2024 (Bin-2J-) stems, and the
        # SMHiggs renames, all resolve to the right xsec.
        xsec = _find_xsec(stem, cross_sections)
        if xsec is None:
            xsec = _find_xsec(stem.replace("_EVENTSELECTION", ""), cross_sections)
        try:
            with uproot.open(str(file_path)) as f:
                wte = 0.0
                for key in ("weighted_total_events", "weighted_total_events;1"):
                    if key in f:
                        try:
                            wte = float(f[key].values()[0])
                            break
                        except Exception:
                            pass
                if "Events" not in f:
                    logging.warning("No Events tree in %s — skipping", stem)
                    continue
                raw = f["Events"].arrays(library="np")
                branches = dict(raw)
        except Exception as exc:
            logging.warning("Could not load %s: %s — skipping", stem, exc)
            continue

        # Scale wte by xsec if provided (consistent with make-event-plots normalisation)
        effective_wte = wte if wte > 0 else 1.0
        try:
            result = analyzer.process_from_eventselection(
                branches=branches,
                weighted_total_events=effective_wte,
                is_data=is_data,
                dnn_inference=dnn_inference,
                dnn_mass_scan=dnn_mass_scan,
            )
        except Exception as exc:
            logging.error("Region analysis failed for %s: %s", stem, exc, exc_info=True)
            continue

        if dnn_inference is not None:
            if is_data or _fname_is_data(file_path):
                n_data_scored += 1
            elif _is_signal_heuristic(file_path, (), None):
                n_sig_scored += 1
            else:
                n_bkg_scored += 1

        # Tag result with xsec for downstream normalisation
        result.setdefault("metadata", {})["xsec"] = xsec
        result["metadata"]["sample"] = stem

        # Save per-sample PKL
        if args.output:
            import os as _os2
            _out_dir = args.output if _os2.path.isdir(args.output) else _os2.path.dirname(args.output) or '.'
            _pkl_dir = _os2.path.join(_out_dir, 'per_sample')
            _os2.makedirs(_pkl_dir, exist_ok=True)
            _pkl_path = _os2.path.join(_pkl_dir, stem + '.pkl')
            import pickle as _pkl2
            with open(_pkl_path, 'wb') as _pf:
                _pkl2.dump(result, _pf)
        if merged_result is None:
            merged_result = result
        else:
            merged_result = _merge_region_results(merged_result, result)
        logging.debug("Processed %s (wte=%.1f)", stem, effective_wte)

    if dnn_inference is not None:
        logging.info(
            "DNN scores added to events (ml_score) for %d background file(s), "
            "%d signal file(s), %d data file(s)",
            n_bkg_scored, n_sig_scored, n_data_scored,
        )

    if merged_result is None:
        logging.error("No files processed — nothing to save.")
        return

    # Save merged result (for backward compat) + per-sample PKLs already saved above
    if args.output:
        _out_dir2 = args.output if _os.path.isdir(args.output) else _os.path.dirname(args.output) or '.'
        _merged_out = _os.path.join(_out_dir2, 'merged.pkl')
        analyzer.accumulator = merged_result
        analyzer.save_results(_merged_out, output_format=args.output_format)
        logging.info("Region analysis saved: merged=%s, per_sample=%s", _merged_out, _os.path.join(_out_dir2, 'per_sample'))


def _merge_region_results(a: Dict, b: Dict) -> Dict:
    """Shallow-merge two region-analysis result dicts (add histogram counts)."""
    import copy
    merged = copy.deepcopy(a)
    for region, hists in b.get("region_histograms", {}).items():
        if region not in merged.setdefault("region_histograms", {}):
            merged["region_histograms"][region] = hists
        else:
            for hname, hdata in hists.items():
                if hname not in merged["region_histograms"][region]:
                    merged["region_histograms"][region][hname] = hdata
                else:
                    try:
                        merged["region_histograms"][region][hname] = (
                            merged["region_histograms"][region][hname] + hdata
                        )
                    except Exception:
                        pass
    for region, res in b.get("regions", {}).items():
        merged.setdefault("regions", {})[region] = res
    wte_a = merged.get("metadata", {}).get("weighted_total_events", 0.0)
    wte_b = b.get("metadata", {}).get("weighted_total_events", 0.0)
    merged.setdefault("metadata", {})["weighted_total_events"] = wte_a + wte_b
    return merged


def _trigger_plots(args):
    """Call make_event_plots for --make-region-plots and/or --make-event-selection-plots."""
    make_region_plots_flag    = getattr(args, "make_region_plots", False)
    make_evsel_flag    = getattr(args, "make_event_selection_plots", False)
    if not make_region_plots_flag and not make_evsel_flag:
        return
    # If --region-results, create temp folder with pattern-named PKL copies
    import os as _os2
    region_pkl = getattr(args, "region_results", None)
    if region_pkl and _os2.path.isfile(str(region_pkl)):
        import tempfile, shutil, yaml as _yaml
        td = tempfile.mkdtemp(prefix='rplot_')
        _cfg = {}
        if getattr(args, 'plot_config', None):
            with open(args.plot_config) as _f:
                _cfg = _yaml.safe_load(_f)
        _pats = set()
        for _g in ['process_groups','signal_groups','data_groups']:
            for _v in _cfg.get(_g,{}).values():
                if isinstance(_v,dict) and 'patterns' in _v:
                    _pats.update(_v['patterns'])
        if not _pats:
            _pats = {'merged'}
        for _p in _pats:
            shutil.copy(str(region_pkl), f'{td}/{_p}.pkl')
        args.input_folder = td
    # make_event_plots needs input_folder
    if not getattr(args, "input_folder", None):
        import os
        # region-analysis/event-selection-plots: --input is the folder
        raw = getattr(args, "input", None)
        if raw:
            candidate = raw[0] if isinstance(raw, list) else raw
            if os.path.isdir(candidate):
                args.input_folder = candidate
        # full mode: derive from --event-selection-output
        if not getattr(args, "input_folder", None):
            evsel_out = getattr(args, "event_selection_output", None)
            if evsel_out:
                args.input_folder = os.path.dirname(evsel_out)
        if not getattr(args, "input_folder", None):
            logging.warning("--make-region-plots requested but no input_folder derivable from --input or --event-selection-output")
            return
    args.data_folder   = None
    args.process_groups = getattr(args, "process_groups", None)
    # map plot-specific args that may differ in name
    if not getattr(args, "variables", None):
        args.variables = getattr(args, "plot_variables", None)
    if not getattr(args, "regions", None):
        args.regions   = getattr(args, "plot_regions", None)
    if make_region_plots_flag:
        logging.info("Producing region plots (ROOT→PKL→plot)...")
        args.mode = "region" if getattr(args, "region_results", None) else "region-from-events"
        make_event_plots(args)
    if make_evsel_flag:
        logging.info("Producing event-selection plots...")
        args.mode = "event-selection"
        make_event_plots(args)


def run_analyzer(args):
    """Run analysis pipeline."""
    # Translate --mode into internal flags
    mode = getattr(args, "mode", "full")
    # backward compat: old --event-selection-only true still works
    _old_esel = getattr(args, "event_selection_only", "false")
    if isinstance(_old_esel, str) and _old_esel.lower() == "true":
        mode = "event-selection"
    # old --from-eventselection still works
    if getattr(args, "from_eventselection", False):
        mode = "region-analysis"

    args.event_selection_only = "true" if mode == "event-selection" else "false"
    args.from_eventselection   = (mode == "region-analysis")

    make_evsel_plots = getattr(args, "make_event_selection_plots", False)

    mode_labels = {
        "event-selection": (
            "Producing event-selection plots (EVENTSELECTION.root → stacked plots)..."
            if make_evsel_plots else
            "Running event selection (NanoAOD → EVENTSELECTION.root)..."
        ),
        "region-analysis": "Running region analysis (EVENTSELECTION.root → region plots (ROOT→PKL→plot))...",
        "full":            "Running full pipeline (NanoAOD → event-selection + regions)...",
    }
    logging.info(mode_labels.get(mode, "Running analysis..."))

    # event-selection + --make-event-selection-plots → plot only, no NanoAOD processing
    if mode == "event-selection" and make_evsel_plots:
        args.input_folder   = args.input[0]
        args.data_folder    = None
        args.process_groups = getattr(args, "process_groups", None)
        args.variables      = getattr(args, "plot_variables", None)
        args.regions        = getattr(args, "plot_regions", None)
        args.mode           = "event-selection"
        make_event_plots(args)
        return

    # region-analysis mode: delegate to _run_analyzer_from_eventselection then plot
    if mode == "region-analysis":
        if getattr(args, "region_results", None):
            _trigger_plots(args)
        else:
            _run_analyzer_from_eventselection(args)
            if not getattr(args, "skip_plots", False):
                _trigger_plots(args)
        return

    # Unified pipeline flags
    dnn_model = getattr(args, "dnn_model", None)
    dnn_config = getattr(args, "dnn_config", None)
    train_dnn_config = getattr(args, "train_dnn", None)
    dnn_outdir = getattr(args, "dnn_outdir", "data/dnn")
    dnn_plotdir = getattr(args, "dnn_plotdir", "outputs/dnn")
    dnn_only = getattr(args, "dnn_only", False)

    # DNN training requires full event set → fall back to iterative
    if train_dnn_config and args.executor in ("futures", "dask"):
        logging.warning(
            "DNN training requested but Coffea %s executor cannot train. "
            "Falling back to iterative executor.",
            args.executor,
        )
        args.executor = "iterative"
    # DNN scoring works per-chunk in Coffea path — no fallback needed

    event_selection_only = (mode == "event-selection")

    # Validate arguments
    if event_selection_only:
        if not args.event_selection_output:
            logging.error("--event-selection-output required with --mode event-selection")
            sys.exit(1)
        logging.info("Event selection only mode: will stop after event selection (no region analysis)")
    elif not dnn_only:
        # regions-config always required for full mode
        if not args.regions_config:
            logging.error("--regions-config required for --mode full")
            sys.exit(1)
        # --output required only when saving PKL (not needed if only making plots)
        make_plots_any = getattr(args, "make_region_plots", False) or getattr(args, "make_event_selection_plots", False)
        if not args.output and not make_plots_any:
            logging.error("--output or --make-region-plots/--make-event-selection-plots required for --mode full")
            sys.exit(1)

    config = load_config(args.config)
    if args.data:
        config.setdefault("data", {})["is_data"] = True
    config.setdefault("data", {})["is_signal"] = _is_signal_sample(args.input)

    try:
        import uproot
        import awkward as ak
        import os

        is_txt_input = len(args.input) == 1 and args.input[0].endswith(".txt")
        input_files = _get_input_files(args.input)

        input_total_events = None
        try:
            input_total_events = 0
            for file_path in input_files:
                tree = uproot.open(f"{file_path}:Events")
                input_total_events += int(tree.num_entries)
            logging.info(f"Computed input_total_events={input_total_events} from input files")
        except Exception as e:
            logging.warning(f"Could not compute input_total_events from input files: {e}")
            input_total_events = None

        # -1 (or any negative) means "no limit" — treat as None throughout
        if args.max_events is not None and args.max_events < 0:
            args.max_events = None

        # Total events before selection to be saved into event-selection-output metadata.
        # Rule: use --max-events when specified; otherwise use total events from input files.
        total_events = args.max_events if args.max_events is not None else None
        if total_events is None and args.event_selection_output:
            try:
                total_events = 0
                for file_path in input_files:
                    tree = uproot.open(f"{file_path}:Events")
                    total_events += int(tree.num_entries)
                logging.info(f"Computed total_events={total_events} from input files")
            except Exception as e:
                logging.warning(f"Could not compute total_events from input files: {e}")
                total_events = None

        # Parse chunk size argument (can be "auto" or int)
        chunk_size = None
        if hasattr(args, 'chunk_size') and args.chunk_size is not None:
            if isinstance(args.chunk_size, str):
                chunk_size = parse_chunk_size_arg(args.chunk_size)
            else:
                chunk_size = args.chunk_size

        # Auto-optimize chunk size if requested
        if chunk_size is None and args.executor in ["futures", "dask"]:
            logging.info("Auto-optimizing chunk size based on input files...")
            try:
                # Estimate available memory (default: 8GB per worker, conservative)
                # This is a rough estimate - users can override with explicit chunk-size
                available_memory_mb = 8000  # 8GB default
                chunk_size = optimize_chunk_size_for_files(
                    input_files=input_files,
                    available_memory_mb=available_memory_mb,
                    num_workers=args.workers,
                    executor=args.executor,
                )
                logging.info(f"Auto-optimized chunk size: {chunk_size:,} events")
            except Exception as e:
                logging.warning(f"Failed to auto-optimize chunk size: {e}")
                # Fallback to defaults
                chunk_size = 50000 if args.executor == "futures" else 200000
                logging.info(f"Using default chunk size: {chunk_size:,} events")

        # Check if we should use Coffea run_uproot_job with chunk-size
        use_coffea_chunking = (
            COFFEA_AVAILABLE and
            args.executor in ["futures", "dask"] and
            chunk_size is not None
        )

        if use_coffea_chunking:
            # Use Coffea run_uproot_job for chunked processing
            # Import the Coffea processor wrapper (only available if Coffea is installed)
            try:
                from .analyzer import DarkBottomLineAnalyzerCoffeaProcessor
            except ImportError:
                logging.error("DarkBottomLineAnalyzerCoffeaProcessor not available. Coffea may not be installed.")
                raise

            logging.info(f"Using Coffea {args.executor} executor with chunk-size={chunk_size}")

            fileset = {"dataset": {"treename": "Events", "files": input_files}}
            chunksize = chunk_size
            maxchunks = None
            if args.max_events is not None and chunksize > 0:
                maxchunks = max(1, (args.max_events + chunksize - 1) // chunksize)
                logging.info(
                    f"Applying event limit: max-events={args.max_events}, chunk-size={chunksize}, maxchunks={maxchunks}"
                )

            # For event_selection_only mode, use a dummy regions_config
            regions_config_for_coffea = args.regions_config if not event_selection_only else None

            # Auto-detect output format from event_selection_output extension if not explicitly set
            output_format_to_use = args.output_format
            if args.event_selection_output and output_format_to_use == "pkl":
                # Check if explicit format is needed or can be auto-detected
                if args.event_selection_output.endswith('.root'):
                    output_format_to_use = 'root'
                elif args.event_selection_output.endswith('.parquet'):
                    output_format_to_use = 'parquet'

            coffea_analyzer = DarkBottomLineAnalyzerCoffeaProcessor(
                config, regions_config_for_coffea, event_selection_output=args.event_selection_output,
                event_selection_only=event_selection_only, output_format=output_format_to_use,
                max_events=args.max_events, total_events=total_events,
                input_total_events=input_total_events,
                dnn_model=dnn_model, dnn_config=dnn_config,
            )

            if args.executor == "futures":
                runner = Runner(
                    executor=FuturesExecutor(workers=args.workers),
                    chunksize=chunksize,
                    maxchunks=maxchunks,
                    schema=BaseSchema,
                )
                result = runner(fileset, coffea_analyzer)
            elif args.executor == "dask" and DASK_AVAILABLE:
                client = None
                try:
                    client = Client(n_workers=args.workers, timeout=120)
                    try:
                        client.wait_for_workers(args.workers, timeout=60)
                        logging.info(f"Dask client ready with {len(client.scheduler_info()['workers'])} workers")
                    except Exception as e:
                        logging.warning(f"Timeout waiting for workers, continuing anyway: {e}")

                    dask_chunksize = chunksize if chunksize != 50000 else 200000
                    runner = Runner(
                        executor=DaskExecutor(client=client),
                        chunksize=dask_chunksize,
                        maxchunks=maxchunks,
                        schema=BaseSchema,
                    )
                    result = runner(fileset, coffea_analyzer)
                except Exception as e:
                    logging.error(f"Dask execution error: {e}")
                    raise
                finally:
                    if client is not None:
                        try:
                            client.close()
                        except Exception as e:
                            logging.warning(f"Error closing Dask client: {e}")
            else:
                raise ValueError(f"Executor {args.executor} not available or not supported")

            # Runner calls postprocess automatically; call again only as safety net
            if hasattr(coffea_analyzer, 'postprocess'):
                logging.info("Calling postprocess to finalize event_selection_output if needed...")
                result = coffea_analyzer.postprocess(result)

            # Save results (only if not in event_selection_only mode and --output given)
            if not event_selection_only:
                if args.output:
                    analyzer = DarkBottomLineAnalyzer(config, args.regions_config)
                    analyzer.accumulator = result
                    outdir = os.path.dirname(args.output)
                    if outdir:
                        os.makedirs(outdir, exist_ok=True)
                    analyzer.save_results(args.output, output_format=args.output_format)
            else:
                logging.info("Event selection only mode: skipping region analysis and main output save")

        else:
            # Original processing without chunking
            analyzer = DarkBottomLineAnalyzer(config, args.regions_config) if not event_selection_only else None

            if is_txt_input and len(input_files) > 1 and not event_selection_only:
                logging.info("Processing multiple files from .txt file iteratively.")
                temp_files = []
                output_dir = os.path.dirname(args.output)
                os.makedirs(output_dir, exist_ok=True)

                for i, file_path in enumerate(input_files):
                    logging.info(f"Processing file {i+1}/{len(input_files)}: {file_path}")
                    temp_output_path = os.path.join(output_dir, f"temp_{i}.pkl")
                    temp_files.append(temp_output_path)

                    events = uproot.open(f"{file_path}:Events")

                    if args.max_events and args.max_events > 0:
                        events = events.arrays(entry_stop=args.max_events)
                    else:
                        events = events.arrays()

                    events = ak.Array(events)

                    logging.info(f"Loaded {len(events)} events")

                    results = analyzer.process(events, event_selection_output=None) # No event selection output for partial files

                    analyzer.accumulator = results
                    analyzer.save_results(temp_output_path, output_format=args.output_format)

                _merge_pickle_outputs(temp_files, args.output)

            else:
                logging.info(f"Loading events from {len(input_files)} files")
                events = uproot.concatenate([f"{path}:Events" for path in input_files])

                if args.max_events and args.max_events > 0 and len(events) > args.max_events:
                    events = events[:args.max_events]
                    logging.info(f"Limited to {args.max_events} events")

                logging.info(f"Loaded {len(events)} events")

                # Auto-detect output format from event_selection_output extension if not explicitly set
                output_format_to_use = args.output_format
                if args.event_selection_output and output_format_to_use == "pkl":
                    # Check if explicit format is needed or can be auto-detected
                    if args.event_selection_output.endswith('.root'):
                        output_format_to_use = 'root'
                    elif args.event_selection_output.endswith('.parquet'):
                        output_format_to_use = 'parquet'

                if event_selection_only:
                    # Event selection only mode
                    logging.info("Event selection only mode: performing event selection...")
                    analyzer = DarkBottomLineAnalyzer(config, None)
                    results = analyzer.process(events, event_selection_output=args.event_selection_output,
                                              event_selection_only=True, output_format=output_format_to_use,
                                              total_events=total_events, input_total_events=input_total_events)
                    logging.info(f"Event selection completed, saved to {args.event_selection_output}")

                elif train_dnn_config:
                    # ── Apply preselection + compute MC weights BEFORE DNN training ──
                    # Physics rationale: DNN must train on the same phase space
                    # (post-preselection) and with correct event weights
                    # (generator × pileup × btag SF × …).
                    from .objects import build_objects
                    from .selections import _build_event_cut_masks
                    from dnn.make_trees import _is_data, _is_signal_heuristic
                    import awkward as _ak

                    n_total = len(events)

                    # Step 0: Build per-file labels + keep-mask for ORIGINAL events
                    # (before preselection) so the alignment is correct.  We then
                    # apply the preselection mask to get per-event labels for the
                    # events that actually enter DNN training.
                    sig_patterns = tuple(getattr(args, "signal_pattern", None) or ())
                    sig_prefix = getattr(args, "signal_prefix", None)
                    y_parts_full = []
                    keep_parts_full = []
                    n_data_total = 0
                    for fpath in input_files:
                        is_data = _is_data(fpath)
                        with uproot.open(fpath) as f:
                            cnt = int(f["Events"].num_entries)
                        if is_data:
                            y_parts_full.append(np.full(cnt, -1, dtype="i4"))
                            keep_parts_full.append(np.zeros(cnt, dtype=bool))
                            n_data_total += cnt
                        else:
                            sig = 1 if _is_signal_heuristic(fpath, sig_patterns, sig_prefix) else 0
                            y_parts_full.append(np.full(cnt, sig, dtype="i4"))
                            keep_parts_full.append(np.ones(cnt, dtype=bool))
                    y_full_orig = np.concatenate(y_parts_full)[:n_total]
                    keep_full_orig = np.concatenate(keep_parts_full)[:n_total]
                    if n_data_total > 0:
                        logging.info(
                            "Identified %d data events (%d total) — will be excluded from DNN training",
                            n_data_total, n_total,
                        )

                    # Step 1: Build physics objects & compute preselection mask
                    objects_all = build_objects(events, config)
                    cut_masks, _ = _build_event_cut_masks(events, objects_all, config)
                    presel_mask = _ak.ones_like(events["event"], dtype=bool)
                    for m in cut_masks.values():
                        presel_mask = presel_mask & _ak.fill_none(m, False, axis=0)

                    # Apply preselection to events, objects, and labels
                    selected_events = events[presel_mask]
                    selected_objects = {k: v[presel_mask] for k, v in objects_all.items()}
                    presel_mask_np = np.asarray(_ak.to_numpy(presel_mask), dtype=bool)
                    y_presel = y_full_orig[presel_mask_np]
                    keep_presel = keep_full_orig[presel_mask_np]
                    n_sel = len(selected_events)
                    logging.info(
                        "Preselection for DNN training: %d / %d events pass (%.1f%%)",
                        n_sel, n_total, 100.0 * n_sel / max(n_total, 1),
                    )

                    if n_sel == 0:
                        logging.error("No events pass preselection — cannot train DNN")
                        sys.exit(1)

                    # Step 2: Filter to MC-only events + compute weights
                    mc_mask = keep_presel  # True = non-data
                    n_mc = int(mc_mask.sum())
                    if n_mc == 0:
                        logging.error("No MC events after preselection — cannot train DNN")
                        sys.exit(1)
                    logging.info("MC events for DNN training: %d / %d preselected", n_mc, n_sel)

                    selected_events_mc = selected_events[mc_mask]
                    selected_objects_mc = {k: v[mc_mask] for k, v in selected_objects.items()}
                    y_train = y_presel[mc_mask]

                    proc_temp = DarkBottomLineProcessor(config)
                    if not config.get("data", {}).get("is_data", False):
                        try:
                            weight_results = proc_temp.correction_manager.compute_event_weights(
                                selected_events_mc, selected_objects_mc
                            )
                            full_weight = _ak.fill_none(
                                weight_results["full_event_weight"], 1.0, axis=0
                            )
                        except Exception as _w_exc:
                            logging.warning(
                                "Weight computation failed, using unit weights: %s", _w_exc
                            )
                            full_weight = _ak.ones_like(
                                selected_events_mc[selected_events_mc.fields[0]]
                            )
                    else:
                        full_weight = _ak.ones_like(
                            selected_events_mc[selected_events_mc.fields[0]]
                        )
                    selected_events_mc = _ak.with_field(
                        selected_events_mc, full_weight, "full_event_weight"
                    )
                    logging.info("MC weights attached to %d training events", n_mc)

                    # Step 3: Train DNN on MC-only preselected + weighted events
                    logging.info("DNN training on %d MC preselected events...", n_mc)
                    _scores_train, model_path = _train_dnn_on_events(
                        selected_events_mc, train_dnn_config, dnn_outdir, args,
                        objects=selected_objects_mc, config=config,
                        y_train=y_train,
                        dnn_plotdir=getattr(args, "dnn_plotdir", "outputs/dnn"),
                    )

                    # Step 4: Score ALL original events using the saved model
                    # (training was on preselected; scoring covers the full dataset
                    #  so ml_score is available for every event entering region analysis)
                    trainer_scoring = DNNTrainer(train_dnn_config)
                    trainer_scoring.load_model(model_path)

                    # Use standard variable pipeline for consistent feature values
                    _dnn_yaml = load_config(train_dnn_config)
                    feat_list = _dnn_yaml.get("features")
                    if not feat_list:
                        raise ValueError(
                            f"'{train_dnn_config}' has no features: list — required for DNN scoring."
                        )
                    X_full_df = _build_dnn_feature_matrix_from_events(
                        events, objects_all, config, feat_list
                    )
                    scores = (
                        trainer_scoring
                        .predict(
                            X_full_df.to_numpy(dtype="f8"),
                            np.zeros(n_total, dtype="f8"),
                        )
                        .ravel()
                        .astype("float32")
                    )
                    events = _ak.with_field(events, _ak.Array(scores), "ml_score")
                    logging.info(
                        "ml_score injected for all %d events "
                        "(model trained on %d preselected events)",
                        n_total, n_sel,
                    )

                    if dnn_only:
                        _plot_dnn_score_only(scores, dnn_plotdir)
                        logging.info(
                            "--dnn-only set: stopping after DNN scoring. "
                            "Training plots in %s", dnn_plotdir
                        )
                    else:
                        results = analyzer.process(
                            events,
                            event_selection_output=args.event_selection_output,
                            event_selection_only=False,
                            output_format=output_format_to_use,
                            total_events=total_events,
                        )
                        if args.output:
                            outdir = os.path.dirname(args.output)
                            if outdir:
                                os.makedirs(outdir, exist_ok=True)
                            analyzer.accumulator = results
                            analyzer.save_results(
                                args.output, output_format=args.output_format
                            )

                elif dnn_model:
                    # Score with existing model → inject ml_score in-memory → optionally region analysis
                    logging.info("Scoring events with DNN model: %s", dnn_model)
                    # Build objects for proper feature-name resolution (MET→MET_pt etc.)
                    from .objects import build_objects as _build_obj
                    _obj_for_dnn = _build_obj(events, config)
                    events = _add_dnn_scores_to_events(
                        events, dnn_model, dnn_config,
                        objects=_obj_for_dnn, config=config,
                    )
                    scores = np.asarray(ak.to_numpy(events["ml_score"]), dtype="f4")

                    if dnn_only:
                        _plot_dnn_score_only(scores, dnn_plotdir)
                        logging.info("--dnn-only set: stopping after DNN scoring. Plots in %s", dnn_plotdir)
                    else:
                        results = analyzer.process(events, event_selection_output=args.event_selection_output,
                                                  event_selection_only=False, output_format=output_format_to_use,
                                                  total_events=total_events)
                        if args.output:
                            outdir = os.path.dirname(args.output)
                            if outdir:
                                os.makedirs(outdir, exist_ok=True)
                            analyzer.accumulator = results
                            analyzer.save_results(args.output, output_format=args.output_format)

                else:
                    results = analyzer.process(events, event_selection_output=args.event_selection_output,
                                              event_selection_only=False, output_format=output_format_to_use,
                                              total_events=total_events, input_total_events=input_total_events)
                    if args.output:
                        outdir = os.path.dirname(args.output)
                        if outdir:
                            os.makedirs(outdir, exist_ok=True)
                        analyzer.accumulator = results
                        analyzer.save_results(args.output, output_format=args.output_format)

    except Exception as e:
        logging.error("Error in multi-region analysis: %s", e, exc_info=True)
        raise

    logging.info("Multi-region analysis completed!")
    _trigger_plots(args)


def make_trees(args):
    """Convert per-sample event-selection ROOT files → ppbbchichi-trees.root."""
    from dnn.make_trees import convert_files

    input_files = _get_input_files(args.input)
    summary = convert_files(
        input_files=input_files,
        output_path=args.output,
        signal_patterns=(args.signal_pattern or None),
        signal_prefix=args.signal_prefix,
        label_csv=args.label_csv,
        weight_branch=args.weight_branch,
        region_name=args.region,
        max_events_per_file=args.max_events,
        verbose=True,
    )
    n_sig = sum(1 for v in summary.values() if v["signal"])
    n_bkg = sum(1 for v in summary.values() if not v["signal"] and not v["isdata"])
    n_data = sum(1 for v in summary.values() if v["isdata"])
    logging.info(
        "make-trees done: %d samples (%d signal, %d background, %d data) → %s",
        len(summary), n_sig, n_bkg, n_data, args.output,
    )




def _load_one_eventsel_file(task: tuple):
    """Load + label + weight one Events ROOT file into a feature frame.

    Module-level (picklable) so it can run as a multiprocessing worker — each
    input file is read and processed completely independently of every other
    file, so this is dispatched across CPU cores instead of the serial
    per-file loop in ``_load_training_data_from_eventsel``.

    Returns either ("ok", fpath, sample, df, y_arr, w_arr, mass_arr, src, n)
    or ("skip", fpath, reason) or ("data", fpath) for a data file to skip silently.
    """
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    import uproot
    from dnn.make_trees import _sample_name, _is_data, _is_signal_heuristic
    from dnn.feature_engineering import build_feature_frame_from_tree
    from dnn.common import sanitize_feature_frame
    from .plotting import _find_xsec

    (fpath, feat_list, sig_patterns, signal_prefix, label_map, weight_branch,
     max_events_per_file, signal_cross_sections, background_cross_sections,
     lumi, parametric_input, mass_grid_list, seed, file_index) = task

    sample = _sample_name(fpath)

    if label_map:
        key = fpath if fpath in label_map else os.path.basename(fpath)
        if key not in label_map:
            return ("error", fpath, f"'{fpath}' not in label-csv")
        sig_flag = bool(label_map[key] == 1)
        data_flag = False
    else:
        data_flag = _is_data(fpath)
        sig_flag = False if data_flag else _is_signal_heuristic(fpath, sig_patterns, signal_prefix)

    if data_flag:
        return ("data", fpath)

    # Independent RNG per file (seeded deterministically from the run seed
    # and the file's position in the input list) so parametric mass sampling
    # stays reproducible across runs without serializing across files.
    _rng = np.random.default_rng((seed, file_index)) if parametric_input else None
    _mass_grid_arr = np.asarray(mass_grid_list, dtype="f8") if parametric_input else None

    try:
        with uproot.open(fpath) as in_f:
            if "Events" not in in_f:
                return ("skip", fpath, "No Events tree")
            tree = in_f["Events"]
            df, src, _ = build_feature_frame_from_tree(
                tree, feat_list,
                max_events=max_events_per_file,
            )
            df = sanitize_feature_frame(df)
            n = len(df)

            avail = set(tree.keys())
            if weight_branch in avail:
                w_arr = tree[weight_branch].array(entry_stop=n, library="np").astype("f8")
                w_arr = np.where(np.isfinite(w_arr), np.maximum(w_arr, 0.0), 0.0)
            else:
                w_arr = np.ones(n, dtype="f8")

            n = min(n, len(w_arr))

            wte = 0.0
            if signal_cross_sections or background_cross_sections:
                for key in ("weighted_total_events", "weighted_total_events;1"):
                    if key in in_f:
                        try:
                            wte = float(in_f[key].values()[0])
                            break
                        except Exception:
                            pass

            if sig_flag and signal_cross_sections:
                gm_cols = sorted(k for k in avail if str(k).startswith("GenModel_"))
                if gm_cols:
                    if wte > 0:
                        gm_arr = tree.arrays(gm_cols, entry_stop=n, library="np")
                        mp_scale = np.ones(n, dtype="f8")
                        for gmc in gm_cols:
                            mask = gm_arr[gmc][:n].astype(bool)
                            mp_label = gmc[len("GenModel_"):]
                            mp_xsec = _find_xsec(mp_label, signal_cross_sections)
                            if mp_xsec is not None:
                                mp_scale[mask] = (lumi * mp_xsec * 1000.0) / wte
                        w_arr = w_arr[:n] * mp_scale
                    else:
                        logging.warning(
                            "GenModel branches found in %s but no weighted_total_events — "
                            "skipping per-masspoint weighting", Path(fpath).name,
                        )
            elif (not sig_flag) and background_cross_sections:
                bkg_xsec = _find_xsec(sample, background_cross_sections)
                if bkg_xsec is not None and wte > 0:
                    bkg_scale = (lumi * bkg_xsec * 1000.0) / wte
                    w_arr = w_arr[:n] * bkg_scale
                else:
                    logging.warning(
                        "No xsec match or weighted_total_events for background file %s "
                        "(sample=%s) — using raw weight_branch only", Path(fpath).name, sample,
                    )

            mass_arr = None
            if parametric_input:
                if sig_flag:
                    gm_cols_mass = sorted(k for k in avail if str(k).startswith("GenModel_"))
                    mass_arr = np.full((n, 2), np.nan, dtype="f8")
                    if gm_cols_mass:
                        gm_arr_mass = tree.arrays(gm_cols_mass, entry_stop=n, library="np")
                        for gmc in gm_cols_mass:
                            parsed = _parse_masspoint_label(gmc[len("GenModel_"):])
                            if parsed is None:
                                continue
                            mask = gm_arr_mass[gmc][:n].astype(bool)
                            mass_arr[mask] = parsed
                    n_unlabeled = int(np.isnan(mass_arr).any(axis=1).sum())
                    if n_unlabeled:
                        idx = _rng.integers(0, len(_mass_grid_arr), size=n_unlabeled)
                        mass_arr[np.isnan(mass_arr).any(axis=1)] = _mass_grid_arr[idx]
                        logging.warning(
                            "%d/%d signal events in %s had no matching GenModel masspoint — "
                            "assigned a random grid mass instead", n_unlabeled, n, Path(fpath).name,
                        )
                else:
                    idx = _rng.integers(0, len(_mass_grid_arr), size=n)
                    mass_arr = _mass_grid_arr[idx]
    except Exception as _exc:
        return ("error", fpath, str(_exc)[:120])

    return ("ok", fpath, sample, df.iloc[:n], int(sig_flag), w_arr[:n], mass_arr, src, n)


def _load_training_data_from_eventsel(
    input_files: list,
    region: str,
    signal_patterns,
    signal_prefix,
    label_csv,
    weight_branch: str,
    max_events_per_file,
    signal_cross_sections: Optional[Dict[str, float]] = None,
    background_cross_sections: Optional[Dict[str, float]] = None,
    lumi: float = 1.0,
    features: Optional[List[str]] = None,
    parametric_input: bool = False,
    mass_grid: Optional[List[Tuple[float, float]]] = None,
    seed: int = 7,
) -> tuple:
    """In-memory conversion of flat Events ROOT files to labelled numpy arrays.

    Returns (X_df, y, w, feature_sources) — same format train_from_root expects
    after the data-loading phase, bypassing the intermediate ppbbchichi-trees.root.

    When *signal_cross_sections* is given, signal files with GenModel_* masspoint
    branches get their per-event weight additionally scaled by
    (lumi * masspoint_xsec * 1000 / weighted_total_events), mirroring the
    per-masspoint scaling already used for signal stacked plots
    (plotting.py _create_region_from_events_plots).

    When *background_cross_sections* is given, background files get their
    per-event weight scaled by (lumi * file_xsec * 1000 / weighted_total_events),
    so different background processes contribute in proportion to their true
    physical yield rather than their raw MC event count.

    Without either, behavior is unchanged: every event keeps the plain
    weight_branch value.

    When *parametric_input* is true, an extra (N, 2) numpy array of (MH3, MH4)
    values is returned as the 5th tuple element: signal events get their true
    masspoint (parsed from the GenModel_* branch that is set for that event),
    background events get a masspoint drawn uniformly at random (with
    replacement, seeded by *seed*) from *mass_grid*. When false, the 5th
    element is None.
    """
    import pandas as pd
    import os
    import csv as _csv

    if not features:
        raise ValueError(
            "No features provided — configs/dnn.yaml must have a features: list."
        )
    feat_list = list(features)
    sig_patterns = tuple(signal_patterns) if signal_patterns else ()

    if parametric_input and not mass_grid:
        raise ValueError(
            "parametric_input is true but no mass_grid was provided — "
            "pass --xsection-signal-json so the (MH3, MH4) grid can be derived."
        )

    label_map: dict = {}
    if label_csv:
        with open(label_csv, "r", newline="") as fp:
            reader = _csv.DictReader(fp)
            for row in reader:
                label_map[str(row["path"]).strip()] = int(row["label"])

    X_parts, y_parts, w_parts = [], [], []
    mass_parts: list = []
    feature_sources: dict = {}
    skipped_files: list = []  # (filepath, reason)

    # Each input file is read and processed completely independently — parallelize
    # across CPU cores via multiprocessing (uproot read + feature-frame build is
    # CPU/IO-bound, so a thread pool would just serialize on the GIL for the
    # numpy/pandas-heavy parts). Override with DNN_LOAD_WORKERS.
    tasks = [
        (
            fpath, feat_list, sig_patterns, signal_prefix, label_map, weight_branch,
            max_events_per_file, signal_cross_sections, background_cross_sections,
            lumi, parametric_input, mass_grid, seed, idx,
        )
        for idx, fpath in enumerate(input_files)
    ]
    num_workers = int(os.environ.get("DNN_LOAD_WORKERS", max(1, (os.cpu_count() or 1))))
    num_workers = max(1, min(num_workers, len(tasks)))

    if num_workers > 1 and len(tasks) > 1:
        import multiprocessing as mp

        logging.info(f"Loading {len(tasks)} input file(s) with {num_workers} worker processes")
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=num_workers) as pool:
            load_results = pool.map(_load_one_eventsel_file, tasks)
    else:
        load_results = [_load_one_eventsel_file(t) for t in tasks]

    for result in load_results:
        kind = result[0]
        if kind == "data":
            logging.info("Skipping data file for DNN training: %s", result[1])
            continue
        if kind == "skip":
            _, fpath, reason = result
            logging.warning("%s in %s — skipping", reason, fpath)
            skipped_files.append((fpath, reason))
            continue
        if kind == "error":
            _, fpath, reason = result
            if "not in label-csv" in reason:
                raise KeyError(reason)
            logging.warning("Failed to read %s — skipping (%s)", fpath, reason)
            skipped_files.append((fpath, reason))
            continue

        _, fpath, sample, df, sig_flag_int, w_arr, mass_arr, src, n = result
        X_parts.append(df)
        y_parts.append(np.full(n, sig_flag_int, dtype="i4"))
        w_parts.append(w_arr)
        if parametric_input:
            mass_parts.append(mass_arr)
        for feat in df.columns:
            if feat not in feature_sources:
                feature_sources[feat] = src.get(feat, "unknown")

        logging.info("Loaded %s: n=%d signal=%d", sample, n, sig_flag_int)

    if not X_parts:
        logging.error("No training events loaded — check input files and signal/background flags.")
        if skipped_files:
            logging.error("Skipped files summary (%d total):", len(skipped_files))
            for fpath, reason in skipped_files:
                logging.error("  SKIPPED: %s", fpath)
                logging.error("    Reason: %s", reason)
        raise ValueError("No training events loaded — check input files and signal/background flags.")

    # Report skipped files
    if skipped_files:
        logging.warning("=" * 60)
        logging.warning("SKIPPED FILES SUMMARY (%d total):", len(skipped_files))
        for fpath, reason in skipped_files:
            logging.warning("  SKIP: %s", fpath)
            logging.warning("        %s", reason)
        logging.warning("=" * 60)

    import pandas as pd
    X = pd.concat(X_parts, axis=0, ignore_index=True)
    y = np.concatenate(y_parts)
    w = np.concatenate(w_parts)
    mass = np.concatenate(mass_parts, axis=0) if parametric_input else None
    return X, y, w, feature_sources, mass


def train_dnn(args):
    """Train DNN from event-selection ROOT files (no intermediate file needed)."""
    import os
    import json

    trainer = DNNTrainer(args.config)

    raw_inputs = _get_input_files(args.input)
    input_files = []
    for p in raw_inputs:
        if os.path.isdir(p):
            input_files.extend(sorted(str(f) for f in Path(p).iterdir() if f.suffix == ".root"))
        else:
            input_files.append(p)
    logging.info("Training DNN from %d input file(s)", len(input_files))

    # Signal cross sections for GenModel per-masspoint event weighting (optional).
    signal_cross_sections: Dict[str, float] = {}
    xsec_signal_json = getattr(args, "xsection_signal_json", None)
    if xsec_signal_json:
        with open(xsec_signal_json) as f:
            _sig_raw = json.load(f)
        for _model, _entries in _sig_raw.items():
            if isinstance(_entries, dict):
                for _k, _v in _entries.items():
                    if _k.startswith("_") or not isinstance(_v, (int, float)):
                        continue
                    signal_cross_sections[_k] = float(_v)

    # Background cross sections for per-file lumi*xsec/wte event weighting (optional).
    background_cross_sections: Dict[str, float] = {}
    xsec_json = getattr(args, "xsection_json", None)
    if xsec_json:
        from .plotting import PlotManager
        with open(xsec_json) as f:
            _bkg_raw = json.load(f)
        background_cross_sections = PlotManager._normalize_cross_sections(_bkg_raw)

    # Lumi: training is year-independent by default (lumi=1.0) unless dnn.yaml's
    # training.use_lumi is true and --config-year is given.
    lumi = 1.0
    if trainer.training_config.get("use_lumi", False) and getattr(args, "config_year", None):
        _year_cfg = load_config(args.config_year)
        lumi = float(_year_cfg.get("lumi", _year_cfg.get("luminosity", 1.0)))

    # Parametric-DNN: derive the (MH3, MH4) mass grid from the signal xsec
    # masspoint labels, needed to sample background masses and to record the
    # grid in the checkpoint.
    parametric_input = bool(trainer.model_config.get("parametric_input", False))
    mass_grid = None
    if parametric_input:
        mass_grid = sorted({
            parsed for label in signal_cross_sections
            if (parsed := _parse_masspoint_label(label)) is not None
        })
        if not mass_grid:
            raise ValueError(
                "model.parametric_input is true in dnn.yaml but no (MH3, MH4) "
                "masspoints could be parsed — pass --xsection-signal-json."
            )

    # Load feature matrix directly from flat Events trees — no ppbbchichi-trees.root written
    X, y, w, feature_sources, mass = _load_training_data_from_eventsel(
        input_files=input_files,
        region=getattr(args, "region", "preselection"),
        signal_patterns=(args.signal_pattern or None),
        signal_prefix=args.signal_prefix,
        label_csv=args.label_csv,
        weight_branch=getattr(args, "weight_branch", "full_event_weight"),
        max_events_per_file=args.max_events_per_sample,
        signal_cross_sections=signal_cross_sections or None,
        background_cross_sections=background_cross_sections or None,
        lumi=lumi,
        features=trainer.config.get("features") or None,
        parametric_input=parametric_input,
        mass_grid=mass_grid,
        seed=int(trainer.training_config.get("seed", 7)),
    )

    metrics = trainer.train_from_arrays(
        X=X,
        y=y,
        w=w,
        feature_sources=feature_sources,
        outdir=args.outdir,
        plot_dir=args.plot_dir,
        mass=mass,
    )

    logging.info(
        "DNN training complete — AUC(val)=%.4f  AUC(test)=%.4f",
        metrics.get("auc_val", float("nan")),
        metrics.get("auc_test", float("nan")),
    )


def apply_dnn(args):
    """Score events in event-selection ROOT files with a trained DNN model.

    Reads each flat Events ROOT file, applies the trained model, and writes
    a new branch (default: ml_score) back to the file — or to a new output
    ROOT file when --output-dir is given.

    For a parametric model, by default every event is scored once at a single
    benchmark masspoint (the checkpoint's mass_grid[0]). --dnn-mass-scan
    requests scoring at multiple grid points instead, writing one branch per
    point named "<score_branch>_mh3_<a>_mh4_<b>".
    """
    import pandas as pd
    from dnn.feature_engineering import build_feature_frame_from_tree
    from dnn.common import sanitize_feature_frame

    inference = DNNInference(args.model, config_path=args.config)
    features = inference.features
    score_branch = args.score_branch

    mass_scan = _resolve_mass_scan(getattr(args, "dnn_mass_scan", None), inference)

    input_files = _get_input_files(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    for fpath in input_files:
        with uproot.open(fpath) as in_f:
            if "Events" not in in_f:
                logging.warning("No 'Events' tree in %s, skipping", fpath)
                continue
            tree = in_f["Events"]
            df, _, _ = build_feature_frame_from_tree(tree, features)
            df = sanitize_feature_frame(df)
            n = len(df)
            X = df.to_numpy(dtype="f8")

            if mass_scan is None:
                scores = inference.predict(X, None).ravel()
                score_branches = {score_branch: scores.astype("f4")}
            else:
                score_branches = {}
                for mh3, mh4 in mass_scan:
                    masses = np.tile(np.asarray([mh3, mh4], dtype="f8"), (n, 1))
                    scores = inference.predict(X, masses).ravel()
                    score_branches[_mass_branch_name(score_branch, mh3, mh4)] = scores.astype("f4")

            # Collect all existing branches
            arrays = tree.arrays(library="np")

        arrays.update(score_branches)

        if output_dir:
            out_path = output_dir / Path(fpath).name
        else:
            out_path = fpath  # overwrite in-place

        with uproot.recreate(str(out_path)) as out_f:
            out_f["Events"] = arrays

        logging.info("Scored %s: n=%d → %s (branches: %s)", Path(fpath).name, n, out_path, list(score_branches))


def make_plots(args):
    """Create data/MC plots."""
    logging.info("Creating plots...")

    # Load results
    import pickle
    with open(args.input, 'rb') as f:
        results = pickle.load(f)

    # Load plotting config if provided
    plot_config = None
    if args.plot_config:
        plot_config = load_config(args.plot_config)
        logging.info(f"Loaded plotting configuration from {args.plot_config}")
    else:
        # Try to load default plotting config
        default_plot_config_path = Path(__file__).parent.parent / "configs" / "plotting.yaml"
        if default_plot_config_path.exists():
            plot_config = load_config(str(default_plot_config_path))
            logging.info(f"Loaded default plotting configuration from {default_plot_config_path}")

    # Determine luminosity: prefer --config arg, fall back to metadata stored in pkl
    luminosity = None
    if getattr(args, "config", None):
        year_cfg = load_config(args.config)
        luminosity = float(year_cfg.get("lumi", 1.0))
        logging.info(f"Luminosity from --config: {luminosity} fb-1")
    if luminosity is None:
        luminosity = float(results.get("metadata", {}).get("luminosity", 1.0))
        if luminosity != 1.0:
            logging.info(f"Luminosity from pkl metadata: {luminosity} fb-1")

    # Load cross sections if provided
    cross_sections = {}
    if getattr(args, "xsection_json", None):
        import json
        with open(args.xsection_json) as _f:
            cross_sections = json.load(_f)
        logging.info(f"Loaded {len(cross_sections)} cross sections from {args.xsection_json}")

    # Initialize plot manager with config
    plot_manager = PlotManager(config=plot_config)

    # Generate version string if not provided (format: YYYYMMDD_HHMM)
    if not args.version:
        version = _default_version()
    else:
        version = args.version

    # Create output directory
    import os
    os.makedirs(args.save_dir, exist_ok=True)

    # Create plots with all formats automatically (PNG, PDF, ROOT, TXT)
    plot_files = plot_manager.create_all_plots(
        results, args.save_dir, args.show_data, args.regions, version,
        formats=None, luminosity=luminosity, cross_sections=cross_sections,
    )

    logging.info(f"Plots saved to {args.save_dir}")
    logging.info("Plot creation completed!")


def make_single_plots(args):
    """Create plots from a single event-level analysis file."""
    logging.info("Creating single plots from event-level file...")

    # Load results (which are 'events' and 'objects' in this case)
    import pickle
    import numpy as np
    import awkward as ak
    with open(args.input, 'rb') as f:
        loaded_data = pickle.load(f)

    events_list = loaded_data.get('events')
    objects_dict_list = loaded_data.get('objects')

    if events_list is None or objects_dict_list is None:
        logging.error(f"Input file {args.input} does not contain 'events' or 'objects' keys required for single plotting.")
        sys.exit(1)

    # Convert lists back to awkward arrays as HistogramManager expects them
    events = ak.Array(events_list)
    objects = {}
    for k, v in objects_dict_list.items():
        if v is not None: # Ensure the list is not None before converting
            objects[k] = ak.Array(v)
        else:
            objects[k] = ak.Array([]) # Or an empty awkward array if None

    # Initialize HistogramManager and define histograms
    # A minimal config might be needed for HistogramManager if it relies on it.
    # For now, let's assume it can be initialized without extensive config,
    # or that default parameters are sufficient.
    from .histograms import HistogramManager
    histogram_manager = HistogramManager()

    # Define histograms
    defined_histograms = histogram_manager.define_histograms()

    # Create dummy weights for filling histograms
    # Event-level files from selection might not contain weights
    dummy_weights = np.ones(len(events))

    # Fill histograms with the loaded events and objects
    # Note: DarkBottomLineProcessor.process usually handles this with full corrections/weights
    # Here, we do a minimal filling for plotting purposes.
    filled_histograms = histogram_manager.fill_histograms(
        events, objects, dummy_weights
    )

    # Construct a results dictionary that the PlotManager expects
    # For event-level plots, we create a pseudo-results dict with only the 'histograms'
    pseudo_results = {"histograms": filled_histograms}

    # Load plotting config if provided
    plot_config = None
    if args.plot_config:
        plot_config = load_config(args.plot_config)
        logging.info(f"Loaded plotting configuration from {args.plot_config}")
    else:
        # Try to load default plotting config
        default_plot_config_path = Path(__file__).parent.parent / "configs" / "plotting.yaml"
        if default_plot_config_path.exists():
            plot_config = load_config(str(default_plot_config_path))
            logging.info(f"Loaded default plotting configuration from {default_plot_config_path}")

    # Initialize plot manager with config
    plot_manager = PlotManager(config=plot_config)

    # Generate version string if not provided (format: YYYYMMDD_HHMM)
    if not args.version:
        version = _default_version()
    else:
        version = args.version

    # Create output directory
    import os
    os.makedirs(args.save_dir, exist_ok=True)

    # Call the new event-level plotting function
    plot_files = plot_manager.create_event_level_variable_plots(
        pseudo_results, args.save_dir, args.show_data, version
    )

    logging.info(f"Single plots saved to {args.save_dir}")
    logging.info("Single plot creation completed!")


def make_stacked_plots(args):
    """Create stacked Data/MC plots with ratio and uncertainty band."""
    logging.info("Creating stacked plots...")

    # Load plotting config if provided
    plot_config = None
    if args.plot_config:
        plot_config = load_config(args.plot_config)
        logging.info(f"Loaded plotting configuration from {args.plot_config}")
    else:
        # Try to load default plotting config
        default_plot_config_path = Path(__file__).parent.parent / "configs" / "plotting.yaml"
        if default_plot_config_path.exists():
            plot_config = load_config(str(default_plot_config_path))
            logging.info(f"Loaded default plotting configuration from {default_plot_config_path}")

    plot_manager = PlotManager(config=plot_config)

    # Generate version string if not provided (format: YYYYMMDD_HHMM)
    if not args.version:
        version = _default_version()
    else:
        version = args.version

    # Parse inputs
    data_file = args.data
    bkg_files = args.backgrounds or []
    signal_file = args.signal
    output = args.output
    variable = args.variable
    region = args.region
    xlabel = args.xlabel
    title_tag = args.title

    # Run with multi-format saving
    out = plot_manager.create_stacked_plot_from_files(
        data_file=data_file,
        background_files=bkg_files,
        signal_file=signal_file,
        output_path=output,
        variable=variable,
        region=region,
        xlabel=xlabel,
        title_tag=title_tag,
        version=version,
        formats=None  # All formats generated automatically
    )

    logging.info(f"Stacked plot saved to {out}")


def make_event_plots(args):
    """Create stacked event-selection or region plots (ROOT→PKL→plot)."""
    import json

    config = load_config(args.config)
    luminosity = float(config.get("lumi", config.get("luminosity", 1.0)))
    year = str(config.get("year", ""))

    plot_config = None
    if args.plot_config:
        plot_config = load_config(args.plot_config)
    else:
        default_cfg = Path(__file__).parent.parent / "configs" / "plotting.yaml"
        if default_cfg.exists():
            plot_config = load_config(str(default_cfg))

    plot_manager = PlotManager(config=plot_config)

    # Process groups: CLI JSON overrides plotting.yaml process_groups entirely
    if args.process_groups:
        with open(args.process_groups) as f:
            raw_groups = json.load(f)
        # Re-parse through PlotManager logic by injecting into a fresh config
        from .plotting import PlotManager as _PM
        _tmp = _PM(config={"process_groups": raw_groups})
        process_groups = _tmp.process_groups
        signal_groups  = _tmp.signal_groups
        data_groups    = _tmp.data_groups
    else:
        process_groups = plot_manager.process_groups
        signal_groups  = plot_manager.signal_groups
        data_groups    = plot_manager.data_groups

    if not process_groups:
        raise SystemExit(
            "No background process_groups defined. "
            "Add process_groups to plotting.yaml or pass --process-groups JSON."
        )

    cross_sections: dict = {}
    if args.xsection_json:
        with open(args.xsection_json) as f:
            _raw = json.load(f)
        # Flatten nested format {category: [{full_dataset, xsection, year}]}
        # into {full_dataset_stem: xsec_pb} for direct stem lookup.
        # Also keep flat {stem: xsec} format if already flat.
        year_str = str(config.get("year", ""))
        for _cat, _entries in _raw.items():
            if isinstance(_entries, list):
                for _e in _entries:
                    if not isinstance(_e, dict):
                        continue
                    _ds = _e.get("full_dataset") or _e.get("dataset")
                    _xsec = _e.get("xsection")
                    _yr = str(_e.get("year", ""))
                    if _ds is None or _xsec is None:
                        continue
                    # full_dataset may be a single name or a list of per-era
                    # dataset-name variants — register each variant as a key.
                    _names = _ds if isinstance(_ds, (list, tuple)) else [_ds]
                    for _name in _names:
                        if not _name:
                            continue
                        # prefer matching year; always overwrite so last match wins
                        if not year_str or _yr == year_str or _name not in cross_sections:
                            cross_sections[_name] = float(_xsec)
            elif isinstance(_entries, (int, float)):
                # already flat: {stem: xsec}
                cross_sections[_cat] = float(_entries)

    # Signal cross sections: {model: {masspoint: xsec}} — flatten all models into cross_sections
    if getattr(args, "xsection_signal_json", None):
        with open(args.xsection_signal_json) as f:
            _sig_raw = json.load(f)
        for _model, _entries in _sig_raw.items():
            if isinstance(_entries, dict):
                for _k, _v in _entries.items():
                    if _k.startswith("_"):
                        continue  # skip _comment etc.
                    if isinstance(_v, (int, float)):
                        cross_sections[_k] = float(_v)
            elif isinstance(_entries, (int, float)):
                cross_sections[_model] = float(_entries)
        logging.info("Loaded signal cross sections from %s (%d masspoints)",
                     args.xsection_signal_json,
                     sum(1 for k in cross_sections if k.startswith("MH")))

    version = args.version
    if not version:
        version = _default_version()

    out_files = plot_manager.create_stacked_plots(
        mode=args.mode,
        input_folder=args.input_folder,
        process_groups=process_groups,
        signal_groups=signal_groups,
        data_groups=data_groups,
        output_dir=args.output_dir,
        luminosity=luminosity,
        year=year,
        version=version,
        cross_sections=cross_sections if cross_sections else None,
        variables=args.variables or None,
        regions=args.regions or None,
        save_root=args.save_root,
        regions_config=getattr(args, "regions_config", None),
        weight_systematic=getattr(args, "weight_systematic", None),
        show_data=getattr(args, "show_data", False),
        signal_scale=float(getattr(args, "signal_scale", 1.0) or 1.0),
        make_syst_plots=getattr(args, "make_syst_plots", False),
        apply_dnn=getattr(args, "apply_dnn", False),
        dnn_model=getattr(args, "dnn_model", None),
        dnn_config=getattr(args, "dnn_config", None),
        dnn_mass_scan=getattr(args, "dnn_mass_scan", None),
    )
    logging.info(f"analyze-regions: {len(out_files)} plot(s) written to {args.output_dir}")


def make_datacard(args):
    """Generate Combine datacard."""
    logging.info("Generating datacard...")

    # Load results
    # results = load_results(args.input)

    # Generate datacard (placeholder)
    # datacard_writer = CombineDatacardWriter()
    # datacard_writer.write_datacard(results, args.output)

    logging.info("Datacard generation completed!")


def run_combine(args):
    """Run Combine fits."""
    logging.info("Running Combine fits...")

    # Run Combine command (placeholder)
    # combine_runner = CombineRunner()
    # results = combine_runner.run_fit(args.mode, args.datacard, args.options)

    logging.info("Combine execution completed!")


def make_impact(args):
    """Create impact plots."""
    logging.info("Creating impact plots...")

    # Load fit results
    # results = load_fit_results(args.input)

    # Create impact plots (placeholder)
    # diagnostic_plotter = DiagnosticPlotter()
    # diagnostic_plotter.plot_impacts(results, args.output)

    logging.info("Impact plot creation completed!")


def make_pulls(args):
    """Create pull plots."""
    logging.info("Creating pull plots...")

    # Load fit results
    # results = load_fit_results(args.input)

    # Create pull plots (placeholder)
    # diagnostic_plotter = DiagnosticPlotter()
    # diagnostic_plotter.plot_pulls(results, args.output)

    logging.info("Pull plot creation completed!")


def make_gof(args):
    """Create goodness-of-fit plots."""
    logging.info("Creating GOF plots...")

    # Load GOF results
    # results = load_gof_results(args.input)

    # Create GOF plots (placeholder)
    # diagnostic_plotter = DiagnosticPlotter()
    # diagnostic_plotter.plot_gof(results, args.output)

    logging.info("GOF plot creation completed!")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="DarkBottomLine Framework - Advanced Analysis Tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run basic analysis
  darkbottomline run --config configs/2023.yaml --input data.root --output results.coffea

  # Run multi-region analysis
  darkbottomline analyze --config configs/2023.yaml --regions-config configs/regions.yaml --input data.root --output results.coffea

  # Train DNN
  darkbottomline train-dnn --config configs/dnn.yaml --signal signals.root --background bkg.root --output model.pt

  # Create plots
  darkbottomline make-plots --year 2023 --region SR --show-data False --save-dir outputs/plots/

  # Generate datacard
  darkbottomline make-datacard --region SR --output outputs/combine/ --year 2023

  # Run Combine fits
  darkbottomline run-combine --mode FitDiagnostics --datacard outputs/combine/datacard.txt

  # Create diagnostic plots
  darkbottomline make-impact --input outputs/combine/fitDiagnostics.root --output outputs/plots/
        """
    )

    # Global arguments
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Logging level")

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run basic analysis")
    run_parser.add_argument("--config", required=True, help="Configuration file")
    run_parser.add_argument("--input", nargs="+", required=True, help="Input file(s), can be a single .txt file listing paths")
    run_parser.add_argument("--output", required=True, help="Output file")
    run_parser.add_argument("--event-selection-output", help="Path to save events that pass event-level selection (optional)")
    run_parser.add_argument("--executor", choices=["iterative", "futures", "dask"],
                           default="iterative", help="Execution backend")
    run_parser.add_argument("--workers", type=int, default=4, help="Number of workers")
    run_parser.add_argument("--max-events", type=int, help="Maximum events to process")
    run_parser.add_argument("--data", action="store_true",
                           help="Input is collision data: apply golden JSON lumi mask and skip MC-only weights")
    run_parser.set_defaults(func=run_analysis)

    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Run analysis pipeline (event-selection, region analysis, or both)",
    )
    analyze_parser.add_argument("--config", required=True, help="Base configuration file")
    analyze_parser.add_argument("--regions-config", required=False,
                               help="Regions config YAML (required for full/region-analysis modes)")
    analyze_parser.add_argument("--input", nargs="+", required=True,
                               help="Input file(s): NanoAOD ROOT, EVENTSELECTION.root, folder, or .txt list")
    analyze_parser.add_argument(
        "--mode", default="full",
        choices=["full", "event-selection", "region-analysis"],
        help=(
            "Pipeline mode: "
            "'full' = NanoAOD → event-selection + region analysis (default); "
            "'event-selection' = NanoAOD → EVENTSELECTION.root (add --make-event-selection-plots to plot instead); "
            "'region-analysis' = EVENTSELECTION.root → region cuts + plots"
        ),
    )
    analyze_parser.add_argument("--output", required=False,
                               help="Output PKL file (full/region-analysis modes)")
    analyze_parser.add_argument("--event-selection-output",
                               help="Path to save EVENTSELECTION.root (full and event-selection modes)")
    analyze_parser.add_argument("--output-format", default="pkl",
                               choices=["pkl", "root", "parquet"],
                               help="Output file format (default: pkl)")
    analyze_parser.add_argument("--executor", choices=["iterative", "futures", "dask"],
                               default="iterative", help="Execution backend")
    analyze_parser.add_argument("--workers", type=int, default=4, help="Number of workers")
    analyze_parser.add_argument("--chunk-size", type=str, default=None,
                               help="Events per chunk for futures/dask (default: auto)")
    analyze_parser.add_argument("--max-events", type=int,
                               help="Maximum events to process across all chunks")
    analyze_parser.add_argument("--data", action="store_true",
                               help="Collision data: apply golden JSON lumi mask, skip MC weights")
    analyze_parser.add_argument("--xsection-json", default=None, metavar="JSON",
                               help="JSON mapping filename stem → cross-section in pb (region-analysis mode)")
    analyze_parser.add_argument("--xsection-signal-json", default=None, metavar="JSON",
                               help="JSON with signal cross sections: {model: {masspoint: xsec_pb}} e.g. data/cross-section/xsection_signal.json")
    analyze_parser.add_argument("--signal-scale", type=float, default=1.0, metavar="N",
                               help="Multiply all signal histograms by N for shape visibility (shown as ×N in legend, default: 1)")
    analyze_parser.add_argument("--make-syst-plots", action="store_true", default=False,
                               help="Produce systematic comparison plots (central+up+down per uncertainty) in outputs/plots/{version}/systematics/")
    # Plotting flags
    analyze_parser.add_argument("--make-region-plots", action="store_true", default=False,
                               help="Produce stacked region plots (ROOT→PKL→plot) (pdf/png/txt/root) — region-analysis and full modes")
    analyze_parser.add_argument("--skip-plots", action="store_true", default=False,
                               help="Skip plot generation")
    analyze_parser.add_argument("--make-event-selection-plots", action="store_true", default=False,
                               help="Produce stacked event-selection plots (before region cuts)")
    analyze_parser.add_argument("--plot-config", default=None,
                               help="Plotting YAML (default: configs/plotting.yaml)")
    analyze_parser.add_argument("--output-dir", default="outputs",
                               help="Output directory for plots (default: outputs)")
    analyze_parser.add_argument("--show-data", action="store_true", default=False,
                               help="Unblind SR: show real data in SR plots (default: bkg-sum as pseudo-data)")
    analyze_parser.add_argument("--save-root", action="store_true",
                               help="Also save ROOT TH1 files for plots")
    analyze_parser.add_argument("--plot-variables", nargs="+", default=None, metavar="VAR",
                               help="Variables to plot (default: all)")
    analyze_parser.add_argument("--region-results", default=None, metavar="PKL",
                               help="Use pre-computed PKL")
    analyze_parser.add_argument("--plot-regions", nargs="+", default=None, metavar="REGION",
                               help="Regions to plot (default: all)")
    analyze_parser.add_argument("--version", default=None,
                               help="Version tag for plot output subdirectory (default: timestamp)")
    analyze_parser.add_argument("--weight-systematic", default=None, metavar="BRANCH",
                               help="Weight branch override for plots (e.g. weight_pileupUP)")
    # DNN integration flags
    analyze_parser.add_argument(
        "--apply-dnn", action="store_true",
        help="Score events with --dnn-model/--dnn-config in the stacked-plot "
             "(--make-region-plots) path and add ml_score as a plotted variable.",
    )
    analyze_parser.add_argument(
        "--dnn-model", default=None,
        help="Path to trained DNN checkpoint (.pt). Scores events before region analysis.",
    )
    analyze_parser.add_argument(
        "--dnn-config", default=None,
        help="DNN config YAML (e.g. configs/dnn.yaml).",
    )
    analyze_parser.add_argument(
        "--dnn-mass-scan", default=None,
        help="Parametric models only. Omit (default) to score once at the "
             "checkpoint's benchmark masspoint (mass_grid[0]). Pass 'all' to "
             "score at every grid point (produces one ml_score_mh3_<a>_mh4_<b> "
             "branch and one full set of region plots per masspoint), or a "
             "comma list of MH3_<a>_MH4_<b>_Mchi_<c> labels to scan a subset. "
             "Ignored for non-parametric models.",
    )
    analyze_parser.add_argument(
        "--train-dnn", default=None,
        help="DNN config YAML path: train DNN on event-selection output before region analysis.",
    )
    analyze_parser.add_argument(
        "--dnn-outdir", default="data/dnn",
        help="Output directory for DNN model artifacts: dnn_model.pt, scaler.json, etc. (default: data/dnn)",
    )
    analyze_parser.add_argument(
        "--dnn-plotdir", default="outputs/dnn",
        help="Output directory for DNN training plots (default: outputs/dnn)",
    )
    analyze_parser.add_argument(
        "--dnn-only", action="store_true",
        help="Stop after DNN scoring — produce score plot, skip region analysis.",
    )
    analyze_parser.add_argument(
        "--signal-pattern", action="append", default=None, dest="signal_pattern",
        help="Regex to identify signal files for DNN training (repeatable)",
    )
    analyze_parser.add_argument("--signal-prefix", default=None,
                               help="Filename prefix marking signal samples for DNN training")
    analyze_parser.add_argument("--label-csv", default=None,
                               help="CSV with columns path,label for DNN training label assignment")
    analyze_parser.set_defaults(func=run_analyzer)

    # Make trees command
    # Converts per-sample flat Events ROOT files → ppbbchichi-trees.root (sample/region structure)
    # Run this between `analyze --event-selection-only` and `train-dnn`
    make_trees_parser = subparsers.add_parser(
        "make-trees",
        help="Convert event-selection ROOT outputs to ppbbchichi-trees.root for DNN training",
    )
    make_trees_parser.add_argument(
        "--input", nargs="+", required=True,
        help="Event-selection ROOT files (one per sample), or a .txt file listing paths",
    )
    make_trees_parser.add_argument(
        "--output", required=True,
        help="Output ppbbchichi-trees.root path",
    )
    make_trees_parser.add_argument(
        "--region", default="preselection",
        help="Region name used as TTree name inside each sample dir (default: preselection)",
    )
    make_trees_parser.add_argument(
        "--signal-pattern", action="append", default=None, dest="signal_pattern",
        help="Regex to identify signal files (repeatable). Default: keyword heuristic",
    )
    make_trees_parser.add_argument(
        "--signal-prefix", default=None,
        help="Filename prefix that marks signal, e.g. 'bbDM'",
    )
    make_trees_parser.add_argument(
        "--label-csv", default=None,
        help="CSV with columns path,label (1=signal, 0=background) — overrides pattern/prefix",
    )
    make_trees_parser.add_argument(
        "--weight-branch", default="full_event_weight",
        help="Branch name to use as event weight (default: full_event_weight)",
    )
    make_trees_parser.add_argument(
        "--max-events", type=int, default=None,
        help="Max events per input file (default: all)",
    )
    make_trees_parser.set_defaults(func=make_trees)

    # Train DNN command
    # Input: flat per-sample ROOT files from `analyze --event-selection-only`
    # (no intermediate ppbbchichi-trees.root needed)
    train_dnn_parser = subparsers.add_parser(
        "train-dnn",
        help="Train DNN classifier from event-selection ROOT output files",
    )
    train_dnn_parser.add_argument("--dnn-config", dest="config", required=True,
                                   help="DNN configuration YAML (configs/dnn.yaml)")
    train_dnn_parser.add_argument(
        "--input", nargs="+", required=True,
        help="Event-selection ROOT files (one per sample), a folder of them, or a .txt file listing paths",
    )
    train_dnn_parser.add_argument("--region", default="preselection", help="Region label (default: preselection)")
    train_dnn_parser.add_argument("--outdir", default="data/dnn", help="Output directory for model + metrics (default: data/dnn)")
    train_dnn_parser.add_argument("--plot-dir", default="outputs/dnn", help="Output directory for plots (default: outputs/dnn)")
    train_dnn_parser.add_argument(
        "--signal-pattern", action="append", default=None, dest="signal_pattern",
        help="Regex to identify signal files (repeatable). Default: keyword heuristic",
    )
    train_dnn_parser.add_argument(
        "--signal-prefix", default=None,
        help="Filename prefix that marks signal, e.g. 'bbDM'",
    )
    train_dnn_parser.add_argument(
        "--label-csv", default=None,
        help="CSV with columns path,label (1=signal, 0=background) — overrides pattern/prefix",
    )
    train_dnn_parser.add_argument(
        "--weight-branch", default="full_event_weight",
        help="Branch name to use as event weight (default: full_event_weight)",
    )
    train_dnn_parser.add_argument(
        "--max-events-per-sample", type=int, default=200000,
        help="Cap events loaded per sample (default: 200000)",
    )
    train_dnn_parser.add_argument(
        "--xsection-signal-json", default=None,
        help="Signal cross-section JSON (data/cross-section/xsection_signal.json) for "
             "GenModel per-masspoint event weighting. Omit to keep uniform per-file weight.",
    )
    train_dnn_parser.add_argument(
        "--xsection-json", default=None,
        help="Background cross-section JSON (data/cross-section/xsection_background_run3.json) "
             "for per-file lumi*xsec/weighted_total_events event weighting. "
             "Omit to keep raw weight_branch only.",
    )
    train_dnn_parser.add_argument(
        "--config-year", default=None,
        help="Optional year config (e.g. configs/2024.yaml) to source lumi from. "
             "Only used when dnn.yaml training.use_lumi is true; otherwise lumi=1.0.",
    )
    train_dnn_parser.set_defaults(func=train_dnn)

    # Apply DNN command — score events with a trained model, write ml_score branch
    apply_dnn_parser = subparsers.add_parser(
        "apply-dnn",
        help="Apply trained DNN to event-selection ROOT files and write per-event score",
    )
    apply_dnn_parser.add_argument(
        "--input", nargs="+", required=True,
        help="Event-selection ROOT files (one per sample), or a .txt file listing paths",
    )
    apply_dnn_parser.add_argument(
        "--model", required=True,
        help="Path to trained model checkpoint (.pt file from train-dnn)",
    )
    apply_dnn_parser.add_argument(
        "--config", default=None,
        help="DNN config YAML (optional — used to resolve feature list if not in checkpoint)",
    )
    apply_dnn_parser.add_argument(
        "--output-dir", default=None,
        help="Write scored files here (default: overwrite input files in-place)",
    )
    apply_dnn_parser.add_argument(
        "--score-branch", default="ml_score",
        help="Name of the new score branch (default: ml_score)",
    )
    apply_dnn_parser.add_argument(
        "--dnn-mass-scan", default=None,
        help="Parametric models only. Omit (default) to score once at the "
             "checkpoint's benchmark masspoint (mass_grid[0]). Pass 'all' to "
             "score at every grid point, or a comma list of MH3_<a>_MH4_<b>_Mchi_<c> "
             "labels to score at specific points. Ignored for non-parametric models.",
    )
    apply_dnn_parser.set_defaults(func=apply_dnn)

    # Make plots command
    plots_parser = subparsers.add_parser("make-plots", help="Create data/MC plots")
    plots_parser.add_argument("--input", required=True, help="Input results file")
    plots_parser.add_argument("--save-dir", required=True, help="Output directory")
    plots_parser.add_argument("--year", help="Data-taking year")
    plots_parser.add_argument("--region", help="Specific region to plot")
    plots_parser.add_argument("--show-data", action="store_true", help="Show data points")
    plots_parser.add_argument("--regions", nargs="+", help="List of regions to plot")
    plots_parser.add_argument("--version", help="Version string (default: auto-generate timestamp)")
    plots_parser.add_argument("--plot-config", help="Path to plotting configuration YAML file (default: configs/plotting.yaml)")
    plots_parser.add_argument("--config", help="Year config YAML (e.g. configs/2024.yaml) — provides luminosity for histogram normalisation")
    plots_parser.add_argument("--xsection-json", help="JSON file mapping process names to cross sections in pb — applies lumi×xsec/wte normalisation to region histograms")
    # All formats (PNG, PDF, ROOT, TXT) are generated automatically in batch mode
    plots_parser.set_defaults(func=make_plots)

    # Make single plots command (for event-level analysis files)
    single_plots_parser = subparsers.add_parser("make-single-plots", help="Create plots from a single analysis file (pre-region)")
    single_plots_parser.add_argument("--input", required=True, help="Input event-level results file (e.g., from 'run' command)")
    single_plots_parser.add_argument("--save-dir", required=True, help="Output directory")
    single_plots_parser.add_argument("--show-data", action="store_true", help="Show data points")
    single_plots_parser.add_argument("--version", help="Version string (default: auto-generate timestamp)")
    single_plots_parser.add_argument("--plot-config", help="Path to plotting configuration YAML file (default: configs/plotting.yaml)")
    single_plots_parser.set_defaults(func=make_single_plots)

    # Make stacked plots command
    stacked_parser = subparsers.add_parser("make-stacked-plots", help="Create stacked Data/MC plots with ratio")
    stacked_parser.add_argument("--data", help="Data results pickle path")
    stacked_parser.add_argument("--backgrounds", nargs="+", help="Background results pickle paths")
    stacked_parser.add_argument("--signal", help="Signal results pickle path")
    stacked_parser.add_argument("--output", required=True, help="Output plot file (e.g. outputs/plots/stacked_met.pdf)")
    stacked_parser.add_argument("--variable", default="met", help="Variable key to plot (default: met)")
    stacked_parser.add_argument("--region", default=None, help="Analysis region to plot (e.g., '1b:SR'). If not provided, attempts to plot from top-level histograms (for pre-region analysis results).")
    stacked_parser.add_argument("--xlabel", default="MET [GeV]", help="X-axis label")
    stacked_parser.add_argument("--title", default="CMS Preliminary  (13.6 TeV, 2023)", help="Title tag with CMS text")
    stacked_parser.add_argument("--version", help="Version string (default: auto-generate timestamp)")
    stacked_parser.add_argument("--plot-config", help="Path to plotting configuration YAML file (default: configs/plotting.yaml)")
    # All formats (PNG, PDF, ROOT, TXT) are generated automatically in batch mode
    stacked_parser.set_defaults(func=make_stacked_plots)

    # Make datacard command
    datacard_parser = subparsers.add_parser("make-datacard", help="Generate Combine datacard")
    datacard_parser.add_argument("--input", required=True, help="Input results file")
    datacard_parser.add_argument("--output", required=True, help="Output directory")
    datacard_parser.add_argument("--region", help="Specific region for datacard")
    datacard_parser.add_argument("--year", help="Data-taking year")
    datacard_parser.set_defaults(func=make_datacard)

    # Run Combine command
    combine_parser = subparsers.add_parser("run-combine", help="Run Combine fits")
    combine_parser.add_argument("--mode", required=True,
                               choices=["AsymptoticLimits", "FitDiagnostics", "GoodnessOfFit"],
                               help="Combine mode")
    combine_parser.add_argument("--datacard", required=True, help="Datacard file")
    combine_parser.add_argument("--output", help="Output directory")
    combine_parser.add_argument("--fit-region", help="Fit region")
    combine_parser.add_argument("--include-signal", action="store_true", help="Include signal in fit")
    combine_parser.add_argument("--toys", type=int, help="Number of toys for GOF")
    combine_parser.set_defaults(func=run_combine)

    # Make impact command
    impact_parser = subparsers.add_parser("make-impact", help="Create impact plots")
    impact_parser.add_argument("--input", required=True, help="Input fit results file")
    impact_parser.add_argument("--output", required=True, help="Output directory")
    impact_parser.set_defaults(func=make_impact)

    # Make pulls command
    pulls_parser = subparsers.add_parser("make-pulls", help="Create pull plots")
    pulls_parser.add_argument("--input", required=True, help="Input fit results file")
    pulls_parser.add_argument("--output", required=True, help="Output directory")
    pulls_parser.set_defaults(func=make_pulls)

    # Make GOF command
    gof_parser = subparsers.add_parser("make-gof", help="Create goodness-of-fit plots")
    gof_parser.add_argument("--input", required=True, help="Input GOF results file")
    gof_parser.add_argument("--output", required=True, help="Output directory")
    gof_parser.set_defaults(func=make_gof)

    # Parse arguments
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Check if command was provided
    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute command
    try:
        args.func(args)
    except Exception as e:
        logging.error(f"Command failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
