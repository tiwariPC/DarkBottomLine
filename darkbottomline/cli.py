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
from typing import Dict, Any, Optional, List

from .processor import DarkBottomLineProcessor
from .analyzer import DarkBottomLineAnalyzer
from .dnn_trainer import DNNTrainer
from .dnn_inference import DNNInference
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


def run_analysis(args):
    """Run basic analysis."""
    logging.info("Running basic analysis...")

    # Load configuration
    config = load_config(args.config)
    if args.data:
        config.setdefault("data", {})["is_data"] = True

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


def _add_dnn_scores_to_events(events, model_path: str, config_path: Optional[str], score_branch: str = "ml_score"):
    """Score all events with trained DNN; return ak.Array with ml_score field added."""
    import awkward as ak
    from dnn.feature_engineering import build_feature_frame_from_tree, REQUESTED_FEATURES_25
    from dnn.common import sanitize_feature_frame

    inference = DNNInference(model_path, config_path=config_path)
    model_info = inference.get_model_info()
    features = model_info.get("features") or list(REQUESTED_FEATURES_25)

    # Build feature DataFrame from the ak.Array event record
    # Treat events like a flat tree: extract each feature branch directly
    n = len(events)
    X_parts = {}
    for feat in features:
        if feat in events.fields:
            X_parts[feat] = np.asarray(ak.to_numpy(events[feat]), dtype="f8")
        else:
            X_parts[feat] = np.full(n, -9999.0, dtype="f8")

    import pandas as pd
    df = pd.DataFrame(X_parts)
    from dnn.common import sanitize_feature_frame
    df = sanitize_feature_frame(df)

    X = df.to_numpy(dtype="f8")
    masses = np.zeros(n, dtype="f8")
    scores = inference.predict(X, masses).ravel().astype("float32")

    # Append ml_score to events ak.Array
    events_with_score = ak.with_field(events, ak.Array(scores), score_branch)
    logging.info("DNN scoring complete: n=%d score_branch=%s", n, score_branch)
    return events_with_score


def _train_dnn_on_events(events, train_dnn_config: str, dnn_outdir: str, args) -> Tuple[np.ndarray, str]:
    """Train DNN on selected events (in-memory); return (scores, model_path).

    Scores array aligns 1:1 with events — same length, float32.
    All training plots (AUC, ranking, score distributions, loss curves) are
    written to dnn_outdir/plots/ by train_from_arrays.
    """
    import awkward as ak
    import pandas as pd
    from dnn.feature_engineering import REQUESTED_FEATURES_25
    from dnn.common import sanitize_feature_frame
    from dnn.make_trees import _is_data, _is_signal_heuristic

    n = len(events)
    sig_patterns = tuple(getattr(args, "signal_pattern", None) or ())
    sig_prefix = getattr(args, "signal_prefix", None)
    label_csv_path = getattr(args, "label_csv", None)

    # Build label array from input file heuristics
    # For in-memory events we need a per-event signal label.
    # Use the first input file's classification applied to all events
    # (single-sample mode) or per-file if multiple files loaded separately.
    # Here events are already concatenated — label from args.signal_prefix/pattern
    # applied to the first input file name as proxy.
    input_files = _get_input_files(args.input)
    if label_csv_path:
        import csv as _csv
        label_map = {}
        with open(label_csv_path, "r", newline="") as fp:
            for row in _csv.DictReader(fp):
                label_map[str(row["path"]).strip()] = int(row["label"])
        # Can't map per-event without per-file info; fall back to file-level
        y_parts = []
        for fpath in input_files:
            import os as _os
            key = fpath if fpath in label_map else _os.path.basename(fpath)
            lbl = label_map.get(key, 0)
            with uproot.open(fpath) as f:
                y_parts.append(np.full(int(f["Events"].num_entries), lbl, dtype="i4"))
        y = np.concatenate(y_parts)[:n]
    else:
        # Build per-file labels then concatenate to match events length
        y_parts = []
        for fpath in input_files:
            is_data = _is_data(fpath)
            if is_data:
                sig = 0
            else:
                sig = 1 if _is_signal_heuristic(fpath, sig_patterns, sig_prefix) else 0
            with uproot.open(fpath) as f:
                cnt = int(f["Events"].num_entries)
            y_parts.append(np.full(cnt, sig, dtype="i4"))
        y = np.concatenate(y_parts)[:n]

    if np.unique(y).size < 2:
        logging.warning("Only one class present — skipping DNN training, scores set to 0.5")
        return np.full(n, 0.5, dtype="float32"), ""

    # Build feature DataFrame from events ak.Array
    feat_list = list(REQUESTED_FEATURES_25)
    X_dict = {}
    for feat in feat_list:
        if feat in events.fields:
            X_dict[feat] = np.asarray(ak.to_numpy(events[feat]), dtype="f8")
        else:
            X_dict[feat] = np.full(n, -9999.0, dtype="f8")
    X_df = pd.DataFrame(X_dict)
    X_df = sanitize_feature_frame(X_df)

    # Weights
    if "full_event_weight" in events.fields:
        w = np.asarray(ak.to_numpy(events["full_event_weight"]), dtype="f8")
        w = np.where(np.isfinite(w), np.maximum(w, 0.0), 0.0)
    else:
        w = np.ones(n, dtype="f8")

    trainer = DNNTrainer(train_dnn_config)
    metrics = trainer.train_from_arrays(
        X=X_df, y=y, w=w,
        feature_sources={},
        outdir=dnn_outdir,
        plot_dir=str(Path(dnn_outdir) / "plots"),
    )
    model_path = str(Path(dnn_outdir) / "dnn_model.pt")
    logging.info("DNN trained — AUC(val)=%.4f  model=%s", metrics.get("auc_val", float("nan")), model_path)

    # Score all events with the just-trained model
    scores = trainer.predict(X_df.to_numpy(dtype="f8"), np.zeros(n, dtype="f8")).ravel().astype("float32")
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
    plt.savefig(out, dpi=160)
    plt.close()
    logging.info("DNN score plot saved to %s", out)


def run_analyzer(args):
    """Run multi-region analysis."""
    logging.info("Running multi-region analysis...")

    # Unified pipeline flags
    dnn_model = getattr(args, "dnn_model", None)
    dnn_config = getattr(args, "dnn_config", None)
    train_dnn_config = getattr(args, "train_dnn", None)
    dnn_outdir = getattr(args, "dnn_outdir", "outputs_dnn")
    dnn_only = getattr(args, "dnn_only", False)

    # Convert string boolean to actual boolean
    event_selection_only = args.event_selection_only.lower() == "true"

    # Validate arguments
    if event_selection_only:
        if not args.event_selection_output:
            logging.error("--event-selection-output must be provided when using --event-selection-only")
            sys.exit(1)
        logging.info("Event selection only mode: will stop after event selection (no region analysis)")
    elif not dnn_only:
        # Region analysis required unless stopping at DNN level
        if not (dnn_model or train_dnn_config) and (not args.regions_config or not args.output):
            logging.error("--regions-config and --output are required when --event-selection-only is false")
            sys.exit(1)
        if (dnn_model or train_dnn_config) and (not args.regions_config or not args.output):
            logging.error("--regions-config and --output are required for region analysis after DNN scoring")
            sys.exit(1)

    config = load_config(args.config)
    if args.data:
        config.setdefault("data", {})["is_data"] = True

    try:
        import uproot
        import awkward as ak
        import os

        is_txt_input = len(args.input) == 1 and args.input[0].endswith(".txt")
        input_files = _get_input_files(args.input)

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
                max_events=args.max_events, total_events=total_events
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

            # Save results (only if not in event_selection_only mode)
            if not event_selection_only:
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
                                              total_events=total_events)
                    logging.info(f"Event selection completed, saved to {args.event_selection_output}")

                elif train_dnn_config:
                    # Train DNN on selected events → inject ml_score in-memory → optionally region analysis
                    logging.info("DNN training on %d events...", len(events))
                    scores, model_path = _train_dnn_on_events(events, train_dnn_config, dnn_outdir, args)
                    events = ak.with_field(events, ak.Array(scores), "ml_score")
                    logging.info("ml_score injected in-memory (no disk write)")

                    if dnn_only:
                        _plot_dnn_score_only(scores, str(Path(dnn_outdir) / "plots"))
                        logging.info("--dnn-only set: stopping after DNN scoring. Training plots in %s/plots/", dnn_outdir)
                    else:
                        results = analyzer.process(events, event_selection_output=args.event_selection_output,
                                                   event_selection_only=False, output_format=output_format_to_use,
                                                   total_events=total_events)
                        outdir = os.path.dirname(args.output)
                        if outdir:
                            os.makedirs(outdir, exist_ok=True)
                        analyzer.accumulator = results
                        analyzer.save_results(args.output, output_format=args.output_format)

                elif dnn_model:
                    # Score with existing model → inject ml_score in-memory → optionally region analysis
                    logging.info("Scoring events with DNN model: %s", dnn_model)
                    events = _add_dnn_scores_to_events(events, dnn_model, dnn_config)
                    scores = np.asarray(ak.to_numpy(events["ml_score"]), dtype="f4")

                    if dnn_only:
                        _plot_dnn_score_only(scores, str(Path(dnn_outdir) / "plots"))
                        logging.info("--dnn-only set: stopping after DNN scoring. Plot in %s/plots/", dnn_outdir)
                    else:
                        results = analyzer.process(events, event_selection_output=args.event_selection_output,
                                                  event_selection_only=False, output_format=output_format_to_use,
                                                  total_events=total_events)
                        outdir = os.path.dirname(args.output)
                        if outdir:
                            os.makedirs(outdir, exist_ok=True)
                        analyzer.accumulator = results
                        analyzer.save_results(args.output, output_format=args.output_format)

                else:
                    results = analyzer.process(events, event_selection_output=args.event_selection_output,
                                              event_selection_only=False, output_format=output_format_to_use,
                                              total_events=total_events)
                    outdir = os.path.dirname(args.output)
                    if outdir:
                        os.makedirs(outdir, exist_ok=True)
                    analyzer.accumulator = results
                    analyzer.save_results(args.output, output_format=args.output_format)

    except Exception as e:
        logging.error("Error in multi-region analysis: %s", e, exc_info=True)
        raise

    logging.info("Multi-region analysis completed!")


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


def _load_training_data_from_eventsel(
    input_files: list,
    region: str,
    signal_patterns,
    signal_prefix,
    label_csv,
    weight_branch: str,
    max_events_per_file,
) -> tuple:
    """In-memory conversion of flat Events ROOT files to labelled numpy arrays.

    Returns (X_df, y, w, feature_sources) — same format train_from_root expects
    after the data-loading phase, bypassing the intermediate ppbbchichi-trees.root.
    """
    import pandas as pd
    from dnn.make_trees import convert_files
    import tempfile, os

    # Write to a temp file then read back via train_from_root's loader,
    # OR build arrays directly here without touching disk.
    # We build directly — no temp file.
    import csv as _csv
    import re as _re
    from dnn.make_trees import _sample_name, _is_data, _is_signal_heuristic
    from dnn.feature_engineering import build_feature_frame_from_tree, REQUESTED_FEATURES_25
    from dnn.common import sanitize_feature_frame

    sig_patterns = tuple(signal_patterns) if signal_patterns else ()

    label_map: dict = {}
    if label_csv:
        with open(label_csv, "r", newline="") as fp:
            reader = _csv.DictReader(fp)
            for row in reader:
                label_map[str(row["path"]).strip()] = int(row["label"])

    X_parts, y_parts, w_parts = [], [], []
    feature_sources: dict = {}

    for fpath in input_files:
        sample = _sample_name(fpath)

        if label_map:
            key = fpath if fpath in label_map else os.path.basename(fpath)
            if key not in label_map:
                raise KeyError(f"'{fpath}' not in label-csv")
            sig_flag = bool(label_map[key] == 1)
            data_flag = False
        else:
            data_flag = _is_data(fpath)
            sig_flag = False if data_flag else _is_signal_heuristic(fpath, sig_patterns, signal_prefix)

        if data_flag:
            logging.info("Skipping data file for DNN training: %s", fpath)
            continue

        with uproot.open(fpath) as in_f:
            if "Events" not in in_f:
                raise KeyError(f"No 'Events' tree in {fpath}")
            tree = in_f["Events"]
            df, src, _ = build_feature_frame_from_tree(
                tree, list(REQUESTED_FEATURES_25),
                max_events=max_events_per_file,
            )
            df = sanitize_feature_frame(df)
            n = len(df)

            # weight
            avail = set(tree.keys())
            if weight_branch in avail:
                w_arr = tree[weight_branch].array(entry_stop=n, library="np").astype("f8")
                w_arr = np.where(np.isfinite(w_arr), np.maximum(w_arr, 0.0), 0.0)
            else:
                w_arr = np.ones(n, dtype="f8")

            n = min(n, len(w_arr))

        X_parts.append(df.iloc[:n])
        y_parts.append(np.full(n, int(sig_flag), dtype="i4"))
        w_parts.append(w_arr[:n])
        for feat in df.columns:
            if feat not in feature_sources:
                feature_sources[feat] = src.get(feat, "unknown")

        logging.info("Loaded %s: n=%d signal=%d", sample, n, int(sig_flag))

    if not X_parts:
        raise ValueError("No training events loaded — check input files and signal/background flags.")

    import pandas as pd
    X = pd.concat(X_parts, axis=0, ignore_index=True)
    y = np.concatenate(y_parts)
    w = np.concatenate(w_parts)
    return X, y, w, feature_sources


def train_dnn(args):
    """Train DNN from event-selection ROOT files (no intermediate file needed)."""
    trainer = DNNTrainer(args.config)

    input_files = _get_input_files(args.input)
    logging.info("Training DNN from %d input file(s)", len(input_files))

    # Load feature matrix directly from flat Events trees — no ppbbchichi-trees.root written
    X, y, w, feature_sources = _load_training_data_from_eventsel(
        input_files=input_files,
        region=getattr(args, "region", "preselection"),
        signal_patterns=(args.signal_pattern or None),
        signal_prefix=args.signal_prefix,
        label_csv=args.label_csv,
        weight_branch=getattr(args, "weight_branch", "full_event_weight"),
        max_events_per_file=args.max_events_per_sample,
    )

    metrics = trainer.train_from_arrays(
        X=X,
        y=y,
        w=w,
        feature_sources=feature_sources,
        outdir=args.outdir,
        plot_dir=args.plot_dir,
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
    """
    import pandas as pd
    from dnn.feature_engineering import build_feature_frame_from_tree, REQUESTED_FEATURES_25
    from dnn.common import sanitize_feature_frame

    inference = DNNInference(args.model, config_path=args.config)
    model_info = inference.get_model_info()
    features = model_info.get("features") or list(REQUESTED_FEATURES_25)
    score_branch = args.score_branch

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
            masses = np.zeros(n, dtype="f8")
            scores = inference.predict(X, masses).ravel()

            # Collect all existing branches
            arrays = tree.arrays(library="np")

        arrays[score_branch] = scores.astype("f4")

        if output_dir:
            out_path = output_dir / Path(fpath).name
        else:
            out_path = fpath  # overwrite in-place

        with uproot.recreate(str(out_path)) as out_f:
            out_f["Events"] = arrays

        logging.info("Scored %s: n=%d → %s (branch: %s)", Path(fpath).name, n, out_path, score_branch)


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
        from datetime import datetime
        version = datetime.now().strftime("%Y%m%d_%H%M")
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
        from datetime import datetime
        version = datetime.now().strftime("%Y%m%d_%H%M")
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
        from datetime import datetime
        version = datetime.now().strftime("%Y%m%d_%H%M")
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
    """Create stacked event-selection or region plots."""
    import json

    config = load_config(args.config)
    luminosity = float(config["luminosity"])
    year = str(config["year"])

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
            cross_sections = json.load(f)

    version = args.version
    if not version:
        from datetime import datetime
        version = datetime.now().strftime("%Y%m%d_%H%M")

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
    )
    logging.info(f"make-event-plots: {len(out_files)} plot(s) written to {args.output_dir}")


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
    analyze_parser = subparsers.add_parser("analyze", help="Run multi-region analysis")
    analyze_parser.add_argument("--config", required=True, help="Base configuration file")
    analyze_parser.add_argument("--regions-config", required=False, help="Regions configuration file (not required if --event-selection-only is used)")
    analyze_parser.add_argument("--input", nargs="+", required=True, help="Input file(s), can be a single .txt file listing paths")
    analyze_parser.add_argument("--output", required=False, help="Output file (not required if --event-selection-only is used)")
    analyze_parser.add_argument("--event-selection-output", help="Path to save events that pass event-level selection (optional)")
    analyze_parser.add_argument("--event-selection-only", type=str.lower, default="false", 
                               choices=["true", "false"], 
                               help="If true, stop after event selection and don't perform region analysis (default: false)")
    analyze_parser.add_argument("--output-format", default="pkl", 
                               choices=["pkl", "root", "parquet"],
                               help="Output file format (default: pkl)")
    analyze_parser.add_argument("--executor", choices=["iterative", "futures", "dask"],
                               default="iterative", help="Execution backend")
    analyze_parser.add_argument("--workers", type=int, default=4, help="Number of workers")
    analyze_parser.add_argument("--chunk-size", type=str, default=None,
                               help="Number of events per chunk for futures/dask executors. Use 'auto' for automatic optimization, or specify an integer (default: 50000 for futures, 200000 for dask). Only used with futures/dask executors.")
    analyze_parser.add_argument("--max-events", type=int, help="Maximum events to process across all chunks")
    analyze_parser.add_argument("--data", action="store_true",
                               help="Input is collision data: apply golden JSON lumi mask and skip MC-only weights")
    # DNN integration flags
    analyze_parser.add_argument(
        "--dnn-model", default=None,
        help="Path to trained DNN checkpoint (.pt). When provided, scores each passing event "
             "with the model before region analysis (adds ml_score branch).",
    )
    analyze_parser.add_argument(
        "--dnn-config", default=None,
        help="DNN config YAML (e.g. configs/dnn.yaml). Used to resolve feature list when --dnn-model is set.",
    )
    analyze_parser.add_argument(
        "--train-dnn", default=None,
        help="If set to a DNN config YAML path, train the DNN on the event-selection output before "
             "scoring and region analysis (implies --event-selection-only first, then full pipeline). "
             "Requires --signal-pattern or --signal-prefix or --label-csv to label samples.",
    )
    analyze_parser.add_argument(
        "--dnn-outdir", default="outputs_dnn",
        help="Output directory for DNN model + metrics when --train-dnn is used (default: outputs_dnn)",
    )
    analyze_parser.add_argument(
        "--dnn-only", action="store_true",
        help="Stop after DNN scoring — produce score distribution plot, skip region analysis. "
             "Training plots (AUC, ranking, loss) are always produced when --train-dnn is set.",
    )
    analyze_parser.add_argument(
        "--signal-pattern", action="append", default=None, dest="signal_pattern",
        help="Regex to identify signal files for DNN training (repeatable)",
    )
    analyze_parser.add_argument(
        "--signal-prefix", default=None,
        help="Filename prefix that marks signal samples for DNN training",
    )
    analyze_parser.add_argument(
        "--label-csv", default=None,
        help="CSV with columns path,label for DNN training label assignment",
    )
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
    train_dnn_parser.add_argument("--config", required=True, help="DNN configuration YAML (configs/dnn.yaml)")
    train_dnn_parser.add_argument(
        "--input", nargs="+", required=True,
        help="Event-selection ROOT files (one per sample), or a .txt file listing paths",
    )
    train_dnn_parser.add_argument("--region", default="preselection", help="Region label (default: preselection)")
    train_dnn_parser.add_argument("--outdir", default="outputs_dnn", help="Output directory for model + metrics (default: outputs_dnn)")
    train_dnn_parser.add_argument("--plot-dir", default="outputs_dnn/plots", help="Output directory for plots (default: outputs_dnn/plots)")
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

    # Make event plots command
    event_plots_parser = subparsers.add_parser(
        "make-event-plots",
        help="Create stacked MC+data plots (event-selection or region mode)",
    )
    event_plots_parser.add_argument(
        "--mode", required=True,
        choices=["event-selection", "region", "region-from-events"],
        help=(
            "Plot mode: "
            "event-selection (flat per-sample files → stacked variable plots), "
            "region (region histogram PKLs from 'analyze'), "
            "region-from-events (event-selected files → apply region cuts in-memory → stacked region plots)"
        ),
    )
    event_plots_parser.add_argument("--config", required=True,
                                    help="Year config YAML (provides year + luminosity)")
    event_plots_parser.add_argument("--input-folder", required=True, metavar="DIR",
                                    help="Single folder containing all background ROOT/PKL files")
    event_plots_parser.add_argument("--process-groups", default=None, metavar="JSON",
                                    help="JSON file mapping process label → {files: [...]} groups "
                                         "(default: read process_groups from plotting.yaml)")
    event_plots_parser.add_argument("--data-folder", default=None, metavar="DIR",
                                    help="Separate folder for data files (optional; "
                                         "ignored if data groups defined in plotting.yaml)")
    event_plots_parser.add_argument("--output-dir", required=True,
                                    help="Root output directory")
    event_plots_parser.add_argument("--xsection-json", default=None, metavar="JSON",
                                    help="JSON file mapping filename stem → cross-section in pb")
    event_plots_parser.add_argument("--variables", nargs="+", default=None, metavar="VAR",
                                    help="Variables to plot (default: all)")
    event_plots_parser.add_argument("--regions", nargs="+", default=None, metavar="REGION",
                                    help="Regions to plot in region mode (default: all)")
    event_plots_parser.add_argument("--save-root", action="store_true",
                                    help="Also save ROOT TH1 files")
    event_plots_parser.add_argument("--plot-config", default=None,
                                    help="Plotting YAML (default: configs/plotting.yaml)")
    event_plots_parser.add_argument("--version", default=None,
                                    help="Version tag for output subdirectory (default: timestamp)")
    event_plots_parser.add_argument(
        "--regions-config", default=None, metavar="YAML",
        help="Path to regions.yaml — required for region-from-events mode",
    )
    event_plots_parser.add_argument(
        "--weight-systematic", default=None, metavar="BRANCH",
        help=(
            "Weight branch to use instead of 'full_event_weight' (nominal). "
            "E.g. 'weight_pileupUP', 'weight_btagDOWN'. "
            "Only applies to region-from-events mode."
        ),
    )
    event_plots_parser.set_defaults(func=make_event_plots)

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
