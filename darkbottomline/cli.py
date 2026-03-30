"""
Command-line interface for DarkBottomLine framework.
"""

import argparse
import logging
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List

from .processor import DarkBottomLineProcessor
from .analyzer import DarkBottomLineAnalyzer
from .dnn_trainer import DNNTrainer
from .dnn_inference import DNNInference
from .plotting import PlotManager
from .regions import RegionManager
from .utils.chunk_optimizer import (
    optimize_chunk_size_for_files,
    parse_chunk_size_arg,
)

# Try to import Coffea for chunk-size support
try:
    from coffea import processor
    from coffea.processor import run_uproot_job, FuturesExecutor
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

    # Initialize processor
    processor = DarkBottomLineProcessor(config)

    # Load events from ROOT file
    try:
        import uproot
        import awkward as ak

        input_files = _get_input_files(args.input)
        logging.info(f"Loading events from {str(input_files)} files")

        events = uproot.concatenate([f"{path}:Events" for path in input_files])

        # Limit events if specified
        if args.max_events and len(events) > args.max_events:
            events = events[:args.max_events]
            logging.info(f"Limited to {args.max_events} events")

        logging.info(f"Loaded {len(events)} events")

        # Process events (optionally save event-level selection)
        results = processor.process(events, event_selection_output=args.event_selection_output)

        # Save results
        import pickle
        import os

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

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


def run_analyzer(args):
    """Run multi-region analysis."""
    logging.info("Running multi-region analysis...")

    # Convert string boolean to actual boolean
    event_selection_only = args.event_selection_only.lower() == "true"

    # Validate arguments
    if event_selection_only:
        if not args.event_selection_output:
            logging.error("--event-selection-output must be provided when using --event-selection-only")
            sys.exit(1)
        logging.info("Event selection only mode: will stop after event selection (no region analysis)")
    else:
        if not args.regions_config or not args.output:
            logging.error("--regions-config and --output are required when --event-selection-only is false")
            sys.exit(1)

    config = load_config(args.config)

    try:
        import uproot
        import awkward as ak
        import os

        is_txt_input = len(args.input) == 1 and args.input[0].endswith(".txt")
        input_files = _get_input_files(args.input)

        # Total events before selection to be saved into event-selection-output metadata.
        # Rule: use --max-events when specified; otherwise use total events from input files.
        n_events_total = args.max_events if args.max_events is not None else None
        if n_events_total is None and args.event_selection_output:
            try:
                n_events_total = 0
                for file_path in input_files:
                    tree = uproot.open(f"{file_path}:Events")
                    n_events_total += int(tree.num_entries)
                logging.info(f"Computed n_events_total={n_events_total} from input files")
            except Exception as e:
                logging.warning(f"Could not compute n_events_total from input files: {e}")
                n_events_total = None

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

            fileset = {"dataset": input_files}
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
                max_events=args.max_events, n_events_total=n_events_total
            )

            if args.executor == "futures":
                result = run_uproot_job(
                    fileset,
                    "Events",
                    coffea_analyzer,
                    executor=FuturesExecutor,
                    executor_args={"workers": args.workers},
                    chunksize=chunksize,
                    maxchunks=maxchunks,
                )
            elif args.executor == "dask" and DASK_AVAILABLE:
                client = None
                try:
                    # Start Dask client
                    client = Client(n_workers=args.workers, timeout=120)

                    # Wait for workers to be ready (with timeout)
                    try:
                        client.wait_for_workers(args.workers, timeout=60)
                        logging.info(f"Dask client ready with {len(client.scheduler_info()['workers'])} workers")
                    except Exception as e:
                        logging.warning(f"Timeout waiting for workers, continuing anyway: {e}")

                    result = run_uproot_job(
                        fileset,
                        "Events",
                        coffea_analyzer,
                        executor=DaskExecutor,
                        executor_args={"client": client},
                        chunksize=chunksize if chunksize != 50000 else 200000,  # Default 200k for dask
                        maxchunks=maxchunks,
                    )
                except Exception as e:
                    logging.error(f"Dask execution error: {e}")
                    raise
                finally:
                    # Ensure client is properly closed
                    if client is not None:
                        try:
                            client.close()
                        except Exception as e:
                            logging.warning(f"Error closing Dask client: {e}")
            else:
                raise ValueError(f"Executor {args.executor} not available or not supported")

            # Ensure postprocess is called (run_uproot_job should call it, but be explicit)
            if hasattr(coffea_analyzer, 'postprocess'):
                logging.info("Calling postprocess to finalize event_selection_output if needed...")
                result = coffea_analyzer.postprocess(result)

            # Save results (only if not in event_selection_only mode)
            if not event_selection_only:
                analyzer = DarkBottomLineAnalyzer(config, args.regions_config)
                analyzer.accumulator = result
                os.makedirs(os.path.dirname(args.output), exist_ok=True)
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

                    if args.max_events:
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

                if args.max_events and len(events) > args.max_events:
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
                    # Event selection only mode: create analyzer just to use its process method
                    logging.info("Event selection only mode: performing event selection...")
                    analyzer = DarkBottomLineAnalyzer(config, None)  # No regions config needed for event selection only
                    results = analyzer.process(events, event_selection_output=args.event_selection_output,
                                              event_selection_only=True, output_format=output_format_to_use,
                                              n_events_total=n_events_total)
                    logging.info(f"Event selection completed, saved to {args.event_selection_output}")
                else:
                    results = analyzer.process(events, event_selection_output=args.event_selection_output,
                                              event_selection_only=False, output_format=output_format_to_use,
                                              n_events_total=n_events_total)
                    os.makedirs(os.path.dirname(args.output), exist_ok=True)
                    analyzer.accumulator = results
                    analyzer.save_results(args.output, output_format=args.output_format)

    except Exception as e:
        logging.error("Error in multi-region analysis: %s", e, exc_info=True)
        raise

    logging.info("Multi-region analysis completed!")


def train_dnn(args):
    """Train DNN model."""
    logging.info("Training DNN model...")

    # Initialize trainer
    trainer = DNNTrainer(args.config)

    # Load data
    signal_files = _get_input_files(args.signal)
    background_files = _get_input_files(args.background)

    features, labels, masses = trainer.load_data(signal_files, background_files)

    # Preprocess data
    train_features, train_labels, train_masses, val_features, val_labels, val_masses = trainer.preprocess(features, labels, masses)

    # Train model
    results = trainer.train(train_features, train_labels, train_masses, val_features, val_labels, val_masses)

    # Save model
    trainer.save_model(args.output)

    # Plot training history
    if args.plot_history:
        trainer.plot_training_history(f"{args.output}_history.png")

    logging.info("DNN training completed!")


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
        results, args.save_dir, args.show_data, args.regions, version, formats=None
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
    analyze_parser.set_defaults(func=run_analyzer)

    # Train DNN command
    train_dnn_parser = subparsers.add_parser("train-dnn", help="Train DNN model")
    train_dnn_parser.add_argument("--config", required=True, help="DNN configuration file")
    train_dnn_parser.add_argument("--signal", nargs="+", required=True, help="Signal files, can be a single .txt file listing paths")
    train_dnn_parser.add_argument("--background", nargs="+", required=True, help="Background files, can be a single .txt file listing paths")
    train_dnn_parser.add_argument("--output", required=True, help="Output model file")
    train_dnn_parser.add_argument("--plot-history", action="store_true", help="Plot training history")
    train_dnn_parser.set_defaults(func=train_dnn)

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
