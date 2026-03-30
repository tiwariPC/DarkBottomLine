"""
Main processor class for DarkBottomLine framework.
"""

import awkward as ak
import numpy as np
from typing import Dict, Any, Optional, Union
import logging
import time

try:
    from coffea import processor
    from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
    COFFEA_AVAILABLE = True
except ImportError:
    COFFEA_AVAILABLE = False
    logging.warning("Coffea not available. Using fallback implementation.")

from .objects import build_objects
from .selections import apply_selection
from .corrections import CorrectionManager
from .weights import WeightCalculator
from .histograms import HistogramManager


def _build_event_weights_for_save(
    events: ak.Array,
    corrections: Dict[str, Any],
    weight_calculator: WeightCalculator,
) -> Dict[str, Any]:
    """
    Build a dict of per-event weights (central/up/down per systematic) for saving.

    Returns dict with: generator, pileup, weight_* (central/up/down per systematic only),
    weight_total_nominal (combined nominal; no total up/down saved).
    """
    n = len(events)
    out = {}

    # Generator (central only)
    if hasattr(events, "genWeight"):
        out["generator"] = np.asarray(ak.to_numpy(np.abs(events.genWeight)))
    else:
        out["generator"] = np.ones(n, dtype=np.float64)

    # Corrections: each key is either a single array (e.g. pileup) or dict with central/up/down
    for name, val in corrections.items():
        if isinstance(val, dict) and "central" in val:
            out[name] = {}
            for k in ("central", "up", "down"):
                if val.get(k) is not None:
                    out[name][k] = np.asarray(ak.to_numpy(val[k]))
        else:
            arr = val
            if hasattr(arr, "__len__") and not isinstance(arr, (str, dict)):
                out[name] = np.asarray(ak.to_numpy(arr))
            else:
                out[name] = np.full(n, float(arr), dtype=np.float64)

    # Combined nominal total weight only (no total up/down)
    out["weight_total_nominal"] = np.asarray(ak.to_numpy(weight_calculator.get_weight("central")))

    return out


def _event_weights_flat_columns(event_weights: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Flatten event_weights into a dict of 1D arrays for parquet/ROOT columns."""
    cols = {}
    for name, val in event_weights.items():
        if isinstance(val, dict):
            for var, arr in val.items():
                if isinstance(arr, np.ndarray):
                    cols[f"{name}_{var}"] = arr
        elif isinstance(val, np.ndarray):
            cols[name] = val
    return cols


class DarkBottomLineProcessor:
    """
    Main processor class for DarkBottomLine analysis.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the processor with configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.correction_manager = CorrectionManager(config)
        self.histogram_manager = HistogramManager()
        self.histograms = self.histogram_manager.define_histograms()

        # Initialize accumulators
        self.accumulator = {
            "histograms": self.histograms,
            "cutflow": {},
            "metadata": {},
            "event_weights": {},
        }

        logging.info(f"Initialized DarkBottomLine processor for year {config.get('year', 'unknown')}")

    def process(self, events: ak.Array, event_selection_output: Optional[str] = None) -> Dict[str, Any]:
        """
        Process events through the analysis chain.

        Args:
            events: Awkward Array of events

        Returns:
            Dictionary containing analysis results
        """
        start_time = time.time()
        print(f"=== PROCESSING EVENTS ===")
        print(f"Total events loaded: {len(events)}")

        # Build physics objects
        print("Step 1: Building physics objects...")
        logging.info("Building physics objects...")
        objects = build_objects(events, self.config)

        # Print object counts
        n_muons = ak.sum(ak.num(objects["muons"], axis=1))
        n_electrons = ak.sum(ak.num(objects["electrons"], axis=1))
        n_taus = ak.sum(ak.num(objects["taus"], axis=1))
        n_jets = ak.sum(ak.num(objects["jets"], axis=1))
        n_bjets = ak.sum(ak.num(objects["bjets"], axis=1))
        print(f"  Selected objects: {n_muons} muons, {n_electrons} electrons, {n_taus} taus, {n_jets} jets, {n_bjets} b-jets")

        # Apply event selection
        print("Step 2: Applying event selection...")
        logging.info("Applying event selection...")
        selected_events, selected_objects, cutflow = apply_selection(
            events, objects, self.config
        )
        print(f"  Events passing selection: {len(selected_events)} / {len(events)} ({len(selected_events)/len(events)*100:.1f}%)")

        # Optionally save event-level selection results immediately and continue
        if event_selection_output:
            try:
                logging.info(f"Saving event-level selection to {event_selection_output} ({len(selected_events)} events)")
                self._save_event_selection(event_selection_output, selected_events, selected_objects, max_events=self.config.get("max_events"))
                # Verify file was created
                import os
                if os.path.exists(event_selection_output):
                    file_size = os.path.getsize(event_selection_output)
                    logging.info(f"✓ Event selection saved successfully to {event_selection_output} ({file_size} bytes)")
                else:
                    logging.error(f"✗ File {event_selection_output} was not created!")
            except Exception as e:
                logging.error(f"Failed to save event-level selection to {event_selection_output}: {e}", exc_info=True)

        # Calculate corrections and weights
        print("Step 3: Calculating corrections and weights...")
        print(f"  Selected events for weight calculation: {len(selected_events)}")
        logging.info("Calculating corrections and weights...")

        try:
            corrections = self.correction_manager.get_all_corrections(
                selected_events, selected_objects
            )

            # Initialize weight calculator
            weight_calculator = WeightCalculator(len(selected_events))

            # Add generator weights
            weight_calculator.add_generator_weight(selected_events)

            # Add corrections
            weight_calculator.add_corrections(corrections)

            # Get final weights (nominal = central for histogram filling)
            event_weights = weight_calculator.get_weight("central")
            print(f"  Weights calculated successfully: {len(event_weights)} weights")

            # Build per-event weight dict for each systematic (central/up/down) for saving
            event_weights_save = _build_event_weights_for_save(
                selected_events, corrections, weight_calculator
            )
        except Exception as e:
            print(f"  Error in weight calculation: {e}")
            print(f"  Selected events length: {len(selected_events)}")
            raise

        # Fill histograms with nominal total weight
        print("Step 4: Filling histograms...")
        print(f"  Selected events: {len(selected_events)}")
        print(f"  Event weights shape: {event_weights.shape if hasattr(event_weights, 'shape') else len(event_weights)}")
        logging.info("Filling histograms with nominal total weight...")
        try:
            filled_histograms = self.histogram_manager.fill_histograms(
                selected_events, selected_objects, event_weights
            )
            print(f"  Histograms filled: {list(filled_histograms.keys())}")
        except Exception as e:
            print(f"  Error in histogram filling: {e}")
            print(f"  Selected events length: {len(selected_events)}")
            print(f"  Objects keys: {list(selected_objects.keys())}")
            for key, obj in selected_objects.items():
                if isinstance(obj, ak.Array):
                    try:
                        n_total = ak.num(obj, axis=1).sum()
                    except Exception:
                        n_total = len(obj)
                    print(f"    {key}: {len(obj)} events, {n_total} total objects")
            raise

        # Calculate processing statistics
        processing_time = time.time() - start_time

        # Update accumulator (histograms with nominal weight; all systematic weights saved)
        self.accumulator["histograms"] = filled_histograms
        self.accumulator["cutflow"] = cutflow
        self.accumulator["event_weights"] = event_weights_save
        self.accumulator["metadata"] = {
            "n_events_processed": len(events),
            "n_events_selected": len(selected_events),
            "processing_time": processing_time,
            "weight_statistics": weight_calculator.get_weight_statistics(),
        }

        # Optionally save skimmed events
        if self.config.get("save_skims", False):
            print("Step 5: Creating skimmed events...")
            self.accumulator["skimmed_events"] = self._create_skimmed_events(
                selected_events, selected_objects, event_weights
            )

        print("=== ANALYSIS COMPLETE ===")
        print(f"Processed {len(events)} events, selected {len(selected_events)} events")
        print(f"Processing time: {processing_time:.2f} seconds")
        logging.info(f"Processed {len(events)} events, selected {len(selected_events)} events")
        logging.info(f"Processing time: {processing_time:.2f} seconds")

        return self.accumulator

    def _create_skimmed_events(
        self,
        events: ak.Array,
        objects: Dict[str, Any],
        weights: Union[ak.Array, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Create skimmed events for downstream analysis.

        Args:
            events: Selected events
            objects: Selected objects
            weights: Event weights

        Returns:
            Dictionary containing skimmed event data
        """
        skimmed = {
            "event": events.event,
            "run": events.run,
            "luminosityBlock": events.luminosityBlock,
            "MET": {
                "pt": events["PFMET_pt"] if "PFMET_pt" in events.fields else events["MET_pt"],
                "phi": events["PFMET_phi"] if "PFMET_phi" in events.fields else events["MET_phi"],
                "significance": events["PFMET_significance"] if "PFMET_significance" in events.fields else events["MET_significance"],
            },
            "weights": weights,
        }

        # Add selected objects
        for obj_name, obj_data in objects.items():
            if isinstance(obj_data, ak.Array):
                skimmed[obj_name] = obj_data

        return skimmed

    def _save_event_selection(self, output_file: str, events: ak.Array, objects: Dict[str, Any], 
                              max_events: Optional[int] = None, n_events_total: Optional[int] = None,
                              event_weights: Optional[Dict[str, Any]] = None, output_format: str = "pkl"):
        """
        Save selected events and corresponding objects to a file.

        Currently supports pickle, ROOT formats.
        
        Args:
            output_file: Path to save the output file
            events: Selected events (after event-level selection)
            objects: Selected objects (after event-level selection)
            max_events: Maximum events parameter from config (optional)
            n_events_total: Total number of events BEFORE selection (optional)
            event_weights: Event weights dictionary (optional, includes all corrections)
            output_format: Output format ("pkl", "root"). Default: "pkl" (saves both pkl and root)
        """
        import os
        import pickle
        import numpy as np

        # Ensure output directory exists
        outdir = os.path.dirname(output_file)
        if outdir:
            os.makedirs(outdir, exist_ok=True)

        # Auto-detect strict output intent from file extension.
        # If user explicitly provides a .root target, enforce root-only output.
        output_file_lower = output_file.lower()
        if output_file_lower.endswith('.root'):
            output_format = 'root'
        elif output_file_lower.endswith('.pkl') and output_format not in ('root',):
            output_format = 'pkl'

        # Determine actual output files based on format
        # For backward compatibility, pkl format saves both pickle and ROOT files
        save_pkl = output_format in ("pkl", "both")
        save_root = output_format in ("root", "pkl", "both")  # pkl format also saves ROOT
        
        # Explicitly adjust extension if format is specified
        if output_format == "root":
            # Force root extension
            if not output_file.endswith('.root'):
                output_file_pkl = os.path.splitext(output_file)[0] + '.root'
                output_file_root = output_file_pkl
            else:
                output_file_pkl = output_file
                output_file_root = output_file
            save_pkl = False  # Don't save pickle when root format is specified
        elif output_format == "pkl":
            if not output_file.endswith('.pkl'):
                output_file_pkl = os.path.splitext(output_file)[0] + '.pkl'
            else:
                output_file_pkl = output_file
            output_file_root = os.path.splitext(output_file)[0] + '.root'
        else:
            output_file_pkl = output_file + ".pkl" if not output_file.endswith('.pkl') else output_file
            output_file_root = os.path.splitext(output_file)[0] + '.root'

        # Save pickle file if specified
        if save_pkl:
            # 1) Save a human/inspection-friendly representation where all
            #    Awkward arrays are converted to plain Python lists (via
            #    ak.to_list). This is what we write to `output_file`.
            # 2) Also attempt to save the raw awkward data as a backup next to
            #    the main file with suffix `.awk_raw.pkl` (best-effort).

            # Build serializable dict
            serializable = {}
            
            # Add n_events: total events before selection
            if n_events_total is not None:
                serializable["n_events"] = n_events_total
            
            # Keep max_events for backward compatibility
            if max_events is not None:
                serializable["total_event"] = max_events
                
            try:
                # Convert events (awkward array) to list-of-records where possible
                try:
                    serializable["events"] = ak.to_list(events)
                except Exception:
                    # Fallback: try to coerce to Python list directly
                    try:
                        serializable["events"] = list(events)
                    except Exception:
                        serializable["events"] = None

                # Convert objects: for each awkward array, convert to list
                serializable_objects = {}
                for k, v in objects.items():
                    if isinstance(v, ak.Array):
                        try:
                            serializable_objects[k] = ak.to_list(v)
                        except Exception:
                            try:
                                serializable_objects[k] = list(v)
                            except Exception:
                                serializable_objects[k] = None
                    else:
                        # Keep non-Awkward values as-is (likely small metadata)
                        serializable_objects[k] = v

                serializable["objects"] = serializable_objects

                # Add event weights if provided (now with corrections)
                if event_weights:
                    serializable_weights = {}
                    for name, val in event_weights.items():
                        if isinstance(val, dict):
                            # Dictionary of different variations (e.g., central/up/down)
                            serializable_weights[name] = {}
                            for k, v in val.items():
                                if isinstance(v, np.ndarray):
                                    serializable_weights[name][k] = v.tolist()
                                else:
                                    serializable_weights[name][k] = v
                        elif isinstance(val, np.ndarray):
                            # Single numpy array
                            serializable_weights[name] = val.tolist()
                        else:
                            serializable_weights[name] = val
                    serializable["event_weights"] = serializable_weights

                # Ensure output directory exists
                outdir = os.path.dirname(output_file_pkl)
                if outdir:
                    os.makedirs(outdir, exist_ok=True)

                # Write the safe version
                with open(output_file_pkl, "wb") as f:
                    pickle.dump(serializable, f, protocol=pickle.HIGHEST_PROTOCOL)

                logging.info(f"Event selection (serializable) saved to {output_file_pkl}")

            except Exception as e:
                logging.warning(f"Failed to save serializable event selection to {output_file_pkl}: {e}")

            # Try to save raw awkward structures as a backup for advanced users
            try:
                raw_backup = output_file_pkl + ".awk_raw.pkl"
                with open(raw_backup, "wb") as f:
                    pickle.dump({"events": events, "objects": objects, "event_weights": event_weights}, f, protocol=pickle.HIGHEST_PROTOCOL)
                logging.info(f"Raw awkward backup saved to {raw_backup}")
            except Exception as e:
                logging.warning(f"Failed to save raw awkward backup to {raw_backup}: {e}")

        # Save ROOT file if specified
        if save_root:
            try:
                import uproot

                logging.info(f"Creating ROOT file: {output_file_root}")

                # Prepare flat scalar branches
                branches = {}
                # Standard event identifiers
                try:
                    branches['event'] = ak.to_numpy(events['event'])
                except Exception:
                    branches['event'] = np.asarray(ak.to_list(events.get('event', [])))
                try:
                    branches['run'] = ak.to_numpy(events['run'])
                except Exception:
                    branches['run'] = np.asarray(ak.to_list(events.get('run', [])))
                try:
                    branches['luminosityBlock'] = ak.to_numpy(events['luminosityBlock'])
                except Exception:
                    branches['luminosityBlock'] = np.asarray(ak.to_list(events.get('luminosityBlock', [])))

                # MET scalars
                def _get_met_field_array(field_name_v15, field_name_v12):
                    if field_name_v15 in events.fields:
                        return ak.to_numpy(events[field_name_v15])
                    elif field_name_v12 in events.fields:
                        return ak.to_numpy(events[field_name_v12])
                    else:
                        # Return an empty array of appropriate size if neither field exists
                        return np.asarray([]) if len(events) == 0 else np.zeros(len(events), dtype=float)

                branches['PFMET_pt'] = _get_met_field_array('PFMET_pt', 'MET_pt')
                branches['PFMET_phi'] = _get_met_field_array('PFMET_phi', 'MET_phi')
                branches['PFMET_significance'] = _get_met_field_array('PFMET_significance', 'MET_significance')

                # Recoil (event-level scalar)
                try:
                    met_pt = events['PFMET_pt'] if 'PFMET_pt' in events.fields else events['MET_pt']
                    met_phi = events['PFMET_phi'] if 'PFMET_phi' in events.fields else events['MET_phi']

                    # Use tight pt>30 leptons for CR-consistent recoil definition
                    muons = objects.get('tight_muons_pt30', ak.Array([]))
                    electrons = objects.get('tight_electrons_pt30', ak.Array([]))

                    lep_px = ak.zeros_like(met_pt)
                    lep_py = ak.zeros_like(met_pt)
                    try:
                        if len(ak.flatten(muons)) > 0:
                            lep_px = lep_px + ak.sum(muons.pt * np.cos(muons.phi), axis=1)
                            lep_py = lep_py + ak.sum(muons.pt * np.sin(muons.phi), axis=1)
                        if len(ak.flatten(electrons)) > 0:
                            lep_px = lep_px + ak.sum(electrons.pt * np.cos(electrons.phi), axis=1)
                            lep_py = lep_py + ak.sum(electrons.pt * np.sin(electrons.phi), axis=1)
                    except (Exception, BaseException):
                        pass

                    recoil_px = -(met_pt * np.cos(met_phi) + lep_px)
                    recoil_py = -(met_pt * np.sin(met_phi) + lep_py)
                    recoil = np.sqrt(recoil_px**2 + recoil_py**2)
                    branches['recoil'] = ak.to_numpy(ak.fill_none(recoil, 0.0))
                except Exception:
                    branches['recoil'] = np.zeros(len(events), dtype=float)

                # Object multiplicities
                def safe_num(obj_key):
                    arr = objects.get(obj_key, ak.Array([]))
                    try:
                        return ak.to_numpy(ak.num(arr, axis=1))
                    except Exception:
                        return np.asarray([0] * len(branches.get('event', [])))

                branches['n_muons'] = safe_num('muons')
                branches['n_electrons'] = safe_num('electrons')
                branches['n_taus'] = safe_num('taus')
                branches['n_jets'] = safe_num('jets')
                branches['n_bjets'] = safe_num('bjets')

                # Add key object kinematics (jagged per-event branches where available)
                def _add_object_branch(obj_key, field, branch_name):
                    obj_arr = objects.get(obj_key)
                    if obj_arr is None:
                        return
                    try:
                        if hasattr(obj_arr, 'fields') and field in obj_arr.fields:
                            branches[branch_name] = obj_arr[field]
                    except Exception:
                        pass

                object_field_map = {
                    'muons': {
                        'pt': 'muon_pt',
                        'eta': 'muon_eta',
                        'phi': 'muon_phi',
                    },
                    'electrons': {
                        'pt': 'electron_pt',
                        'eta': 'electron_eta',
                        'phi': 'electron_phi',
                    },
                    'jets': {
                        'pt': 'jet_pt',
                        'eta': 'jet_eta',
                        'phi': 'jet_phi',
                        'btagDeepFlavB': 'jet_btag',
                    },
                    'bjets': {
                        'pt': 'bjet_pt',
                        'eta': 'bjet_eta',
                        'phi': 'bjet_phi',
                    },
                    'taus': {
                        'pt': 'tau_pt',
                        'eta': 'tau_eta',
                        'phi': 'tau_phi',
                    },
                    'fatjets': {
                        'pt': 'fatjet_pt',
                        'eta': 'fatjet_eta',
                        'phi': 'fatjet_phi',
                        'mass': 'fatjet_mass',
                    },
                }

                for object_key, field_map in object_field_map.items():
                    for src_field, dst_branch in field_map.items():
                        _add_object_branch(object_key, src_field, dst_branch)

                # Add event weights if provided
                if event_weights:
                    for name, val in event_weights.items():
                        if isinstance(val, dict):
                            # Dictionary with central/up/down variations
                            for var, arr in val.items():
                                if isinstance(arr, np.ndarray):
                                    branches[f'{name}_{var}'] = arr
                        elif isinstance(val, np.ndarray):
                            branches[name] = val

                # Write to ROOT
                outdir = os.path.dirname(output_file_root)
                if outdir:
                    os.makedirs(outdir, exist_ok=True)

                with uproot.recreate(output_file_root) as f:
                    f['Events'] = branches

                    # Add metadata tree with n_events (total events before selection)
                    if n_events_total is not None:
                        metadata_dict = {
                            'n_events': np.array([n_events_total], dtype=np.int64),
                            'n_selected_events': np.array([len(events)], dtype=np.int64),
                        }
                        f['Metadata'] = metadata_dict
                        logging.info(f"Added n_events={n_events_total} to ROOT file metadata")

                # Verify file was created
                if os.path.exists(output_file_root):
                    file_size = os.path.getsize(output_file_root)
                    logging.info(f"✓ Event selection exported to ROOT file {output_file_root} ({file_size} bytes, {len(events)} events)")
                else:
                    logging.error(f"✗ ROOT file {output_file_root} was not created!")
            except Exception as e:
                logging.error(f"Failed to write ROOT event selection to {output_file_root}: {e}", exc_info=True)

    def get_histogram_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for all histograms.

        Returns:
            Dictionary with statistics for each histogram
        """
        return self.histogram_manager.get_histogram_statistics(self.histograms)

    def save_results(self, output_file: str):
        """
        Save analysis results to file.

        Args:
            output_file: Output file path
        """
        if output_file.endswith('.parquet'):
            self._save_parquet(output_file)
        elif output_file.endswith('.root'):
            self._save_root(output_file)
        else:
            self._save_pickle(output_file)

    def _save_parquet(self, output_file: str):
        """Save results as Parquet file (histograms + per-event weights)."""
        import pandas as pd

        # Histogram data
        data = {}
        for name, hist in self.accumulator["histograms"].items():
            if hasattr(hist, 'values'):
                data[name] = hist.values().flatten()
            else:
                data[name] = hist.get("values", [])

        # Per-event weights (central/up/down per systematic)
        event_weights = self.accumulator.get("event_weights", {})
        if event_weights:
            flat = _event_weights_flat_columns(event_weights)
            for k, v in flat.items():
                data[k] = v

        df = pd.DataFrame(data)
        df.to_parquet(output_file)
        logging.info(f"Saved results to {output_file}")

    def _save_root(self, output_file: str):
        """Save results as ROOT file (histograms + per-event weight tree)."""
        try:
            import uproot
            with uproot.recreate(output_file) as f:
                # Save histograms
                for name, hist in self.accumulator["histograms"].items():
                    if hasattr(hist, 'values'):
                        f[name] = hist

                # Save metadata
                f["metadata"] = self.accumulator["metadata"]

                # Per-event weights as TTree (flat columns)
                event_weights = self.accumulator.get("event_weights", {})
                if event_weights:
                    flat = _event_weights_flat_columns(event_weights)
                    if flat:
                        f["event_weights"] = flat

            logging.info(f"Saved results to {output_file}")
        except ImportError:
            logging.warning("uproot not available. Falling back to pickle.")
            self._save_pickle(output_file)

    def _save_pickle(self, output_file: str):
        """Save results as pickle file."""
        import pickle
        with open(output_file, 'wb') as f:
            pickle.dump(self.accumulator, f)
        logging.info(f"Saved results to {output_file}")

    def get_cutflow_summary(self) -> str:
        """
        Get a formatted cutflow summary.

        Returns:
            Formatted cutflow string
        """
        cutflow = self.accumulator.get("cutflow", {})
        if not cutflow:
            return "No cutflow data available"

        summary = "Cutflow Summary:\n"
        summary += "=" * 50 + "\n"

        for cut_name, n_events in cutflow.items():
            summary += f"{cut_name:20s}: {n_events:8d}\n"

        return summary

    def get_processing_summary(self) -> str:
        """
        Get a formatted processing summary.

        Returns:
            Formatted processing summary string
        """
        metadata = self.accumulator.get("metadata", {})
        if not metadata:
            return "No processing data available"

        summary = "Processing Summary:\n"
        summary += "=" * 50 + "\n"
        summary += f"Events processed: {metadata.get('n_events_processed', 0)}\n"
        summary += f"Events selected:  {metadata.get('n_events_selected', 0)}\n"
        summary += f"Processing time:   {metadata.get('processing_time', 0):.2f} seconds\n"

        weight_stats = metadata.get('weight_statistics', {})
        if weight_stats:
            summary += f"Weight statistics:\n"
            summary += f"  Mean: {weight_stats.get('mean', 0):.4f}\n"
            summary += f"  Std:  {weight_stats.get('std', 0):.4f}\n"
            summary += f"  Sum:  {weight_stats.get('sum', 0):.2f}\n"

        return summary


# Coffea processor wrapper for compatibility
if COFFEA_AVAILABLE:
    class DarkBottomLineCoffeaProcessor(processor.ProcessorABC):
        """
        Coffea-compatible processor wrapper.
        """

        def __init__(self, config: Dict[str, Any]):
            self.config = config
            self.processor = DarkBottomLineProcessor(config)
            # Initialize accumulator for Coffea
            self.accumulator = processor.dict_accumulator({
                "histograms": processor.defaultdict_accumulator(float),
                "cutflow": processor.defaultdict_accumulator(int),
                "metadata": processor.dict_accumulator({}),
            })

        def process(self, events: ak.Array) -> Dict[str, Any]:
            """Process events using the main processor."""
            result = self.processor.process(events)
            # Update accumulator with results
            if "histograms" in result:
                for key, value in result["histograms"].items():
                    self.accumulator["histograms"][key] += value
            if "cutflow" in result:
                for key, value in result["cutflow"].items():
                    self.accumulator["cutflow"][key] += value
            if "metadata" in result:
                self.accumulator["metadata"].update(result["metadata"])
            return self.accumulator

        def postprocess(self, accumulator: Dict[str, Any]) -> Dict[str, Any]:
            """Post-process results."""
            return accumulator
