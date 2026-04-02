"""
Multi-region analyzer for DarkBottomLine framework.
"""

import awkward as ak
import numpy as np
import time
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path
import json

from .processor import (
    DarkBottomLineProcessor,
    _build_event_weights_for_save,
    _event_weights_flat_columns,
)
from .objects import build_objects
from .regions import RegionManager
from .histograms import HistogramManager
from .selections import apply_selection

# Try to import Coffea for processor wrapper
try:
    from coffea import processor
    COFFEA_AVAILABLE = True
except ImportError:
    COFFEA_AVAILABLE = False


class _BoostHistAccumulator:
    """Thin accumulator wrapper for hist.Hist / boost_histogram objects.

    coffea's dict_accumulator.add() requires every leaf value to implement
    .identity() (returns a zero-valued copy) and __iadd__. hist.Hist has
    __iadd__ but not .identity(), so it cannot be stored directly in a
    dict_accumulator that is merged across futures/dask workers.
    """

    def __init__(self, h: Any) -> None:
        self._h = h

    def identity(self) -> "_BoostHistAccumulator":
        import copy
        h_copy = copy.deepcopy(self._h)
        h_copy.reset()
        return _BoostHistAccumulator(h_copy)

    def add(self, other: "_BoostHistAccumulator") -> None:
        if isinstance(other, _BoostHistAccumulator):
            self._h += other._h

    def __iadd__(self, other: "_BoostHistAccumulator") -> "_BoostHistAccumulator":
        self.add(other)
        return self

    @property
    def value(self) -> Any:
        return self._h


class DarkBottomLineAnalyzer:
    """
    Multi-region analyzer extending the base processor.
    """

    def __init__(self, config: Dict[str, Any], regions_config_path: Optional[str] = None):
        """
        Initialize analyzer with configuration and regions.

        Args:
            config: Base configuration dictionary
            regions_config_path: Path to regions configuration file (optional for event-selection-only mode)
        """
        # Initialize base processor
        self.base_processor = DarkBottomLineProcessor(config)

        # Initialize region manager (only if regions_config_path is provided)
        if regions_config_path:
            self.region_manager = RegionManager(regions_config_path)
            self.histogram_manager = HistogramManager()
            self.region_histograms = self._create_region_histograms()
        else:
            self.region_manager = None
            self.histogram_manager = None
            self.region_histograms = {}

        # Initialize accumulator
        self.accumulator = {
            "regions": {},
            "region_histograms": self.region_histograms,
            "region_cutflow": {},
            "region_validation": {},
            "metadata": {},
            "event_weights": {},
        }

        logging.info(f"Initialized analyzer with {len(self.region_manager.regions) if self.region_manager else 0} regions")

    def _create_region_histograms(self) -> Dict[str, Dict[str, Any]]:
        """
        Create histograms for each region.

        Returns:
            Dictionary of region histograms
        """
        region_histograms = {}

        for region_name in self.region_manager.regions.keys():
            # Create standard histograms for each region
            histograms = self.histogram_manager.define_histograms()

            # Add region-specific histograms
            histograms[f"{region_name}_dnn_score"] = self._create_dnn_histogram()
            histograms[f"{region_name}_region_variables"] = self._create_region_variables_histogram()

            region_histograms[region_name] = histograms

        return region_histograms

    def _create_dnn_histogram(self) -> Any:
        """
        Create DNN score histogram.

        Returns:
            DNN score histogram
        """
        try:
            import hist
            return hist.Hist(
                hist.axis.Regular(50, 0, 1, name="dnn_score", label="DNN Score"),
                storage=hist.storage.Weight()
            )
        except ImportError:
            return {
                "bins": np.linspace(0, 1, 51),
                "label": "DNN Score",
                "values": [],
                "weights": []
            }

    def _create_region_variables_histogram(self) -> Any:
        """
        Create region-specific variables histogram.

        Returns:
            Region variables histogram
        """
        try:
            import hist
            return hist.Hist(
                hist.axis.Regular(50, 0, 500, name="met", label="MET [GeV]"),
                hist.axis.Regular(10, 0, 10, name="nbjets", label="Number of B-jets"),
                storage=hist.storage.Weight()
            )
        except ImportError:
            return {
                "bins": [np.linspace(0, 500, 51), np.arange(0, 11)],
                "label": "Region Variables",
                "values": [],
                "weights": []
            }

    def process(self, events: ak.Array, event_selection_output: Optional[str] = None,
                n_events_total: Optional[int] = None, event_selection_only: bool = False,
                output_format: str = "pkl") -> Dict[str, Any]:
        """
        Process events through all regions.

        Cuts are applied in series: (1) preselection (event selection), then
        (2) region cuts. Regions therefore only see events that passed preselection.

        Args:
            events: Awkward Array of events
            event_selection_output: If set, save preselected events to this path (AFTER weight corrections)
            n_events_total: Total number of events before selection (from input files)
            event_selection_only: If True, stop after saving event_selection_output (skip region analysis)
            output_format: Output format for event selection ("pkl", "root", "parquet")

        Returns:
            Analysis results with per-region histograms (or event selection only if event_selection_only=True)
        """
        start_time = time.time()

        # Guard: if region_manager is None and we're not in event-selection-only mode with output, return early
        if self.region_manager is None and not event_selection_output:
            if event_selection_only:
                logging.warning("event_selection_only=True but no event_selection_output provided and region_manager not initialized. Returning safe accumulator.")
                return {
                    "regions": {},
                    "region_histograms": {},
                    "region_cutflow": {},
                    "region_validation": {},
                    "metadata": {},
                    "event_weights": {},
                }
            else:
                logging.error("Region manager not initialized and no event_selection_output provided. Cannot process events.")
                raise ValueError("Region manager must be initialized or event_selection_output must be provided")

        # Apply golden JSON lumi mask (data only; no-op for MC)
        if self.base_processor.is_data:
            events = self.base_processor.apply_lumi_mask(events)
            logging.info(f"Events after golden JSON filter: {len(events)}")

        # Compute h_total_weight from raw events before any selection
        h_total_weight = self.base_processor.correction_manager.get_h_total_weight(events)

        # Build physics objects
        logging.info("Building physics objects...")
        objects = build_objects(events, self.base_processor.config)

        # Step 1 (series): Apply preselection first — regions get events after preselection
        logging.info("Applying preselection (event selection)...")
        try:
            selected_events, selected_objects, cutflow = apply_selection(
                events, objects, self.base_processor.config
            )
            events = selected_events
            objects = selected_objects
            logging.info(f"Events after preselection: {len(events)}")
        except Exception as e:
            logging.error(f"Preselection failed: {e}", exc_info=True)
            raise

        # Step 2: Compute corrections and nominal total weight (for histograms + saving)
        # NOTE: This must happen BEFORE saving event_selection_output to ensure event weights are corrected
        # For data, all weights are 1 — no MC corrections should be applied.
        logging.info("Computing corrections and event weights...")
        event_weights_nominal = None
        event_weights_save = {}
        n_ev = len(events)
        if self.base_processor.is_data:
            logging.info("Data mode: skipping MC corrections, using unit weights.")
            event_weights_nominal = np.ones(n_ev, dtype=np.float64)
            event_weights_save = {
                "generator": np.ones(n_ev, dtype=np.float64),
                "pileup": np.ones(n_ev, dtype=np.float64),
                "weight_total_nominal": np.ones(n_ev, dtype=np.float64),
            }
        else:
            try:
                weight_results = self.base_processor.correction_manager.compute_event_weights(
                    events, objects
                )
                event_weights_nominal = np.asarray(ak.to_numpy(weight_results["total_weight"]))
                event_weights_save = _build_event_weights_for_save(weight_results)
            except Exception as e:
                logging.warning(f"Weight calculation failed, using unit weights: {e}", exc_info=True)
                event_weights_nominal = np.ones(n_ev, dtype=np.float64)
                event_weights_save = {
                    "generator": np.ones(n_ev, dtype=np.float64),
                    "pileup": np.ones(n_ev, dtype=np.float64),
                    "weight_total_nominal": np.ones(n_ev, dtype=np.float64),
                }

        # Step 2b: Optionally save preselected+weighted events to file AFTER weight corrections
        if event_selection_output:
            try:
                logging.info(f"Saving event-selected (with weights) to {event_selection_output} ({len(events)} events)")
                self.base_processor._save_event_selection(
                    event_selection_output, events, objects,
                    max_events=self.base_processor.config.get("max_events"),
                    n_events_total=n_events_total,
                    h_total_weight=h_total_weight,
                    event_weights=event_weights_save,
                    output_format=output_format
                )
                import os
                if os.path.exists(event_selection_output):
                    file_size = os.path.getsize(event_selection_output)
                    logging.info(f"✓ Event selection (with weights) saved to {event_selection_output} ({file_size} bytes)")
                else:
                    logging.error(f"✗ File {event_selection_output} was not created!")
            except Exception as e:
                logging.error(f"Failed to save event selection to {event_selection_output}: {e}", exc_info=True)
            
            # If event_selection_only mode is enabled, return early
            if event_selection_only:
                logging.info("event_selection_only mode: stopping after event selection (no region analysis)")
                processing_time = time.time() - start_time
                # Return empty structure (not to be merged by Coffea in event_selection_only mode,
                # but maintain compatibility for accumulator merging if multiple chunks occur)
                return {
                    "regions": {},
                    "region_histograms": {},
                    "region_cutflow": {},
                    "region_validation": {},
                    "metadata": {},
                    "event_weights": {},
                }

        # Step 3 (series): Apply region cuts to preselected events only
        if self.region_manager is None:
            logging.warning("No region manager configured. Returning early without region analysis.")
            return {
                "regions": {},
                "region_histograms": {},
                "region_cutflow": {},
                "region_validation": {},
                "metadata": {},
                "event_weights": {},
            }
        
        logging.info("Applying region cuts...")
        region_masks = self.region_manager.apply_regions(events, objects)

        # Process each region
        region_results = {}
        for region_name, region_mask in region_masks.items():
            # Skip processing if no events pass cuts
            n_events = ak.sum(region_mask)
            if n_events == 0:
                logging.info(f"Skipping region {region_name}: 0 events")
                region_results[region_name] = {
                    "n_events": 0,
                    "variables": {},
                    "dnn_scores": None
                }
                continue

            logging.info(f"Processing region {region_name}...")
            region_results[region_name] = self._process_region(
                events, objects, region_mask, region_name
            )

        # Validate regions
        logging.info("Validating regions...")
        validation_results = self.region_manager.validate_regions(events, objects)

        # Calculate processing statistics
        processing_time = time.time() - start_time

        # Update accumulator (histograms filled with nominal total weight; all systematics saved)
        self.accumulator["regions"] = region_results
        self.accumulator["region_histograms"] = self._fill_region_histograms(
            events, objects, region_masks, event_weights_nominal
        )
        self.accumulator["region_cutflow"] = self._calculate_region_cutflow(region_masks)
        self.accumulator["region_validation"] = validation_results
        self.accumulator["event_weights"] = event_weights_save
        self.accumulator["metadata"] = {
            "n_events_processed": len(events),
            "n_regions": len(self.region_manager.regions),
            "processing_time": processing_time
        }

        logging.info(f"Analysis completed in {processing_time:.2f} seconds")

        return self.accumulator

    def _extract_objects_from_results(self, base_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract objects from base processor results.

        Args:
            base_results: Results from base processor

        Returns:
            Dictionary of objects
        """
        # This would extract objects from the base processor results
        # For now, return empty dict as placeholder
        return {}

    def _process_region(
        self,
        events: ak.Array,
        objects: Dict[str, Any],
        region_mask: ak.Array,
        region_name: str
    ) -> Dict[str, Any]:
        """
        Process a single region.

        Args:
            events: Awkward Array of events
            objects: Dictionary containing selected objects
            region_mask: Boolean mask for region
            region_name: Name of the region

        Returns:
            Region processing results
        """
        # Apply region mask
        region_events = events[region_mask]
        region_objects = {}

        # Handle empty regions
        if len(region_events) == 0:
            # Create empty objects for empty regions with proper structure
            for obj_name, obj_data in objects.items():
                if isinstance(obj_data, ak.Array):
                    # Create empty array with same structure (empty list of lists)
                    # Use ak.Array([]) but check if it's a list-type first
                    try:
                        # Try to create empty array with same layout
                        if hasattr(obj_data, 'type') and obj_data.type is not None:
                            # Create empty array with same type structure
                            region_objects[obj_name] = ak.Array([])
                        else:
                            region_objects[obj_name] = ak.Array([])
                    except Exception:
                        region_objects[obj_name] = ak.Array([])
                else:
                    region_objects[obj_name] = obj_data
        else:
            # Extract objects for non-empty regions
            for obj_name, obj_data in objects.items():
                if isinstance(obj_data, ak.Array):
                    region_objects[obj_name] = obj_data[region_mask]
                else:
                    region_objects[obj_name] = obj_data

        # Calculate region-specific variables
        region_variables = self._calculate_region_variables(region_events, region_objects)

        # Apply DNN if available
        dnn_scores = self._apply_dnn(region_events, region_objects)

        return {
            "n_events": len(region_events),
            "variables": region_variables,
            "dnn_scores": dnn_scores,
            "region_name": region_name
        }

    def _calculate_region_variables(
        self,
        events: ak.Array,
        objects: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate region-specific variables.

        Args:
            events: Awkward Array of events
            objects: Dictionary containing selected objects

        Returns:
            Dictionary of region variables
        """
        variables = {}

        # Handle empty events
        if len(events) == 0:
            return {}

        # MET
        variables["met"] = events["PFMET_pt"] if "PFMET_pt" in events.fields else events["MET_pt"]

        # Jet multiplicity; use tight pt>30 leptons for region-consistent variables
        jets = objects.get("jets", ak.Array([]))
        bjets = objects.get("bjets", ak.Array([]))
        muons = objects.get("tight_muons_pt30", ak.Array([]))
        electrons = objects.get("tight_electrons_pt30", ak.Array([]))
        taus = objects.get("tight_taus_pt30", ak.Array([]))

        # Check if objects are empty
        # Safe per-event count: use axis=1 for jagged arrays; depth-1 (sliced region) can raise
        def _safe_num_jagged(arr, n_ev: int):
            if arr is None or (isinstance(arr, ak.Array) and len(arr) == 0):
                return np.zeros(n_ev, dtype=np.int64)
            try:
                return np.asarray(ak.num(arr, axis=1))
            except (Exception, BaseException):
                return np.zeros(n_ev, dtype=np.int64)

        n_ev = len(events) # Total events in this region

        variables["n_jets"] = _safe_num_jagged(objects.get("jets"), n_ev)
        variables["n_bjets"] = _safe_num_jagged(objects.get("bjets"), n_ev)
        variables["n_muons"] = _safe_num_jagged(objects.get("tight_muons_pt30"), n_ev)
        variables["n_electrons"] = _safe_num_jagged(objects.get("tight_electrons_pt30"), n_ev)
        variables["n_taus"] = _safe_num_jagged(objects.get("tight_taus_pt30"), n_ev)

        # DeltaPhi between MET and jets (use safe count; avoid axis=1 on depth-1 arrays)
        jets = objects.get("jets", ak.Array([]))
        met_phi = events["PFMET_phi"] if "PFMET_phi" in events.fields else events["MET_phi"]
        n_jets_per_event = _safe_num_jagged(jets, n_ev)
        has_jets = np.any(n_jets_per_event > 0)

        if has_jets:
            try:
                jet_phi = jets.phi
                delta_phi = ak.min(ak.abs(jet_phi - met_phi), axis=1)
                delta_phi = ak.fill_none(delta_phi, 0.0)
                variables["delta_phi"] = delta_phi
            except (Exception, BaseException):
                variables["delta_phi"] = np.zeros(n_ev, dtype=np.float64)
        else:
            variables["delta_phi"] = np.zeros(n_ev, dtype=np.float64)

        return variables

    def _apply_dnn(
        self,
        events: ak.Array,
        objects: Dict[str, Any]
    ) -> Optional[ak.Array]:
        """
        Apply DNN to events (placeholder implementation).

        Args:
            events: Awkward Array of events
            objects: Dictionary containing selected objects

        Returns:
            DNN scores or None if DNN not available
        """
        # Placeholder for DNN application
        # In real implementation, this would load and apply the trained DNN model
        return None

    def _fill_region_histograms(
        self,
        events: ak.Array,
        objects: Dict[str, Any],
        region_masks: Dict[str, ak.Array],
        event_weights_nominal: Optional[Union[ak.Array, np.ndarray]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fill histograms for all regions with nominal total weight.

        Args:
            events: Awkward Array of events
            objects: Dictionary containing selected objects
            region_masks: Dictionary of region masks
            event_weights_nominal: Per-event nominal total weight (optional); if None, use ones.

        Returns:
            Dictionary of filled region histograms
        """
        filled_histograms = {}
        n_ev = len(events)
        # Convert to numpy once for slicing per region
        if event_weights_nominal is not None:
            try:
                w_full = np.asarray(ak.to_numpy(event_weights_nominal))
            except Exception:
                w_full = np.ones(n_ev, dtype=np.float64)
        else:
            w_full = np.ones(n_ev, dtype=np.float64)

        for region_name, region_mask in region_masks.items():
            # Check if region mask is valid and has any passing events
            n_passing = ak.sum(region_mask) if hasattr(region_mask, '__len__') else 0

            # Handle empty regions
            if n_passing == 0:
                # Create empty objects for empty regions
                region_objects = {}
                for obj_name, obj_data in objects.items():
                    if isinstance(obj_data, ak.Array):
                        region_objects[obj_name] = ak.Array([])
                    else:
                        region_objects[obj_name] = obj_data

                filled_histograms[region_name] = self.histogram_manager.define_histograms()
            else:
                # Apply region mask to events
                try:
                    region_events = events[region_mask]
                except Exception as e:
                    logging.warning(f"Error slicing events for region {region_name}: {e}, skipping")
                    filled_histograms[region_name] = self.histogram_manager.define_histograms()
                    continue

                # Extract objects for non-empty regions
                region_objects = {}
                for obj_name, obj_data in objects.items():
                    if isinstance(obj_data, ak.Array):
                        try:
                            region_objects[obj_name] = obj_data[region_mask]
                        except Exception as e:
                            logging.warning(f"Error slicing {obj_name} for region {region_name}: {e}, using empty array")
                            region_objects[obj_name] = ak.Array([])
                    else:
                        region_objects[obj_name] = obj_data

                # Fill histograms with nominal total weight (sliced for this region)
                if len(region_events) > 0:
                    n_r = len(region_events)
                    try:
                        mask_np = np.asarray(ak.to_numpy(region_mask))
                        w = w_full[mask_np].astype(np.float64)
                    except (Exception, BaseException):
                        w = np.ones(n_r, dtype=np.float64)
                    filled_histograms[region_name] = self.histogram_manager.fill_histograms(
                        region_events, region_objects, w
                    )
                else:
                    filled_histograms[region_name] = self.histogram_manager.define_histograms()

        return filled_histograms

    def _calculate_region_cutflow(self, region_masks: Dict[str, ak.Array]) -> Dict[str, Any]:
        """
        Calculate cutflow for all regions.

        Args:
            region_masks: Dictionary of region masks

        Returns:
            Cutflow results
        """
        cutflow = {
            "total_events": len(region_masks[list(region_masks.keys())[0]]) if region_masks else 0,
            "regions": {}
        }

        for region_name, mask in region_masks.items():
            n_events = ak.sum(mask)
            cutflow["regions"][region_name] = {
                "n_events": n_events,
                "fraction": float(n_events) / cutflow["total_events"] if cutflow["total_events"] > 0 else 0.0
            }

        return cutflow

    def get_region_summary(self) -> str:
        """
        Get formatted region summary.

        Returns:
            Formatted region summary string
        """
        summary = "Region Analysis Summary:\n"
        summary += "=" * 50 + "\n"

        # Region summary (only if region_manager is initialized)
        if self.region_manager:
            region_summary = self.region_manager.get_region_summary()
            summary += f"Number of regions: {region_summary['n_regions']}\n"
            summary += f"Signal regions: {self.region_manager.get_signal_regions()}\n"
            summary += f"Control regions: {self.region_manager.get_control_regions()}\n"
            summary += f"Validation regions: {self.region_manager.get_validation_regions()}\n\n"

            # Region details
            for region_name, region in self.region_manager.regions.items():
                summary += f"{region_name}:\n"
                summary += f"  Description: {region.description}\n"
                summary += f"  Cuts: {len(region.cuts)} cuts\n"
                summary += f"  Expected backgrounds: {region.expected_backgrounds}\n"
                summary += f"  Blind data: {region.blind_data}\n"
                summary += f"  Priority: {region.priority}\n"
                if region.transfer_factor_to_SR:
                    summary += f"  Transfer factor: {region.transfer_factor_to_SR}\n"
                summary += "\n"
        else:
            summary += "No region manager initialized (event selection only mode)\n"

        return summary

    def get_region_validation_summary(self) -> str:
        """
        Get formatted region validation summary.

        Returns:
            Formatted validation summary string
        """
        validation = self.accumulator.get("region_validation", {})
        if not validation:
            return "No validation data available"

        summary = "Region Validation Summary:\n"
        summary += "=" * 50 + "\n"

        if validation.get("status") == "completed":
            # Region statistics
            summary += "Region Statistics:\n"
            for region_name, stats in validation.get("regions", {}).items():
                summary += f"  {region_name}: {stats['n_events']} events ({stats['fraction']:.2%})\n"

            # Overlaps
            if validation.get("overlaps"):
                summary += "\nRegion Overlaps:\n"
                for overlap_name, overlap_stats in validation["overlaps"].items():
                    summary += f"  {overlap_name}: {overlap_stats['n_overlap']} events ({overlap_stats['fraction']:.2%})\n"

            # Warnings
            if validation.get("warnings"):
                summary += "\nWarnings:\n"
                for warning in validation["warnings"]:
                    summary += f"  - {warning}\n"
        else:
            summary += f"Validation status: {validation.get('status', 'unknown')}\n"

        return summary

    def save_results(self, output_file: str, output_format: str = "pkl"):
        """
        Save analysis results to file.

        Args:
            output_file: Output file path
            output_format: Output format ("pkl", "root", "parquet"). If output_format is specified,
                          use it; otherwise infer from output_file extension.
        """
        # If output_format is explicitly specified, add extension if needed
        if output_format and output_format != "pkl":
            if not output_file.endswith(f'.{output_format}'):
                output_file_with_format = f"{output_file.rsplit('.', 1)[0]}.{output_format}" if '.' in output_file else f"{output_file}.{output_format}"
            else:
                output_file_with_format = output_file
        else:
            output_file_with_format = output_file
        
        # Route to appropriate save function based on format
        if output_file_with_format.endswith('.parquet'):
            self._save_parquet(output_file_with_format)
        elif output_file_with_format.endswith('.root'):
            self._save_root(output_file_with_format)
        else:
            # Default to pickle
            self._save_pickle(output_file_with_format)

    def _save_parquet(self, output_file: str):
        """Save results as Parquet file (region variables + per-event weights)."""
        import pandas as pd

        # Convert region results to DataFrame format
        data = {}
        for region_name, region_data in self.accumulator.get("regions", {}).items():
            for var_name, var_data in region_data.get("variables", {}).items():
                col_name = f"{region_name}_{var_name}"
                if isinstance(var_data, ak.Array):
                    data[col_name] = ak.to_numpy(var_data)
                else:
                    data[col_name] = var_data

        # Per-event weights (central/up/down per systematic)
        event_weights = self.accumulator.get("event_weights", {})
        if event_weights:
            flat = _event_weights_flat_columns(event_weights)
            for k, v in flat.items():
                data[k] = v

        df = pd.DataFrame(data)
        df.to_parquet(output_file)
        logging.info(f"Saved region results to {output_file}")

    def _save_root(self, output_file: str):
        """Save results as ROOT file (histograms + per-event weight tree)."""
        try:
            import uproot
            outdir = os.path.dirname(output_file)
            if outdir:
                os.makedirs(outdir, exist_ok=True)
            with uproot.recreate(output_file) as f:
                # Save region histograms
                for region_name, histograms in self.accumulator.get("region_histograms", {}).items():
                    for hist_name, hist in histograms.items():
                        f[f"{region_name}_{hist_name}"] = hist

                # Save metadata as a JSON string
                metadata_str = json.dumps(self.accumulator.get("metadata", {}))
                f["metadata"] = metadata_str

                # Per-event weights as TTree
                event_weights = self.accumulator.get("event_weights", {})
                if event_weights:
                    flat = _event_weights_flat_columns(event_weights)
                    if flat:
                        f["event_weights"] = flat

            logging.info(f"Saved region results to {output_file}")
        except ImportError:
            logging.warning("uproot not available. Falling back to pickle.")
            self._save_pickle(output_file)

    def _save_pickle(self, output_file: str):
        """Save results as pickle file."""
        import pickle
        outdir = os.path.dirname(output_file)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        with open(output_file, 'wb') as f:
            pickle.dump(self.accumulator, f)
        logging.info(f"Saved region results to {output_file}")


# Coffea processor wrapper for analyzer compatibility
if COFFEA_AVAILABLE:
    class DarkBottomLineAnalyzerCoffeaProcessor(processor.ProcessorABC):
        """
        Coffea-compatible processor wrapper for multi-region analyzer.
        """

        def __init__(self, config: Dict[str, Any], regions_config_path: Optional[str] = None, 
                     event_selection_output: Optional[str] = None,
                     n_events_total: Optional[int] = None,
                     event_selection_only: bool = False,
                     output_format: Optional[str] = None,
                     max_events: Optional[int] = None):
            self.config = config
            self.regions_config_path = regions_config_path
            self.event_selection_output = event_selection_output
            self.n_events_total = n_events_total  # Total events before selection
            self.h_total_weight = None  # Set on first chunk in process()
            self.event_selection_only = event_selection_only
            self.max_events = max_events  # Maximum events to process
            self.processed_events = 0  # Track number of events processed
            # Auto-detect output_format from event_selection_output extension if not specified
            if output_format is None and event_selection_output:
                if event_selection_output.endswith('.root'):
                    output_format = 'root'
                elif event_selection_output.endswith('.parquet'):
                    output_format = 'parquet'
                else:
                    output_format = 'pkl'
            self.output_format = output_format or 'pkl'
            self.analyzer = DarkBottomLineAnalyzer(config, regions_config_path)
            # Initialize accumulator for Coffea
            self.accumulator = processor.dict_accumulator({
                "regions": processor.dict_accumulator({}),
                "region_histograms": processor.dict_accumulator({}),
                "region_cutflow": processor.dict_accumulator({}),
                "region_validation": processor.dict_accumulator({}),
                "metadata": processor.dict_accumulator({}),
                "event_weights": processor.dict_accumulator({}),
                "_event_selection_chunk_files": processor.dict_accumulator({}),  # Use dict_accumulator for proper merging
            })
            
            # Store selected events/objects for event_selection_output
            # Use file-based approach for cross-worker compatibility
            # Always create _temp_dir so event_weights can be saved to chunk files
            # regardless of whether event_selection_output is requested.
            import tempfile
            import os
            import uuid
            unique_id = str(uuid.uuid4())[:8]
            self._temp_dir = tempfile.mkdtemp(prefix=f"dbl_chunks_{unique_id}_")

        def process(self, events: ak.Array) -> Dict[str, Any]:
            """Process events using the analyzer."""
            metadata = getattr(events, "metadata", {}) if events is not None else {}
            events_to_process = events

            if self.max_events is not None and isinstance(metadata, dict):
                entry_start = metadata.get("entrystart")
                entry_stop = metadata.get("entrystop")
                if isinstance(entry_start, int) and isinstance(entry_stop, int):
                    if entry_start >= self.max_events:
                        logging.info(
                            f"Skipping chunk [{entry_start}, {entry_stop}) due to max-events={self.max_events}"
                        )
                        return self.accumulator
                    if entry_stop > self.max_events:
                        keep_events = self.max_events - entry_start
                        if keep_events <= 0:
                            logging.info(
                                f"Skipping chunk [{entry_start}, {entry_stop}) due to max-events={self.max_events}"
                            )
                            return self.accumulator
                        events_to_process = events[:keep_events]
                        logging.info(
                            f"Trimming chunk [{entry_start}, {entry_stop}) to {keep_events} events for max-events={self.max_events}"
                        )

            # Fallback path for executors/chunks without metadata
            if self.max_events is not None:
                events_remaining = self.max_events - self.processed_events
                if events_remaining <= 0:
                    logging.info(f"Skipping chunk: already processed {self.processed_events}/{self.max_events} max events")
                    return self.accumulator
                if len(events_to_process) > events_remaining:
                    events_to_process = events_to_process[:events_remaining]
                    logging.info(f"Limiting chunk to {len(events_to_process)} events (max_events={self.max_events}, already processed={self.processed_events})")
            
            # Track processed events
            self.processed_events += len(events_to_process)
            logging.info(f"Processing {len(events_to_process)} events (total processed: {self.processed_events}/{self.max_events if self.max_events else 'unlimited'})")

            # Accumulate h_total_weight across chunks
            chunk_h = self.analyzer.base_processor.correction_manager.get_h_total_weight(events_to_process)
            self.h_total_weight = (self.h_total_weight or 0.0) + chunk_h

            # Call analyzer.process() with appropriate parameters
            # In event_selection_only mode, analyzer will skip region analysis
            result = self.analyzer.process(
                events_to_process,
                # Never write the final output file per-chunk: the temp_dir mechanism
                # collects chunks and postprocess() does the single final save.
                event_selection_output=None,
                event_selection_only=self.event_selection_only,
                output_format=self.output_format,
                n_events_total=self.n_events_total
            )

            # If event_selection_output is requested, collect selected events from this chunk
            if self.event_selection_output and self._temp_dir:
                try:
                    import os
                    import pickle
                    import awkward as ak
                    logging.info(f"Collecting selected events for event_selection_output from chunk ({len(events_to_process)} events)")
                    from .selections import apply_selection
                    from .objects import build_objects
                    # Apply event-level selection to get selected events
                    objects = build_objects(events_to_process, self.config)
                    selected_events, selected_objects, _ = apply_selection(
                        events_to_process, objects, self.config
                    )
                    logging.info(f"Chunk: {len(selected_events)}/{len(events_to_process)} events passed selection")

                    # Save this chunk to a temporary file (for cross-worker compatibility)
                    # Use unique filename to avoid conflicts across workers
                    import time
                    import uuid
                    chunk_id = f"{int(time.time() * 1000000)}_{uuid.uuid4().hex[:8]}"
                    chunk_file = os.path.join(self._temp_dir, f"chunk_{chunk_id}.pkl")
                    with open(chunk_file, 'wb') as f:
                        pickle.dump({"events": selected_events, "objects": selected_objects}, f, protocol=pickle.HIGHEST_PROTOCOL)

                    # Add to accumulator dict (this will be merged across workers)
                    # Note: Don't store in accumulator because Coffea can't properly merge True values
                    # Instead, we'll dir-scan for chunk files in postprocess
                    # self.accumulator["_event_selection_chunk_files"][chunk_file] = True
                    logging.info(f"Saved chunk to {chunk_file}, accumulator will scan dir in postprocess")
                except Exception as e:
                    logging.warning(f"Failed to collect selected events for event_selection_output: {e}", exc_info=True)
            
            # If event_selection_only mode is enabled, don't merge results (no region analysis was done)
            if self.event_selection_only:
                logging.info("event_selection_only mode: skipping accumulator merge (no region analysis)")
                return self.accumulator

            # Save region analysis results to a temp file instead of merging via coffea's
            # dict_accumulator.add(). coffea's accumulator requires every leaf value to
            # implement .identity(), which plain int/float/str/None/hist.Hist do not. This
            # causes AttributeError when multiple workers merge accumulators (futures/dask).
            # The same pattern is already used for event_weights and event_selection output.
            if self._temp_dir:
                self._save_analysis_result_to_temp(result, chunk_h=chunk_h)

            # event_weights still go through the temp-file mechanism
            if "event_weights" in result and result["event_weights"]:
                self._merge_event_weights_into_accumulator(result["event_weights"])
            return self.accumulator

        def _to_dict_accumulator(self, value: Any):
            """Recursively convert nested dicts to coffea dict_accumulator.

            ALL leaf values must be wrapped in a coffea accumulator type —
            otherwise dict_accumulator.add() calls value.identity() on new
            keys, which fails for plain Python int/float/str/None (no identity
            method). This is only triggered when multiple workers merge
            accumulators (futures/dask on lxplus), not in single-process runs.
            """
            if isinstance(value, dict):
                return processor.dict_accumulator(
                    {k: self._to_dict_accumulator(v) for k, v in value.items()}
                )
            # bool must be checked before int (bool is a subclass of int)
            if isinstance(value, bool):
                return processor.value_accumulator(int, int(value))
            if isinstance(value, (int, np.integer)):
                return processor.value_accumulator(int, int(value))
            if isinstance(value, (float, np.floating)):
                return processor.value_accumulator(float, float(value))
            if isinstance(value, np.ndarray):
                return processor.list_accumulator(value.tolist())
            if isinstance(value, list):
                return processor.list_accumulator(value)
            # hist.Hist / boost_histogram objects: have __iadd__ but no .identity().
            # Wrap in _BoostHistAccumulator so coffea can merge them across workers.
            if hasattr(value, "axes") and hasattr(value, "__iadd__"):
                return _BoostHistAccumulator(value)
            # str, None, and other types: use a 0-valued int accumulator as a
            # safe no-op placeholder so identity() and add() don't crash.
            # These fields (dnn_scores=None, region_name=str) are metadata
            # that should not be summed; they are retrieved from non-accumulated
            # paths (e.g. postprocess chunk files) instead.
            return processor.value_accumulator(int, 0)

        def _merge_event_weights_into_accumulator(self, new_weights: Dict[str, Any]):
            """Save per-chunk event weights to a temp file.

            Instead of storing plain Python dicts in the coffea accumulator
            (which causes ``dict += dict`` TypeError during inter-worker
            merging), we write each chunk's event weights to a temp pickle
            file.  They are collected and concatenated in ``postprocess``.
            """
            if not self._temp_dir:
                return
            import os
            import pickle
            import time
            import uuid
            chunk_id = f"{int(time.time() * 1000000)}_{uuid.uuid4().hex[:8]}"
            ew_file = os.path.join(self._temp_dir, f"ew_chunk_{chunk_id}.pkl")
            try:
                # Serialise: convert any non-ndarray values to ndarray
                serialisable = {}
                for k, v in new_weights.items():
                    if isinstance(v, dict):
                        serialisable[k] = {
                            kk: np.asarray(vv)
                            for kk, vv in v.items()
                            if isinstance(vv, np.ndarray) or hasattr(vv, '__len__')
                        }
                    elif isinstance(v, np.ndarray):
                        serialisable[k] = v
                    elif hasattr(v, '__len__'):
                        serialisable[k] = np.asarray(v)
                with open(ew_file, 'wb') as f:
                    pickle.dump(serialisable, f, protocol=pickle.HIGHEST_PROTOCOL)
                logging.debug(f"Saved event_weights chunk to {ew_file}")
            except Exception as e:
                logging.warning(f"Failed to save event_weights chunk: {e}")
            # Keep the coffea accumulator entry as an empty dict_accumulator so
            # coffea can merge it across workers without errors.
            # (do NOT assign a plain dict here)

        def _save_analysis_result_to_temp(self, result: Dict[str, Any], chunk_h: float = 0.0) -> None:
            """Save per-chunk analysis results (regions, histograms, etc.) to a temp file.

            Bypasses coffea's accumulator type system entirely. hist.Hist and plain
            Python scalars are picklable; they just can't satisfy coffea's .identity()
            requirement. Merging is done manually in postprocess().
            """
            import os, pickle, time, uuid
            chunk_id = f"{int(time.time() * 1000000)}_{uuid.uuid4().hex[:8]}"
            result_file = os.path.join(self._temp_dir, f"analysis_chunk_{chunk_id}.pkl")
            payload = {k: result[k] for k in ("regions", "region_histograms",
                                               "region_cutflow", "region_validation",
                                               "metadata") if k in result}
            payload["h_total_weight"] = float(chunk_h)
            try:
                with open(result_file, "wb") as f:
                    pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
                logging.debug(f"Saved analysis chunk to {result_file}")
            except Exception as e:
                logging.warning(f"Failed to save analysis chunk to temp file: {e}")

        def _merge_analysis_chunks(self, result_files: list) -> Dict[str, Any]:
            """Load and merge per-chunk analysis results from temp pkl files."""
            import pickle
            merged: Dict[str, Any] = {
                "regions": {},
                "region_histograms": {},
                "region_cutflow": {"total_events": 0, "regions": {}},
                "region_validation": {"status": "completed", "regions": {},
                                      "overlaps": {}, "warnings": []},
                "metadata": {"n_events_processed": 0, "n_regions": 0,
                             "processing_time": 0.0},
                "h_total_weight": 0.0,
            }
            for rf in sorted(result_files):
                try:
                    with open(rf, "rb") as f:
                        chunk = pickle.load(f)
                except Exception as e:
                    logging.warning(f"Failed to load analysis chunk {rf}: {e}")
                    continue

                # regions: sum n_events; concatenate per-event variable arrays
                for region, data in chunk.get("regions", {}).items():
                    if region not in merged["regions"]:
                        merged["regions"][region] = {
                            "n_events": 0, "variables": {},
                            "dnn_scores": None, "region_name": region,
                        }
                    merged["regions"][region]["n_events"] += int(data.get("n_events", 0))
                    for var, val in data.get("variables", {}).items():
                        if var not in merged["regions"][region]["variables"]:
                            merged["regions"][region]["variables"][var] = val
                        else:
                            ex = merged["regions"][region]["variables"][var]
                            try:
                                if isinstance(ex, np.ndarray) and isinstance(val, np.ndarray):
                                    merged["regions"][region]["variables"][var] = np.concatenate([ex, val])
                                elif isinstance(ex, list) and isinstance(val, list):
                                    merged["regions"][region]["variables"][var] = ex + val
                            except Exception:
                                pass

                # region_histograms: add hist objects with +
                for region, hists in chunk.get("region_histograms", {}).items():
                    if region not in merged["region_histograms"]:
                        merged["region_histograms"][region] = {}
                    for hname, h in hists.items():
                        if hname not in merged["region_histograms"][region]:
                            merged["region_histograms"][region][hname] = h
                        else:
                            try:
                                merged["region_histograms"][region][hname] = (
                                    merged["region_histograms"][region][hname] + h
                                )
                            except Exception as e:
                                logging.warning(f"Failed to merge histogram {hname}/{region}: {e}")

                # region_cutflow: sum total_events and per-region n_events
                cutflow = chunk.get("region_cutflow", {})
                merged["region_cutflow"]["total_events"] += int(cutflow.get("total_events", 0))
                for region, cf in cutflow.get("regions", {}).items():
                    if region not in merged["region_cutflow"]["regions"]:
                        merged["region_cutflow"]["regions"][region] = {"n_events": 0, "fraction": 0.0}
                    merged["region_cutflow"]["regions"][region]["n_events"] += int(cf.get("n_events", 0))

                # region_validation: sum n_events, collect warnings
                validation = chunk.get("region_validation", {})
                for region, vd in validation.get("regions", {}).items():
                    if region not in merged["region_validation"]["regions"]:
                        merged["region_validation"]["regions"][region] = {"n_events": 0, "fraction": 0.0}
                    merged["region_validation"]["regions"][region]["n_events"] += int(vd.get("n_events", 0))
                for okey, od in validation.get("overlaps", {}).items():
                    if okey not in merged["region_validation"]["overlaps"]:
                        merged["region_validation"]["overlaps"][okey] = dict(od)
                    else:
                        merged["region_validation"]["overlaps"][okey]["n_overlap"] = (
                            int(merged["region_validation"]["overlaps"][okey].get("n_overlap", 0))
                            + int(od.get("n_overlap", 0))
                        )
                merged["region_validation"]["warnings"].extend(validation.get("warnings", []))

                # metadata: sum n_events_processed and processing_time
                meta = chunk.get("metadata", {})
                merged["metadata"]["n_events_processed"] += int(meta.get("n_events_processed", 0))
                merged["metadata"]["processing_time"] += float(meta.get("processing_time", 0.0))
                if "n_regions" in meta:
                    merged["metadata"]["n_regions"] = meta["n_regions"]

                # h_total_weight: sum across all chunks
                merged["h_total_weight"] += float(chunk.get("h_total_weight", 0.0))

            # Recompute fractions
            total_cf = merged["region_cutflow"]["total_events"]
            if total_cf > 0:
                for region in merged["region_cutflow"]["regions"]:
                    merged["region_cutflow"]["regions"][region]["fraction"] = (
                        merged["region_cutflow"]["regions"][region]["n_events"] / total_cf
                    )
            total_ev = merged["metadata"]["n_events_processed"]
            if total_ev > 0:
                for region in merged["region_validation"]["regions"]:
                    merged["region_validation"]["regions"][region]["fraction"] = (
                        merged["region_validation"]["regions"][region]["n_events"] / total_ev
                    )
            return merged

        def postprocess(self, accumulator: Dict[str, Any]) -> Dict[str, Any]:
            """Post-process results."""
            import os
            import pickle
            import awkward as ak

            # Get chunk files from accumulator (merged across all workers)
            # File paths are stored as keys in the dict accumulator
            # Note: Since we stopped storing in accumulator to fix merge issues,
            # now we scan the temp_dir directly for chunk files
            chunk_files = []
            if self._temp_dir and os.path.exists(self._temp_dir):
                chunk_files = [os.path.join(self._temp_dir, f) for f in os.listdir(self._temp_dir) if f.startswith("chunk_") and f.endswith(".pkl")]
            else:
                # Fallback: try to get from accumulator in case it was populated before the change
                chunk_files_acc = accumulator.get("_event_selection_chunk_files", {})
                if isinstance(chunk_files_acc, dict):
                    chunk_files = list(chunk_files_acc.keys())
                else:
                    chunk_files = []

            logging.info(f"postprocess called: event_selection_output={self.event_selection_output}, chunk_files={len(chunk_files)}")

            # ── Merge region analysis results from per-chunk temp files ───────
            if self._temp_dir and os.path.exists(self._temp_dir):
                analysis_files = [
                    os.path.join(self._temp_dir, f)
                    for f in os.listdir(self._temp_dir)
                    if f.startswith("analysis_chunk_") and f.endswith(".pkl")
                ]
                if analysis_files:
                    merged_analysis = self._merge_analysis_chunks(analysis_files)
                    accumulator["regions"] = merged_analysis["regions"]
                    accumulator["region_histograms"] = merged_analysis["region_histograms"]
                    accumulator["region_cutflow"] = merged_analysis["region_cutflow"]
                    accumulator["region_validation"] = merged_analysis["region_validation"]
                    accumulator["metadata"] = merged_analysis["metadata"]
                    # Propagate summed h_total_weight back to the processor instance so
                    # _save_event_selection (called below) receives the correct value.
                    if merged_analysis.get("h_total_weight"):
                        self.h_total_weight = merged_analysis["h_total_weight"]
                    logging.info(f"Merged region analysis from {len(analysis_files)} chunks: "
                                 f"{len(merged_analysis['regions'])} regions, "
                                 f"{merged_analysis['metadata']['n_events_processed']} events processed")
                    for af in analysis_files:
                        try:
                            os.remove(af)
                        except Exception:
                            pass

            # ── Merge event_weights from per-chunk temp files ─────────────────
            if self._temp_dir and os.path.exists(self._temp_dir):
                ew_files = sorted(
                    os.path.join(self._temp_dir, f)
                    for f in os.listdir(self._temp_dir)
                    if f.startswith("ew_chunk_") and f.endswith(".pkl")
                )
                if ew_files:
                    merged_ew: Dict[str, Any] = {}
                    for ew_file in ew_files:
                        try:
                            with open(ew_file, 'rb') as f:
                                chunk_ew = pickle.load(f)
                            for k, v in chunk_ew.items():
                                if isinstance(v, dict):
                                    if k not in merged_ew:
                                        merged_ew[k] = {}
                                    for kk, vv in v.items():
                                        if isinstance(vv, np.ndarray):
                                            if kk in merged_ew[k]:
                                                merged_ew[k][kk] = np.concatenate([merged_ew[k][kk], vv])
                                            else:
                                                merged_ew[k][kk] = vv
                                elif isinstance(v, np.ndarray):
                                    if k in merged_ew:
                                        merged_ew[k] = np.concatenate([merged_ew[k], v])
                                    else:
                                        merged_ew[k] = v
                        except Exception as e:
                            logging.warning(f"Failed to load event_weights chunk {ew_file}: {e}")
                    if merged_ew:
                        accumulator["event_weights"] = merged_ew
                        logging.info(f"Merged event_weights from {len(ew_files)} chunk files")
                    # Clean up ew chunk files
                    for ew_file in ew_files:
                        try:
                            os.remove(ew_file)
                        except Exception:
                            pass

            # Save accumulated event selection if requested
            if self.event_selection_output and chunk_files:
                try:
                    # Load all chunks from files
                    all_selected_events_list = []
                    all_selected_objects_list = []

                    for chunk_file in chunk_files:
                        if os.path.exists(chunk_file):
                            try:
                                with open(chunk_file, 'rb') as f:
                                    chunk_data = pickle.load(f)
                                    events = chunk_data.get("events")
                                    objects = chunk_data.get("objects")
                                    if events is not None and len(events) > 0:
                                        all_selected_events_list.append(events)
                                        all_selected_objects_list.append(objects)
                            except Exception as e:
                                logging.warning(f"Failed to load chunk file {chunk_file}: {e}")

                    if all_selected_events_list:
                        # Concatenate all selected events and objects from all chunks
                        all_selected_events = ak.concatenate(all_selected_events_list)

                        # Merge selected objects dictionaries
                        all_selected_objects = {}
                        if all_selected_objects_list:
                            all_keys = set()
                            for obj_dict in all_selected_objects_list:
                                all_keys.update(obj_dict.keys())
                            for key in all_keys:
                                arrays_to_concat = []
                                for obj_dict in all_selected_objects_list:
                                    if key in obj_dict and len(obj_dict[key]) > 0:
                                        arrays_to_concat.append(obj_dict[key])
                                if arrays_to_concat:
                                    all_selected_objects[key] = ak.concatenate(arrays_to_concat)
                    else:
                        # No events passed selection across all chunks — still write the output
                        # file so Metadata (n_events, h_total_weight) is present for downstream tools.
                        logging.warning(
                            f"No selected events found in {len(chunk_files)} chunk files — "
                            f"writing metadata-only output to {self.event_selection_output}"
                        )
                        all_selected_events = ak.Array([])
                        all_selected_objects = {}

                    # Save using base processor helper (handles empty events gracefully)
                    logging.info(f"Saving accumulated event-level selection to {self.event_selection_output}")
                    self.analyzer.base_processor._save_event_selection(
                        self.event_selection_output, all_selected_events, all_selected_objects,
                        max_events=self.config.get("max_events"),
                        n_events_total=self.n_events_total,
                        h_total_weight=self.h_total_weight,
                        output_format=self.output_format
                    )
                    # Verify file was created
                    if os.path.exists(self.event_selection_output):
                        file_size = os.path.getsize(self.event_selection_output)
                        n_saved = len(all_selected_events)
                        logging.info(f"✓ Saved accumulated event-level selection from {len(chunk_files)} chunks ({n_saved} events) to {self.event_selection_output} ({file_size} bytes)")
                    else:
                        logging.error(f"✗ File {self.event_selection_output} was not created!")

                    # Clean up temporary files and directories
                    # Collect all unique temp directories from chunk file paths
                    temp_dirs = set()
                    for chunk_file in chunk_files:
                        if os.path.exists(chunk_file):
                            temp_dirs.add(os.path.dirname(chunk_file))

                    # Remove chunk files and temp directories
                    for chunk_file in chunk_files:
                        try:
                            if os.path.exists(chunk_file):
                                os.remove(chunk_file)
                        except Exception as e:
                            logging.warning(f"Failed to remove chunk file {chunk_file}: {e}")

                    for temp_dir in temp_dirs:
                        try:
                            if os.path.exists(temp_dir) and os.path.isdir(temp_dir):
                                os.rmdir(temp_dir)
                                logging.debug(f"Removed temp directory {temp_dir}")
                        except Exception as e:
                            logging.warning(f"Failed to remove temp directory {temp_dir}: {e}")

                except Exception as e:
                    logging.error(f"Failed to save accumulated event selection to {self.event_selection_output}: {e}", exc_info=True)

            return accumulator
