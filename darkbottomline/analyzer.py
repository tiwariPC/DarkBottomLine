"""
Multi-region analyzer for DarkBottomLine framework.
"""

# ── Compatibility patch for coffea 2025.12.0 + uproot 5.x (LCG_109) ─────────
# uproot 5.x removed uproot.behaviors.RNTuple. coffea.util._is_interpretable
# references it unconditionally → AttributeError in every loky worker process.
# This module is imported by workers (via cloudpickle), so the patch runs there.
try:
    import uproot.behaviors as _uproot_behaviors
    if not hasattr(_uproot_behaviors, "RNTuple"):
        import types as _types, sys as _sys
        _rnt = _types.ModuleType("uproot.behaviors.RNTuple")
        class _HasFields:
            pass
        _rnt.HasFields = _HasFields
        _uproot_behaviors.RNTuple = _rnt
        _sys.modules.setdefault("uproot.behaviors.RNTuple", _rnt)
        # Re-patch coffea.util._is_interpretable if already imported without the fix
        try:
            import coffea.util as _coffea_util
            import uproot as _uproot
            import warnings as _warnings
            def _is_interpretable_patched(branch, emit_warning=True):
                try:
                    _rnt_cls = _uproot_behaviors.RNTuple.HasFields
                    if isinstance(branch, _rnt_cls):
                        if branch.path.startswith("_collection") or "." in branch.path:
                            return False
                        return True
                except Exception:
                    pass
                if isinstance(branch.interpretation,
                              _uproot.interpretation.identify.uproot.AsGrouped):
                    for _, interp in branch.interpretation.subbranches.items():
                        if isinstance(interp,
                                      _uproot.interpretation.identify.UnknownInterpretation):
                            if emit_warning:
                                _warnings.warn(f"Skipping {branch.name} as it is not interpretable by Uproot")
                            return False
                if isinstance(branch.interpretation,
                              _uproot.interpretation.identify.UnknownInterpretation):
                    if emit_warning:
                        _warnings.warn(f"Skipping {branch.name} as it is not interpretable by Uproot")
                    return False
                try:
                    branch.interpretation.awkward_form(None)
                except _uproot.interpretation.objects.CannotBeAwkward:
                    if emit_warning:
                        _warnings.warn(f"Skipping {branch.name} as it cannot be represented as an Awkward array")
                    return False
                return True
            _coffea_util._is_interpretable = _is_interpretable_patched
            # Also patch the reference inside the nanoevents mapping module
            try:
                import coffea.nanoevents.mapping.uproot as _nano_uproot
                _nano_uproot._is_interpretable = _is_interpretable_patched
            except Exception:
                pass
        except Exception:
            pass
except Exception:
    pass
# ─────────────────────────────────────────────────────────────────────────────

import awkward as ak
import numpy as np
import os
import time
import logging
from typing import Dict, Any, Optional, Union, List, Tuple
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
            "region_cutflow_histograms": {},
            "event_selection_cutflow": {},
            "region_validation": {},
            "metadata": {},
            "event_weights": {},
        }

        logging.debug(f"Initialized analyzer with {len(self.region_manager.regions) if self.region_manager else 0} regions")

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

    def _create_cutflow_histogram(self, cutflow: Dict[str, int]) -> Any:
        """Create a cutflow histogram with labeled steps."""
        try:
            import hist
            labels = list(cutflow.keys())
            h = hist.Hist(
                hist.axis.StrCategory(labels, name="step", label="Cut Step"),
                storage=hist.storage.Weight(),
            )
            for label, value in cutflow.items():
                h.fill(step=label, weight=float(value))
            return h
        except ImportError:
            return {"labels": list(cutflow.keys()), "values": list(cutflow.values())}

    def process(self, events: ak.Array, event_selection_output: Optional[str] = None,
                total_events: Optional[int] = None, input_total_events: Optional[int] = None,
                event_selection_only: bool = False,
                output_format: str = "pkl") -> Dict[str, Any]:
        """
        Process events through all regions.

        Cuts are applied in series: (1) preselection (event selection), then
        (2) region cuts. Regions therefore only see events that passed preselection.

        Args:
            events: Awkward Array of events
            event_selection_output: If set, save preselected events to this path (AFTER weight corrections)
            total_events: Total number of events before selection (legacy semantic used by event-selection output)
            input_total_events: Total number of events in the input files before any max-events slicing
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

        # Compute weighted_total_events from raw events before any selection
        weighted_total_events = self.base_processor.correction_manager.get_weighted_total_events(events)

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
        # Store event-selection cutflow in accumulator so region plots can prepend it
        self.accumulator["event_selection_cutflow"] = dict(cutflow)

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
                event_weights_nominal = np.asarray(ak.to_numpy(weight_results["full_event_weight"]))
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
                    total_events=total_events,
                    weighted_total_events=weighted_total_events,
                    event_weights=event_weights_save,
                    cutflow=cutflow,
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

        # Build cutflow histograms for control regions only (8 CRs)
        region_cutflows = self.region_manager.get_region_cutflows(events, objects, only_control_regions=False)
        region_cutflow_histograms = {
            region_name: self._create_cutflow_histogram(cutflow)
            for region_name, cutflow in region_cutflows.items()
        }

        # Calculate processing statistics
        processing_time = time.time() - start_time

        # Update accumulator (histograms filled with nominal total weight; all systematics saved)
        self.accumulator["regions"] = region_results
        self.accumulator["region_histograms"] = self._fill_region_histograms(
            events, objects, region_masks, event_weights_nominal
        )
        self.accumulator["region_cutflow"] = self._calculate_region_cutflow(region_masks)
        self.accumulator["region_cutflow_histograms"] = region_cutflow_histograms
        # event_selection_cutflow already set after apply_selection above
        self.accumulator["region_validation"] = validation_results
        self.accumulator["event_weights"] = event_weights_save
        self.accumulator["metadata"] = {
            "n_events_processed": len(events),
            "n_events_input": int(input_total_events if input_total_events is not None else (total_events if total_events is not None else len(events))),
            "weighted_total_events": weighted_total_events,
            "n_regions": len(self.region_manager.regions),
            "processing_time": processing_time,
            "weighted_total_events": float(weighted_total_events),
            "luminosity": float(self.base_processor.config.get("lumi", 1.0)),
        }

        logging.info(f"Analysis completed in {processing_time:.2f} seconds")

        return self.accumulator

    def process_from_eventselection(
        self,
        branches: Dict[str, np.ndarray],
        weighted_total_events: float,
        is_data: bool = False,
        dnn_model: Optional[str] = None,
        dnn_config: Optional[str] = None,
        dnn_inference: Optional[Any] = None,
        dnn_mass_scan: Optional[List[Tuple[float, float]]] = None,
    ) -> Dict[str, Any]:
        """
        Run region analysis starting from a pre-selected flat-branch dict (EVENTSELECTION.root).

        Skips NanoAOD object building and event-level preselection — input events have already
        passed the event selection.  Region cuts are applied via RegionManager using the flat
        branches (same fallback logic as make-event-plots region-from-events mode).

        Args:
            branches:                flat {name: np.ndarray} from EVENTSELECTION.root Events tree
            weighted_total_events:   sum of generator weights before preselection (from metadata TH1)
            is_data:                 True for collision data (unit weights)
            dnn_model:               path to trained DNN checkpoint (optional; ignored if
                                      dnn_inference is given)
            dnn_config:              path to DNN config YAML (optional; ignored if
                                      dnn_inference is given)
            dnn_inference:           pre-built DNNInference instance (optional). Callers looping
                                      over many files should build this once and pass it in here
                                      instead of dnn_model/dnn_config, to avoid reloading the
                                      checkpoint from disk on every file.
            dnn_mass_scan:           optional list of (MH3, MH4) points (already resolved via
                                      dnn_inference._resolve_mass_scan by the caller). None (default)
                                      scores once at the benchmark masspoint into a single "ml_score"
                                      field, matching non-parametric behavior. When given, writes one
                                      "ml_score_mh3_<a>_mh4_<b>" field per point instead.

        Returns:
            Same accumulator structure as process()
        """
        import time
        start_time = time.time()

        if self.region_manager is None:
            raise ValueError("RegionManager not initialised — pass --regions-config")

        n_ev = len(next(iter(branches.values()))) if branches else 0

        # Build ak.Array from flat branches; convert object-dtype (jagged) arrays individually
        ak_dict: Dict[str, Any] = {}
        for k, v in branches.items():
            if not isinstance(v, np.ndarray) or v.ndim != 1:
                continue
            if v.dtype == object:
                try:
                    ak_dict[k] = ak.Array(list(v))
                except Exception:
                    pass
            else:
                ak_dict[k] = v
        events = ak.Array(ak_dict)

        # Weights: read from branches if present, else unit
        if is_data:
            event_weights_nominal = np.ones(n_ev, dtype=np.float64)
        else:
            fw = branches.get("full_event_weight")
            if fw is not None and len(fw) == n_ev:
                event_weights_nominal = np.asarray(fw, dtype=np.float64)
            else:
                event_weights_nominal = np.ones(n_ev, dtype=np.float64)

        # Optional DNN scoring
        if dnn_inference is not None or dnn_model:
            try:
                from dnn.common import sanitize_feature_frame
                import pandas as _pd

                if dnn_inference is not None:
                    inferencer = dnn_inference
                else:
                    from .dnn_inference import DNNInference
                    inferencer = DNNInference(dnn_model, dnn_config)
                features = inferencer.features

                # Feature names are exact EVENTSELECTION.root branch names
                # (configs/dnn.yaml features: is the source of truth).
                X_parts = {}
                for feat in features:
                    if feat in events.fields:
                        X_parts[feat] = np.asarray(
                            ak.to_numpy(events[feat]), dtype="f8"
                        )
                    else:
                        X_parts[feat] = np.full(n_ev, -9999.0, dtype="f8")

                X_df = _pd.DataFrame(X_parts)
                X_df = sanitize_feature_frame(X_df)
                X = X_df.to_numpy(dtype="f8")
                if dnn_mass_scan is None:
                    scores = inferencer.predict(X, None).ravel().astype("float32")
                    events = ak.with_field(events, ak.Array(scores), "ml_score")
                else:
                    from .dnn_inference import _mass_branch_name
                    for mh3, mh4 in dnn_mass_scan:
                        masses = np.tile(np.asarray([mh3, mh4], dtype="f8"), (n_ev, 1))
                        scores = inferencer.predict(X, masses).ravel().astype("float32")
                        events = ak.with_field(events, ak.Array(scores), _mass_branch_name("ml_score", mh3, mh4))
            except Exception as _dnn_exc:
                logging.warning("DNN scoring failed, continuing without scores: %s", _dnn_exc)

        # Apply region cuts (objects={} — regions.py uses flat-branch fallbacks)
        logging.debug("Applying region cuts from event-selection branches (%d events)...", n_ev)
        region_masks = self.region_manager.apply_regions(events, objects={})

        region_results: Dict[str, Any] = {}
        for region_name, region_mask in region_masks.items():
            n_passing = int(ak.sum(region_mask))
            if n_passing == 0:
                logging.debug("Skipping region %s: 0 events", region_name)
                region_results[region_name] = {"n_events": 0, "variables": {}, "dnn_scores": None}
                continue
            logging.debug("Processing region %s (%d events)...", region_name, n_passing)
            region_results[region_name] = self._process_region(events, {}, region_mask, region_name)

        region_histograms = self._fill_region_histograms(events, {}, region_masks, event_weights_nominal)
        region_cutflow    = self._calculate_region_cutflow(region_masks)

        # Per-cut-step cutflow for each region (for cutflow waterfall plots)
        region_cutflow_steps: Dict[str, Dict[str, float]] = {}
        for _rname in region_masks:
            _robj = self.region_manager.regions.get(_rname)
            if _robj is not None:
                try:
                    _steps = _robj.apply_cuts_with_yields(events, objects={},
                                                           weight=event_weights_nominal)
                    region_cutflow_steps[_rname] = _steps
                except Exception:
                    pass

        validation        = self.region_manager.validate_regions(events, {})
        processing_time   = time.time() - start_time

        self.accumulator["regions"]          = region_results
        self.accumulator["region_histograms"] = region_histograms
        self.accumulator["region_cutflow"]    = region_cutflow
        self.accumulator["region_cutflow_steps"] = region_cutflow_steps
        self.accumulator["region_validation"] = validation
        self.accumulator["event_weights"]     = {}
        self.accumulator["metadata"] = {
            "n_events_processed":  n_ev,
            "n_regions":           len(self.region_manager.regions),
            "processing_time":     processing_time,
            "weighted_total_events": float(weighted_total_events),
            "luminosity":          float(self.base_processor.config.get("lumi", 1.0)),
        }
        logging.debug("Region analysis from event-selection completed in %.2f s", processing_time)
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

        # MET — PuppiMET preferred
        variables["met"] = next((events[v] for v in ("PuppiMET_pt", "PFMET_pt", "MET_pt") if v in events.fields), None)

        # Jet multiplicity; use tight pt>30 leptons for region-consistent variables
        jets = objects.get("jets", ak.Array([]))
        bjets = objects.get("bjets", ak.Array([]))
        muons = objects.get("tight_muons", ak.Array([]))
        electrons = objects.get("tight_electrons", ak.Array([]))
        taus = objects.get("tight_taus", ak.Array([]))

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
        variables["n_muons"] = _safe_num_jagged(objects.get("tight_muons"), n_ev)
        variables["n_electrons"] = _safe_num_jagged(objects.get("tight_electrons"), n_ev)
        variables["n_taus"] = _safe_num_jagged(objects.get("tight_taus"), n_ev)

        # DeltaPhi between MET and jets (use safe count; avoid axis=1 on depth-1 arrays)
        jets = objects.get("jets", ak.Array([]))
        met_phi = next((events[v] for v in ("PuppiMET_phi", "PFMET_phi", "MET_phi") if v in events.fields), None)
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
        """Return per-event DNN scores from ml_score field if present, else None."""
        if "ml_score" in events.fields:
            return events["ml_score"]
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
                    filled = self.histogram_manager.fill_histograms(
                        region_events, region_objects, w
                    )
                    if "ml_score" in region_events.fields and "dnn_score" in filled:
                        scores = np.clip(ak.to_numpy(region_events["ml_score"]).astype("f8"), 0., 1.)
                        filled["dnn_score"].fill(dnn_score=scores, weight=w)
                    filled_histograms[region_name] = filled
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

                # Save cutflow histograms in a ROOT directory named Metadata
                for region_name, hist in self.accumulator.get("region_cutflow_histograms", {}).items():
                    f[f"Metadata/{region_name}_cutflow"] = hist

                # Save metadata as 1-bin TH1 per scalar — hadd sums correctly
                metadata = self.accumulator.get("metadata", {})
                edges_1bin = np.array([0.0, 1.0])
                for mk, mv in metadata.items():
                    try:
                        f[f"Metadata/h_{mk}"] = (np.array([float(mv)]), edges_1bin)
                    except (TypeError, ValueError):
                        pass

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
        except Exception as e:
            logging.error(f"ROOT write failed: {e}. Falling back to pickle.")
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
                        total_events: Optional[int] = None,
                        input_total_events: Optional[int] = None,
                        event_selection_only: bool = False,
                        output_format: Optional[str] = None,
                        max_events: Optional[int] = None,
                        dnn_model: Optional[str] = None,
                        dnn_config: Optional[str] = None):
            self.config = config
            self.regions_config_path = regions_config_path
            self.event_selection_output = event_selection_output
            self.total_events = total_events
            self.input_total_events = input_total_events
            self.weighted_total_events = None
            self.event_selection_only = event_selection_only
            self.max_events = max_events
            self.processed_events = 0
            self.dnn_model = dnn_model
            self.dnn_config = dnn_config
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
                "region_cutflow_histograms": processor.dict_accumulator({}),
                "region_cutflow": processor.dict_accumulator({}),
                "event_selection_cutflow": processor.dict_accumulator({}),
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

            # Accumulate weighted_total_events across chunks.
            # Also persist to a temp file so postprocess() can sum across
            # workers (futures/dask workers run in separate processes and do
            # not share self.weighted_total_events with the main process).
            chunk_h = self.analyzer.base_processor.correction_manager.get_weighted_total_events(events_to_process)
            self.weighted_total_events = (self.weighted_total_events or 0.0) + chunk_h
            if self._temp_dir:
                import os, pickle, time, uuid
                _wte_id = f"{int(time.time() * 1000000)}_{uuid.uuid4().hex[:8]}"
                _wte_file = os.path.join(self._temp_dir, f"wte_chunk_{_wte_id}.pkl")
                try:
                    with open(_wte_file, "wb") as _f:
                        pickle.dump({"weighted_total_events": float(chunk_h)}, _f, protocol=pickle.HIGHEST_PROTOCOL)
                except Exception as _e:
                    logging.warning(f"Failed to persist wte chunk: {_e}")

            # --- DNN scoring (per-chunk, before preselection) ---
            if self.dnn_model:
                try:
                    from .objects import build_objects as _build_obj
                    _obj = _build_obj(events_to_process, self.config)
                    # Inline feature extraction + scoring (avoids cli.py import)
                    from .dnn_inference import DNNInference
                    from dnn.common import sanitize_feature_frame
                    import pandas as _pd, awkward as _ak

                    _inf = DNNInference(self.dnn_model, self.dnn_config)
                    _feats = _inf.features

                    # Feature names are exact compute_event_variables() output
                    # keys (configs/dnn.yaml features: is the source of truth).
                    from .variables import compute_event_variables
                    _all_vars = compute_event_variables(events_to_process, _obj, self.config)
                    _n_ch = len(events_to_process)
                    _Xd = {}
                    for _f in _feats:
                        _arr = _all_vars.get(_f)
                        _Xd[_f] = np.asarray(_arr, dtype="f8").ravel() if _arr is not None else np.full(_n_ch, -9999.0, dtype="f8")
                    _Xdf = _pd.DataFrame(_Xd)
                    _Xdf = sanitize_feature_frame(_Xdf)
                    _scores = _inf.predict(_Xdf.to_numpy(dtype="f8")).ravel().astype("float32")
                    events_to_process = _ak.with_field(events_to_process, _ak.Array(_scores), "ml_score")
                    logging.info("DNN scored %d events in Coffea chunk", _n_ch)
                except Exception as _de:
                    logging.warning("DNN scoring failed in Coffea chunk, continuing: %s", _de)

            # Call analyzer.process() with appropriate parameters
            # In event_selection_only mode, analyzer will skip region analysis
            result = self.analyzer.process(
                events_to_process,
                # Never write the final output file per-chunk: the temp_dir mechanism
                # collects chunks and postprocess() does the single final save.
                event_selection_output=None,
                event_selection_only=self.event_selection_only,
                output_format=self.output_format,
                total_events=self.total_events,
                input_total_events=self.input_total_events
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
                    selected_events, selected_objects, chunk_cutflow = apply_selection(
                        events_to_process, objects, self.config
                    )
                    logging.info(f"Chunk: {len(selected_events)}/{len(events_to_process)} events passed selection")

                    # Compute event weights so futures output matches iterative output
                    chunk_event_weights = {}
                    n_sel = len(selected_events)
                    if self.analyzer.base_processor.is_data:
                        ones = np.ones(n_sel, dtype=np.float64)
                        chunk_event_weights = {
                            "generator": ones,
                            "pileup": ones,
                            "weight_total_nominal": ones,
                        }
                    else:
                        try:
                            _wr = self.analyzer.base_processor.correction_manager.compute_event_weights(
                                selected_events, selected_objects
                            )
                            chunk_event_weights = _build_event_weights_for_save(_wr)
                        except Exception as _we:
                            logging.warning(f"Chunk weight calculation failed, using unit weights: {_we}")
                            ones = np.ones(n_sel, dtype=np.float64)
                            chunk_event_weights = {
                                "generator": ones,
                                "pileup": ones,
                                "weight_total_nominal": ones,
                            }

                    # Convert selected_events to flat numpy branch dict before pickling.
                    # NanoEvents-derived awkward arrays are tied to the uproot file handle
                    # which is closed in the main process → pickle.load fails with
                    # "zero-size array to reduction operation" on awkward 2.8.9.
                    from .variables import compute_event_variables
                    try:
                        branches = compute_event_variables(
                            selected_events, selected_objects,
                            self.config, chunk_event_weights
                        )
                    except Exception as _be:
                        logging.warning(f"compute_event_variables failed: {_be}; storing raw arrays")
                        branches = {}

                    import time
                    import uuid
                    chunk_id = f"{int(time.time() * 1000000)}_{uuid.uuid4().hex[:8]}"
                    chunk_file = os.path.join(self._temp_dir, f"chunk_{chunk_id}.pkl")
                    with open(chunk_file, 'wb') as f:
                        pickle.dump(
                            {
                                "branches": branches,   # flat numpy dict — safe to pickle
                                "n_events": n_sel,
                                "cutflow": chunk_cutflow,
                                "event_weights": chunk_event_weights,
                            },
                            f,
                            protocol=pickle.HIGHEST_PROTOCOL,
                        )

                    logging.info(f"Saved chunk to {chunk_file}: {n_sel} events, {len(branches)} branches")
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
                                               "region_cutflow_histograms", "region_cutflow",
                                               "event_selection_cutflow", "region_validation",
                                               "metadata") if k in result}
            payload["weighted_total_events"] = float(chunk_h)
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
                "region_cutflow_histograms": {},
                "region_cutflow": {"total_events": 0, "regions": {}},
                "event_selection_cutflow": {},
                "region_validation": {"status": "completed", "regions": {},
                                      "overlaps": {}, "warnings": []},
                "metadata": {"n_events_processed": 0, "n_events_input": 0,
                             "n_regions": 0, "processing_time": 0.0,
                             "weighted_total_events": 0.0, "luminosity": 0.0},
                "weighted_total_events": 0.0,
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

                # region_cutflow_histograms: add hist objects with +
                for region, h in chunk.get("region_cutflow_histograms", {}).items():
                    if region not in merged["region_cutflow_histograms"]:
                        merged["region_cutflow_histograms"][region] = h
                    else:
                        try:
                            merged["region_cutflow_histograms"][region] = merged["region_cutflow_histograms"][region] + h
                        except Exception as e:
                            logging.warning(f"Failed to merge cutflow histogram {region}: {e}")

                # event_selection_cutflow: sum per-step counts across chunks
                for step, count in chunk.get("event_selection_cutflow", {}).items():
                    merged["event_selection_cutflow"][step] = (
                        int(merged["event_selection_cutflow"].get(step, 0)) + int(count)
                    )

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

                # metadata: sum n_events_processed and processing_time; take lumi from first chunk
                meta = chunk.get("metadata", {})
                merged["metadata"]["n_events_processed"] += int(meta.get("n_events_processed", 0))
                if "n_events_input" in meta:
                    merged["metadata"]["n_events_input"] = int(meta.get("n_events_input", 0))
                merged["metadata"]["processing_time"] += float(meta.get("processing_time", 0.0))
                if "n_regions" in meta:
                    merged["metadata"]["n_regions"] = meta["n_regions"]
                if merged["metadata"]["luminosity"] == 0.0 and meta.get("luminosity", 0.0) > 0:
                    merged["metadata"]["luminosity"] = float(meta["luminosity"])

                # weighted_total_events: sum across all chunks
                merged["weighted_total_events"] += float(chunk.get("weighted_total_events", 0.0))

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

            # ── Sum weighted_total_events from per-worker temp files ──────────
            # Futures/dask workers run in separate processes; self.weighted_total_events
            # on the main-process instance is None. Summing the wte_chunk files is the
            # only reliable source of truth across all executors.
            if self._temp_dir and os.path.exists(self._temp_dir):
                wte_files = [
                    os.path.join(self._temp_dir, f)
                    for f in os.listdir(self._temp_dir)
                    if f.startswith("wte_chunk_") and f.endswith(".pkl")
                ]
                if wte_files:
                    _wte_total = 0.0
                    for _wf in wte_files:
                        try:
                            with open(_wf, "rb") as _f:
                                _wte_total += float(pickle.load(_f).get("weighted_total_events", 0.0))
                            os.remove(_wf)
                        except Exception as _e:
                            logging.warning(f"Failed to load wte chunk {_wf}: {_e}")
                    self.weighted_total_events = _wte_total
                    logging.info(f"Summed weighted_total_events from {len(wte_files)} worker files: {_wte_total}")

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
                    accumulator["region_cutflow_histograms"] = merged_analysis["region_cutflow_histograms"]
                    accumulator["region_cutflow"] = merged_analysis["region_cutflow"]
                    accumulator["event_selection_cutflow"] = merged_analysis["event_selection_cutflow"]
                    accumulator["region_validation"] = merged_analysis["region_validation"]
                    # weighted_total_events is already set from wte_chunk files above;
                    # only fall back to analysis-chunk value if wte files were absent.
                    if self.weighted_total_events is None and merged_analysis.get("weighted_total_events"):
                        self.weighted_total_events = merged_analysis["weighted_total_events"]
                    # Inject wte into metadata so region plots can normalise histograms
                    merged_analysis["metadata"]["weighted_total_events"] = float(
                        self.weighted_total_events or merged_analysis.get("weighted_total_events", 0.0)
                    )
                    # luminosity already in metadata from chunk merge (taken from first chunk)
                    accumulator["metadata"] = merged_analysis["metadata"]
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
                    # Load all chunks from files.
                    # Chunks now contain flat numpy branch dicts (key "branches") — NOT
                    # NanoEvents awkward arrays — so they deserialise safely across processes.
                    merged_branches: Dict[str, list] = {}   # branch_name → list of np.ndarray
                    merged_cutflow: Dict[str, int] = {}
                    total_selected = 0

                    for chunk_file in sorted(chunk_files):
                        if not os.path.exists(chunk_file):
                            continue
                        try:
                            with open(chunk_file, 'rb') as f:
                                chunk_data = pickle.load(f)
                        except Exception as e:
                            logging.warning(f"Failed to unpickle chunk file {chunk_file}: {e}")
                            continue
                        try:
                            branches = chunk_data.get("branches", {})
                            n_ev = int(chunk_data.get("n_events", 0))
                            cutflow = chunk_data.get("cutflow")

                            if n_ev > 0 and branches:
                                total_selected += n_ev
                                for bname, barr in branches.items():
                                    if bname not in merged_branches:
                                        merged_branches[bname] = []
                                    merged_branches[bname].append(barr)

                            if isinstance(cutflow, dict):
                                for key, value in cutflow.items():
                                    try:
                                        merged_cutflow[key] = int(merged_cutflow.get(key, 0)) + int(value)
                                    except Exception:
                                        pass
                        except Exception as e:
                            logging.warning(f"Failed to process chunk file {chunk_file}: {e}")

                    # Concatenate per-branch arrays across all chunks.
                    # Branches are np.ndarray (flat) or list-of-lists (jagged).
                    flat_branches: Dict[str, Any] = {}
                    for bname, arrs in merged_branches.items():
                        if not arrs:
                            continue
                        if isinstance(arrs[0], np.ndarray):
                            flat_branches[bname] = np.concatenate(arrs)
                        else:
                            # jagged: list-of-lists per chunk → flatten one level
                            merged_list = []
                            for a in arrs:
                                merged_list.extend(a)
                            flat_branches[bname] = merged_list

                    if not flat_branches:
                        logging.warning(
                            f"No selected events found in {len(chunk_files)} chunk files — "
                            f"writing metadata-only output to {self.event_selection_output}"
                        )

                    # Write directly to ROOT using flat numpy branch arrays.
                    # Bypasses _save_event_selection entirely — no awkward arrays needed.
                    logging.info(
                        f"Saving {total_selected} selected events "
                        f"({len(flat_branches)} branches) to {self.event_selection_output}"
                    )
                    self._write_event_selection_root(
                        self.event_selection_output,
                        flat_branches,
                        total_events=self.total_events,
                        weighted_total_events=self.weighted_total_events,
                        n_selected=total_selected,
                        cutflow=merged_cutflow if merged_cutflow else None,
                    )

                    if os.path.exists(self.event_selection_output):
                        file_size = os.path.getsize(self.event_selection_output)
                        logging.info(
                            f"✓ Saved accumulated event-level selection from "
                            f"{len(chunk_files)} chunks ({total_selected} events) "
                            f"to {self.event_selection_output} ({file_size} bytes)"
                        )
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

            # Ensure weighted_total_events is in metadata before returning
            if "metadata" not in accumulator:
                accumulator["metadata"] = {}
            if self.weighted_total_events is not None:
                accumulator["metadata"]["weighted_total_events"] = self.weighted_total_events

            return accumulator

        def _write_event_selection_root(
            self,
            output_path: str,
            branches: Dict[str, np.ndarray],
            total_events: Optional[int],
            weighted_total_events: Optional[float],
            n_selected: int,
            cutflow: Optional[Dict[str, int]],
        ) -> None:
            """Write flat numpy branch arrays directly to a ROOT file.

            Replaces the old path that reconstructed awkward arrays from
            NanoEvents-derived objects (which can't be deserialized across
            loky worker processes on awkward 2.8.9).
            """
            import uproot
            import os

            outdir = os.path.dirname(output_path)
            if outdir:
                os.makedirs(outdir, exist_ok=True)

            edges_1bin = np.array([0.0, 1.0])

            with uproot.recreate(output_path) as f:
                # Metadata scalars as 1-bin TH1Ds (hadd-summable)
                f["total_events"] = (
                    np.array([float(total_events if total_events is not None else 0)]),
                    edges_1bin,
                )
                f["weighted_total_events"] = (
                    np.array([float(weighted_total_events if weighted_total_events is not None else 0.0)]),
                    edges_1bin,
                )
                # selected_events = sum of full_event_weight (matches iterative path)
                _ew = branches.get("full_event_weight")
                if _ew is not None and len(_ew) > 0:
                    try:
                        _sel_val = float(np.sum(np.asarray(_ew)))
                    except Exception:
                        _sel_val = float(n_selected)
                else:
                    _sel_val = float(n_selected)
                f["selected_events"] = (
                    np.array([_sel_val]),
                    edges_1bin,
                )

                # Cutflow: single labeled TH1 (matches iterative path in processor.py)
                if cutflow:
                    try:
                        import boost_histogram as bh
                        cut_names  = list(cutflow.keys())
                        cut_counts = list(cutflow.values())
                        h_cf = bh.Histogram(bh.axis.StrCategory(cut_names))
                        for i, name in enumerate(cut_names):
                            h_cf[bh.loc(name)] = cut_counts[i]
                        f["cutflow"] = h_cf
                    except Exception:
                        cut_counts = np.array(list(cutflow.values()), dtype=float)
                        cf_edges   = np.arange(len(cut_counts) + 1, dtype=float)
                        f["cutflow"] = (cut_counts, cf_edges)

                # Events TTree — flat numpy branches only (jagged ak.Arrays corrupt uproot output)
                if branches:
                    try:
                        flat_branches = {}
                        for k, v in branches.items():
                            if isinstance(v, np.ndarray):
                                if v.dtype == object or v.ndim != 1:
                                    continue  # jagged or multi-dim numpy
                                flat_branches[k] = v
                            elif isinstance(v, ak.Array):
                                try:
                                    arr = ak.to_numpy(v)
                                    if arr.dtype == object or arr.ndim != 1:
                                        continue  # jagged
                                    flat_branches[k] = arr.astype(np.float32)
                                except Exception:
                                    continue
                        f["Events"] = flat_branches
                        logging.info(f"Wrote Events tree with {n_selected} entries, {len(flat_branches)} branches")
                    except Exception as e:
                        logging.error(f"Failed to write Events tree: {e}", exc_info=True)

            return None
