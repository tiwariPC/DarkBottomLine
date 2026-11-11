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
    from coffea.lumi_tools import LumiMask
    COFFEA_AVAILABLE = True
except ImportError:
    COFFEA_AVAILABLE = False
    LumiMask = None
    logging.warning("Coffea not available. Using fallback implementation.")

from .objects import build_objects
from .selections import apply_selection
from .corrections import CorrectionManager
from .histograms import HistogramManager
from .variables import compute_event_variables
from .mc_year_split import assign_year_bucket, year_lumi_fraction


def _compute_weight_statistics(weights: np.ndarray) -> Dict[str, float]:
    """Compute basic statistics for an array of event weights."""
    if len(weights) == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "sum": 0.0}
    return {
        "mean": float(np.mean(weights)),
        "std": float(np.std(weights)),
        "min": float(np.min(weights)),
        "max": float(np.max(weights)),
        "sum": float(np.sum(weights)),
    }


def _build_event_weights_for_save(weight_results: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Build a flat dict of per-event weight arrays for saving from compute_event_weights output.
    All ak.Array values are converted to numpy.
    """
    out = {}
    for name, val in weight_results.items():
        if isinstance(val, ak.Array):
            out[name] = np.asarray(ak.to_numpy(val))
        elif isinstance(val, np.ndarray):
            out[name] = val
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


def _flatten_metadata(metadata: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Convert the metadata dict to a flat dict of single-element numpy arrays
    suitable for writing as a ROOT tree or parquet file-level metadata.
    Nested dicts (e.g. weight_statistics) are flattened with underscore-joined keys.
    """
    out = {}
    for key, val in metadata.items():
        if isinstance(val, dict):
            for subkey, subval in val.items():
                out[f"{key}_{subkey}"] = np.array([subval], dtype=np.float64)
        elif isinstance(val, (int, np.integer)):
            out[key] = np.array([val], dtype=np.int64)
        elif isinstance(val, (float, np.floating)):
            out[key] = np.array([val], dtype=np.float64)
    return out


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

        # Golden JSON lumi mask (data only)
        data_cfg = config.get("data", {})
        self.is_data = bool(data_cfg.get("is_data", False))
        self._lumi_mask = None
        if self.is_data and LumiMask is not None:
            golden_json = data_cfg.get("golden_json")
            if golden_json:
                from pathlib import Path
                gjp = Path(golden_json)
                if not gjp.exists():
                    # Try relative to project root
                    gjp = Path(__file__).resolve().parent.parent / golden_json
                if gjp.exists():
                    self._lumi_mask = LumiMask(str(gjp))
                    logging.info(f"Loaded golden JSON lumi mask from {gjp}")
                else:
                    logging.warning(f"Golden JSON not found: {golden_json}")
        elif self.is_data and LumiMask is None:
            logging.warning("is_data=True but coffea.lumi_tools not available; golden JSON filter will be skipped")

        # Initialize accumulators
        self.accumulator = {
            "histograms": self.histograms,
            "cutflow": {},
            "metadata": {},
            "event_weights": {},
        }

        # one-mc-for-all: deterministic disjoint year-split of shared MC (see
        # mc_year_split.py). Off by default — config must explicitly set
        # data.one_mc_for_all=True (wired from --one-mc-for-all in cli.py).
        # When enabled, this year's job keeps only its ~lumi-fraction slice
        # of events and rescales weights by 1/f_y, so that per-year Combine
        # channels become statistically disjoint (autoMCStats becomes valid
        # again) instead of all three channels reusing ~100% of the same MC.
        # MC-only: real collision data is never duplicated/reweighted across
        # years the way this shared MC is, so there is nothing to split —
        # force the flag off for data rather than silently no-op on a
        # half-initialized state.
        if data_cfg.get("one_mc_for_all", False) and self.is_data:
            logging.warning(
                "data.one_mc_for_all=True but is_data=True — one_mc_for_all only "
                "applies to MC (the same sample reprocessed per year); ignoring it "
                "for this data file."
            )
        self._one_mc_for_all = bool(data_cfg.get("one_mc_for_all", False)) and not self.is_data
        self._mc_year_split_scale = 1.0
        if self._one_mc_for_all:
            split_cfg = config.get("one_mc_for_all", {})
            years = split_cfg.get("years")
            lumi_by_year = split_cfg.get("lumi_by_year")
            if not years or not lumi_by_year:
                raise ValueError(
                    "data.one_mc_for_all=True requires a top-level 'one_mc_for_all: "
                    "{years: [...], lumi_by_year: {...}}' block in the config."
                )
            this_year = str(config.get("year"))
            if this_year not in years:
                raise ValueError(
                    f"one_mc_for_all.years={years} does not contain this config's "
                    f"year={this_year!r}."
                )
            self._one_mc_for_all_years = [str(y) for y in years]
            self._one_mc_for_all_lumis = [float(lumi_by_year[str(y)]) for y in self._one_mc_for_all_years]
            self._one_mc_for_all_salt = int(split_cfg.get("salt", 12345))
            self._one_mc_for_all_this_year = this_year
            self._mc_year_split_scale = 1.0 / year_lumi_fraction(
                this_year, {str(y): float(lumi_by_year[str(y)]) for y in years}
            )
            logging.info(
                f"one_mc_for_all enabled for year={this_year}: lumi_fraction="
                f"{1.0/self._mc_year_split_scale:.4f}, weight_rescale={self._mc_year_split_scale:.4f}"
            )

        logging.debug(f"Initialized DarkBottomLine processor for year {config.get('year', 'unknown')} (is_data={self.is_data})")

    def apply_lumi_mask(self, events: ak.Array) -> ak.Array:
        """
        Apply golden JSON lumi mask to data events.
        No-op for MC (is_data=False) or when mask is not loaded.
        Returns filtered events and the number removed.
        """
        if self._lumi_mask is None:
            return events
        n_before = len(events)

        # Detect runs entirely absent from the golden JSON (distinct from runs
        # present but with some lumis outside certified ranges). A file whose
        # runs are not JSON keys at all yields zero passing events regardless
        # of real selection efficiency, which looks identical to a normal cut
        # in the "N -> 0" log line unless flagged explicitly.
        known_runs = set(self._lumi_mask._masks.keys())
        input_runs = set(np.unique(ak.to_numpy(events.run)).tolist())
        missing_runs = sorted(r for r in input_runs if r not in known_runs)
        if missing_runs:
            n_missing_events = int(ak.sum(np.isin(ak.to_numpy(events.run), missing_runs)))
            logging.warning(
                f"Golden JSON does not contain {len(missing_runs)} run(s) present in input "
                f"({n_missing_events}/{n_before} events): {missing_runs[:10]}"
                f"{'...' if len(missing_runs) > 10 else ''}. "
                f"These events will be dropped entirely (not a lumi-range cut, but missing "
                f"run coverage) — check that the golden JSON covers the full run range of "
                f"the dataset being processed."
            )

        good_lumi = self._lumi_mask(events.run, events.luminosityBlock)
        events = events[good_lumi]
        logging.info(f"Golden JSON: {n_before} -> {len(events)} events ({n_before - len(events)} removed)")
        if n_before > 0 and len(events) == 0 and missing_runs:
            logging.warning(
                "Golden JSON filter removed ALL events, entirely due to missing run coverage "
                "in the golden JSON (see warning above), not due to certified-lumi selection."
            )
        return events

    def apply_mc_year_split(self, events: ak.Array) -> ak.Array:
        """
        Filter MC events down to this year's deterministic disjoint lumi-fraction
        bucket (see mc_year_split.assign_year_bucket). No-op if one_mc_for_all is
        not enabled. Same (run, luminosityBlock, event) always hashes to the same
        bucket, so the kept subsets across years are provably disjoint.
        """
        if not self._one_mc_for_all:
            return events
        n_before = len(events)
        run = ak.to_numpy(events.run)
        lumiblock = ak.to_numpy(events.luminosityBlock)
        event_no = ak.to_numpy(events.event)
        buckets = assign_year_bucket(
            run, lumiblock, event_no,
            self._one_mc_for_all_years, self._one_mc_for_all_lumis,
            salt=self._one_mc_for_all_salt,
        )
        keep = buckets == self._one_mc_for_all_this_year
        events = events[keep]
        logging.info(
            f"one_mc_for_all: {n_before} -> {len(events)} events "
            f"({len(events)/n_before*100:.1f}% kept for year={self._one_mc_for_all_this_year})"
            if n_before > 0 else "one_mc_for_all: 0 events before split, nothing to filter"
        )
        return events

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

        # Cross-check --data flag against actual file content: real NanoAOD
        # data never has genWeight, MC always does. is_data is user-asserted
        # (from --data CLI flag) and never otherwise verified, so a wrong
        # flag silently skips the golden JSON mask (or applies it to MC) with
        # no other symptom.
        has_gen_weight = "genWeight" in events.fields
        if self.is_data and has_gen_weight:
            logging.warning(
                "is_data=True but input events have a 'genWeight' branch, which real "
                "collision data does not have. Check that --data was not passed for an "
                "MC file — golden JSON mask and unit-weight logic are being applied."
            )
        elif not self.is_data and not has_gen_weight:
            logging.warning(
                "is_data=False but input events have no 'genWeight' branch, which is "
                "expected for collision data. Check that --data is not missing for a "
                "real data file — golden JSON mask will NOT be applied."
            )

        # Apply golden JSON lumi mask (data only; no-op for MC)
        if self.is_data:
            events = self.apply_lumi_mask(events)
            print(f"  Events after golden JSON filter: {len(events)}")

            if len(events) == 0:
                logging.warning(
                    "No events survive the golden JSON lumi mask; stopping here for this "
                    "data file (nothing to select, weight, or histogram)."
                )
                if event_selection_output:
                    self._save_event_selection(
                        event_selection_output, events, {},
                        max_events=self.config.get("max_events"),
                        total_events=len(events),
                        weighted_total_events=0.0,
                        cutflow={},
                    )
                self.accumulator["histograms"] = self.histograms
                self.accumulator["cutflow"] = {}
                self.accumulator["event_weights"] = {}
                self.accumulator["metadata"] = {
                    "n_events_processed": 0,
                    "n_events_selected": 0,
                    "weighted_total_events": 0.0,
                    "processing_time": time.time() - start_time,
                    "weight_statistics": {},
                }
                return self.accumulator

        # one-mc-for-all: disjoint year-split (no-op unless enabled; see __init__)
        events = self.apply_mc_year_split(events)
        if self._one_mc_for_all:
            print(f"  Events after one-mc-for-all year split: {len(events)}")

        # weighted_total_events: sum of sign(genWeight) over ALL events before any selection
        weighted_total_events = self.correction_manager.get_weighted_total_events(events)
        print(f"  weighted_total_events (normalization): {weighted_total_events}")
        # Multi-point signal samples: the same sum per GenModel_<point> flag (empty otherwise)
        weighted_total_events_per_genmodel = self.correction_manager.get_weighted_total_events_per_genmodel(events)
        if weighted_total_events_per_genmodel:
            print(f"  weighted_total_events per GenModel: {len(weighted_total_events_per_genmodel)} mass points")

        # Build physics objects
        print("Step 1: Building physics objects...")
        logging.info("Building physics objects...")
        objects = build_objects(events, self.config, corrections=self.correction_manager)

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
                self._save_event_selection(event_selection_output, selected_events, selected_objects,
                                            max_events=self.config.get("max_events"),
                                            total_events=len(events),
                                            weighted_total_events=weighted_total_events,
                                            weighted_total_events_per_genmodel=weighted_total_events_per_genmodel,
                                            cutflow=cutflow)
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
            weight_results = self.correction_manager.compute_event_weights(
                selected_events, selected_objects
            )
            if self._one_mc_for_all and self._mc_year_split_scale != 1.0:
                # Compensate for keeping only a lumi-fraction f_y of events:
                # multiply every weight component by 1/f_y so each year's
                # total predicted yield matches the unsplit (flag-off) result.
                weight_results = {
                    k: (v * self._mc_year_split_scale if isinstance(v, ak.Array) else v)
                    for k, v in weight_results.items()
                }
            event_weights = np.asarray(ak.to_numpy(weight_results["full_event_weight"]))
            print(f"  Weights calculated successfully: {len(event_weights)} weights")

            event_weights_save = _build_event_weights_for_save(weight_results)
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
            "weighted_total_events": weighted_total_events,
            "weighted_total_events_per_genmodel": weighted_total_events_per_genmodel,
            "processing_time": processing_time,
            "weight_statistics": _compute_weight_statistics(event_weights),
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
                              max_events: Optional[int] = None, total_events: Optional[int] = None,
                              weighted_total_events: Optional[float] = None,
                              weighted_total_events_per_genmodel: Optional[Dict[str, float]] = None,
                              event_weights: Optional[Dict[str, Any]] = None,
                              cutflow: Optional[Dict[str, int]] = None,
                              output_format: str = "pkl"):
        """
        Save selected events and corresponding objects to a file.

        Currently supports pickle, ROOT formats.

        Args:
            output_file: Path to save the output file
            events: Selected events (after event-level selection)
            objects: Selected objects (after event-level selection)
            max_events: Maximum events parameter from config (optional)
            total_events: Total number of events BEFORE selection (optional)
            event_weights: Event weights dictionary (optional, includes all corrections)
            cutflow: Event-selection cutflow dictionary (optional)
            output_format: Output format ("pkl", "root"). Default: "pkl" (saves both pkl and root)
        """
        import os
        import pickle
        import numpy as np

        # Ensure output directory exists before any early return so that
        # downstream processes can rely on the directory being present even
        # when no events pass the selection (e.g. data after golden JSON filter).
        outdir = os.path.dirname(output_file)
        if outdir:
            os.makedirs(outdir, exist_ok=True)

        no_events = events is None or len(events) == 0

        # Compute all output variables once — used by every format
        branches = {}
        if not no_events:
            branches = compute_event_variables(events, objects, self.config, event_weights)

        # Auto-detect strict output intent from file extension.
        # If user explicitly provides a .root target, enforce root-only output.
        output_file_lower = output_file.lower()
        if output_file_lower.endswith('.root'):
            output_format = 'root'
        elif output_file_lower.endswith('.parquet') and output_format not in ('root',):
            output_format = 'parquet'
        elif output_file_lower.endswith('.pkl') and output_format not in ('root',):
            output_format = 'pkl'

        # Determine actual output files based on format
        base = os.path.splitext(output_file)[0]
        save_root    = output_format in ("root", "pkl", "both")
        save_pkl     = output_format in ("pkl",  "both")
        save_parquet = output_format == "parquet"

        output_file_root    = base + '.root'
        output_file_pkl     = base + '.pkl'
        output_file_parquet = base + '.parquet'

        # Save pickle file if specified
        if save_pkl:
            # Build serializable dict — always include metadata even when no events passed selection.
            serializable = {}
            if total_events is not None:
                serializable["total_events"] = total_events
            if max_events is not None:
                serializable["total_event"] = max_events
            serializable["selected_events"] = 0 if no_events else len(events)
            if weighted_total_events is not None:
                serializable["weighted_total_events"] = weighted_total_events
            if weighted_total_events_per_genmodel:
                serializable["weighted_total_events_per_genmodel"] = dict(weighted_total_events_per_genmodel)

            if not no_events:
                try:
                    try:
                        serializable["events"] = ak.to_list(events)
                    except Exception:
                        try:
                            serializable["events"] = list(events)
                        except Exception:
                            serializable["events"] = []

                    serializable_objects = {}
                    for k, v in objects.items():
                        if isinstance(v, ak.Array):
                            try:
                                serializable_objects[k] = ak.to_list(v)
                            except Exception:
                                try:
                                    serializable_objects[k] = list(v)
                                except Exception:
                                    serializable_objects[k] = []
                        else:
                            serializable_objects[k] = v
                    serializable["objects"] = serializable_objects

                    if event_weights:
                        serializable_weights = {}
                        for name, val in event_weights.items():
                            if isinstance(val, dict):
                                serializable_weights[name] = {}
                                for k, v in val.items():
                                    serializable_weights[name][k] = v.tolist() if isinstance(v, np.ndarray) else v
                            elif isinstance(val, np.ndarray):
                                serializable_weights[name] = val.tolist()
                            else:
                                serializable_weights[name] = val
                        serializable["event_weights"] = serializable_weights
                except Exception as e:
                    logging.warning(f"Failed to serialise events/objects for pkl: {e}")
            else:
                serializable["events"] = []
                serializable["objects"] = {}
                logging.info("No selected events — writing metadata-only pickle")

            try:
                with open(output_file_pkl, "wb") as f:
                    pickle.dump(serializable, f, protocol=pickle.HIGHEST_PROTOCOL)
                logging.info(f"Event selection (serializable) saved to {output_file_pkl}")
            except Exception as e:
                logging.warning(f"Failed to save serializable event selection to {output_file_pkl}: {e}")

            # Best-effort raw awkward backup (only meaningful when events exist)
            if not no_events:
                try:
                    raw_backup = output_file_pkl + ".awk_raw.pkl"
                    with open(raw_backup, "wb") as f:
                        pickle.dump({"events": events, "objects": objects, "event_weights": event_weights}, f, protocol=pickle.HIGHEST_PROTOCOL)
                    logging.info(f"Raw awkward backup saved to {raw_backup}")
                except Exception as e:
                    logging.warning(f"Failed to save raw awkward backup to {raw_backup}: {e}")

        # Save ROOT file if specified (always write even with zero events — Metadata must be present)

        if save_root:
            try:
                import uproot

                logging.info(f"Creating ROOT file: {output_file_root}")

                from .variables import get_empty_branch_types, prepare_event_branches, write_events_tree

                # Compute all output branches via variables module
                branches = {}
                if not no_events:
                    branches = compute_event_variables(
                        events, objects, self.config, event_weights
                    )
                elif events is not None:
                    # Zero selected events: compute the branches on the zero-length
                    # array anyway, so the empty Events has exactly the fields and
                    # dtypes a file with events gets. Objects are missing when the
                    # golden JSON mask removed everything; build them on the empty
                    # events too.
                    try:
                        if not objects:
                            objects = build_objects(events, self.config, corrections=self.correction_manager)
                        branches = compute_event_variables(events, objects, self.config, event_weights)
                    except Exception as e:
                        logging.warning(f"Could not derive the Events schema from zero events ({e}); "
                                        "using the static schema from get_empty_branch_types()")
                        branches = {}
                    # Forward extra event fields (e.g. ml_score from DNN) not in standard variable set
                    for _ef in ("ml_score",):
                        if _ef in events.fields:
                            try:
                                branches[_ef] = np.asarray(ak.to_numpy(events[_ef]), dtype="f8")
                            except Exception:
                                pass

                # Write to ROOT
                outdir = os.path.dirname(output_file_root)
                if outdir:
                    os.makedirs(outdir, exist_ok=True)

                _ew = event_weights.get("full_event_weight") if event_weights else None
                selected_events = float(np.sum(_ew)) if _ew is not None and len(_ew) > 0 else float(0 if no_events else len(events))
                with uproot.recreate(output_file_root) as f:
                    if branches:
                        # Same conversion as the futures/dask writer (analyzer.py).
                        write_branches = prepare_event_branches(branches)
                        n_written = write_events_tree(f, write_branches)
                        if no_events:
                            logging.info(f"No selected events — wrote empty Events TTree with "
                                         f"{len(write_branches)} branches for hadd compatibility")
                        else:
                            logging.info(f"Wrote Events TTree with {n_written} entries, {len(write_branches)} branches")
                    else:
                        # Last resort (no events object to derive the schema from):
                        # static schema, which may not match files with events exactly.
                        logging.warning("No selected events and no derivable schema — writing empty "
                                        "Events TTree from get_empty_branch_types()")
                        f.mktree("Events", get_empty_branch_types())

                    # Flat 1-bin TH1 per scalar — hadd sums bin contents correctly
                    edges_1bin = np.array([0.0, 1.0])
                    f['total_events']             = (np.array([float(total_events) if total_events is not None else 0.0]), edges_1bin)
                    f['selected_events'] = (np.array([selected_events]),                                     edges_1bin)
                    f['weighted_total_events']    = (np.array([float(weighted_total_events) if weighted_total_events is not None else 0.0]), edges_1bin)
                    # Per-mass-point generated weighted events (multi-point signal): hadd-summable TH1s
                    for _gm, _gv in (weighted_total_events_per_genmodel or {}).items():
                        f[f'weighted_total_events_{_gm}'] = (np.array([float(_gv)]), edges_1bin)
                    logging.info(f"Saved Metadata histograms: total_events={total_events}, selected_events={selected_events:.2f}, weighted_total_events={weighted_total_events}")

                    # Labeled TH1 cutflow — bin labels = cut names, hadd sums correctly
                    if cutflow:
                        try:
                            import boost_histogram as bh
                            cut_names  = list(cutflow.keys())
                            cut_counts = list(cutflow.values())
                            h_cf = bh.Histogram(bh.axis.StrCategory(cut_names))
                            for i, name in enumerate(cut_names):
                                h_cf[bh.loc(name)] = cut_counts[i]
                            f['cutflow'] = h_cf
                        except Exception:
                            # fallback: unlabeled TH1
                            cut_counts = np.array(list(cutflow.values()), dtype=float)
                            cf_edges   = np.arange(len(cut_counts) + 1, dtype=float)
                            f['cutflow'] = (cut_counts, cf_edges)
                        logging.info(f"Saved Cutflow histogram with {len(cutflow)} bins")

                # Verify file was created
                if os.path.exists(output_file_root):
                    file_size = os.path.getsize(output_file_root)
                    logging.info(f"✓ Event selection exported to ROOT file {output_file_root} ({file_size} bytes, {selected_events:.0f} events)")
                else:
                    logging.error(f"✗ ROOT file {output_file_root} was not created!")
            except Exception as e:
                logging.error(f"Failed to write ROOT event selection to {output_file_root}: {e}", exc_info=True)

        # Save parquet file if specified
        if save_parquet:
            try:
                import pandas as pd
                scalar_data = {}
                for k, v in branches.items():
                    if isinstance(v, ak.Array) and v.ndim > 1:
                        continue  # skip jagged in parquet
                    a = np.asarray(v) if not isinstance(v, np.ndarray) else v
                    if a.ndim == 1:
                        scalar_data[k] = a
                df = pd.DataFrame(scalar_data)
                df.attrs.update({
                    'total_events':             float(total_events or 0),
                    'selected_events': selected_events,
                    'weighted_total_events':    float(weighted_total_events or 0),
                })
                df.to_parquet(output_file_parquet, index=False)
                logging.info(f"✓ Event selection exported to parquet {output_file_parquet} ({len(df)} rows)")
            except Exception as e:
                logging.error(f"Failed to write parquet event selection to {output_file_parquet}: {e}", exc_info=True)

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
        """Save results as Parquet file (histograms + per-event weights + metadata)."""
        import pandas as pd
        import json

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

        # Metadata as parquet file-level key-value pairs (scalars only)
        metadata = self.accumulator.get("metadata", {})
        if metadata:
            flat_meta = _flatten_metadata(metadata)
            existing = df.attrs if hasattr(df, "attrs") else {}
            existing.update({k: float(v[0]) for k, v in flat_meta.items()})
            df.attrs = existing

        df.to_parquet(output_file)
        logging.info(f"Saved results to {output_file}")

    def _save_root(self, output_file: str):
        """Save results as ROOT file (histograms + per-event weight tree)."""
        try:
            import uproot
            outdir = os.path.dirname(output_file)
            if outdir:
                os.makedirs(outdir, exist_ok=True)
            with uproot.recreate(output_file) as f:
                # Save histograms
                for name, hist in self.accumulator["histograms"].items():
                    if hasattr(hist, 'values'):
                        f[name] = hist

                # Save metadata as a flat single-entry TTree
                metadata = self.accumulator.get("metadata", {})
                if metadata:
                    f["metadata"] = _flatten_metadata(metadata)

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
