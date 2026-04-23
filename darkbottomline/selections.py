"""
Event selection functions for DarkBottomLine framework.
"""

import awkward as ak
import numpy as np
from typing import Dict, Any, List, Tuple


def _build_event_cut_masks(
    events: ak.Array,
    objects: Dict[str, Any],
    config: Dict[str, Any],
    logger: Any = None,
) -> Tuple[Dict[str, ak.Array], Dict[str, float]]:
    """Build per-cut masks for event-level selection and optional diagnostics."""
    selection = config["event_selection"]

    # Count objects per event
    n_muons = ak.num(objects["muons"], axis=1)
    n_electrons = ak.num(objects["electrons"], axis=1)
    n_taus = ak.num(objects["taus"], axis=1)
    n_jets = ak.num(objects["jets"], axis=1)
    n_bjets = ak.num(objects["bjets"], axis=1)

    # Multiplicity masks
    muon_cut = (n_muons >= selection["min_muons"]) & (n_muons <= selection["max_muons"])
    electron_cut = (n_electrons >= selection["min_electrons"]) & (n_electrons <= selection["max_electrons"])
    tau_cut = (n_taus >= selection["min_taus"]) & (n_taus <= selection["max_taus"])
    jet_cut = (n_jets >= selection["min_jets"]) & (n_jets <= selection["max_jets"])
    bjet_cut = n_bjets >= selection["min_bjets"]

    # Recoil mask
    recoil = calculate_recoil(events, objects)
    recoil_min = selection.get("recoil_min", 250.0)
    recoil_cut = recoil > recoil_min

    # Leading jet pt mask
    jet1_pt_min = selection.get("jet1_pt_min")
    if jet1_pt_min is not None:
        jets = objects.get("jets", ak.Array([]))
        if len(ak.flatten(jets)) > 0:
            leading_jet_pt = ak.fill_none(ak.max(jets.pt, axis=1), 0.0)
            jet1_pt_cut = leading_jet_pt > jet1_pt_min
        else:
            jet1_pt_cut = ak.zeros_like(events["event"], dtype=bool)
    else:
        jet1_pt_cut = ak.ones_like(events["event"], dtype=bool)

    # DeltaPhi mask
    delta_phi_min = selection.get("delta_phi_min")
    if delta_phi_min is not None:
        jets = objects.get("jets", ak.Array([]))
        met_phi = events["PFMET_phi"] if "PFMET_phi" in events.fields else events["MET_phi"]
        if len(ak.flatten(jets)) > 0:
            dphi = np.abs(jets.phi - met_phi)
            dphi = np.where(dphi > np.pi, 2 * np.pi - dphi, dphi)
            min_dphi = ak.fill_none(ak.min(dphi, axis=1), 0.0)
            delta_phi_cut = min_dphi > delta_phi_min
        else:
            delta_phi_cut = ak.zeros_like(events["event"], dtype=bool)
    else:
        delta_phi_cut = ak.ones_like(events["event"], dtype=bool)

    masks = {
        "Pass muon multiplicity": muon_cut,
        "Pass electron multiplicity": electron_cut,
        "Pass tau multiplicity": tau_cut,
        "Pass jet multiplicity": jet_cut,
        "Pass bjet multiplicity": bjet_cut,
        "Pass recoil": recoil_cut,
        "Pass leading jet pt": jet1_pt_cut,
        "Pass delta phi": delta_phi_cut,
    }

    diagnostics = {
        "recoil_min": recoil_min,
        "jet1_pt_min": jet1_pt_min,
        "delta_phi_min": delta_phi_min,
        "n_muons_min": selection["min_muons"],
        "n_muons_max": selection["max_muons"],
        "n_electrons_min": selection["min_electrons"],
        "n_electrons_max": selection["max_electrons"],
        "n_taus_min": selection["min_taus"],
        "n_taus_max": selection["max_taus"],
        "n_jets_min": selection["min_jets"],
        "n_jets_max": selection["max_jets"],
        "n_bjets_min": selection["min_bjets"],
        "recoil_min_obs": float(ak.min(recoil)),
        "recoil_max_obs": float(ak.max(recoil)),
        "recoil_mean_obs": float(ak.mean(recoil)),
        "n_muons_obs_min": int(ak.min(n_muons)),
        "n_muons_obs_max": int(ak.max(n_muons)),
        "n_electrons_obs_min": int(ak.min(n_electrons)),
        "n_electrons_obs_max": int(ak.max(n_electrons)),
        "n_taus_obs_min": int(ak.min(n_taus)),
        "n_taus_obs_max": int(ak.max(n_taus)),
        "n_jets_obs_min": int(ak.min(n_jets)),
        "n_jets_obs_max": int(ak.max(n_jets)),
        "n_bjets_obs_min": int(ak.min(n_bjets)),
        "n_bjets_obs_max": int(ak.max(n_bjets)),
    }

    if logger:
        logger.info("  Object counts per event:")
        logger.info(
            f"    n_muons: min={diagnostics['n_muons_obs_min']}, max={diagnostics['n_muons_obs_max']}, mean={ak.mean(n_muons):.2f}"
        )
        logger.info(
            f"    n_electrons: min={diagnostics['n_electrons_obs_min']}, max={diagnostics['n_electrons_obs_max']}, mean={ak.mean(n_electrons):.2f}"
        )
        logger.info(
            f"    n_taus: min={diagnostics['n_taus_obs_min']}, max={diagnostics['n_taus_obs_max']}, mean={ak.mean(n_taus):.2f}"
        )
        logger.info(
            f"    n_jets: min={diagnostics['n_jets_obs_min']}, max={diagnostics['n_jets_obs_max']}, mean={ak.mean(n_jets):.2f}"
        )
        logger.info(
            f"    n_bjets: min={diagnostics['n_bjets_obs_min']}, max={diagnostics['n_bjets_obs_max']}, mean={ak.mean(n_bjets):.2f}"
        )
        logger.info("  Multiplicity cuts (standalone):")
        logger.info(
            f"    muon_cut ({selection['min_muons']} <= n <= {selection['max_muons']}): {ak.sum(muon_cut)} pass"
        )
        logger.info(
            f"    electron_cut ({selection['min_electrons']} <= n <= {selection['max_electrons']}): {ak.sum(electron_cut)} pass"
        )
        logger.info(f"    tau_cut (n <= {selection['max_taus']}): {ak.sum(tau_cut)} pass")
        logger.info(
            f"    jet_cut ({selection['min_jets']} <= n <= {selection['max_jets']}): {ak.sum(jet_cut)} pass"
        )
        logger.info(f"    bjet_cut (n >= {selection['min_bjets']}): {ak.sum(bjet_cut)} pass [CRITICAL FOR Z->NU]")
        logger.info(f"  Recoil cut (threshold > {recoil_min} GeV):")
        logger.info(
            f"    Recoil: min={diagnostics['recoil_min_obs']:.1f}, max={diagnostics['recoil_max_obs']:.1f}, mean={diagnostics['recoil_mean_obs']:.1f}"
        )
        logger.info(f"    recoil_cut (Recoil > {recoil_min}): {ak.sum(recoil_cut)} pass")
        if jet1_pt_min is not None:
            logger.info(f"  Leading jet pT cut (> {jet1_pt_min} GeV): {ak.sum(jet1_pt_cut)} pass")
        else:
            logger.info("  Leading jet pT cut: DISABLED")
        if delta_phi_min is not None:
            logger.info(f"  DeltaPhi cut (min > {delta_phi_min}): {ak.sum(delta_phi_cut)} pass")
        else:
            logger.info("  DeltaPhi cut: DISABLED")

    return masks, diagnostics


def calculate_recoil(events: ak.Array, objects: Dict[str, Any]) -> ak.Array:
    """
    Calculate event-level recoil with the same object definition used by saved output.

    Recoil = |-(MET + sum pT(leptons))|, with leptons from tight pt>30 collections.
    """
    met_pt = events["PFMET_pt"] if "PFMET_pt" in events.fields else events["MET_pt"]
    met_phi = events["PFMET_phi"] if "PFMET_phi" in events.fields else events["MET_phi"]

    muons = objects.get("tight_muons_pt30", ak.Array([]))
    electrons = objects.get("tight_electrons_pt30", ak.Array([]))

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

    recoil_px = -(met_pt * np.cos(met_phi) + ak.fill_none(lep_px, 0.0))
    recoil_py = -(met_pt * np.sin(met_phi) + ak.fill_none(lep_py, 0.0))
    recoil = np.sqrt(recoil_px**2 + recoil_py**2)
    return ak.fill_none(recoil, 0.0)


def pass_triggers(events: ak.Array, trigger_paths: List[str]) -> ak.Array:
    """
    Check if events pass any of the specified trigger paths.

    Args:
        events: Awkward Array of events
        trigger_paths: List of trigger path names to check

    Returns:
        Boolean mask for events passing triggers
    """
    if not trigger_paths:
        return ak.ones_like(events["event"], dtype=bool)

    # Initialize with False
    trigger_mask = ak.zeros_like(events["event"], dtype=bool)

    # Check each trigger path
    for trigger in trigger_paths:
        if trigger in events.fields:
            trigger_branch = events[trigger]
            trigger_mask = trigger_mask | trigger_branch

    return trigger_mask


def pass_met_filters(events: ak.Array, filter_names: List[str]) -> ak.Array:
    """
    Check if events pass all specified MET filters.

    Args:
        events: Awkward Array of events
        filter_names: List of MET filter names to check

    Returns:
        Boolean mask for events passing all filters
    """
    if not filter_names:
        return ak.ones_like(events["event"], dtype=bool)

    # Initialize with True
    filter_mask = ak.ones_like(events["event"], dtype=bool)

    # Check each filter
    for filter_name in filter_names:
        if filter_name in events.fields:
            filter_branch = events[filter_name]
            filter_mask = filter_mask & filter_branch

    return filter_mask


def select_events(
    events: ak.Array,
    objects: Dict[str, Any],
    config: Dict[str, Any]
) -> ak.Array:
    """
    Apply event-level selection cuts.

    Args:
        events: Awkward Array of events
        objects: Dictionary containing selected objects
        config: Configuration dictionary with event selection cuts

    Returns:
        Boolean mask for events passing selection
    """
    import logging
    logger = logging.getLogger(__name__)
    
    cut_masks, _ = _build_event_cut_masks(events, objects, config, logger=logger)

    # === FINAL MASK ===
    cumulative_mask = ak.ones_like(events["event"], dtype=bool)
    for cut_mask in cut_masks.values():
        cumulative_mask = cumulative_mask & cut_mask
    logger.info(f"  FINAL cumulative mask: {ak.sum(cumulative_mask)} events pass ALL cuts")

    # Debug which cuts are failing (sequential cumulative counts)
    cumulative_counts = {}
    running_mask = ak.ones_like(events["event"], dtype=bool)
    for name, cut_mask in cut_masks.items():
        running_mask = running_mask & cut_mask
        cumulative_counts[name] = ak.sum(running_mask)
    logger.info(f"  Cumulative pass counts:")
    for name, value in cumulative_counts.items():
        logger.info(f"    After {name}: {value}")

    event_mask = cumulative_mask

    return event_mask


def get_cutflow(
    events: ak.Array,
    objects: Dict[str, Any],
    config: Dict[str, Any]
) -> Dict[str, int]:
    """
    Calculate cutflow for event selection.

    Args:
        events: Awkward Array of events
        objects: Dictionary containing selected objects
        config: Configuration dictionary

    Returns:
        Dictionary with cut names and event counts
    """
    cutflow: Dict[str, int] = {}
    cutflow["Total events"] = int(len(events))

    combined_trigger_mask = ak.zeros_like(events["event"], dtype=bool)
    for trigger_paths in config["triggers"].values():
        if trigger_paths:
            combined_trigger_mask = combined_trigger_mask | pass_triggers(events, trigger_paths)

    final_mask = ak.ones_like(events["event"], dtype=bool)
    final_mask = final_mask & combined_trigger_mask
    cutflow["Pass trigger"] = int(ak.sum(final_mask))

    filter_mask = pass_met_filters(events, config["met_filters"])
    final_mask = final_mask & filter_mask
    cutflow["Pass filters"] = int(ak.sum(final_mask))

    event_cut_masks, _ = _build_event_cut_masks(events, objects, config)
    for step_name, step_mask in event_cut_masks.items():
        final_mask = final_mask & step_mask
        cutflow[step_name] = int(ak.sum(final_mask))

    cutflow["Final selection"] = int(ak.sum(final_mask))
    return cutflow


def apply_selection(
    events: ak.Array,
    objects: Dict[str, Any],
    config: Dict[str, Any]
) -> Tuple[ak.Array, Dict[str, Any], Dict[str, int]]:
    """
    Apply complete event selection including triggers, filters, and cuts.

    Args:
        events: Awkward Array of events
        objects: Dictionary containing selected objects
        config: Configuration dictionary

    Returns:
        Tuple of (selected_events, selected_objects, cutflow)
    """
    combined_trigger_mask = ak.zeros_like(events["event"], dtype=bool)
    for trigger_type, trigger_paths in config["triggers"].items():
        if trigger_paths: # Only apply if there are paths for this type
            type_mask = pass_triggers(events, trigger_paths)
            combined_trigger_mask = combined_trigger_mask | type_mask
    trigger_mask = combined_trigger_mask

    print("  Applying MET filter selection...")
    # Apply MET filter selection
    filter_mask = pass_met_filters(events, config["met_filters"])
    print(f"    Events passing MET filters: {ak.sum(filter_mask)}")

    print("  Applying event selection...")
    # Apply event selection
    event_mask = select_events(events, objects, config)
    print(f"    Events passing event selection: {ak.sum(event_mask)}")

    # Combine all masks
    final_mask = trigger_mask & filter_mask & event_mask
    print(f"    Events passing all selections: {ak.sum(final_mask)}")

    # Apply selection to events and objects
    selected_events = events[final_mask]
    selected_objects = {}
    for key, obj in objects.items():
        if isinstance(obj, ak.Array):
            selected_objects[key] = obj[final_mask]
        else:
            selected_objects[key] = obj

    # Calculate cutflow
    cutflow = get_cutflow(events, objects, config)

    return selected_events, selected_objects, cutflow
