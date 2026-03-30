"""
Event selection functions for DarkBottomLine framework.
"""

import awkward as ak
import numpy as np
from typing import Dict, Any, List, Tuple


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
    
    selection = config["event_selection"]

    # Count objects per event
    n_muons = ak.num(objects["muons"], axis=1)
    n_electrons = ak.num(objects["electrons"], axis=1)
    n_taus = ak.num(objects["taus"], axis=1)
    n_jets = ak.num(objects["jets"], axis=1)
    n_bjets = ak.num(objects["bjets"], axis=1)

    # === DETAILED CUTFLOW DEBUG ===
    logger.info(f"  Object counts per event:")
    logger.info(f"    n_muons: min={ak.min(n_muons)}, max={ak.max(n_muons)}, mean={ak.mean(n_muons):.2f}")
    logger.info(f"    n_electrons: min={ak.min(n_electrons)}, max={ak.max(n_electrons)}, mean={ak.mean(n_electrons):.2f}")
    logger.info(f"    n_taus: min={ak.min(n_taus)}, max={ak.max(n_taus)}, mean={ak.mean(n_taus):.2f}")
    logger.info(f"    n_jets: min={ak.min(n_jets)}, max={ak.max(n_jets)}, mean={ak.mean(n_jets):.2f}")
    logger.info(f"    n_bjets: min={ak.min(n_bjets)}, max={ak.max(n_bjets)}, mean={ak.mean(n_bjets):.2f}")
    
    # Apply multiplicity cuts
    muon_cut = (n_muons >= selection["min_muons"]) & (n_muons <= selection["max_muons"])
    electron_cut = (n_electrons >= selection["min_electrons"]) & (n_electrons <= selection["max_electrons"])
    tau_cut = (n_taus >= selection["min_taus"]) & (n_taus <= selection["max_taus"])
    jet_cut = n_jets >= selection["min_jets"]
    bjet_cut = n_bjets >= selection["min_bjets"]
    
    logger.info(f"  Multiplicity cuts (cumulative):")
    logger.info(f"    muon_cut ({selection['min_muons']} <= n <= {selection['max_muons']}): {ak.sum(muon_cut)} pass")
    logger.info(f"    electron_cut ({selection['min_electrons']} <= n <= {selection['max_electrons']}): {ak.sum(electron_cut)} pass")
    logger.info(f"    tau_cut (n <= {selection['max_taus']}): {ak.sum(tau_cut)} pass")
    logger.info(f"    jet_cut (n >= {selection['min_jets']}): {ak.sum(jet_cut)} pass")
    logger.info(f"    bjet_cut (n >= {selection['min_bjets']}): {ak.sum(bjet_cut)} pass [CRITICAL FOR Z->NU]")

    # === MET/Recoil cut ===
    met_pt = events["PFMET_pt"] if "PFMET_pt" in events.fields else events["MET_pt"]
    met_phi = events["PFMET_phi"] if "PFMET_phi" in events.fields else events["MET_phi"]
    met_min = selection["met_min"]
    met_cut = met_pt > met_min
    logger.info(f"  MET/Recoil cuts (threshold > {met_min} GeV):")
    logger.info(f"    MET: min={ak.min(met_pt):.1f}, max={ak.max(met_pt):.1f}, mean={ak.mean(met_pt):.1f}")
    logger.info(f"    met_cut (MET > {met_min}): {ak.sum(met_cut)} pass")

    # Recoil = | -( pTmiss + sum pT(leptons) ) | (same convention as region Recoil)
    muons = objects.get("muons", ak.Array([]))
    electrons = objects.get("electrons", ak.Array([]))
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
    recoil_cut = recoil > met_min
    logger.info(f"    Recoil: min={ak.min(recoil):.1f}, max={ak.max(recoil):.1f}, mean={ak.mean(recoil):.1f}")
    logger.info(f"    recoil_cut (Recoil > {met_min}): {ak.sum(recoil_cut)} pass")

    met_or_recoil_cut = met_cut | recoil_cut
    logger.info(f"    met_or_recoil_cut: {ak.sum(met_or_recoil_cut)} pass")

    # === Leading jet pT cut ===
    jet1_pt_min = selection.get("jet1_pt_min")
    if jet1_pt_min is not None:
        jets = objects.get("jets", ak.Array([]))
        if len(ak.flatten(jets)) > 0:
            leading_jet_pt = ak.fill_none(ak.max(jets.pt, axis=1), 0.0)
            jet1_pt_cut = leading_jet_pt > jet1_pt_min
            logger.info(f"  Leading jet pT cut (> {jet1_pt_min} GeV): {ak.sum(jet1_pt_cut)} pass")
        else:
            jet1_pt_cut = ak.zeros_like(events["event"], dtype=bool)
            logger.info(f"  Leading jet pT cut: NO JETS (all fail)")
    else:
        jet1_pt_cut = ak.ones_like(events["event"], dtype=bool)
        logger.info(f"  Leading jet pT cut: DISABLED")

    # === DeltaPhi cut ===
    delta_phi_min = selection.get("delta_phi_min")
    if delta_phi_min is not None:
        jets = objects.get("jets", ak.Array([]))
        met_phi = events["PFMET_phi"] if "PFMET_phi" in events.fields else events["MET_phi"]
        if len(ak.flatten(jets)) > 0:
            dphi = np.abs(jets.phi - met_phi)
            dphi = np.where(dphi > np.pi, 2 * np.pi - dphi, dphi)
            min_dphi = ak.fill_none(ak.min(dphi, axis=1), 0.0)
            delta_phi_cut = min_dphi > delta_phi_min
            logger.info(f"  DeltaPhi cut (min > {delta_phi_min}): {ak.sum(delta_phi_cut)} pass")
        else:
            delta_phi_cut = ak.zeros_like(events["event"], dtype=bool)
            logger.info(f"  DeltaPhi cut: NO JETS (all fail)")
    else:
        delta_phi_cut = ak.ones_like(events["event"], dtype=bool)
        logger.info(f"  DeltaPhi cut: DISABLED")

    # === FINAL MASK ===
    cumulative_mask = muon_cut & electron_cut & tau_cut & jet_cut & bjet_cut & met_or_recoil_cut & jet1_pt_cut & delta_phi_cut
    logger.info(f"  FINAL cumulative mask: {ak.sum(cumulative_mask)} events pass ALL cuts")
    
    # Debug which cuts are failing
    cumulative_mask1 = muon_cut
    cumulative_mask2 = cumulative_mask1 & electron_cut
    cumulative_mask3 = cumulative_mask2 & tau_cut
    cumulative_mask4 = cumulative_mask3 & jet_cut
    cumulative_mask5 = cumulative_mask4 & bjet_cut
    cumulative_mask6 = cumulative_mask5 & met_or_recoil_cut
    cumulative_mask7 = cumulative_mask6 & jet1_pt_cut
    cumulative_mask8 = cumulative_mask7 & delta_phi_cut
    logger.info(f"  Cumulative pass counts:")
    logger.info(f"    After muon_cut: {ak.sum(cumulative_mask1)}")
    logger.info(f"    After electron_cut: {ak.sum(cumulative_mask2)}")
    logger.info(f"    After tau_cut: {ak.sum(cumulative_mask3)}")
    logger.info(f"    After jet_cut: {ak.sum(cumulative_mask4)}")
    logger.info(f"    After bjet_cut: {ak.sum(cumulative_mask5)}")
    logger.info(f"    After met_or_recoil_cut: {ak.sum(cumulative_mask6)}")
    logger.info(f"    After jet1_pt_cut: {ak.sum(cumulative_mask7)}")
    logger.info(f"    After delta_phi_cut: {ak.sum(cumulative_mask8)}")

    event_mask = cumulative_mask8

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
    n_total = len(events)

    # Trigger selection
    trigger_mask = pass_triggers(events, config["triggers"]["MET"])
    n_trigger = ak.sum(trigger_mask)

    # MET filter selection
    filter_mask = pass_met_filters(events, config["met_filters"])
    n_filters = ak.sum(trigger_mask & filter_mask)

    # Object selection
    n_muons = ak.sum(ak.num(objects["muons"], axis=1) > 0)
    n_electrons = ak.sum(ak.num(objects["electrons"], axis=1) > 0)
    n_taus = ak.sum(ak.num(objects["taus"], axis=1) > 0)
    n_jets = ak.sum(ak.num(objects["jets"], axis=1) > 0)
    n_bjets = ak.sum(ak.num(objects["bjets"], axis=1) > 0)

    # Event selection
    event_mask = select_events(events, objects, config)
    n_selected = ak.sum(event_mask)

    return {
        "Total events": n_total,
        "Pass trigger": n_trigger,
        "Pass filters": n_filters,
        "Has muons": n_muons,
        "Has electrons": n_electrons,
        "Has taus": n_taus,
        "Has jets": n_jets,
        "Has bjets": n_bjets,
        "Final selection": n_selected,
    }


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
