"""
Physics object selection and cleaning functions for DarkBottomLine framework.
"""

import awkward as ak
import numpy as np
from typing import Dict, Any, Tuple


def select_muons(events: ak.Array, config: Dict[str, Any], wp: str = "loose") -> ak.Array:
    """
    Select muons based on configuration cuts.

    Args:
        events: Awkward Array of events
        config: Configuration dictionary with muon selection cuts
        wp: Working point - "loose" (for event selection) or "tight" (for region selection)

    Returns:
        Boolean mask for selected muons
    """
    # Build muon collection from flat branches (need looseId for loose WP)
    muon_fields = {
        "pt": events["Muon_pt"],
        "eta": events["Muon_eta"],
        "phi": events["Muon_phi"],
        "tightId": events["Muon_tightId"],
        "pfIsoId": events["Muon_pfIsoId"],
    }
    if "Muon_looseId" in events.fields:
        muon_fields["looseId"] = events["Muon_looseId"]
    muons = ak.zip(muon_fields)

    # Basic kinematic cuts: preselection uses pt_min_loose (default 10), region uses pt_min
    pt_min = config.get("pt_min_loose", 10.0) if wp == "loose" else config["pt_min"]
    pt_mask = muons.pt > pt_min
    eta_mask = abs(muons.eta) < config["eta_max"]

    # ID and isolation by working point
    if wp == "loose":
        id_mask = muons.looseId == 1 if "looseId" in muons.fields else (muons.tightId == 1)
        iso_mask = muons.pfIsoId >= 1  # loose isolation
    else:
        id_mask = muons.tightId == 1
        iso_mask = muons.pfIsoId >= 3  # tight isolation

    selection_mask = pt_mask & eta_mask & id_mask & iso_mask
    return selection_mask


def select_electrons(events: ak.Array, config: Dict[str, Any], wp: str = "loose") -> ak.Array:
    """
    Select electrons based on configuration cuts.

    Args:
        events: Awkward Array of events
        config: Configuration dictionary with electron selection cuts
        wp: Working point - "loose" (for event selection) or "tight" (for region selection)

    Returns:
        Boolean mask for selected electrons
    """
    # Build electron collection from flat branches (need mvaIso_WP90 for loose)
    ele_fields = {
        "pt": events["Electron_pt"],
        "eta": events["Electron_eta"],
        "phi": events["Electron_phi"],
        "cutBased": events["Electron_cutBased"],
        "mvaIso_WP80": events["Electron_mvaIso_WP80"],
    }
    if "Electron_mvaIso_WP90" in events.fields:
        ele_fields["mvaIso_WP90"] = events["Electron_mvaIso_WP90"]
    electrons = ak.zip(ele_fields)

    # Basic kinematic cuts: preselection uses pt_min_loose (default 10), region uses pt_min
    pt_min = config.get("pt_min_loose", 10.0) if wp == "loose" else config["pt_min"]
    pt_mask = electrons.pt > pt_min
    eta_mask = abs(electrons.eta) < config["eta_max"]

    # ECAL barrel-endcap gap veto: exclude 1.4442 < |eta| < 1.566
    eta_gap_min = 1.4442
    eta_gap_max = 1.566
    in_gap = (abs(electrons.eta) > eta_gap_min) & (abs(electrons.eta) < eta_gap_max)
    gap_veto_mask = ~in_gap

    # ID and isolation by working point (cutBased: 2=loose, 4=tight)
    if wp == "loose":
        id_mask = electrons.cutBased >= 2
        iso_mask = electrons.mvaIso_WP90 == 1 if "mvaIso_WP90" in electrons.fields else (electrons.mvaIso_WP80 == 1)
    else:
        id_mask = electrons.cutBased >= 4
        iso_mask = electrons.mvaIso_WP80 == 1

    selection_mask = pt_mask & eta_mask & gap_veto_mask & id_mask & iso_mask
    return selection_mask


def select_taus(events: ak.Array, config: Dict[str, Any], wp: str = "loose") -> ak.Array:
    """
    Select taus based on configuration cuts.

    Args:
        events: Awkward Array of events
        config: Configuration dictionary with tau selection cuts
        wp: Working point - "loose" (for event selection) or "tight" (for region selection)

    Returns:
        Boolean mask for selected taus
    """
    # Build tau collection from flat branches
    taus = ak.zip({
        "pt": events["Tau_pt"],
        "eta": events["Tau_eta"],
        "phi": events["Tau_phi"],
        "idDeepTau2018v2p5VSjet": events["Tau_idDeepTau2018v2p5VSjet"],
        "decayMode": events["Tau_decayMode"],
    })

    # Basic kinematic cuts: preselection uses pt_min_loose (default 10), region uses pt_min
    pt_min = config.get("pt_min_loose", 10.0) if wp == "loose" else config["pt_min"]
    pt_mask = taus.pt > pt_min
    eta_mask = abs(taus.eta) < config["eta_max"]

    # ID by working point (DeepTau: 4=medium/loose, 16=tight)
    id_mask = (taus.idDeepTau2018v2p5VSjet >= 4) if wp == "loose" else (taus.idDeepTau2018v2p5VSjet >= 16)
    # Check if decay mode is in allowed modes
    decay_mode_mask = ak.zeros_like(taus.pt, dtype=bool)
    for mode in config["decay_modes"]:
        decay_mode_mask = decay_mode_mask | (taus.decayMode == mode)

    selection_mask = pt_mask & eta_mask & id_mask & decay_mode_mask
    return selection_mask


def select_jets(events: ak.Array, config: Dict[str, Any]) -> ak.Array:
    """
    Select AK4 jets based on configuration cuts.

    Args:
        events: Awkward Array of events
        config: Configuration dictionary with jet selection cuts

    Returns:
        Boolean mask for selected jets
    """
    jet_fields = {
        "pt": events["Jet_pt"],
        "eta": events["Jet_eta"],
        "phi": events["Jet_phi"],
        "btagDeepFlavB": events["Jet_btagDeepFlavB"],
    }
    if "Jet_hadronFlavour" in events.fields:
        jet_fields["hadronFlavour"] = events["Jet_hadronFlavour"]
    elif "Jet_partonFlavour" in events.fields:
        jet_fields["hadronFlavour"] = events["Jet_partonFlavour"]
    jets = ak.zip(jet_fields)

    # Basic kinematic cuts
    pt_mask = jets.pt > config["pt_min"]
    eta_mask = abs(jets.eta) < config["eta_max"]

    # Jet ID cut (simplified - would need actual CMS jet ID implementation)
    id_mask = ak.ones_like(jets.pt, dtype=bool)  # puIdDisc branch not in v12

    # Combine all cuts
    selection_mask = pt_mask & eta_mask & id_mask

    return selection_mask


def select_fatjets(events: ak.Array, config: Dict[str, Any]) -> ak.Array:
    """
    Select AK8 fat jets based on configuration cuts.

    Args:
        events: Awkward Array of events
        config: Configuration dictionary with fat jet selection cuts

    Returns:
        Boolean mask for selected fat jets
    """
    # Build fat jet collection from flat branches
    fatjets = ak.zip({
        "pt": events["FatJet_pt"],
        "eta": events["FatJet_eta"],
        "phi": events["FatJet_phi"],
        "mass": events["FatJet_mass"],
    })

    # Basic kinematic cuts
    pt_mask = fatjets.pt > config["pt_min"]
    eta_mask = abs(fatjets.eta) < config["eta_max"]

    # Fat jet ID cut (simplified - would need actual CMS fat jet ID implementation)
    # For now, just use basic kinematic cuts
    id_mask = ak.ones_like(fatjets.pt, dtype=bool)

    # Combine all cuts
    selection_mask = pt_mask & eta_mask & id_mask

    return selection_mask


def clean_jets_from_leptons(
    jets: ak.Array,
    leptons: ak.Array,
    dr_min: float = 0.4
) -> ak.Array:
    """
    Remove jets that are too close to selected leptons.

    Args:
        jets: Selected jets
        leptons: Selected leptons (muons, electrons, or taus)
        dr_min: Minimum Delta-R separation

    Returns:
        Boolean mask for jets that pass cleaning
    """
    if len(ak.flatten(leptons)) == 0:
        # No leptons, all jets pass
        return ak.ones_like(jets.pt, dtype=bool)

    # Calculate Delta-R between jets and all leptons
    # This is a simplified version - would need proper 2D Delta-R calculation
    jet_eta = jets.eta
    jet_phi = jets.phi
    lep_eta = ak.flatten(leptons.eta)
    lep_phi = ak.flatten(leptons.phi)

    # For each jet, check if it's far enough from any lepton
    # This is a placeholder - actual implementation would use proper Delta-R
    dr_mask = ak.ones_like(jets.pt, dtype=bool)

    # TODO: Implement proper Delta-R calculation
    # For now, return all jets as passing
    return dr_mask


def get_bjet_mask(jets: ak.Array, config: Dict[str, Any]) -> ak.Array:
    """
    Get b-tagging mask for jets based on working point.

    Args:
        jets: Selected jets
        config: Configuration dictionary with b-tagging parameters

    Returns:
        Boolean mask for b-tagged jets
    """
    algorithm = config["algorithm"]
    wp = config["wp"]

    if algorithm == "deepJet":
        if wp == "loose":
            return jets.btagDeepFlavB > 0.0490
        elif wp == "medium":
            return jets.btagDeepFlavB > 0.2783
        elif wp == "tight":
            return jets.btagDeepFlavB > 0.7100
    elif algorithm == "deepCSV":
        if wp == "loose":
            return jets.btagDeepB > 0.1208
        elif wp == "medium":
            return jets.btagDeepB > 0.4941
        elif wp == "tight":
            return jets.btagDeepB > 0.8001

    # Default to medium working point
    return jets.btagDeepFlavB > 0.2783


def _dilepton_mass(l1: ak.Array, l2: ak.Array, m_default: float = 0.105) -> ak.Array:
    """Compute dilepton invariant mass from two leptons (pt, eta, phi; mass optional).

    Uses ak.fill_none on every field access so that optional (?Record) and union-type
    arrays (produced by ak.fill_none(optA, optB)) never reach numpy arithmetic, which
    would otherwise trigger a RecursionError in Python's ABC __instancecheck__.
    """
    pt1  = ak.fill_none(l1.pt,  0.0)
    eta1 = ak.fill_none(l1.eta, 0.0)
    phi1 = ak.fill_none(l1.phi, 0.0)
    pt2  = ak.fill_none(l2.pt,  0.0)
    eta2 = ak.fill_none(l2.eta, 0.0)
    phi2 = ak.fill_none(l2.phi, 0.0)
    m1 = ak.fill_none(ak.values_astype(l1["mass"], float), m_default) if "mass" in l1.fields else ak.full_like(pt1, m_default)
    m2 = ak.fill_none(ak.values_astype(l2["mass"], float), m_default) if "mass" in l2.fields else ak.full_like(pt2, m_default)
    px1 = pt1 * np.cos(phi1)
    py1 = pt1 * np.sin(phi1)
    pz1 = pt1 * np.sinh(eta1)
    e1 = np.sqrt(m1**2 + pt1**2 * np.cosh(eta1)**2)
    px2 = pt2 * np.cos(phi2)
    py2 = pt2 * np.sin(phi2)
    pz2 = pt2 * np.sinh(eta2)
    e2 = np.sqrt(m2**2 + pt2**2 * np.cosh(eta2)**2)
    mll_sq = (e1 + e2)**2 - (px1 + px2)**2 - (py1 + py2)**2 - (pz1 + pz2)**2
    return np.sqrt(ak.where(mll_sq >= 0, mll_sq, 0.0))


def build_z_candidates(
    loose_muons: ak.Array,
    loose_electrons: ak.Array,
    pt_lead_min: float = 30.0,
    pt_sublead_min: float = 10.0,
) -> Tuple[ak.Array, ak.Array, ak.Array, ak.Array, ak.Array, ak.Array, ak.Array, ak.Array]:
    """
    Build Z->ll candidates for Z CR: 2 opposite-sign leptons.
    Leading: tight ID, pt > pt_lead_min (30 GeV).
    Subleading: loose ID, pt > pt_sublead_min (10 GeV).

    Returns:
        n_z_muons, n_z_electrons: 2 if valid Z candidate else 0 per event
        mll_mu, mll_el: invariant mass of pair (0 if no candidate)
        z_lep_sum_x_mu, z_lep_sum_y_mu, z_lep_sum_x_el, z_lep_sum_y_el: pT vector sum for RecoilZ
    """
    n_ev = len(loose_muons)
    # awkward has no ak.zeros; use numpy and wrap for compatibility
    n_z_muons = ak.Array(np.zeros(n_ev, dtype=np.int64))
    n_z_electrons = ak.Array(np.zeros(n_ev, dtype=np.int64))
    mll_mu = ak.Array(np.zeros(n_ev, dtype=float))
    mll_el = ak.Array(np.zeros(n_ev, dtype=float))
    z_lep_sum_x_mu = ak.Array(np.zeros(n_ev, dtype=float))
    z_lep_sum_y_mu = ak.Array(np.zeros(n_ev, dtype=float))
    z_lep_sum_x_el = ak.Array(np.zeros(n_ev, dtype=float))
    z_lep_sum_y_el = ak.Array(np.zeros(n_ev, dtype=float))

    has_charge_mu = "charge" in loose_muons.fields
    has_charge_el = "charge" in loose_electrons.fields

    def _one_flavor(
        loose_lep: ak.Array, is_mu: bool
    ) -> Tuple[ak.Array, ak.Array, ak.Array, ak.Array, ak.Array]:
        n_lep = ak.num(loose_lep, axis=1)
        has_two = n_lep >= 2
        idx = ak.argsort(loose_lep.pt, axis=1, ascending=False)
        ordered = loose_lep[idx]
        lead = ak.firsts(ordered)
        sublead = ak.pad_none(ordered, 2, axis=1)[:, 1]
        # Do NOT ak.fill_none(sublead, lead) — both are optional (?Record), which
        # creates a UnionArray that causes RecursionError in numpy type dispatch.
        # Instead pass sublead directly; _dilepton_mass uses ak.fill_none per-field.
        if (has_charge_mu and is_mu) or (has_charge_el and not is_mu):
            os_pair = (lead.charge + ak.fill_none(sublead.charge, 999)) == 0
        else:
            os_pair = ak.ones_like(has_two, dtype=bool)
        lead_tight_pt = lead.is_tight & (lead.pt > pt_lead_min)
        sublead_pt_ok = ak.fill_none(sublead.pt, 0.0) > pt_sublead_min
        valid = has_two & os_pair & lead_tight_pt & sublead_pt_ok
        n_z = ak.where(valid, 2, 0)
        m_default = 0.105 if is_mu else 0.000511
        mll = ak.where(valid, _dilepton_mass(lead, sublead, m_default), 0.0)
        sum_x = np.cos(lead.phi) * lead.pt + np.cos(ak.fill_none(sublead.phi, 0.0)) * ak.fill_none(sublead.pt, 0.0)
        sum_y = np.sin(lead.phi) * lead.pt + np.sin(ak.fill_none(sublead.phi, 0.0)) * ak.fill_none(sublead.pt, 0.0)
        return n_z, mll, sum_x, sum_y, valid

    if len(ak.flatten(loose_muons)) > 0 and hasattr(loose_muons, "is_tight"):
        n_z_muons, mll_mu, sx_mu, sy_mu, vm = _one_flavor(loose_muons, True)
        z_lep_sum_x_mu = ak.where(vm, sx_mu, 0.0)
        z_lep_sum_y_mu = ak.where(vm, sy_mu, 0.0)
    if len(ak.flatten(loose_electrons)) > 0 and hasattr(loose_electrons, "is_tight"):
        n_z_electrons, mll_el, sx_el, sy_el, ve = _one_flavor(loose_electrons, False)
        z_lep_sum_x_el = ak.where(ve, sx_el, 0.0)
        z_lep_sum_y_el = ak.where(ve, sy_el, 0.0)

    return (
        n_z_muons, n_z_electrons, mll_mu, mll_el,
        z_lep_sum_x_mu, z_lep_sum_y_mu, z_lep_sum_x_el, z_lep_sum_y_el,
    )


def build_objects(events: ak.Array, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build all physics objects with selection and cleaning applied.

    Args:
        events: Awkward Array of events
        config: Full configuration dictionary

    Returns:
        Dictionary containing selected objects and masks
    """
    print("  Building object collections from flat branches...")

    # Select objects: loose for event selection, tight for region selection
    print("  Selecting muons (loose for event sel, tight for regions)...")
    muon_mask_loose = select_muons(events, config["objects"]["muons"], wp="loose")
    muon_mask_tight = select_muons(events, config["objects"]["muons"], wp="tight")
    muon_mask = muon_mask_loose  # main collection = loose (event selection)
    print(f"    Muons (loose): {ak.sum(muon_mask_loose)}, tight: {ak.sum(muon_mask_loose & muon_mask_tight)}")

    print("  Selecting electrons (loose for event sel, tight for regions)...")
    electron_mask_loose = select_electrons(events, config["objects"]["electrons"], wp="loose")
    electron_mask_tight = select_electrons(events, config["objects"]["electrons"], wp="tight")
    electron_mask = electron_mask_loose
    print(f"    Electrons (loose): {ak.sum(electron_mask_loose)}, tight: {ak.sum(electron_mask_loose & electron_mask_tight)}")

    print("  Selecting taus (loose for event sel, tight for regions)...")
    tau_mask_loose = select_taus(events, config["objects"]["taus"], wp="loose")
    tau_mask_tight = select_taus(events, config["objects"]["taus"], wp="tight")
    tau_mask = tau_mask_loose
    print(f"    Taus (loose): {ak.sum(tau_mask_loose)}, tight: {ak.sum(tau_mask_loose & tau_mask_tight)}")

    print("  Selecting jets...")
    jet_mask = select_jets(events, config["objects"]["jets"])
    print(f"    Jets selected: {ak.sum(jet_mask)}")

    # Fat jets disabled — uncomment to re-enable
    # print("  Selecting fat jets...")
    # fatjet_mask = select_fatjets(events, config["objects"]["fatjets"])
    # print(f"    Fat jets selected: {ak.sum(fatjet_mask)}")
    fatjet_mask = ak.zeros_like(events["FatJet_pt"], dtype=bool)  # empty mask placeholder

    # Build collections from flat branches (kinematics + ID + charge for Z CR)
    muon_fields = {
        "pt": events["Muon_pt"],
        "eta": events["Muon_eta"],
        "phi": events["Muon_phi"],
        "tightId": events["Muon_tightId"],
        "pfIsoId": events["Muon_pfIsoId"],
    }
    if "Muon_charge" in events.fields:
        muon_fields["charge"] = events["Muon_charge"]
    if "Muon_mass" in events.fields:
        muon_fields["mass"] = events["Muon_mass"]
    muons = ak.zip(muon_fields)

    electron_fields = {
        "pt": events["Electron_pt"],
        "eta": events["Electron_eta"],
        "phi": events["Electron_phi"],
        "cutBased": events["Electron_cutBased"],
        "mvaIso_WP80": events["Electron_mvaIso_WP80"],
    }
    if "Electron_charge" in events.fields:
        electron_fields["charge"] = events["Electron_charge"]
    if "Electron_mass" in events.fields:
        electron_fields["mass"] = events["Electron_mass"]
    electrons = ak.zip(electron_fields)

    taus = ak.zip({
        "pt": events["Tau_pt"],
        "eta": events["Tau_eta"],
        "phi": events["Tau_phi"],
        "idDeepTau2018v2p5VSjet": events["Tau_idDeepTau2018v2p5VSjet"],
        "decayMode": events["Tau_decayMode"],
    })

    jet_fields = {
        "pt": events["Jet_pt"],
        "eta": events["Jet_eta"],
        "phi": events["Jet_phi"],
        "btagDeepFlavB": events["Jet_btagDeepFlavB"],
    }
    if "Jet_hadronFlavour" in events.fields:
        jet_fields["hadronFlavour"] = events["Jet_hadronFlavour"]
    elif "Jet_partonFlavour" in events.fields:
        jet_fields["hadronFlavour"] = events["Jet_partonFlavour"]
    jets = ak.zip(jet_fields)

    # Fat jets disabled — uncomment to re-enable
    # fatjets = ak.zip({
    #     "pt": events["FatJet_pt"],
    #     "eta": events["FatJet_eta"],
    #     "phi": events["FatJet_phi"],
    #     "mass": events["FatJet_mass"],
    # })
    fatjets = ak.zip({"pt": events["FatJet_pt"], "eta": events["FatJet_eta"],
                      "phi": events["FatJet_phi"], "mass": events["FatJet_mass"]})

    # Apply masks: main collections = loose (for event selection, jet cleaning)
    selected_muons = muons[muon_mask]
    selected_electrons = electrons[electron_mask]
    selected_taus = taus[tau_mask]
    # Per-loose-lepton is_tight flag (for Z CR: leading tight, subleading loose)
    selected_muons = ak.with_field(selected_muons, muon_mask_tight[muon_mask_loose], "is_tight")
    selected_electrons = ak.with_field(selected_electrons, electron_mask_tight[electron_mask_loose], "is_tight")
    # Tight subsets (for region selection)
    tight_muons = muons[muon_mask_loose & muon_mask_tight]
    tight_electrons = electrons[electron_mask_loose & electron_mask_tight]
    tight_taus = taus[tau_mask_loose & tau_mask_tight]
    # Tight pt>30: for all CRs (W, Top, etc.) leading lepton is tight with pt>30
    tight_muons_pt30 = tight_muons[tight_muons.pt > 30.0]
    tight_electrons_pt30 = tight_electrons[tight_electrons.pt > 30.0]
    tight_taus_pt30 = tight_taus[tight_taus.pt > 30.0]

    # Z CR candidates: 2 OS leptons, leading tight pt>30, subleading loose pt>10
    (n_z_muons, n_z_electrons, mll_mu, mll_el,
     z_lep_sum_x_mu, z_lep_sum_y_mu, z_lep_sum_x_el, z_lep_sum_y_el) = build_z_candidates(
        selected_muons, selected_electrons, pt_lead_min=30.0, pt_sublead_min=10.0
    )

    selected_jets = jets[jet_mask]
    selected_fatjets = fatjets[fatjet_mask]  # always empty while fat jets are disabled

    # Clean jets from leptons
    print("  Cleaning jets from leptons...")
    all_leptons = ak.concatenate([
        selected_muons, selected_electrons, selected_taus
    ], axis=1)

    jet_cleaning_mask = clean_jets_from_leptons(
        selected_jets,
        all_leptons,
        config["cleaning"]["dr_muon_jet"]
    )

    # Apply jet cleaning
    cleaned_jets = selected_jets[jet_cleaning_mask]
    print(f"    Jets after cleaning: {ak.sum(ak.num(cleaned_jets, axis=1))}")

    # Get b-tagging mask for cleaned jets
    print("  Applying b-tagging...")
    bjet_mask = get_bjet_mask(cleaned_jets, config["btagging"])
    print(f"    B-jets identified: {ak.sum(ak.num(cleaned_jets[bjet_mask], axis=1))}")

    print("  Object building complete!")
    return {
        "muons": selected_muons,
        "electrons": selected_electrons,
        "taus": selected_taus,
        "tight_muons": tight_muons,
        "tight_electrons": tight_electrons,
        "tight_taus": tight_taus,
        "tight_muons_pt30": tight_muons_pt30,
        "tight_electrons_pt30": tight_electrons_pt30,
        "tight_taus_pt30": tight_taus_pt30,
        "n_z_muons": n_z_muons,
        "n_z_electrons": n_z_electrons,
        "mll_mu": mll_mu,
        "mll_el": mll_el,
        "z_lep_sum_x_mu": z_lep_sum_x_mu,
        "z_lep_sum_y_mu": z_lep_sum_y_mu,
        "z_lep_sum_x_el": z_lep_sum_x_el,
        "z_lep_sum_y_el": z_lep_sum_y_el,
        "jets": cleaned_jets,
        "fatjets": selected_fatjets,
        "bjets": cleaned_jets[bjet_mask],
        "muon_mask": muon_mask,
        "electron_mask": electron_mask,
        "tau_mask": tau_mask,
        "jet_mask": jet_mask,
        "fatjet_mask": fatjet_mask,
        "bjet_mask": bjet_mask,
    }
