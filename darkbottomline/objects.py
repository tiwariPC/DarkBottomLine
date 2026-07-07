"""
Physics object selection and cleaning functions for DarkBottomLine framework.
"""

import awkward as ak
import numpy as np
from typing import Dict, Any, Tuple

SENTINEL = -9.0  # fill value for variables undefined due to insufficient objects


# ---------------------------------------------------------------------------
# Collection builders — single source of truth for each object's field list.
# build_objects zips the real collections once via these builders. The select_*
# mask functions do NOT zip; they read the flat NanoAOD branches directly (same
# pattern as select_jets), so each collection is zipped exactly once per chunk.
# ---------------------------------------------------------------------------

def build_muon_collection(events: ak.Array) -> ak.Array:
    """Zip the muon collection from flat branches (kinematics + ID + iso;
    charge/mass added when present, for Z-candidate reconstruction)."""
    fields = {
        "pt": events["Muon_pt"],
        "eta": events["Muon_eta"],
        "phi": events["Muon_phi"],
        "looseId": events["Muon_looseId"],
        "tightId": events["Muon_tightId"],
        "pfIsoId": events["Muon_pfIsoId"],
    }
    if "Muon_charge" in events.fields:
        fields["charge"] = events["Muon_charge"]
    if "Muon_mass" in events.fields:
        fields["mass"] = events["Muon_mass"]
    return ak.zip(fields)


def build_electron_collection(events: ak.Array) -> ak.Array:
    """Zip the electron collection: cutBased ID + mvaIso WP80/WP90 isolation
    (both required NanoAOD branches — loud KeyError if absent). charge/mass when present."""
    fields = {
        "pt": events["Electron_pt"],
        "eta": events["Electron_eta"],
        "phi": events["Electron_phi"],
        "cutBased": events["Electron_cutBased"],
        "mvaIso_WP80": events["Electron_mvaIso_WP80"],
        "mvaIso_WP90": events["Electron_mvaIso_WP90"],
    }
    if "Electron_charge" in events.fields:
        fields["charge"] = events["Electron_charge"]
    if "Electron_mass" in events.fields:
        fields["mass"] = events["Electron_mass"]
    return ak.zip(fields)


def build_tau_collection(events: ak.Array) -> ak.Array:
    """Zip the tau collection with all three DeepTau2018v2p5 discriminants
    (VSjet/VSe/VSmu) + decayMode."""
    return ak.zip({
        "pt": events["Tau_pt"],
        "eta": events["Tau_eta"],
        "phi": events["Tau_phi"],
        "idDeepTau2018v2p5VSjet": events["Tau_idDeepTau2018v2p5VSjet"],
        "idDeepTau2018v2p5VSe": events["Tau_idDeepTau2018v2p5VSe"],
        "idDeepTau2018v2p5VSmu": events["Tau_idDeepTau2018v2p5VSmu"],
        "decayMode": events["Tau_decayMode"],
    })


def build_jet_collection(events: ak.Array, config: Dict[str, Any]) -> ak.Array:
    """Zip the AK4 jet collection. btagScore comes from the YAML-configured
    branch; hadronFlavour (MC) added when present."""
    fields = {
        "pt": events["Jet_pt"],
        "eta": events["Jet_eta"],
        "phi": events["Jet_phi"],
        "mass": events["Jet_mass"] if "Jet_mass" in events.fields else ak.zeros_like(events["Jet_pt"]),
        "btagScore": events[config["btagging"]["branch"]],
    }
    if "Jet_hadronFlavour" in events.fields:
        fields["hadronFlavour"] = events["Jet_hadronFlavour"]
    elif "Jet_partonFlavour" in events.fields:
        fields["hadronFlavour"] = events["Jet_partonFlavour"]
    return ak.zip(fields)


def build_photon_collection(events: ak.Array) -> ak.Array:
    """Zip the photon collection (veto object): pt/eta/phi/cutBased."""
    return ak.zip({
        "pt": events["Photon_pt"],
        "eta": events["Photon_eta"],
        "phi": events["Photon_phi"],
        "cutBased": events["Photon_cutBased"],
    })


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
    # Mask reads flat branches directly (like select_jets) — no zip needed here.
    # build_objects zips the real collection once via build_muon_collection.
    # Basic kinematic cuts: preselection uses pt_min_loose (default 10), region uses pt_min
    pt_min = config["pt_min_loose"] if wp == "loose" else config["pt_min"]
    pt_mask = events["Muon_pt"] > pt_min
    eta_mask = abs(events["Muon_eta"]) < config["eta_max"]

    # ID and isolation by working point. Both looseId and tightId are required
    # NanoAOD branches (loud KeyError if absent). tight muon is a strict subset
    # of loose, so tight WP uses tightId directly — no looseId fallback.
    iso_wp = config["iso_wp_loose"] if wp == "loose" else config["iso_wp_tight"]
    iso_mask = events["Muon_pfIsoId"] >= iso_wp
    if wp == "loose":
        id_mask = events["Muon_looseId"] == 1
    else:
        id_mask = events["Muon_tightId"] == 1

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
    # Mask reads flat branches directly (like select_jets) — no zip needed here.
    ele_eta = events["Electron_eta"]

    # Basic kinematic cuts: preselection uses pt_min_loose (default 10), region uses pt_min
    pt_min = config["pt_min_loose"] if wp == "loose" else config["pt_min"]
    pt_mask = events["Electron_pt"] > pt_min
    eta_mask = abs(ele_eta) < config["eta_max"]

    # ECAL barrel-endcap gap veto: exclude 1.4442 < |eta| < 1.566
    eta_gap_min = 1.4442
    eta_gap_max = 1.566
    in_gap = (abs(ele_eta) > eta_gap_min) & (abs(ele_eta) < eta_gap_max)
    gap_veto_mask = ~in_gap

    # ID + isolation by working point:
    #   loose  = cutBased >= id_wp_loose (2)  AND mvaIso_WP90
    #   tight  = cutBased >= id_wp_tight (4)  AND mvaIso_WP80
    if wp == "loose":
        id_wp = config["id_wp_loose"]
        iso_mask = events["Electron_mvaIso_WP90"] == 1
    else:
        id_wp = config["id_wp_tight"]
        iso_mask = events["Electron_mvaIso_WP80"] == 1
    id_mask = events["Electron_cutBased"] >= id_wp

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
    # Mask reads flat branches directly (like select_jets) — no zip needed here.
    # Taus use single pt_min (20 GeV) for both loose and tight — no separate loose threshold
    pt_min = config["pt_min"]
    pt_mask = events["Tau_pt"] > pt_min
    eta_mask = abs(events["Tau_eta"]) < config["eta_max"]

    # Tau is a veto object — no tight WP. Single set of DeepTau2018v2p5 WPs
    # applied identically for both loose and tight calls:
    #   VSjet >= id_wp_vsjet (VLoose=3), VSe >= id_wp_vse (VVVLoose=1),
    #   VSmu >= id_wp_vsmu (VLoose=1). All three ANDed.
    id_mask = (
        (events["Tau_idDeepTau2018v2p5VSjet"] >= config["id_wp_vsjet"])
        & (events["Tau_idDeepTau2018v2p5VSe"] >= config["id_wp_vse"])
        & (events["Tau_idDeepTau2018v2p5VSmu"] >= config["id_wp_vsmu"])
    )
    # Check if decay mode is in allowed modes
    tau_decay_mode = events["Tau_decayMode"]
    decay_mode_mask = ak.zeros_like(tau_decay_mode, dtype=bool)
    for mode in config["decay_modes"]:
        decay_mode_mask = decay_mode_mask | (tau_decay_mode == mode)

    selection_mask = pt_mask & eta_mask & id_mask & decay_mode_mask
    return selection_mask


def select_photons(events: ak.Array, config: Dict[str, Any]) -> ak.Array:
    """Select photons for veto: loose ID, pt > 15, |eta| < 2.5.

    Mask reads flat branches directly (like select_jets) — no zip needed here."""
    pt_mask = events["Photon_pt"] > config["pt_min"]
    eta_mask = abs(events["Photon_eta"]) < config["eta_max"]
    id_mask = events["Photon_cutBased"] >= config["id_wp_loose"]
    return pt_mask & eta_mask & id_mask


def select_jets(events: ak.Array, config: Dict[str, Any]) -> ak.Array:
    """
    Select AK4 jets based on configuration cuts.

    Args:
        events: Awkward Array of events
        config: Configuration dictionary with jet selection cuts

    Returns:
        Boolean mask for selected jets
    """
    # Mask needs only pt/eta (no jet ID cut — puIdDisc branch absent in NanoAOD
    # v12+). btagScore/hadronFlavour are added later in build_jet_collection.
    pt_mask = events["Jet_pt"] > config["pt_min"]
    eta_mask = abs(events["Jet_eta"]) < config["eta_max"]

    return pt_mask & eta_mask


def clean_jets_from_leptons(
    jets: ak.Array,
    leptons: ak.Array,
    dr_min: float = 0.4
) -> ak.Array:
    """Remove jets within dr_min of any lepton (muons, electrons, photons)."""
    if len(ak.flatten(leptons)) == 0:
        return ak.ones_like(jets.pt, dtype=bool)

    # nested=True → shape: events × jets × leptons (axis=2 reduces over leptons per jet)
    pairs = ak.cartesian({"jet": jets, "lep": leptons}, axis=1, nested=True)
    deta = pairs["jet"].eta - pairs["lep"].eta
    dphi = pairs["jet"].phi - pairs["lep"].phi
    dphi = ak.where(dphi > np.pi, dphi - 2 * np.pi, dphi)
    dphi = ak.where(dphi < -np.pi, dphi + 2 * np.pi, dphi)
    dr = np.sqrt(deta**2 + dphi**2)

    # True per jet if ANY lepton within dr_min
    too_close = ak.any(dr < dr_min, axis=2)
    return ~too_close


def get_bjet_mask(jets: ak.Array, config: Dict[str, Any]) -> ak.Array:
    """
    Get b-tagging mask for jets based on working point.

    Args:
        jets: Selected jets
        config: Configuration dictionary with b-tagging parameters

    Returns:
        Boolean mask for b-tagged jets
    """
    score = config["score"]           # b-tag discriminant threshold (from YAML)

    return jets.btagScore > score


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
    pt_lead_min_mu: float = 30.0,
    pt_lead_min_el: float = 32.0,
    pt_sublead_min: float = 10.0,
) -> Tuple[ak.Array, ak.Array, ak.Array, ak.Array, ak.Array, ak.Array]:
    """
    Build Z->ll candidates for Z CR: 2 opposite-sign leptons.
    Leading: tight ID, pt > per-flavor threshold (mu 30, el 32 GeV).
    Subleading: loose ID, pt > pt_sublead_min (10 GeV).

    Returns:
        n_z_muons, n_z_electrons: 2 if valid Z candidate else 0 per event
        mll_mu, mll_el: invariant mass of the candidate pair (SENTINEL if none)
        z_pt_mu, z_pt_el: pT of the candidate dilepton system (SENTINEL if none)
    """
    n_ev = len(loose_muons)
    # awkward has no ak.zeros; use numpy and wrap for compatibility
    n_z_muons = ak.Array(np.zeros(n_ev, dtype=np.int64))
    n_z_electrons = ak.Array(np.zeros(n_ev, dtype=np.int64))
    mll_mu = ak.Array(np.full(n_ev, SENTINEL, dtype=float))
    mll_el = ak.Array(np.full(n_ev, SENTINEL, dtype=float))
    z_pt_mu = ak.Array(np.full(n_ev, SENTINEL, dtype=float))
    z_pt_el = ak.Array(np.full(n_ev, SENTINEL, dtype=float))

    has_charge_mu = "charge" in loose_muons.fields
    has_charge_el = "charge" in loose_electrons.fields

    def _one_flavor(
        loose_lep: ak.Array, is_mu: bool
    ) -> Tuple[ak.Array, ak.Array, ak.Array]:
        n_lep = ak.num(loose_lep, axis=1)
        has_two = n_lep >= 2
        idx = ak.argsort(loose_lep.pt, axis=1, ascending=False)
        ordered = loose_lep[idx]
        lead = ak.firsts(ordered)
        sublead = ak.pad_none(ordered, 2, axis=1)[:, 1]
        # Do NOT ak.fill_none(sublead, lead) — both are optional (?Record), which
        # creates a UnionArray that causes RecursionError in numpy type dispatch.
        # Instead pass sublead directly; _dilepton_mass uses ak.fill_none per-field.
        pt_lead_min = pt_lead_min_mu if is_mu else pt_lead_min_el
        # lead/sublead are optional-typed (ak.firsts / pad_none) so per-field
        # comparisons yield ?bool that propagate None where n_lep<2. Wrap every
        # option-typed piece in ak.fill_none(..., False) so `valid` stays plain
        # bool and n_z/mll below are non-optional (int64/float, not ?int64/?float).
        if (has_charge_mu and is_mu) or (has_charge_el and not is_mu):
            os_pair = ak.fill_none((lead.charge + ak.fill_none(sublead.charge, 999)) == 0, False)
        else:
            os_pair = ak.ones_like(has_two, dtype=bool)
        lead_tight_pt = ak.fill_none(lead.is_tight & (lead.pt > pt_lead_min), False)
        sublead_pt_ok = ak.fill_none(sublead.pt, 0.0) > pt_sublead_min
        valid = has_two & os_pair & lead_tight_pt & sublead_pt_ok
        n_z = ak.where(valid, 2, 0)
        m_default = 0.105 if is_mu else 0.000511
        mll = ak.where(valid, _dilepton_mass(lead, sublead, m_default), SENTINEL)
        # Z pT = |lead_vec + sublead_vec| of the SAME candidate pair as mll.
        # fill_none per-field so the optional lead/sublead never reach numpy.
        lead_pt   = ak.fill_none(lead.pt,     0.0)
        lead_phi  = ak.fill_none(lead.phi,    0.0)
        sub_pt    = ak.fill_none(sublead.pt,  0.0)
        sub_phi   = ak.fill_none(sublead.phi, 0.0)
        z_px = lead_pt * np.cos(lead_phi) + sub_pt * np.cos(sub_phi)
        z_py = lead_pt * np.sin(lead_phi) + sub_pt * np.sin(sub_phi)
        z_pt = ak.where(valid, np.sqrt(z_px**2 + z_py**2), SENTINEL)
        return n_z, mll, z_pt

    if len(ak.flatten(loose_muons)) > 0 and hasattr(loose_muons, "is_tight"):
        n_z_muons, mll_mu, z_pt_mu = _one_flavor(loose_muons, True)
    if len(ak.flatten(loose_electrons)) > 0 and hasattr(loose_electrons, "is_tight"):
        n_z_electrons, mll_el, z_pt_el = _one_flavor(loose_electrons, False)

    return n_z_muons, n_z_electrons, mll_mu, mll_el, z_pt_mu, z_pt_el


def calculate_costheta_star(jets: ak.Array) -> ak.Array:
    """
    cos(theta*) = |tanh(dEta_j1j2 / 2)|
    Requires >= 2 jets. Returns SENTINEL when < 2 jets.
    """
    n_jets = ak.num(jets, axis=1)
    has_two = n_jets >= 2

    if not ak.any(has_two):
        return ak.full_like(n_jets, SENTINEL, dtype=float)

    j1 = ak.firsts(jets)
    j2 = ak.pad_none(jets, 2, axis=1)[:, 1]

    deta = ak.fill_none(j1.eta, 0.0) - ak.fill_none(j2.eta, 0.0)
    cos_ts = np.abs(np.tanh(deta / 2.0))

    return ak.where(has_two, cos_ts, SENTINEL)


def _lepton_pt_sums(muons, electrons):
    """Vector pT sum of muons + electrons → (lep_px, lep_py). ak.sum → 0 for empty.
    Lepton kinematics are invariant across MET shifts, so compute once and reuse."""
    lep_px = (ak.sum(muons.pt * np.cos(muons.phi), axis=1)
              + ak.sum(electrons.pt * np.cos(electrons.phi), axis=1))
    lep_py = (ak.sum(muons.pt * np.sin(muons.phi), axis=1)
              + ak.sum(electrons.pt * np.sin(electrons.phi), axis=1))
    return lep_px, lep_py


def _recoil_from_sums(met_pt, met_phi, lep_px, lep_py):
    """Recoil = -(MET_vec + precomputed lepton pT sum). Returns (recoil_pt, recoil_phi)."""
    recoil_px = -(met_pt * np.cos(met_phi) + lep_px)
    recoil_py = -(met_pt * np.sin(met_phi) + lep_py)
    recoil_pt  = np.sqrt(recoil_px**2 + recoil_py**2)
    recoil_phi = np.arctan2(recoil_py, recoil_px)
    return recoil_pt, recoil_phi


def _recoil_from_met(met_pt, met_phi, muons, electrons):
    """Recoil = -(MET_vec + sum pT leptons). Computes lepton sums then delegates."""
    lep_px, lep_py = _lepton_pt_sums(muons, electrons)
    return _recoil_from_sums(met_pt, met_phi, lep_px, lep_py)


def calculate_recoil(events: ak.Array, objects: Dict[str, Any]):
    """Recoil = |-(MET_vec + sum pT(loose muons+electrons))|. Returns (recoil_pt, recoil_phi)."""
    def _met_field(*candidates):
        for v in candidates:
            if v in events.fields:
                return events[v]
        raise KeyError(f"No MET branch found among {candidates}")

    met_pt  = _met_field("PuppiMET_pt",  "PFMET_pt",  "MET_pt")
    met_phi = _met_field("PuppiMET_phi", "PFMET_phi", "MET_phi")
    muons     = objects.get("muons",     ak.Array([]))
    electrons = objects.get("electrons", ak.Array([]))
    return _recoil_from_met(met_pt, met_phi, muons, electrons)


def build_objects(events: ak.Array, config: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
    """
    Build all physics objects with selection and cleaning applied.

    Args:
        events: Awkward Array of events
        config: Full configuration dictionary
        verbose: When True, print per-step diagnostics. These call ak.sum/ak.num,
                 which force eager materialization of jagged arrays — off by default
                 to avoid that cost on full NanoAOD chunks.

    Returns:
        Dictionary containing selected objects and masks
    """
    if verbose:
        print("  Building object collections from flat branches...")

    # Select objects: loose for event selection, tight for region selection
    muon_mask_loose = select_muons(events, config["objects"]["muons"], wp="loose")
    muon_mask_tight = select_muons(events, config["objects"]["muons"], wp="tight")
    muon_mask = muon_mask_loose  # main collection = loose (event selection)

    electron_mask_loose = select_electrons(events, config["objects"]["electrons"], wp="loose")
    electron_mask_tight = select_electrons(events, config["objects"]["electrons"], wp="tight")
    electron_mask = electron_mask_loose

    tau_mask_loose = select_taus(events, config["objects"]["taus"], wp="loose")
    tau_mask_tight = select_taus(events, config["objects"]["taus"], wp="tight")
    tau_mask = tau_mask_loose

    photon_mask = select_photons(events, config["objects"]["photons"])
    photons = build_photon_collection(events)
    selected_photons = photons[photon_mask]

    jet_mask = select_jets(events, config["objects"]["jets"])

    if verbose:
        print(f"    Muons (loose): {ak.sum(muon_mask_loose)}, tight: {ak.sum(muon_mask_loose & muon_mask_tight)}")
        print(f"    Electrons (loose): {ak.sum(electron_mask_loose)}, tight: {ak.sum(electron_mask_loose & electron_mask_tight)}")
        print(f"    Taus (loose): {ak.sum(tau_mask_loose)}, tight: {ak.sum(tau_mask_loose & tau_mask_tight)}")
        print(f"    Photons (loose veto): {ak.sum(photon_mask)}")
        print(f"    Jets selected: {ak.sum(jet_mask)}")

    # Build collections via shared builders (single field-list source of truth)
    muons = build_muon_collection(events)
    electrons = build_electron_collection(events)
    taus = build_tau_collection(events)
    jets = build_jet_collection(events, config)

    # Apply masks: main collections = loose (for event selection, jet cleaning)
    selected_muons = muons[muon_mask]
    selected_electrons = electrons[electron_mask]
    selected_taus = taus[tau_mask]
    # Per-loose-lepton is_tight flag (for Z CR: leading tight, subleading loose)
    selected_muons = ak.with_field(selected_muons, muon_mask_tight[muon_mask_loose], "is_tight")
    selected_electrons = ak.with_field(selected_electrons, electron_mask_tight[electron_mask_loose], "is_tight")
    # Tight subsets (for region selection)
    # Tight subsets already carry pt > config["pt_min"] from select_*(wp="tight"),
    # so no extra pt re-filter is needed here. Thresholds are still read for the
    # per-flavor Z-candidate leading-lepton cut below.
    tight_muons = muons[muon_mask_loose & muon_mask_tight]
    tight_electrons = electrons[electron_mask_loose & electron_mask_tight]
    tight_taus = taus[tau_mask_loose & tau_mask_tight]
    mu_pt_min  = config["objects"]["muons"]["pt_min"]       # 30 GeV
    el_pt_min  = config["objects"]["electrons"]["pt_min"]   # 32 GeV

    # Z CR candidates: leading tight pt > mu/el pt_min, subleading loose pt > 10
    (n_z_muons, n_z_electrons, mll_mu, mll_el, z_pt_mu, z_pt_el) = build_z_candidates(
        selected_muons, selected_electrons,
        pt_lead_min_mu=mu_pt_min, pt_lead_min_el=el_pt_min, pt_sublead_min=10.0
    )

    selected_jets = jets[jet_mask]

    # Clean jets from muons, electrons, photons (taus vetoed, not used for cleaning)
    cleaning_objects = ak.concatenate([
        selected_muons[["eta", "phi"]],
        selected_electrons[["eta", "phi"]],
        selected_photons[["eta", "phi"]],
    ], axis=1)

    dr_jet = config["cleaning"].get("dr_jet", 0.4)
    jet_cleaning_mask = clean_jets_from_leptons(
        selected_jets,
        cleaning_objects,
        dr_jet
    )

    # Apply jet cleaning
    cleaned_jets = selected_jets[jet_cleaning_mask]

    # Get b-tagging mask for cleaned jets
    bjet_mask = get_bjet_mask(cleaned_jets, config["btagging"])
    selected_bjets = cleaned_jets[bjet_mask]

    if verbose:
        print(f"    Jets after cleaning: {ak.sum(ak.num(cleaned_jets, axis=1))}")
        print(f"    B-jets identified: {ak.sum(ak.num(selected_bjets, axis=1))}")

    # Compute central recoil + JES/JER shifted variants
    _lep_objs = {"muons": selected_muons, "electrons": selected_electrons}
    recoil, recoil_phi = calculate_recoil(events, _lep_objs)

    # Lepton pT sums are invariant across MET shifts — compute once, reuse for all
    # 4 JES/JER variants (only met_px/met_py differ per shift).
    _lep_px, _lep_py = _lepton_pt_sums(selected_muons, selected_electrons)

    def _shifted_recoil(pt_branch, phi_branch):
        if pt_branch in events.fields and phi_branch in events.fields:
            rpt, _ = _recoil_from_sums(events[pt_branch], events[phi_branch],
                                       _lep_px, _lep_py)
            return rpt
        return recoil  # fall back to central if branch missing

    recoil_JESUp   = _shifted_recoil("PuppiMET_ptJESUp",   "PuppiMET_phiJESUp")
    recoil_JESDown = _shifted_recoil("PuppiMET_ptJESDown",  "PuppiMET_phiJESDown")
    recoil_JERUp   = _shifted_recoil("PuppiMET_ptJERUp",   "PuppiMET_phiJERUp")
    recoil_JERDown = _shifted_recoil("PuppiMET_ptJERDown",  "PuppiMET_phiJERDown")

    # cos(theta*): helicity angle of leading jet in dijet CoM; SENTINEL when < 2 jets
    costheta_star = calculate_costheta_star(cleaned_jets)

    if verbose:
        print("  Object building complete!")
    return {
        "recoil": recoil,
        "recoil_phi": recoil_phi,
        "recoil_JESUp": recoil_JESUp,
        "recoil_JESDown": recoil_JESDown,
        "recoil_JERUp": recoil_JERUp,
        "recoil_JERDown": recoil_JERDown,
        "costheta_star": costheta_star,
        "photons": selected_photons,
        "muons": selected_muons,
        "electrons": selected_electrons,
        "taus": selected_taus,
        "tight_muons": tight_muons,
        "tight_electrons": tight_electrons,
        "tight_taus": tight_taus,
        "n_z_muons": n_z_muons,
        "n_z_electrons": n_z_electrons,
        "mll_mu": mll_mu,
        "mll_el": mll_el,
        "z_pt_mu": z_pt_mu,
        "z_pt_el": z_pt_el,
        "jets": cleaned_jets,
        "bjets": selected_bjets,
    }
