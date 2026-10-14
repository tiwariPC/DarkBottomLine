"""
Physics object selection and cleaning functions for DarkBottomLine framework.
"""

import awkward as ak
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple

from darkbottomline.corrections import jes_sources

SENTINEL = -9999.0  # fill value for variables undefined due to insufficient objects


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


def build_jet_collection(events: ak.Array, config: Dict[str, Any], corrections: Any = None) -> ak.Array:
    """Zip the AK4 jet collection. btagScore comes from the YAML-configured
    branch; hadronFlavour (MC) added when present.

    Args:
        corrections: Optional CorrectionManager. When given and
            config.corrections.jetJEC is configured, pt/mass are the
            properly-rescaled JEC values (CorrectionManager.
            get_corrected_jet_pt_mass: undo NanoAOD's embedded JEC via
            Jet_rawFactor, reapply the configured JEC from scratch — see
            that method's docstring). Omit (None) or unconfigured -> raw
            NanoAOD Jet_pt/Jet_mass, same behavior as before this
            correction existed.
    """
    pt = events["Jet_pt"]
    mass = events["Jet_mass"] if "Jet_mass" in events.fields else ak.zeros_like(events["Jet_pt"])
    if corrections is not None and hasattr(corrections, "get_corrected_jet_pt_mass"):
        result = corrections.get_corrected_jet_pt_mass(events)
        if result is not None:
            pt, mass = result
    fields = {
        "pt": pt,
        "eta": events["Jet_eta"],
        "phi": events["Jet_phi"],
        "mass": mass,
        "btagScore": events[config["btagging"]["branch"]],
    }
    if "Jet_hadronFlavour" in events.fields:
        fields["hadronFlavour"] = events["Jet_hadronFlavour"]
    elif "Jet_partonFlavour" in events.fields:
        fields["hadronFlavour"] = events["Jet_partonFlavour"]
    # Raw-level inputs needed to re-derive/audit JEC, jet-ID and Type-1 MET
    # from EVENTSELECTION.root later WITHOUT rerunning from raw NanoAOD
    # (these are not otherwise saved anywhere downstream — see
    # darkbottomline/variables.py's _jagged_variables, which reads whatever
    # fields exist on this "jets" object). Carried through unchanged
    # regardless of the JEC correction's own corrected pt/mass above, so the
    # saved raw_pt/rawFactor etc. always reflect the true NanoAOD input.
    for raw_field, src_branch in (
        ("rawFactor", "Jet_rawFactor"),
        ("area", "Jet_area"),
        ("muonSubtrFactor", "Jet_muonSubtrFactor"),
        ("muonSubtrDeltaPhi", "Jet_muonSubtrDeltaPhi"),
        ("chHEF", "Jet_chHEF"),
        ("neHEF", "Jet_neHEF"),
        ("chEmEF", "Jet_chEmEF"),
        ("neEmEF", "Jet_neEmEF"),
        ("muEF", "Jet_muEF"),
        ("chMultiplicity", "Jet_chMultiplicity"),
        ("neMultiplicity", "Jet_neMultiplicity"),
    ):
        if src_branch in events.fields:
            fields[raw_field] = events[src_branch]
    return ak.zip(fields)


def build_photon_collection(events: ak.Array) -> ak.Array:
    """Zip the photon collection (veto object): pt/eta/phi/cutBased/electronVeto."""
    return ak.zip({
        "pt": events["Photon_pt"],
        "eta": events["Photon_eta"],
        "phi": events["Photon_phi"],
        "cutBased": events["Photon_cutBased"],
        "electronVeto": events["Photon_electronVeto"],
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
    """Select photons for veto: loose ID, pt > 15, |eta| < 2.5, pixel-seed
    electron veto (rejects electrons reconstructed as photon candidates).

    Mask reads flat branches directly (like select_jets) — no zip needed here."""
    pt_mask = events["Photon_pt"] > config["pt_min"]
    eta_mask = abs(events["Photon_eta"]) < config["eta_max"]
    id_mask = events["Photon_cutBased"] >= config["id_wp_loose"]
    ele_veto_mask = events["Photon_electronVeto"] == 1
    return pt_mask & eta_mask & id_mask & ele_veto_mask


def select_jets(events: ak.Array, config: Dict[str, Any], corrections: Any = None) -> ak.Array:
    """
    Select AK4 jets based on configuration cuts.

    Args:
        events: Awkward Array of events
        config: Configuration dictionary with jet selection cuts
        corrections: Optional CorrectionManager. When given:
          - config.corrections.jetJEC configured -> the pt_min cut uses the
            properly JEC-rescaled pt (CorrectionManager.
            get_corrected_jet_pt_mass), the SAME pt build_jet_collection
            zips into the returned jet collection — otherwise jets near the
            threshold would never reflect the correction. Falls back to raw
            Jet_pt if unconfigured/unavailable.
          - config.corrections.jetid configured -> applies the BTV tight(+
            lepton-veto) jet-ID correction (CorrectionManager.get_jet_id_mask)
            on top of pt/eta. Jet-ID's energy-fraction inputs are read
            straight from NanoAOD (ratios, invariant under the pt/mass
            rescaling above) — no interaction between the two corrections.
        Left out (None) or unconfigured -> pt/eta only on raw Jet_pt, same
        behavior as before these corrections were wired in.

    Returns:
        Boolean mask for selected jets
    """
    # btagScore/hadronFlavour are added later in build_jet_collection.
    jet_pt = events["Jet_pt"]
    if corrections is not None and hasattr(corrections, "get_corrected_jet_pt_mass"):
        result = corrections.get_corrected_jet_pt_mass(events)
        if result is not None:
            jet_pt = result[0]
    pt_mask = jet_pt > config["pt_min"]
    eta_mask = abs(events["Jet_eta"]) < config["eta_max"]
    mask = pt_mask & eta_mask

    if corrections is not None and hasattr(corrections, "get_jet_id_mask"):
        id_mask = corrections.get_jet_id_mask(events)
        if id_mask is not None:
            mask = mask & id_mask

    return mask


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
) -> Tuple[ak.Array, ak.Array, ak.Array, ak.Array, ak.Array, ak.Array, ak.Array, ak.Array]:
    """
    Build Z->ll candidates for Z CR: 2 opposite-sign leptons.
    Leading: tight ID, pt > per-flavor threshold (mu 30, el 32 GeV).
    Subleading: loose ID, pt > pt_sublead_min (10 GeV).

    Returns:
        n_z_muons, n_z_electrons: 2 if valid Z candidate else 0 per event
        mll_mu, mll_el: invariant mass of the candidate pair (SENTINEL if none)
        z_pt_mu, z_pt_el: pT of the candidate dilepton system (SENTINEL if none)
        z_sublead_muons, z_sublead_electrons: jagged (0-or-1-per-event) pt/eta
            of the valid Z candidate's subleading (loose-only) lepton, so its
            ID scale factor can be applied — it is otherwise never in
            tight_muons/tight_electrons and would receive no SF at all.
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
    ) -> Tuple[ak.Array, ak.Array, ak.Array, ak.Array]:
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
        sub_eta   = ak.fill_none(sublead.eta, 0.0)
        z_px = lead_pt * np.cos(lead_phi) + sub_pt * np.cos(sub_phi)
        z_py = lead_pt * np.sin(lead_phi) + sub_pt * np.sin(sub_phi)
        z_pt = ak.where(valid, np.sqrt(z_px**2 + z_py**2), SENTINEL)
        # Jagged (0-or-1-per-event) sublead-lepton pt/eta, kept only for valid Z
        # candidates, so its ID scale factor can be folded into the event weight
        # (the sublead lepton is loose-only and otherwise never receives an SF).
        sublead_for_sf = ak.zip({
            "pt": ak.singletons(ak.mask(sub_pt, valid)),
            "eta": ak.singletons(ak.mask(sub_eta, valid)),
        })
        return n_z, mll, z_pt, sublead_for_sf

    _no_sublead = ak.zip({
        "pt": ak.singletons(ak.mask(ak.Array(np.zeros(n_ev)), ak.Array(np.zeros(n_ev, dtype=bool)))),
        "eta": ak.singletons(ak.mask(ak.Array(np.zeros(n_ev)), ak.Array(np.zeros(n_ev, dtype=bool)))),
    })
    z_sublead_muons = _no_sublead
    z_sublead_electrons = _no_sublead
    if len(ak.flatten(loose_muons)) > 0 and hasattr(loose_muons, "is_tight"):
        n_z_muons, mll_mu, z_pt_mu, z_sublead_muons = _one_flavor(loose_muons, True)
    if len(ak.flatten(loose_electrons)) > 0 and hasattr(loose_electrons, "is_tight"):
        n_z_electrons, mll_el, z_pt_el, z_sublead_electrons = _one_flavor(loose_electrons, False)

    return (
        n_z_muons, n_z_electrons, mll_mu, mll_el, z_pt_mu, z_pt_el,
        z_sublead_muons, z_sublead_electrons,
    )


def build_transverse_mass(
    tight_muons: ak.Array,
    tight_electrons: ak.Array,
    met_pt: ak.Array,
    met_phi: ak.Array,
) -> Tuple[ak.Array, ak.Array]:
    """
    W transverse mass per flavor: leading tight lepton vs MET.
    SENTINEL if no tight lepton of that flavor.

    Returns:
        mt_mu, mt_el: per-event MT for muon/electron channel (SENTINEL if none)
    """
    def _one_flavor(tight_lep: ak.Array) -> ak.Array:
        has_lep = ak.num(tight_lep, axis=1) > 0
        lead = ak.firsts(tight_lep)
        lpt  = ak.fill_none(lead.pt,  0.0)
        lphi = ak.fill_none(lead.phi, 0.0)
        dphi = np.abs(lphi - met_phi)
        dphi = ak.where(dphi > np.pi, 2 * np.pi - dphi, dphi)
        mt = np.sqrt(2 * lpt * met_pt * (1 - np.cos(dphi)))
        return ak.where(has_lep, mt, SENTINEL)

    return _one_flavor(tight_muons), _one_flavor(tight_electrons)


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


def _met_field(events: ak.Array, *candidates):
    for v in candidates:
        if v in events.fields:
            return events[v]
    raise KeyError(f"No MET branch found among {candidates}")


def _type1_met(events: ak.Array, corrections: Any = None,
               jes_source: Optional[str] = None, jes_direction: Optional[str] = None):
    """(met_pt, met_phi): Type-1 MET recomputed from our corrected jets.

    The Type-1 shift must be added to the RAW MET (RawPuppiMET):
    NanoAOD's PuppiMET is already Type-1 corrected, so adding the shift to
    it applies the jet correction twice (~+9 GeV mean, ~2x the events with
    MET>250 on 2024 test files). Without RawPuppiMET_* or a usable shift,
    returns the stored MET unchanged.
    """
    met_pt  = _met_field(events, "PuppiMET_pt",  "PFMET_pt",  "MET_pt")
    met_phi = _met_field(events, "PuppiMET_phi", "PFMET_phi", "MET_phi")
    if corrections is None or not hasattr(corrections, "get_type1_met_shift"):
        return met_pt, met_phi
    if "RawPuppiMET_pt" not in events.fields or "RawPuppiMET_phi" not in events.fields:
        logging.warning("Type-1 MET: no RawPuppiMET_pt/phi branches — keeping stored PuppiMET "
                        "(the shift cannot be applied to an already Type-1 corrected MET)")
        return met_pt, met_phi
    shift = corrections.get_type1_met_shift(events, jes_source=jes_source, jes_direction=jes_direction)
    if shift is None:
        return met_pt, met_phi
    delta_met_x, delta_met_y = shift
    raw_pt = np.asarray(ak.to_numpy(events["RawPuppiMET_pt"]), dtype=float)
    raw_phi = np.asarray(ak.to_numpy(events["RawPuppiMET_phi"]), dtype=float)
    met_x = raw_pt * np.cos(raw_phi) + delta_met_x
    met_y = raw_pt * np.sin(raw_phi) + delta_met_y
    return ak.Array(np.sqrt(met_x**2 + met_y**2)), ak.Array(np.arctan2(met_y, met_x))


def calculate_recoil(events: ak.Array, objects: Dict[str, Any], corrections: Any = None):
    """Recoil = |-(MET_vec + sum pT(loose muons+electrons))|. Returns (recoil_pt, recoil_phi).

    When corrections is given and has get_type1_met_shift, the stored
    PuppiMET_pt/_phi (built by NanoAOD production from its own embedded,
    possibly-stale JEC payload) is first updated via the Type-1 MET
    recompute (Type1MET.pdf Sec 1.2-1.5: vector-sum jet pt shift from the
    SAME corrected jets used elsewhere in this pipeline) before computing
    recoil — keeping jets and MET on a consistent JEC payload, per the
    official recommendation. Falls back to the stored MET unchanged if
    unconfigured/unavailable.
    """
    met_pt, met_phi = _type1_met(events, corrections)
    muons     = objects.get("muons",     ak.Array([]))
    electrons = objects.get("electrons", ak.Array([]))
    return _recoil_from_met(met_pt, met_phi, muons, electrons)


def build_objects(events: ak.Array, config: Dict[str, Any], verbose: bool = False,
                   corrections: Any = None) -> Dict[str, Any]:
    """
    Build all physics objects with selection and cleaning applied.

    Args:
        events: Awkward Array of events
        config: Full configuration dictionary
        verbose: When True, print per-step diagnostics. These call ak.sum/ak.num,
                 which force eager materialization of jagged arrays — off by default
                 to avoid that cost on full NanoAOD chunks.
        corrections: Optional CorrectionManager, forwarded to select_jets
            (jet-ID cut, config.corrections.jetid; JEC-corrected pt for the
            pt_min cut, config.corrections.jetJEC), build_jet_collection
            (same JEC-corrected pt/mass for the returned jet collection —
            kept consistent with select_jets's mask), and calculate_recoil
            (Type-1 MET recompute from the same corrected jets, so jets and
            MET stay on the same JEC payload). Omit to keep pt/eta-only jet
            selection and NanoAOD's stored MET unchanged (pre-existing
            behavior).

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

    jet_mask = select_jets(events, config["objects"]["jets"], corrections=corrections)

    if verbose:
        print(f"    Muons (loose): {ak.sum(muon_mask_loose)}, tight: {ak.sum(muon_mask_loose & muon_mask_tight)}")
        print(f"    Electrons (loose): {ak.sum(electron_mask_loose)}, tight: {ak.sum(electron_mask_loose & electron_mask_tight)}")
        print(f"    Taus (loose): {ak.sum(tau_mask_loose)}, tight: {ak.sum(tau_mask_loose & tau_mask_tight)}")
        print(f"    Jets selected: {ak.sum(jet_mask)}")

    # Build collections via shared builders (single field-list source of truth)
    muons = build_muon_collection(events)
    electrons = build_electron_collection(events)
    taus = build_tau_collection(events)
    jets = build_jet_collection(events, config, corrections=corrections)

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
    (
        n_z_muons, n_z_electrons, mll_mu, mll_el, z_pt_mu, z_pt_el,
        z_sublead_muons, z_sublead_electrons,
    ) = build_z_candidates(
        selected_muons, selected_electrons,
        pt_lead_min_mu=mu_pt_min, pt_lead_min_el=el_pt_min, pt_sublead_min=10.0
    )

    # W transverse mass per flavor: leading tight lepton vs MET
    _met_pt_mt  = _met_field(events, "PuppiMET_pt",  "PFMET_pt",  "MET_pt")
    _met_phi_mt = _met_field(events, "PuppiMET_phi", "PFMET_phi", "MET_phi")
    mt_mu, mt_el = build_transverse_mass(tight_muons, tight_electrons, _met_pt_mt, _met_phi_mt)

    # Photon veto collection: pixel-seed veto in select_photons, then ΔR-cleaned
    # against loose electrons AND muons (Run 2 parity: SkimTree cleanedPho_ag_ele/ag_mu).
    # Without this, every electron is double-counted as a photon candidate and
    # max_photons=0 vetoes W→eν / Z→ee events.
    photon_mask = select_photons(events, config["objects"]["photons"])
    photons = build_photon_collection(events)
    selected_photons = photons[photon_mask]
    dr_photon_lep = config["cleaning"]["dr_photon_lep"]
    photon_clean_mask = clean_jets_from_leptons(
        selected_photons,
        ak.concatenate([
            selected_muons[["eta", "phi"]],
            selected_electrons[["eta", "phi"]],
        ], axis=1),
        dr_photon_lep,
    )
    selected_photons = selected_photons[photon_clean_mask]
    if verbose:
        print(f"    Photons (loose veto, lepton-cleaned): {ak.sum(ak.num(selected_photons, axis=1))}")

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
    # Type-1 MET computed once here and reused for recoil, the stored MET
    # branches and the dPhi(jet, MET) cut, so all of them see the same MET.
    met_pt, met_phi = _type1_met(events, corrections)
    recoil, recoil_phi = _recoil_from_met(met_pt, met_phi, selected_muons, selected_electrons)

    # Lepton pT sums are invariant across MET shifts — compute once, reuse for all
    # 4 JES/JER variants (only met_px/met_py differ per shift).
    _lep_px, _lep_py = _lepton_pt_sums(selected_muons, selected_electrons)

    def _shifted_recoil(pt_branch, phi_branch):
        if pt_branch in events.fields and phi_branch in events.fields:
            rpt, _ = _recoil_from_sums(events[pt_branch], events[phi_branch],
                                       _lep_px, _lep_py)
            return rpt
        return recoil  # fall back to central if branch missing

    # JER systematic: deferred (JER smearing itself isn't implemented — see
    # CorrectionManager.get_corrected_jet_pt_mass's TODO(JER)). Stays on
    # NanoAOD's own old-payload PuppiMET_ptJERUp/Down branches — a known,
    # documented inconsistency with the now JEC-corrected central jets/MET,
    # not silently fixed.
    recoil_JERUp   = _shifted_recoil("PuppiMET_ptJERUp",   "PuppiMET_phiJERUp")
    recoil_JERDown = _shifted_recoil("PuppiMET_ptJERDown",  "PuppiMET_phiJERDown")

    # JES systematic: reduced 11-source scheme (replaces the old single
    # "Total"-uncertainty PuppiMET_ptJESUp/Down pair). Each source x
    # direction needs its own full shifted-jet rebuild (new corrected pt via
    # get_corrected_jet_pt_mass with that source applied, re-selected/
    # re-cleaned, then a self-consistent Type-1 MET from those shifted jets)
    # — NOT just a MET-level px/py shift like the JER branches above, since
    # the whole point of a per-source JES systematic is that jet pt (and
    # therefore njets/n_bjets/cleaning) can move, not just MET.
    _jes_year = config.get("year", "")
    _jes_source_names = jes_sources(_jes_year)

    def _jes_label(source: str, direction: str) -> str:
        # jes_sources() already gives the real-year-suffixed name (e.g.
        # "EC2_2024") — just prefix/suffix for the branch-name convention
        # (-> "recoil_JECEC2_2024Up"). No further munging needed.
        return f"recoil_JEC{source}{direction.capitalize()}"

    recoil_jes_variants: Dict[str, ak.Array] = {}
    if corrections is not None and hasattr(corrections, "get_type1_met_shift"):
        for source in _jes_source_names:
            for direction in ("up", "down"):
                label = _jes_label(source, direction)
                shifted_met_pt, shifted_met_phi = _type1_met(
                    events, corrections, jes_source=source, jes_direction=direction,
                )
                rpt, _ = _recoil_from_sums(shifted_met_pt, shifted_met_phi, _lep_px, _lep_py)
                recoil_jes_variants[label] = rpt
    else:
        for source in _jes_source_names:
            for direction in ("up", "down"):
                recoil_jes_variants[_jes_label(source, direction)] = recoil

    # cos(theta*): helicity angle of leading jet in dijet CoM; SENTINEL when < 2 jets
    costheta_star = calculate_costheta_star(cleaned_jets)

    if verbose:
        print("  Object building complete!")
    return {
        "met_pt": met_pt,
        "met_phi": met_phi,
        "recoil": recoil,
        "recoil_phi": recoil_phi,
        "recoil_JERUp": recoil_JERUp,
        "recoil_JERDown": recoil_JERDown,
        **recoil_jes_variants,
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
        "z_sublead_muons": z_sublead_muons,
        "z_sublead_electrons": z_sublead_electrons,
        "mll_mu": mll_mu,
        "mll_el": mll_el,
        "z_pt_mu": z_pt_mu,
        "z_pt_el": z_pt_el,
        "mt_mu": mt_mu,
        "mt_el": mt_el,
        "jets": cleaned_jets,
        "bjets": selected_bjets,
        "btag_score": config["btagging"]["score"],
    }
