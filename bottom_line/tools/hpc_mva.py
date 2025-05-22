from typing import List, Optional, Tuple

import awkward as ak
import numpy
import xgboost as xgb
import logging

logger = logging.getLogger(__name__)


def calculate_ch_vs_ggh_mva(
    self,
    mva: Tuple[Tuple[Optional[xgb.Booster], Optional[xgb.Booster]], List[str]],
    diphotons: ak.Array,
    events: ak.Array,
) -> ak.Array:
    """
    Calculate cH vs ggH bdt scores for events.
    """

    if mva[0] is None:
        return diphotons, events
    elif len(diphotons) == 0:
        logger.info("no events surviving event selection, adding fake ch vs ggh bdt score")
        diphotons["ch_vs_ggh_bdt_score"] = ak.zeros_like(diphotons.mass)
        return diphotons, events

    ch_vs_ggh = []
    ch_vs_ggh.append(mva[0][0])
    ch_vs_ggh.append(mva[0][1])

    var_order = mva[1]

    events_bdt = events

    if self.analysis != "tagAndProbe":
        pho_lead = "pho_lead"
        pho_sublead = "pho_sublead"
    else:
        pho_lead = "tag"
        pho_sublead = "probe"

    events_bdt["customLeadingPhotonIDMVA"] = diphotons[pho_lead].mvaID
    events_bdt["customSubLeadingPhotonIDMVA"] = diphotons[pho_sublead].mvaID
    events_bdt["leadingPhoton_eta"] = diphotons[pho_lead].eta
    events_bdt["subleadingPhoton_eta"] = diphotons[pho_sublead].eta
    events_bdt["leadingPhoton_relpt"] = diphotons[pho_lead].pt / diphotons.mass
    events_bdt["subleadingPhoton_relpt"] = diphotons[pho_sublead].pt / diphotons.mass
    events_bdt["leadingJet_pt"] = diphotons.first_jet_pt
    events_bdt["leadingJet_eta"] = diphotons.first_jet_eta

    lead_jets = ak.zip(
        {
            "pt": diphotons.first_jet_pt,
            "eta": diphotons.first_jet_eta,
            "phi": diphotons.first_jet_phi,
            "mass": diphotons.first_jet_mass,
            "charge": diphotons.first_jet_charge
        }
    )
    lead_jets = ak.with_name(lead_jets, "PtEtaPhiMCandidate")

    lpj_dphi = diphotons[pho_lead].delta_phi(lead_jets)
    spj_dphi = diphotons[pho_sublead].delta_phi(lead_jets)

    events_bdt["DeltaPhi_gamma1_cjet"] = lpj_dphi
    events_bdt["DeltaPhi_gamma2_cjet"] = spj_dphi

    events_bdt["nJets_revised"] = ak.where(
        diphotons.n_jets > 3,
        ak.ones_like(diphotons.n_jets) * 3,
        diphotons.n_jets
    )

    for name in var_order:
        events_bdt[name] = ak.fill_none(events_bdt[name], -999.0)

    bdt_features = []
    for x in var_order:
        if isinstance(x, tuple):
            name_flat = "_".join(x)
            events_bdt[name_flat] = events_bdt[x]
            bdt_features.append(name_flat)
        else:
            bdt_features.append(x)

    events_bdt = ak.values_astype(events_bdt, numpy.float64)
    features_bdt = ak.to_numpy(events_bdt[bdt_features])

    features_bdt_matrix = xgb.DMatrix(
        features_bdt.view((float, len(features_bdt.dtype.names)))
    )

    scores = []
    for bdt in ch_vs_ggh:
        scores.append(bdt.predict(features_bdt_matrix))

    for var in bdt_features:
        if "dipho" not in var:
            diphotons[var] = events_bdt[var]

    scores_out = ak.where(
        events.event % 4 < 2,
        scores[0],
        scores[1]
    )

    diphotons["ch_vs_ggh_bdt_score"] = ak.ones_like(diphotons.mass)
    diphotons["ch_vs_ggh_bdt_score"] = scores_out

    return diphotons, events


def calculate_ch_vs_cb_mva(
    self,
    mva: Tuple[Tuple[Optional[xgb.Booster], Optional[xgb.Booster]], List[str]],
    diphotons: ak.Array,
    events: ak.Array,
) -> ak.Array:
    """
    Calculate cH vs ggH bdt scores for events.
    """

    if mva[0] is None:
        return diphotons, events
    elif len(diphotons) == 0:
        logger.info("no events surviving event selection, adding fake ch vs cb bdt score")
        diphotons["ch_vs_cb_bdt_score"] = ak.zeros_like(diphotons.mass)
        return diphotons, events

    ch_vs_cb = []
    ch_vs_cb.append(mva[0][0])
    ch_vs_cb.append(mva[0][1])
    var_order = mva[1]

    events_bdt = events

    if self.analysis != "tagAndProbe":
        pho_lead = "pho_lead"
        pho_sublead = "pho_sublead"
    else:
        pho_lead = "tag"
        pho_sublead = "probe"

    events_bdt["customLeadingPhotonIDMVA"] = diphotons[pho_lead].mvaID
    events_bdt["customSubLeadingPhotonIDMVA"] = diphotons[pho_sublead].mvaID
    events_bdt["leadingPhoton_eta"] = diphotons[pho_lead].eta
    events_bdt["subleadingPhoton_eta"] = diphotons[pho_sublead].eta
    events_bdt["leadingPhoton_relpt"] = diphotons[pho_lead].pt / diphotons.mass
    events_bdt["subleadingPhoton_relpt"] = diphotons[pho_sublead].pt / diphotons.mass
    events_bdt["leadingJet_pt"] = diphotons.first_jet_pt
    events_bdt["leadingJet_eta"] = diphotons.first_jet_eta

    lead_jets = ak.zip(
        {
            "pt": diphotons.first_jet_pt,
            "eta": diphotons.first_jet_eta,
            "phi": diphotons.first_jet_phi,
            "mass": diphotons.first_jet_mass,
            "charge": diphotons.first_jet_charge
        }
    )
    lead_jets = ak.with_name(lead_jets, "PtEtaPhiMCandidate")

    lpj_dphi = diphotons[pho_lead].delta_phi(lead_jets)
    spj_dphi = diphotons[pho_sublead].delta_phi(lead_jets)

    events_bdt["DeltaPhi_gamma1_cjet"] = lpj_dphi
    events_bdt["DeltaPhi_gamma2_cjet"] = spj_dphi

    events_bdt["nJets_revised"] = ak.where(
        diphotons.n_jets > 3,
        ak.ones_like(diphotons.n_jets) * 3,
        diphotons.n_jets
    )

    for name in var_order:
        events_bdt[name] = ak.fill_none(events_bdt[name], -999.0)

    bdt_features = []
    for x in var_order:
        if isinstance(x, tuple):
            name_flat = "_".join(x)
            events_bdt[name_flat] = events_bdt[x]
            bdt_features.append(name_flat)
        else:
            bdt_features.append(x)

    events_bdt = ak.values_astype(events_bdt, numpy.float64)
    features_bdt = ak.to_numpy(events_bdt[bdt_features])

    features_bdt_matrix = xgb.DMatrix(
        features_bdt.view((float, len(features_bdt.dtype.names)))
    )

    scores = []
    for bdt in ch_vs_cb:
        scores.append(bdt.predict(features_bdt_matrix))

    for var in bdt_features:
        if "dipho" not in var:
            diphotons[var] = events_bdt[var]

    scores_out = ak.where(
        events.event % 4 < 2,
        scores[0],
        scores[1]
    )

    diphotons["ch_vs_cb_bdt_score"] = ak.ones_like(diphotons.mass)
    diphotons["ch_vs_cb_bdt_score"] = scores_out

    return diphotons, events


def calculate_ggh_vs_hb_mva(
    self,
    mva: Tuple[Tuple[Optional[xgb.Booster], Optional[xgb.Booster]], List[str]],
    diphotons: ak.Array,
    events: ak.Array,
) -> ak.Array:
    """
    Calculate cH vs ggH bdt scores for events.
    """

    if mva[0] is None:
        return diphotons, events
    elif len(diphotons) == 0:
        logger.info("no events surviving event selection, adding fake ggh vs hb bdt score")
        diphotons["ggh_vs_hb_bdt_score"] = ak.zeros_like(diphotons.mass)
        return diphotons, events

    ggh_vs_hb = []
    ggh_vs_hb.append(mva[0][0])
    ggh_vs_hb.append(mva[0][1])
    var_order = mva[1]

    events_bdt = events

    if self.analysis != "tagAndProbe":
        pho_lead = "pho_lead"
        pho_sublead = "pho_sublead"
    else:
        pho_lead = "tag"
        pho_sublead = "probe"

    events_bdt["pt"] = diphotons.pt
    events_bdt["eta"] = diphotons.eta
    events_bdt["dijet_pt"] = diphotons.dijet_pt
    events_bdt["dijet_eta"] = diphotons.dijet_eta
    events_bdt["dijet_mass"] = diphotons.dijet_mass
    events_bdt["LeadPhoton_eta"] = diphotons[pho_lead].eta
    events_bdt["SubleadPhoton_eta"] = diphotons[pho_sublead].eta
    events_bdt["LeadPhoton_pt_mgg"] = diphotons[pho_lead].pt / diphotons.mass
    events_bdt["SubleadPhoton_pt_mgg"] = diphotons[pho_sublead].pt / diphotons.mass
    events_bdt["first_jet_pt"] = diphotons.first_jet_pt
    events_bdt["first_jet_eta"] = diphotons.first_jet_eta
    events_bdt["first_jet_mass"] = diphotons.first_jet_mass
    events_bdt["second_jet_pt"] = diphotons.second_jet_pt
    events_bdt["second_jet_eta"] = diphotons.second_jet_eta
    events_bdt["second_jet_mass"] = diphotons.second_jet_mass
    events_bdt["third_jet_pt"] = diphotons.third_jet_pt
    events_bdt["third_jet_eta"] = diphotons.third_jet_eta
    events_bdt["third_jet_mass"] = diphotons.third_jet_mass
    events_bdt["MET_sumEt"] = diphotons.MET_sumEt
    events_bdt["MET_pt"] = diphotons.MET_pt
    events_bdt["MET_phi"] = diphotons.MET_phi
    events_bdt["MET_significance"] = diphotons.MET_significance
    events_bdt["first_muon_pt"] = diphotons.first_muon_pt
    events_bdt["first_muon_eta"] = diphotons.first_muon_eta
    events_bdt["first_electron_pt"] = diphotons.first_electron_pt
    events_bdt["first_electron_eta"] = diphotons.first_electron_eta
    events_bdt["nMuon"] = diphotons.nMuon
    events_bdt["nElectron"] = diphotons.nElectron
    events_bdt["nTau"] = diphotons.nTau
    events_bdt["n_jets"] = diphotons.n_jets
    events_bdt["n_b_jets_medium"] = diphotons.n_b_jets_medium
    events_bdt["n_b_jets_loose"] = diphotons.n_b_jets_loose

    lead_jets = ak.zip(
        {
            "pt": diphotons.first_jet_pt,
            "eta": diphotons.first_jet_eta,
            "phi": diphotons.first_jet_phi,
            "mass": diphotons.first_jet_mass,
            "charge": diphotons.first_jet_charge
        }
    )
    lead_jets = ak.with_name(lead_jets, "PtEtaPhiMCandidate")

    lead_pt_jets = ak.zip(
        {
            "pt": diphotons.first_pt_jet_pt,
            "eta": diphotons.first_pt_jet_eta,
            "phi": diphotons.first_pt_jet_phi,
            "mass": diphotons.first_pt_jet_mass,
            "charge": diphotons.first_pt_jet_charge
        }
    )
    lead_pt_jets = ak.with_name(lead_pt_jets, "PtEtaPhiMCandidate")

    sublead_pt_jets = ak.zip(
        {
            "pt": diphotons.second_pt_jet_pt,
            "eta": diphotons.second_pt_jet_eta,
            "phi": diphotons.second_pt_jet_phi,
            "mass": diphotons.second_pt_jet_mass,
            "charge": diphotons.second_pt_jet_charge
        }
    )
    sublead_pt_jets = ak.with_name(sublead_pt_jets, "PtEtaPhiMCandidate")

    subsublead_jets = ak.zip(
        {
            "pt": diphotons.third_jet_pt,
            "eta": diphotons.third_jet_eta,
            "phi": diphotons.third_jet_phi,
            "mass": diphotons.third_jet_mass,
            "charge": diphotons.third_jet_charge
        }
    )
    subsublead_jets = ak.with_name(subsublead_jets, "PtEtaPhiMCandidate")

    lpj_dphi = diphotons[pho_lead].delta_phi(lead_pt_jets)
    spj_dphi = diphotons[pho_sublead].delta_phi(lead_pt_jets)

    dEta_ljh = abs(diphotons.eta - lead_pt_jets.eta)
    dEta_sljh = abs(diphotons.eta - sublead_pt_jets.eta)
    dEta_ljslj = abs(lead_pt_jets.eta - sublead_pt_jets.eta)

    dR_ljlp = lead_jets.delta_r(diphotons[pho_lead])

    lj_ptoM = lead_jets.pt / lead_jets.mass

    events_bdt["DeltaPhi_gamma1_cjet"] = lpj_dphi
    events_bdt["DeltaPhi_gamma2_cjet"] = spj_dphi

    events_bdt["dEta_ljh"] = dEta_ljh
    events_bdt["dEta_sljh"] = dEta_sljh
    events_bdt["dEta_ljslj"] = dEta_ljslj

    events_bdt["dR_ljlp"] = dR_ljlp
    events_bdt["lj_ptoM"] = lj_ptoM

    events_bdt["dEta_ljslj"] = ak.where(
        events_bdt["second_jet_eta"] != -999.,
        events_bdt["dEta_ljslj"],
        ak.ones_like(events_bdt["dEta_ljslj"]) * -1
    )
    events_bdt["dEta_sljh"] = ak.where(
        events_bdt["second_jet_eta"] != -999.,
        events_bdt["dEta_sljh"],
        ak.ones_like(events_bdt["dEta_sljh"]) * -1
    )
    events_bdt["lj_ptoM"] = ak.where(
        events_bdt["lj_ptoM"] > 50000,
        ak.ones_like(events_bdt.lj_ptoM) * 50000,
        events_bdt["lj_ptoM"]
    )
    events_bdt["lj_ptoM"] = ak.where(
        events_bdt["lj_ptoM"] < -1.,
        ak.ones_like(events_bdt.lj_ptoM) * -1,
        events_bdt["lj_ptoM"]
    )
    events_bdt["first_muon_pt"] = ak.where(
        events_bdt["first_muon_pt"] < 0,
        ak.ones_like(events_bdt.first_muon_pt) * -1,
        events_bdt["first_muon_pt"]
    )
    events_bdt["first_electron_pt"] = ak.where(
        events_bdt["first_muon_pt"] < 0,
        ak.ones_like(events_bdt.first_muon_pt) * -1,
        events_bdt["first_muon_pt"]
    )

    for name in var_order:
        events_bdt[name] = ak.fill_none(events_bdt[name], -999.0)

    bdt_features = []
    for x in var_order:
        if isinstance(x, tuple):
            name_flat = "_".join(x)
            events_bdt[name_flat] = events_bdt[x]
            bdt_features.append(name_flat)
        else:
            bdt_features.append(x)

    events_bdt = ak.values_astype(events_bdt, numpy.float64)
    features_bdt = ak.to_numpy(events_bdt[bdt_features])

    features_bdt_matrix = xgb.DMatrix(
        features_bdt.view((float, len(features_bdt.dtype.names))), feature_names=bdt_features
    )

    scores = []
    for bdt in ggh_vs_hb:
        scores.append(bdt.predict(features_bdt_matrix))

    for var in bdt_features:
        if "dipho" not in var:
            diphotons[var] = events_bdt[var]

    scores_out_sig = ak.where(
        events.event % 4 < 2,
        scores[0][:, 0],
        scores[1][:, 0]
    )
    scores_out_tth = ak.where(
        events.event % 4 < 2,
        scores[0][:, 1],
        scores[1][:, 1]
    )
    scores_out_vbf = ak.where(
        events.event % 4 < 2,
        scores[0][:, 2],
        scores[1][:, 2]
    )
    scores_out_vh = ak.where(
        events.event % 4 < 2,
        scores[0][:, 3],
        scores[1][:, 3]
    )

    diphotons["ggh_vs_hb_bdt_sig_score"] = ak.ones_like(diphotons.mass)
    diphotons["ggh_vs_hb_bdt_tth_score"] = ak.ones_like(diphotons.mass)
    diphotons["ggh_vs_hb_bdt_vbf_score"] = ak.ones_like(diphotons.mass)
    diphotons["ggh_vs_hb_bdt_vh_score"] = ak.ones_like(diphotons.mass)
    diphotons["ggh_vs_hb_bdt_sig_score"] = scores_out_sig
    diphotons["ggh_vs_hb_bdt_tth_score"] = scores_out_tth
    diphotons["ggh_vs_hb_bdt_vbf_score"] = scores_out_vbf
    diphotons["ggh_vs_hb_bdt_vh_score"] = scores_out_vh

    return diphotons, events
