from bottom_line.selections.object_selections import delta_r_mask
import awkward as ak
import correctionlib
import os
from coffea.analysis_tools import PackedSelection
from copy import deepcopy
import numpy as np
from correctionlib.highlevel import model_auto, open_auto
import json
import logging

logger = logging.getLogger(__name__)


def jetIdFlags_v1213(jets, nano_version):
    abs_eta = abs(jets.eta)

    if nano_version == 12:
        # Default tight
        passJetIdTight = ak.where(
            abs_eta <= 2.7,
            (jets.jetId & (1 << 1)) > 0,  # Tight criteria for abs_eta <= 2.7
            ak.where(
                (abs_eta > 2.7) & (abs_eta <= 3.0),
                ((jets.jetId & (1 << 1)) > 0) & (jets.neHEF < 0.99),  # Tight criteria for 2.7 < abs_eta <= 3.0
                ((jets.jetId & (1 << 1)) > 0) & (jets.neEmEF < 0.4)  # Tight criteria for 3.0 < abs_eta
            )
        )

        # Default tight lepton veto
        passJetIdTightLepVeto = ak.where(
            abs_eta <= 2.7,
            passJetIdTight & (jets.muEF < 0.8) & (jets.chEmEF < 0.8),  # add lepton veto for abs_eta <= 2.7
            passJetIdTight  # No lepton veto for 2.7 < abs_eta
        )
    else:
        # Default tight for NanoAOD version 13
        passJetIdTight = ak.where(
            abs_eta <= 2.6,
            (jets.neHEF < 0.99)
            & (jets.neEmEF < 0.9)
            & (jets.chMultiplicity + jets.neMultiplicity > 1)
            & (jets.chHEF > 0.01)
            & (jets.chMultiplicity > 0),  # Tight criteria for abs_eta <= 2.6
            ak.where(
                (abs_eta > 2.6) & (abs_eta <= 2.7),
                (jets.neHEF < 0.9) & (jets.neEmEF < 0.99),  # Tight criteria for 2.6 < abs_eta <= 2.7
                ak.where(
                    (abs_eta > 2.7) & (abs_eta <= 3.0),
                    jets.neHEF < 0.99,  # Tight criteria for 2.7 < abs_eta <= 3.0
                    (jets.neMultiplicity >= 2) & (jets.neEmEF < 0.4)  # Tight criteria for abs_eta > 3.0
                )
            )
        )

        # Default tight lepton veto
        passJetIdTightLepVeto = ak.where(
            abs_eta <= 2.7,
            passJetIdTight & (jets.muEF < 0.8) & (jets.chEmEF < 0.8),  # add lepton veto for abs_eta <= 2.7
            passJetIdTight  # No lepton veto for 2.7 < abs_eta
        )

    return passJetIdTight, passJetIdTightLepVeto


def getBTagMVACut(mva_name, mva_wp, year):
    mva_name_to_btag_wp_name = {
        "particleNet": "particleNet_wp_values",
        "deepJet": "deepJet_wp_values",
        "robustParticleTransformer": "robustParticleTransformer_wp_values"
    }

    # Based on recommendations for the tight QCD WP seen here: https://btv-wiki.docs.cern.ch/PerformanceCalibration/#working-points
    btag_correction_configs = {
        "2016preVFP": {
            "file": os.path.join(os.path.dirname(__file__), "..", "systematics", "JSONs", "bTagSF", "2016preVFP_UL/btagging.json.gz")
        },
        "2016postVFP": {
            "file": os.path.join(os.path.dirname(__file__), "..", "systematics", "JSONs", "bTagSF", "2016postVFP_UL/btagging.json.gz")
        },
        "2017": {
            "file": os.path.join(os.path.dirname(__file__), "..", "systematics", "JSONs", "bTagSF", "2017_UL/btagging.json.gz")
        },
        "2018": {
            "file": os.path.join(os.path.dirname(__file__), "..", "systematics", "JSONs", "bTagSF", "2018_UL/btagging.json.gz")
        },
        "2022preEE":{
            "file": os.path.join(os.path.dirname(__file__), "..", "systematics", "JSONs", "bTagSF", "2022_Summer22/btagging.json.gz")
        },
        "2022postEE":{
            "file": os.path.join(os.path.dirname(__file__), "..", "systematics", "JSONs", "bTagSF", "2022_Summer22EE/btagging.json.gz")
        },
        "2023preBPix":{
            "file": os.path.join(os.path.dirname(__file__), "..", "systematics", "JSONs", "bTagSF", "2023_Summer23/btagging.json.gz")
        },
        "2023postBPix":{
            "file": os.path.join(os.path.dirname(__file__), "..", "systematics", "JSONs", "bTagSF", "2023_Summer23BPix/btagging.json.gz")
        },
    }
    avail_years = ["2016preVFP", "2016postVFP", "2017", "2018", "2022preEE", "2022postEE", "2023preBPix", "2023postBPix"]
    if year not in avail_years:
        logger.warning(f"\n BTV correctionlib for {year} not found! Don't cut on the selected B-Tag MVA. The b-related variables are most likely not correct.\n")
        return -999.0

    mva_cut_value = correctionlib.CorrectionSet.from_file(btag_correction_configs[year]['file'])[mva_name_to_btag_wp_name[mva_name]].evaluate(mva_wp)

    return mva_cut_value


def select_jets(
    self,
    jets: ak.highlevel.Array,
    diphotons: ak.highlevel.Array,
    muons: ak.highlevel.Array,
    electrons: ak.highlevel.Array,
    taus: ak.highlevel.Array = None,
) -> ak.highlevel.Array:
    # jet id selection: https://twiki.cern.ch/twiki/bin/view/CMS/JetID13p6TeV#nanoAOD_Flags
    if (self.nano_version == 12) or (self.nano_version == 13):
        passJetIdTight, passJetIdTightLepVeto = jetIdFlags_v1213(jets, self.nano_version)
        if self.jet_jetId == "tight":  # Select jetId 2 or 6
            logger.info("Applying jetID recipe of NanoAOD version %s", self.nano_version)
            jetId_cut = passJetIdTight
        elif self.jet_jetId == "tightLepVeto":  # Select jetId 6
            logger.info("Applying jetID recipe of NanoAOD version %s", self.nano_version)
            jetId_cut = passJetIdTight & passJetIdTightLepVeto
        else:
            jetId_cut = ak.ones_like(jets.pt) > 0
            logger.warning("[ select_jets ] - No JetId applied")
    else:
        if self.jet_jetId == "tight":
            jetId_cut = jets.jetId >= 2
        elif self.jet_jetId == "tightLepVeto":
            jetId_cut = jets.jetId == 6
        else:
            jetId_cut = ak.ones_like(jets.pt) > 0
            logger.warning("[ select_jets ] - No JetId applied")
    logger.debug(
        f"[ select_jets ] - Total: {ak.sum(ak.flatten((ak.ones_like(jets.pt) > 0)))} - Pass tight jetId: {ak.sum(ak.flatten(jetId_cut))}"
    )
    pt_cut = jets.pt > self.jet_pt_threshold
    eta_cut = abs(jets.eta) < self.jet_max_eta
    dr_dipho_cut = ak.ones_like(pt_cut) > 0
    if (self.clean_jet_dipho) & (ak.num(diphotons.pt, axis=0) > 0):
        dr_dipho_cut = delta_r_mask(jets, diphotons, self.jet_dipho_min_dr)

    if (self.clean_jet_pho) & (ak.num(diphotons.pt, axis=0) > 0):
        lead = ak.zip(
            {
                "pt": diphotons.pho_lead.pt,
                "eta": diphotons.pho_lead.eta,
                "phi": diphotons.pho_lead.phi,
                "mass": diphotons.pho_lead.mass,
                "charge": diphotons.pho_lead.charge,
            }
        )
        lead = ak.with_name(lead, "PtEtaPhiMCandidate")
        sublead = ak.zip(
            {
                "pt": diphotons.pho_sublead.pt,
                "eta": diphotons.pho_sublead.eta,
                "phi": diphotons.pho_sublead.phi,
                "mass": diphotons.pho_sublead.mass,
                "charge": diphotons.pho_sublead.charge,
            }
        )
        sublead = ak.with_name(sublead, "PtEtaPhiMCandidate")
        dr_pho_lead_cut = delta_r_mask(jets, lead, self.jet_pho_min_dr)
        dr_pho_sublead_cut = delta_r_mask(jets, sublead, self.jet_pho_min_dr)
    else:
        dr_pho_lead_cut = jets.pt > -1
        dr_pho_sublead_cut = jets.pt > -1

    if (self.clean_jet_ele) & (ak.num(electrons.pt, axis=0) > 0):
        dr_electrons_cut = delta_r_mask(jets, electrons, self.jet_ele_min_dr)
    else:
        dr_electrons_cut = jets.pt > -1

    if (self.clean_jet_muo) & (ak.num(muons.pt, axis=0) > 0):
        dr_muons_cut = delta_r_mask(jets, muons, self.jet_muo_min_dr)
    else:
        dr_muons_cut = jets.pt > -1

    if taus is not None:
        if (self.clean_jet_tau) & (ak.num(taus.pt, axis=0) > 0):
            dr_taus_cut = delta_r_mask(jets, taus, self.jet_tau_min_dr)
        else:
            dr_taus_cut = jets.pt > -1
    else:
        dr_taus_cut = jets.pt > -1

    return (
        (jetId_cut)
        & (pt_cut)
        & (eta_cut)
        & (dr_dipho_cut)
        & (dr_pho_lead_cut)
        & (dr_pho_sublead_cut)
        & (dr_electrons_cut)
        & (dr_muons_cut)
        & (dr_taus_cut)
    )


def select_fatjets(
    self,
    fatjets: ak.highlevel.Array,
    diphotons: ak.highlevel.Array,
    muons: ak.highlevel.Array,
    electrons: ak.highlevel.Array,
) -> ak.highlevel.Array:
    # same as select_jets(), but uses fatjet variables
    pt_cut = fatjets.pt > self.fatjet_pt_threshold
    eta_cut = abs(fatjets.eta) < self.fatjet_max_eta
    dr_dipho_cut = ak.ones_like(pt_cut) > 0
    if self.clean_fatjet_dipho & (ak.num(diphotons.pt, axis=0) > 0):
        dr_dipho_cut = delta_r_mask(fatjets, diphotons, self.fatjet_dipho_min_dr)

    if (self.clean_fatjet_pho) & (ak.num(diphotons.pt, axis=0) > 0):
        lead = ak.zip(
            {
                "pt": diphotons.pho_lead.pt,
                "eta": diphotons.pho_lead.eta,
                "phi": diphotons.pho_lead.phi,
                "mass": diphotons.pho_lead.mass,
                "charge": diphotons.pho_lead.charge,
            }
        )
        lead = ak.with_name(lead, "PtEtaPhiMCandidate")
        sublead = ak.zip(
            {
                "pt": diphotons.pho_sublead.pt,
                "eta": diphotons.pho_sublead.eta,
                "phi": diphotons.pho_sublead.phi,
                "mass": diphotons.pho_sublead.mass,
                "charge": diphotons.pho_sublead.charge,
            }
        )
        sublead = ak.with_name(sublead, "PtEtaPhiMCandidate")
        dr_pho_lead_cut = delta_r_mask(fatjets, lead, self.fatjet_pho_min_dr)
        dr_pho_sublead_cut = delta_r_mask(fatjets, sublead, self.fatjet_pho_min_dr)
    else:
        dr_pho_lead_cut = fatjets.pt > -1
        dr_pho_sublead_cut = fatjets.pt > -1

    if (self.clean_fatjet_ele) & (ak.num(electrons.pt, axis=0) > 0):
        dr_electrons_cut = delta_r_mask(fatjets, electrons, self.fatjet_ele_min_dr)
    else:
        dr_electrons_cut = fatjets.pt > -1

    if (self.clean_fatjet_muo) & (ak.num(muons.pt, axis=0) > 0):
        dr_muons_cut = delta_r_mask(fatjets, muons, self.fatjet_muo_min_dr)
    else:
        dr_muons_cut = fatjets.pt > -1

    return (
        (pt_cut)
        & (eta_cut)
        & (dr_dipho_cut)
        & (dr_pho_lead_cut)
        & (dr_pho_sublead_cut)
        & (dr_electrons_cut)
        & (dr_muons_cut)
    )


def jetvetomap(self, events, logger, dataset_name, year="2022preEE"):
    """
    Jet veto map
    """
    systematic = "jetvetomap"
    sel_obj = PackedSelection()

    json_dict = {
        "2016preVFP": os.path.join(
            os.path.dirname(__file__),
            "../systematics/JSONs/POG/JME/2016preVFP_UL/jetvetomaps.json.gz",
        ),
        "2016postVFP": os.path.join(
            os.path.dirname(__file__),
            "../systematics/JSONs/POG/JME/2016postVFP_UL/jetvetomaps.json.gz",
        ),
        "2017": os.path.join(
            os.path.dirname(__file__),
            "../systematics/JSONs/POG/JME/2017_UL/jetvetomaps.json.gz",
        ),
        "2018": os.path.join(
            os.path.dirname(__file__),
            "../systematics/JSONs/POG/JME/2018_UL/jetvetomaps.json.gz",
        ),
        "2022preEE": os.path.join(
            os.path.dirname(__file__),
            "../systematics/JSONs/POG/JME/2022_Summer22/jetvetomaps.json.gz",
        ),
        "2022postEE": os.path.join(
            os.path.dirname(__file__),
            "../systematics/JSONs/POG/JME/2022_Summer22EE/jetvetomaps.json.gz",
        ),
        "2023preBPix": os.path.join(
            os.path.dirname(__file__),
            "../systematics/JSONs/POG/JME/2023_Summer23/jetvetomaps.json.gz",
        ),
        "2023postBPix": os.path.join(
            os.path.dirname(__file__),
            "../systematics/JSONs/POG/JME/2023_Summer23BPix/jetvetomaps.json.gz",
        ),
    }
    key_map = {
        "2016preVFP": "Summer19UL16_V1",
        "2016postVFP": "Summer19UL16_V1",
        "2017": "Summer19UL17_V1",
        "2018": "Summer19UL18_V1",
        "2022preEE": "Summer22_23Sep2023_RunCD_V1",
        "2022postEE": "Summer22EE_23Sep2023_RunEFG_V1",
        "2023preBPix": "Summer23Prompt23_RunC_V1",
        "2023postBPix": "Summer23BPixPrompt23_RunD_V1",
    }

    logger.debug(
        f"[{systematic}] {key_map[year]}, year: {year} to dataset: {dataset_name}"
    )

    # Edge check of input variables. The eta and phi variables don't enable flow
    # https://cms-nanoaod-integration.web.cern.ch/commonJSONSFs/summaries/JME_2022_Prompt_jetvetomaps.html
    _cset = model_auto(open_auto(json_dict[year]))
    _cset_json = json.loads(_cset.json())
    low_eta, high_eta = (
        _cset_json["corrections"][0]["data"]["content"][0]["value"]["edges"][0][0],
        _cset_json["corrections"][0]["data"]["content"][0]["value"]["edges"][0][-1],
    )
    # phi value must be within [-np.pi,np.pi]. Though values beyond are observed.
    # Might due to the accuracy of nanoaod format. So clip the values to be within the first and last bin centers
    low_phi, high_phi = (
        (
            _cset_json["corrections"][0]["data"]["content"][0]["value"]["edges"][1][0]
            + _cset_json["corrections"][0]["data"]["content"][0]["value"]["edges"][1][1]
        )
        / 2,
        (
            _cset_json["corrections"][0]["data"]["content"][0]["value"]["edges"][1][-1]
            + _cset_json["corrections"][0]["data"]["content"][0]["value"]["edges"][1][
                -2
            ]
        )
        / 2,
    )
    jets_jagged = deepcopy(events.Jet)
    # remove jets out of bin edges
    # https://cms-nanoaod-integration.web.cern.ch/commonJSONSFs/summaries/JME_2022_Prompt_jetvetomaps.html
    jets_jagged = jets_jagged[
        (jets_jagged.eta >= low_eta) & (jets_jagged.eta < high_eta)
    ]
    count = ak.num(jets_jagged)
    jets = ak.flatten(jets_jagged)

    cset = correctionlib.CorrectionSet.from_file(json_dict[year])

    # ref: https://twiki.cern.ch/twiki/bin/viewauth/CMS/PdmVRun3Analysis#From_JME
    # and: https://cms-talk.web.cern.ch/t/jet-veto-maps-for-run3/57850/6

    input_dict = {
        "type": "jetvetomap",
        "eta": jets.eta,
        "phi": np.clip(jets.phi, low_phi, high_phi),
    }

    if (self.nano_version == 12) or (self.nano_version == 13):
        passJetIdTight, _ = jetIdFlags_v1213(jets, self.nano_version)
        jetId_cut = passJetIdTight
    else:
        jetId_cut = ((jets.jetId == 2) | (jets.jetId == 6))

    input_dict["type"] = "jetvetomap"
    inputs = [input_dict[input.name] for input in cset[key_map[year]].inputs]
    vetomap = cset[key_map[year]].evaluate(*(inputs))
    flag_veto_jet = (np.abs(vetomap) > 0) & ((jets.pt > 15) & (jetId_cut) & ((jets.chEmEF + jets.neEmEF) < 0.9) & (jets.muonIdx1 == -1) & (jets.muonIdx2 == -1))
    sel_obj.add("vetomap", flag_veto_jet)

    sel_veto_jet = sel_obj.all(*(sel_obj.names))
    sel_good_jet = ~ak.Array(sel_veto_jet)
    logger.debug(
        f"[{systematic}] total: {len(sel_good_jet)}, pass: {ak.sum(sel_good_jet)}"
    )
    sel_good_jet_jagged = ak.unflatten(sel_good_jet, count)
    flag_veto_jet_jagged = ak.unflatten(flag_veto_jet, count)

    sel_event_veto = ~ak.any(flag_veto_jet_jagged, axis=1)

    # Apply the veto mask, preserving all fields
    filtered_events = events[sel_event_veto]

    filtered_events["Photon"] = events.Photon[sel_event_veto]
    filtered_events["Jet"] = (jets_jagged[sel_good_jet_jagged])[sel_event_veto]
    # * Need to add Muon varibles for muon s&s uncertainties
    filtered_events["Muon"] = events.Muon[sel_event_veto]

    logger.debug(
        f"[{systematic}] total event: {len(sel_event_veto)}, pass event: {ak.sum(sel_event_veto)}"
    )
    return filtered_events
