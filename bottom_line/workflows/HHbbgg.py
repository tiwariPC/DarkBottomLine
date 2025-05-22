from bottom_line.workflows.base import bbMETBaseProcessor
from bottom_line.tools.SC_eta import add_photon_SC_eta
from bottom_line.tools.EELeak_region import veto_EEleak_flag
from bottom_line.tools.EcalBadCalibCrystal_events import remove_EcalBadCalibCrystal_events
from bottom_line.tools.gen_helpers import get_fiducial_flag, get_higgs_gen_attributes, match_jet
from bottom_line.tools.sigma_m_tools import compute_sigma_m
from bottom_line.selections.photon_selections import photon_preselection
from bottom_line.selections.diphoton_selections import build_diphoton_candidates, apply_fiducial_cut_det_level
from bottom_line.selections.lepton_selections import select_electrons, select_muons
from bottom_line.selections.jet_selections import select_jets, select_fatjets, jetvetomap
from bottom_line.selections.lumi_selections import select_lumis
from bottom_line.utils.dumping_utils import (
    diphoton_ak_array,
    dump_ak_array,
    diphoton_list_to_pandas,
    dump_pandas,
    get_obj_syst_dict,
)
from bottom_line.utils.misc_utils import choose_jet
from bottom_line.selections.HHbbgg_selections import (
    getCosThetaStar_CS,
    get_HHbbgg,
    getCosThetaStar_gg,
    getCosThetaStar_jj,
    DeltaR,
    Cxx,
    getChi_t0,
    getChi_t1,
    DeltaPhi,
)
from bottom_line.tools.flow_corrections import apply_flow_corrections_to_photons

from bottom_line.tools.mass_decorrelator import decorrelate_mass_resolution
from bottom_line.tools.truth_info import get_truth_info_dict

# from bottom_line.utils.dumping_utils import diphoton_list_to_pandas, dump_pandas
from bottom_line.systematics import object_systematics as available_object_systematics
from bottom_line.systematics import object_corrections as available_object_corrections
from bottom_line.systematics import weight_systematics as available_weight_systematics
from bottom_line.systematics import weight_corrections as available_weight_corrections
from bottom_line.systematics import apply_systematic_variations_object_level

import warnings
from typing import Any, Dict, List, Optional
import awkward as ak
import numpy
import sys
import os
import json
import vector
from coffea.analysis_tools import Weights
from copy import deepcopy

import logging

logger = logging.getLogger(__name__)

vector.register_awkward()


class HHbbggProcessor(bbMETBaseProcessor):
    def __init__(
        self,
        metaconditions: Dict[str, Any],
        systematics: Dict[str, List[Any]] = None,
        corrections: Dict[str, List[Any]] = None,
        apply_trigger: bool = False,
        nano_version: int = None,
        bTagEffFileName: Optional[str] = None,
        output_location: Optional[str] = None,
        taggers: Optional[List[Any]] = None,
        trigger_group=".*DoubleEG.*",
        analysis="mainAnalysis",
        applyCQR: bool = False,
        skipJetVetoMap: bool = False,
        year: Dict[str, List[str]] = None,
        fiducialCuts: str = "store_flag",
        doDeco: bool = False,
        Smear_sigma_m: bool = False,
        doFlow_corrections: bool = False,
        output_format: str = "parquet"
    ) -> None:
        super().__init__(
            metaconditions,
            systematics=systematics,
            corrections=corrections,
            nano_version=nano_version,
            apply_trigger=apply_trigger,
            output_location=output_location,
            bTagEffFileName=bTagEffFileName,
            taggers=taggers,
            trigger_group=trigger_group,
            analysis=analysis,
            applyCQR=applyCQR,
            skipJetVetoMap=skipJetVetoMap,
            year=year,
            fiducialCuts=fiducialCuts,
            doDeco=doDeco,
            Smear_sigma_m=Smear_sigma_m,
            doFlow_corrections=doFlow_corrections,
            output_format=output_format
        )

        self.nano_version = nano_version

        # muon selection cuts
        self.muon_pt_threshold = 15
        self.global_muon = True
        self.mu_iso_wp = "loose"
        self.muon_photon_min_dr = 0.4
        self.muon_max_dxy = None
        self.muon_max_dz = None

        # electron selection cuts
        self.el_id_wp = "WP80"
        self.electron_photon_min_dr = 0.3
        self.electron_max_dxy = None
        self.electron_max_dz = None

        # jet selection cuts
        self.jet_jetId = "tightLepVeto"  # can be "tightLepVeto" or "tight": https://twiki.cern.ch/twiki/bin/view/CMS/JetID13p6TeV#nanoAOD_Flags
        self.jet_max_eta = 4.7

        self.clean_jet_dipho = False
        self.clean_jet_pho = True
        self.clean_jet_ele = False
        self.clean_jet_muo = False

        # fatjet selection cuts, same as jets but for photon dR and pT
        self.fatjet_dipho_min_dr = 0.8
        self.fatjet_pho_min_dr = 0.8
        self.fatjet_ele_min_dr = 0.8
        self.fatjet_muo_min_dr = 0.8
        self.fatjet_pt_threshold = 250
        self.fatjet_max_eta = 4.7

        self.clean_fatjet_dipho = False
        self.clean_fatjet_pho = True
        self.clean_fatjet_ele = False
        self.clean_fatjet_muo = False

        # Objects to save
        self.num_jets_to_store = 10
        self.num_fatjets_to_store = 4
        self.num_leptons_to_store = 4

        # Choose which type of analysis :
        self.bbgg_analysis = ["nonRes", "Res"]

        # Choose fiducial cut
        self.fiducialCuts = "store_flag"  # right now, this is needed even though default for HHbbgg workflow is store_flag as the defualt command line argument for fiducialCuts ('classical') over rides the default of the workflow

    def process_extra(self, events: ak.Array) -> ak.Array:
        return events, {}

    def process(self, events: ak.Array) -> Dict[Any, Any]:
        dataset_name = events.metadata["dataset"]
        filename = events.metadata["filename"]

        # data or monte carlo?
        self.data_kind = "mc" if hasattr(events, "GenPart") else "data"

        # here we start recording possible coffea accumulators
        # most likely histograms, could be counters, arrays, ...
        histos_etc = {}
        histos_etc[dataset_name] = {}
        if self.data_kind == "mc":
            histos_etc[dataset_name]["nTot"] = int(
                ak.num(events.genWeight, axis=0)
            )
            histos_etc[dataset_name]["nPos"] = int(ak.sum(events.genWeight > 0))
            histos_etc[dataset_name]["nNeg"] = int(ak.sum(events.genWeight < 0))
            histos_etc[dataset_name]["nEff"] = int(
                histos_etc[dataset_name]["nPos"] - histos_etc[dataset_name]["nNeg"]
            )
            histos_etc[dataset_name]["genWeightSum"] = float(
                ak.sum(events.genWeight)
            )
        else:
            histos_etc[dataset_name]["nTot"] = int(len(events))
            histos_etc[dataset_name]["nPos"] = int(histos_etc[dataset_name]["nTot"])
            histos_etc[dataset_name]["nNeg"] = int(0)
            histos_etc[dataset_name]["nEff"] = int(histos_etc[dataset_name]["nTot"])
            histos_etc[dataset_name]["genWeightSum"] = float(len(events))

        # lumi mask
        if self.data_kind == "data":
            try:
                lumimask = select_lumis(self.year[dataset_name][0], events, logger)
                events = events[lumimask]
            except:
                logger.info(
                    f"[ lumimask ] Skip now! Unable to find year info of {dataset_name}"
                )
        # apply jetvetomap: only retain events that without any jets in the EE leakage region
        if not self.skipJetVetoMap:
            events = jetvetomap(
                self, events, logger, dataset_name, year=self.year[dataset_name][0]
            )
        # metadata array to append to higgsdna output
        metadata = {}

        if self.data_kind == "mc":
            # Add sum of gen weights before selection for normalisation in postprocessing
            metadata["sum_genw_presel"] = str(ak.sum(events.genWeight))
        else:
            metadata["sum_genw_presel"] = "Data"

        # apply filters and triggers
        events = self.apply_filters_and_triggers(events)

        # remove events affected by EcalBadCalibCrystal
        if self.data_kind == "data":
            events = remove_EcalBadCalibCrystal_events(events)

        # we need ScEta for corrections and systematics, it is present in NanoAODv13+ and can be calculated using PV for older versions
        events.Photon = add_photon_SC_eta(events.Photon, events.PV)

        # Need to add ScEta variables to electrons for scale and smearing corrections
        electrons = events.Electron
        electrons["ScEta"] = electrons.eta + electrons.deltaEtaSC
        electrons["isScEtaEB"] = numpy.abs(electrons.ScEta) < 1.4442
        electrons["isScEtaEE"] = numpy.abs(electrons.ScEta) > 1.566
        events.Electron = electrons

        # add veto EE leak branch for photons, could also be used for electrons
        if (
            self.year[dataset_name][0] == "2022EE"
            or self.year[dataset_name][0] == "2022postEE"
        ):
            events.Photon = veto_EEleak_flag(self, events.Photon)
            events.Photon = events.Photon[events.Photon.vetoEELeak]
            events.Electron = veto_EEleak_flag(self, events.Electron)
            events.Electron = events.Electron[events.Electron.vetoEELeak]

        # read which systematics and corrections to process
        try:
            correction_names = self.corrections[dataset_name]
        except KeyError:
            correction_names = []
        try:
            systematic_names = self.systematics[dataset_name]
        except KeyError:
            systematic_names = []

        # If --Smear_sigma_m == True and no Smearing correction in .json for MC throws an error, since the pt scpectrum need to be smeared in order to properly calculate the smeared sigma_m_m
        if (
            self.data_kind == "mc"
            and self.Smear_sigma_m
            and ("Smearing_Trad" not in correction_names and "Smearing_IJazZ" not in correction_names and "Smearing2G_IJazZ" not in correction_names)
        ):
            warnings.warn(
                "Smearing_Trad or  Smearing_IJazZ or Smearing2G_IJazZ should be specified in the corrections field in .json in order to smear the mass!"
            )
            sys.exit(0)

        # save raw pt if we use scale/smearing corrections
        # These needs to be before the smearing of the mass resolution in order to have the raw pt for the function
        s_or_s_applied = False
        s_or_s_ele_applied = False
        for correction in correction_names:
            if "scale" or "smearing" in correction.lower():
                if "Electron" in correction:
                    s_or_s_ele_applied = True
                else:
                    s_or_s_applied = True
        if s_or_s_applied:
            events.Photon["pt_raw"] = ak.copy(events.Photon.pt)
        if s_or_s_ele_applied:
            events.Electron["pt_raw"] = ak.copy(events.Electron.pt)

        # Since now we are applying Smearing term to the sigma_m_over_m i added this portion of code
        # specially for the estimation of smearing terms for the data events [data pt/energy] are not smeared!
        if self.data_kind == "data" and self.Smear_sigma_m:
            if "Scale_Trad" in correction_names:
                correction_name = "Smearing_Trad"
            elif "Scale_IJazZ" in correction_names:
                correction_name = "Smearing_IJazZ"
            elif "Scale2G_IJazZ" in correction_names:
                correction_name = "Smearing2G_IJazZ"
            else:
                logger.info('Specify a scale correction for the data in the corrections field in .json in order to smear the mass!')
                sys.exit(0)

            logger.info(
                f"""
                \nApplying correction {correction_name} to dataset {dataset_name}\n
                This is only for the addition of the smearing term to the sigma_m_over_m in data\n
                """
            )
            varying_function = available_object_corrections[correction_name]
            events = varying_function(events=events, year=self.year[dataset_name][0])

        # Keep a copy of the original, JES-corrected, non-PNet-regressed variables
        pt_orig = ak.copy(events.Jet["pt"])

        for correction_name in correction_names:
            if correction_name in available_object_corrections.keys():
                logger.info(
                    f"Applying correction {correction_name} to dataset {dataset_name}"
                )
                varying_function = available_object_corrections[correction_name]
                events = varying_function(
                    events=events, year=self.year[dataset_name][0]
                )
            elif correction_name in available_weight_corrections:
                # event weight corrections will be applied after photon preselection / application of further taggers
                continue
            else:
                # may want to throw an error instead, needs to be discussed
                warnings.warn(f"Could not process correction {correction_name}.")
                continue

        original_photons = events.Photon
        original_electrons = events.Electron
        # NOTE: jet jerc systematics are added in the correction functions and handled later
        original_jets = events.Jet
        original_jets["pt_orig"] = pt_orig

        # Computing the normalizing flow correction
        if self.data_kind == "mc" and self.doFlow_corrections:
            original_photons = apply_flow_corrections_to_photons(
                original_photons,
                events,
                self.meta,
                self.year[dataset_name][0],
                self.add_photonid_mva_run3,
                logger
            )

        # Add additional collections if object systematics should be applied
        collections = {
            "Photon": original_photons,
            "Electron": original_electrons
        }

        # Apply the systematic variations.
        collections = apply_systematic_variations_object_level(
            systematic_names,
            events,
            self.year[dataset_name][0],
            logger,
            available_object_systematics,
            available_weight_systematics,
            collections
        )

        original_photons = collections["Photon"]
        original_electrons = collections["Electron"]

        # Write systematic variations to dicts
        photons_dct = {}
        photons_dct["nominal"] = original_photons
        logger.debug(original_photons.systematics.fields)
        for systematic in original_photons.systematics.fields:
            for variation in original_photons.systematics[systematic].fields:
                # deepcopy to allow for independent calculations on photon variables with CQR
                photons_dct[f"{systematic}_{variation}"] = deepcopy(
                    original_photons.systematics[systematic][variation]
                )

        electrons_dct = {}
        electrons_dct["nominal"] = original_electrons
        logger.debug(original_electrons.systematics.fields)
        for systematic in original_electrons.systematics.fields:
            for variation in original_electrons.systematics[systematic].fields:
                # no deepcopy here unless we find a case where it's actually needed
                electrons_dct[f"{systematic}_{variation}"] = original_electrons.systematics[systematic][variation]

        # NOTE: jet jerc systematics are added in the corrections, now extract those variations and create the dictionary
        jerc_syst_list, jets_dct = get_obj_syst_dict(original_jets, ["pt", "mass"])
        # object systematics dictionary
        logger.debug(f"[ jerc systematics ] {jerc_syst_list}")

        # Build the flattened array of all possible variations
        variations_combined = []
        variations_combined.append(original_photons.systematics.fields)
        variations_combined.append(original_electrons.systematics.fields)
        # NOTE: jet jerc systematics are not added with add_systematics
        variations_combined.append(jerc_syst_list)
        # Flatten
        variations_flattened = sum(variations_combined, [])  # Begin with empty list and keep concatenating
        # Attach _down and _up
        variations = [item + suffix for item in variations_flattened for suffix in ['_down', '_up']]
        # Add nominal to the list
        variations.append('nominal')
        logger.debug(f"[systematics variations] {variations}")

        for variation in variations:
            # The PNet corrections are applied during the JES application,
            # using the proper corrections tag in the runner.json.
            # As a result, nominal JES need to be applied to the PNet-corrected jets
            photons, jets = photons_dct["nominal"], jets_dct["nominal"]
            if variation == "nominal":
                pass  # Do nothing since we already get the unvaried, but nominally corrected objects above
            elif variation in [*photons_dct]:  # [*dict] gets the keys of the dict since Python >= 3.5
                photons = photons_dct[variation]
            elif variation in [*electrons_dct]:
                electrons = electrons_dct[variation]
            elif variation in [*jets_dct]:
                jets = jets_dct[variation]
            do_variation = variation  # We can also simplify this a bit but for now it works

            if self.chained_quantile is not None:
                photons = self.chained_quantile.apply(photons, events)
            # recompute photonid_mva on the fly
            if self.photonid_mva_EB and self.photonid_mva_EE:
                photons = self.add_photonid_mva(photons, events)

            # photon preselection
            photons = photon_preselection(self, photons, events, year=self.year[dataset_name][0])

            diphotons = build_diphoton_candidates(photons, self.min_pt_lead_photon)

            # add genWeight column for claculating efficiencies
            if self.data_kind == "mc":
                diphotons["genWeight"] = events.genWeight

            self.calc_cut_flow("diphoton_selection", diphotons, metadata)

            # Apply the fiducial cut at detector level with helper function
            logger.info(f"Using fiducial cuts: {self.fiducialCuts}")
            diphotons = apply_fiducial_cut_det_level(self, diphotons)
            if self.fiducialCuts == "store_flag":
                diphotons = diphotons[(diphotons.pass_fiducial_classical)
                                      | (diphotons.pass_fiducial_geometric)]
            self.calc_cut_flow("fiducial_cut", diphotons, metadata)

            if self.data_kind == "mc":

                # Add the fiducial flags for particle level
                diphotons['fiducialClassicalFlag'] = get_fiducial_flag(events, flavour='Classical')
                diphotons['fiducialGeometricFlag'] = get_fiducial_flag(events, flavour='Geometric')

                # Did not completely update the genJet part of the base processor because HHbbgg has its own way of dealing with it. We may think of a way to use what was developped to improve our gen selection.
                # Changes that were not replicated here : https://gitlab.cern.ch/HiggsDNA-project/HiggsDNA/-/commit/55846b80a83619a9112b95fb8824dfdb71eee0b2
                GenPTH, GenYH, GenPhiH = get_higgs_gen_attributes(events)
                diphotons['GenPTH'] = ak.fill_none(GenPTH, -999.0)

            # baseline modifications to diphotons
            if self.diphoton_mva is not None:
                diphotons = self.add_diphoton_mva(diphotons, events)

            # workflow specific processing
            events, process_extra = self.process_extra(events)
            histos_etc.update(process_extra)

            # jet_variables
            jets = ak.zip(
                {
                    "pt": jets.pt,
                    "eta": jets.eta,
                    "phi": jets.phi,
                    "mass": jets.mass,
                    "charge": ak.zeros_like(
                        jets.pt
                    ),  # added this because jet charge is not a property of photons in nanoAOD v11. We just need the charge to build jet collection.
                    "hFlav": jets.hadronFlavour
                    if self.data_kind == "mc"
                    else ak.zeros_like(jets.pt),
                    "btagDeepFlav_B": jets.btagDeepFlavB,
                    "btagDeepFlav_CvB": jets.btagDeepFlavCvB,
                    "btagDeepFlav_CvL": jets.btagDeepFlavCvL,
                    "btagDeepFlav_QG": jets.btagDeepFlavQG,
                    "btagPNetB": jets.btagPNetB,
                    "btagPNetQvG": jets.btagPNetQvG,
                    "PNetRegPtRawCorr": jets.PNetRegPtRawCorr,
                    "PNetRegPtRawCorrNeutrino": jets.PNetRegPtRawCorrNeutrino,
                    "PNetRegPtRawRes": jets.PNetRegPtRawRes,
                    "jetId": jets.jetId,
                    "rawFactor": jets.rawFactor,
                    "pt_orig": jets.pt_orig,
                    **(
                        {"btagRobustParTAK4B": jets.btagRobustParTAK4B, "btagRobustParTAK4QG": jets.btagRobustParTAK4QG, "neHEF": jets.neHEF, "neEmEF": jets.neEmEF, "chEmEF": jets.chEmEF, "muEF": jets.muEF} if self.nano_version == 12 else {}
                    ),
                    **(
                        {"btagRobustParTAK4B": jets.btagRobustParTAK4B, "btagRobustParTAK4QG": jets.btagRobustParTAK4QG, "neHEF": jets.neHEF, "neEmEF": jets.neEmEF, "chMultiplicity": jets.chMultiplicity, "neMultiplicity": jets.neMultiplicity, "chEmEF": jets.chEmEF, "chHEF": jets.chHEF, "muEF": jets.muEF} if self.nano_version == 13 else {}
                    ),

                }
            )
            jets = ak.with_name(jets, "PtEtaPhiMCandidate")

            electrons = ak.zip(
                {
                    "pt": events.Electron.pt,
                    "eta": events.Electron.eta,
                    "phi": events.Electron.phi,
                    "mass": events.Electron.mass,
                    "charge": events.Electron.charge,
                    "cutBased": events.Electron.cutBased,
                    "mvaIso_WP90": events.Electron.mvaIso_WP90,
                    "mvaIso_WP80": events.Electron.mvaIso_WP80,
                    "mvaID": events.Electron.mvaIso,
                    "pfRelIso03_all": events.Electron.pfRelIso03_all,
                    "pfRelIso04_all": ak.zeros_like(events.Electron.pt),  # Iso04 does not exist in nanoAOD, hence filling in zeros
                    "pfIsoId": ak.ones_like(events.Electron.pt) * 10,  # IsoId does not exist in nanoAOD, hence filling in 10
                    "dxy": events.Electron.dxy,
                    "dz": events.Electron.dz,
                }
            )
            electrons = ak.with_name(electrons, "PtEtaPhiMCandidate")

            # Special cut for base workflow to replicate iso cut for electrons also for muons
            # events['Muon'] = events.Muon[events.Muon.pfRelIso03_all < 0.2]

            muons = ak.zip(
                {
                    "pt": events.Muon.pt,
                    "eta": events.Muon.eta,
                    "phi": events.Muon.phi,
                    "mass": events.Muon.mass,
                    "charge": events.Muon.charge,
                    "tightId": events.Muon.tightId,
                    "mediumId": events.Muon.mediumId,
                    "looseId": events.Muon.looseId,
                    "isGlobal": events.Muon.isGlobal,
                    "mvaID": events.Muon.mvaMuID,
                    "pfRelIso03_all": events.Muon.pfRelIso03_all,
                    "pfRelIso04_all": events.Muon.pfRelIso04_all,
                    "pfIsoId": events.Muon.pfIsoId,
                    "dxy": events.Muon.dxy,
                    "dz": events.Muon.dz,
                }
            )
            muons = ak.with_name(muons, "PtEtaPhiMCandidate")

            # create PuppiMET objects
            puppiMET = events.PuppiMET
            puppiMET = ak.with_name(puppiMET, "PtEtaPhiMCandidate")

            # FatJet variables
            fatjets = events.FatJet
            fatjets["charge"] = ak.zeros_like(fatjets.pt)
            fatjets = ak.with_name(fatjets, "PtEtaPhiMCandidate")

            # SubJet variables
            subjets = events.SubJet
            subjets["charge"] = ak.zeros_like(subjets.pt)
            subjets = ak.with_name(subjets, "PtEtaPhiMCandidate")

            # GenJetAK8 variables
            if self.data_kind == "mc":
                genjetsAK8 = events.GenJetAK8
                genjetsAK8["charge"] = ak.zeros_like(genjetsAK8.pt)
                genjetsAK8 = ak.with_name(genjetsAK8, "PtEtaPhiMCandidate")

            # lepton cleaning
            sel_electrons = electrons[
                select_electrons(self, electrons, diphotons)
            ]
            sel_muons = muons[
                select_muons(self, muons, diphotons)
            ]

            n_electrons = ak.num(sel_electrons)
            n_muons = ak.num(sel_muons)
            diphotons["n_electrons"] = n_electrons
            diphotons["n_muons"] = n_muons
            diphotons["n_electrons_after_dxy_dz_cuts"] = ak.num(
                sel_electrons[(sel_electrons.dxy < 0.2) & (sel_electrons.dz < 0.5)]
            )
            diphotons["n_muons_after_dxy_dz_cuts"] = ak.num(
                sel_muons[(sel_muons.dxy < 0.2) & (sel_muons.dz < 0.5)]
            )

            # adding selected electrons and muons of the specific variation to events to be used in SF calculations
            events["sel_muons"] = sel_muons
            events["sel_electrons"] = sel_electrons

            # jet selection and pt ordering
            jets = jets[
                select_jets(self, jets, diphotons, sel_muons, sel_electrons)
            ]
            # remove eta spikes (no final recipe): applying pT > 50 GeV for jets with abs(eta) in (2.5, 3)
            jets = jets[
                ~((jets.pt < 50) & (numpy.abs(jets.eta) > 2.5) & (numpy.abs(jets.eta) < 3))
            ]
            jets = jets[ak.argsort(jets.pt, ascending=False)]
            jets["index"] = ak.local_index(jets.pt)

            # fatjet selection and pt ordering
            fatjets = fatjets[select_fatjets(self, fatjets, diphotons, sel_muons, sel_electrons)]  # For now, having the same preselection as jet. Can be changed later

            try:
                fatjets = fatjets[ak.argsort(fatjets.particleNetWithMass_HbbvsQCD, ascending=False)]
            except ValueError as e:
                logger.warning(f"Error sorting fatjets: {e}")

            # adding selected jets to events to be used in ctagging SF calculation
            events["sel_jets"] = jets
            n_jets = ak.num(jets)
            Njets2p5 = ak.num(jets[(jets.pt > 30) & (numpy.abs(jets.eta) < 2.5)])

            diphotons["n_jets"] = n_jets
            diphotons["Njets2p5"] = Njets2p5

            ## --------- Beginning of the part added for the HHTobbgg analysis -------
            # Store  N jets with their infos -> Taken from the top workflow. This part of the code was orignally written by Florain Mausolf
            jets_properties = jets.fields
            for i in range(self.num_jets_to_store):  # Number of jets to select
                for prop in jets_properties:
                    key = f"jet{i+1}_{prop}"
                    # Retrieve the value using the choose_jet function
                    value = choose_jet(getattr(jets, prop), i, -999.0)
                    # Store the value in the diphotons dictionary
                    diphotons[key] = value

            if self.data_kind == "mc":
                # add in gen jet info here if data is mc #
                #   - boolean array of matched jet: 1 if matched, 0 if not matched but recoJet exists, -999 if recoJet doesn't exist
                #   - genPartonFlav array of matched genJet: genFlav if matched, -999 if no matching genJet or if recoJet doesn't exist
                genjets = events.GenJet
                genjets["charge"] = ak.zeros_like(genjets.pt)
                genjets = ak.with_name(genjets, "PtEtaPhiMCandidate")
                for i in range(self.num_jets_to_store):  # Number of jets to select
                    for key, jet_flav in [(f"jet{i+1}_genMatched", False), (f"jet{i+1}_genFlav", True)]:
                        # Retrieve the matching boolean using the match_jet function
                        value = match_jet(jets, genjets, i, -999.0, jet_flav=jet_flav)
                        # Store the value in the diphotons dictionary
                        diphotons[key] = value

                gen_b_mask = ((events.GenPart.pdgId == 5) | (events.GenPart.pdgId == -5))
                if ak.sum(ak.num(events.GenPart[gen_b_mask])) != 0:
                    gen_b = ak.pad_none(events.GenPart[gen_b_mask], self.num_jets_to_store)
                    motheridx = gen_b.genPartIdxMother
                    mask_Hbb = (ak.any(events.GenPart[motheridx].pdgId == 25, axis=1))
                    if ak.sum(ak.num(gen_b[mask_Hbb])) != 0:
                        gen_b_hbb = gen_b[mask_Hbb]
                        gen_b_hbb["charge"] = ak.zeros_like(gen_b_hbb.pt)
                        gen_b_hbb = ak.with_name(gen_b_hbb, "PtEtaPhiMCandidate")
                        for i in range(self.num_jets_to_store):  # Number of jets to select
                            key = f"jet{i+1}_genMatched_Hbb"
                            value = match_jet(jets, gen_b_hbb, i, -999.0)
                            diphotons[key] = ak.fill_none(value, -999.0)

                #   - boolean array of matched fatjet
                #   - genPartonFlav array of matched genFatJet
                for i in range(self.num_fatjets_to_store):
                    for key, jet_flav in [(f"fatjet{i+1}_genMatched", False), (f"fatjet{i+1}_genFlav", True)]:
                        value = match_jet(fatjets, genjetsAK8, i, -999.0, jet_size=0.8, jet_flav=jet_flav)
                        diphotons[key] = value

                # add in gen 4-momenta #
                genPart_status = (
                    # events.GenPart.status == 62   # what should status be? 23? 33? 62?
                    events.GenPart.hasFlags(['isPrompt', 'isLastCopy'])
                )
                gen_properties = ["pt", "eta", "phi", "mass"]

                # add in gen 4-momenta #
                #   - genTop (if exists)
                gentop_mask = (
                    (events.GenPart.pdgId == 6) | (events.GenPart.pdgId == -6)
                ) & genPart_status
                if ak.sum(ak.num(events.GenPart[gentop_mask])) != 0:
                    gentops = ak.pad_none(events.GenPart[gentop_mask], 2)
                    gentops["charge"] = ak.where(
                        gentops.pdgId == 6, 2 / 3, 0
                    ) + ak.where(gentops.pdgId == -6, -2 / 3, 0)
                    gentops = ak.with_name(gentops, "PtEtaPhiMCandidate")

                    try:
                        gentops = gentops[ak.argsort(gentops.pt, ascending=False)]
                    except ValueError as e:
                        logger.warning(f"Error sorting gentops: {e}")

                    for i in range(2):
                        for prop in gen_properties:
                            key = f"gentop{i+1}_{prop}"
                            value = choose_jet(getattr(gentops, prop), i, -999.0)
                            diphotons[key] = value

                # add in gen 4-momenta #
                #   - vector bosons (if exists)
                genZ_mask = (events.GenPart.pdgId == 23) & genPart_status
                if ak.sum(ak.num(events.GenPart[genZ_mask])) != 0:
                    genZs = ak.pad_none(events.GenPart[genZ_mask], 2)
                    genZs["charge"] = ak.zeros_like(genZs.pt)
                    genZs = ak.with_name(genZs, "PtEtaPhiMCandidate")

                    try:
                        genZs = genZs[ak.argsort(genZs.pt, ascending=False)]
                    except ValueError as e:
                        logger.warning(f"Error sorting genZs: {e}")

                    for i in range(2):
                        for prop in gen_properties:
                            key = f"genZ{i+1}_{prop}"
                            value = choose_jet(getattr(genZs, prop), i, -999.0)
                            diphotons[key] = value

                genW_mask = (
                    (events.GenPart.pdgId == 24) | (events.GenPart.pdgId == -24)
                ) & genPart_status
                if ak.sum(ak.num(events.GenPart[genW_mask])) != 0:
                    genWs = ak.pad_none(events.GenPart[genW_mask], 2)
                    genWs["charge"] = ak.where(
                        genWs.pdgId == 24, 1, 0
                    ) + ak.where(genWs.pdgId == -24, -1, 0)
                    genWs = ak.with_name(genWs, "PtEtaPhiMCandidate")

                    try:
                        genWs = genWs[ak.argsort(genWs.pt, ascending=False)]
                    except ValueError as e:
                        logger.warning(f"Error sorting genWs: {e}")

                    for i in range(2):
                        for prop in gen_properties:
                            key = f"genW{i+1}_{prop}"
                            value = choose_jet(getattr(genWs, prop), i, -999.0)
                            diphotons[key] = value

                # add in gen 4-momenta #
                #   - gen Higgs (if exists)
                genHiggs_mask = (events.GenPart.pdgId == 25) & genPart_status
                if ak.sum(ak.num(events.GenPart[genHiggs_mask])) != 0:
                    genHiggs = ak.pad_none(events.GenPart[genHiggs_mask], 2)
                    mask_bb = (ak.any(genHiggs.children.pdgId == 5, axis=2)) & (ak.any(genHiggs.children.pdgId == -5, axis=2))
                    mask_gg = (ak.all(genHiggs.children.pdgId == 22, axis=2))
                    mask_two_part = (ak.num(genHiggs.children.pdgId, axis=2) == 2)
                    genHiggs["charge"] = ak.zeros_like(genHiggs.pt)
                    genHiggs = ak.with_name(genHiggs, "PtEtaPhiMCandidate")
                    diHiggs_bool = (
                        ak.num(events.GenPart[genHiggs_mask], axis=1) == 2
                    )
                    try:
                        genHiggs = genHiggs[ak.argsort(genHiggs.pt, ascending=False)]
                    except ValueError as e:
                        logger.warning(f"Error sorting genHiggs: {e}")
                    genHiggsdecay = ak.zip(
                        {
                            "Higgs_toGG": mask_gg & mask_two_part,
                            "Higgs_tobb": mask_bb & mask_two_part,
                        }
                    )
                    for decay in ["Higgs_tobb", "Higgs_toGG"]:
                        for prop in gen_properties:
                            key = f"gen{decay}_{prop}"
                            value = choose_jet(getattr(genHiggs[genHiggsdecay[decay]], prop), 0, -999.0)
                            diphotons[key] = value

                    # add in gen_mHH (if exists)
                    genH1 = genHiggs[ak.local_index(genHiggs, axis=1) == 0]
                    genH2 = genHiggs[ak.local_index(genHiggs, axis=1) == 1]
                    genHH = genH1 + genH2
                    gen_mHH = ak.firsts((genHH).mass)
                    gen_mHH = ak.where(diHiggs_bool, gen_mHH, -999)
                    diphotons["gen_mHH"] = gen_mHH

                    # add in gen_pT_HH (if exists)
                    gen_pT_HH = ak.firsts((genHH).pt)
                    gen_pT_HH = ak.where(diHiggs_bool, gen_pT_HH, -999)
                    diphotons["gen_pT_HH"] = gen_pT_HH

                    # add in gen_cosThetaStar_HH (if exists)
                    genH1_boosted = genH1.boost(-genHH.boostvec)
                    gen_CosThetaStar_HH = ak.firsts(numpy.cos(genH1_boosted.theta))
                    gen_CosThetaStar_HH = ak.where(diHiggs_bool, gen_CosThetaStar_HH, -999)
                    diphotons["gen_CosThetaStar_HH"] = gen_CosThetaStar_HH

                # Add the truth information
                param_values = get_truth_info_dict(filename)
                for key in param_values.keys():
                    diphotons[key] = param_values[key]

            # Store btagPNetB working points for all jets
            json_file = os.path.join(os.path.dirname(__file__), "../tools/WPs_btagging_HHbbgg.json")
            with open(json_file, "r") as jf:
                btagging_wps = json.load(jf)
            nBTight = ak.fill_none((ak.sum(jets.btagPNetB >= btagging_wps[self.year[dataset_name][0]]["PNetTight"], axis=1)), 0)
            nBMedium = ak.fill_none((ak.sum(jets.btagPNetB >= btagging_wps[self.year[dataset_name][0]]["PNetMedium"], axis=1)), 0)
            nBLoose = ak.fill_none((ak.sum(jets.btagPNetB >= btagging_wps[self.year[dataset_name][0]]["PNetLoose"], axis=1)), 0)

            n_fatjets = ak.num(fatjets)
            diphotons["n_fatjets"] = n_fatjets

            # Creatiion a dijet
            dijets_base = ak.combinations(
                jets, 2, fields=("first_jet", "second_jet")
            )
            self.calc_cut_flow("select_two_jets", diphotons[~ak.is_none(ak.firsts(dijets_base))], metadata)
            self.calc_cut_flow("select_two_jets_include_or_atleast_one_fatjet", diphotons[(~ak.is_none(ak.firsts(dijets_base))) | (diphotons["n_fatjets"] > 0)], metadata)

            # HHbbgg :  now turn the dijets into candidates with four momenta and such
            dijets_4mom = dijets_base["first_jet"] + dijets_base["second_jet"]
            dijets_base["pt"] = dijets_4mom.pt
            dijets_base["eta"] = dijets_4mom.eta
            dijets_base["phi"] = dijets_4mom.phi
            dijets_base["mass"] = dijets_4mom.mass
            dijets_base["charge"] = dijets_4mom.charge
            dijets_base["btagPNetB_sum"] = (
                dijets_base["first_jet"].btagPNetB + dijets_base["second_jet"].btagPNetB
            )
            dijets_base["DeltaR_jj"] = DeltaR(dijets_base["first_jet"], dijets_base["second_jet"])
            try:
                dijets_base = dijets_base[ak.argsort(dijets_base.btagPNetB_sum, ascending=False)]
            except ValueError as e:
                logger.warning(f"Error sorting dijets: {e}")

            dijets_base = ak.with_name(dijets_base, "PtEtaPhiMCandidate")

            dijets_for_tth_killer = ak.copy(dijets_base)  # needed for a few ttH killer variables

            # Selection on the dijet
            dijets_base = dijets_base[(numpy.abs(dijets_base["first_jet"].eta) < 2.5) & (numpy.abs(dijets_base["second_jet"].eta) < 2.5)]
            self.calc_cut_flow("jet_eta_cut", diphotons[~ak.is_none(ak.firsts(dijets_base))], metadata)
            self.calc_cut_flow("jet_eta_cut_include_or_atleast_one_fatjet", diphotons[(~ak.is_none(ak.firsts(dijets_base))) | (diphotons["n_fatjets"] > 0)], metadata)
            dijets_base = dijets_base[dijets_base.btagPNetB_sum > 0]
            self.calc_cut_flow("dijet_b_tag_sum_cut", diphotons[~ak.is_none(ak.firsts(dijets_base))], metadata)
            self.calc_cut_flow("dijet_b_tag_sum_cut_include_or_atleast_one_fatjet", diphotons[(~ak.is_none(ak.firsts(dijets_base))) | (diphotons["n_fatjets"] > 0)], metadata)

            # adding MET to parquet, adding all variables for now
            puppiMET_properties = puppiMET.fields
            for prop in puppiMET_properties:
                key = f"puppiMET_{prop}"
                # Retrieve the value using the choose_jet function (which can be used for puppiMET as well)
                value = getattr(puppiMET, prop)
                # Store the value in the diphotons dictionary
                diphotons[key] = value

            for AnType in self.bbgg_analysis :
                dijets = ak.copy(dijets_base)
                if AnType not in ["nonRes", "Res"]:
                    raise NotImplementedError
                if AnType == "nonRes":
                    dijets = dijets[dijets["mass"] > 70]
                    self.calc_cut_flow(f"{AnType}_dijet_lower_mass_cut", diphotons[~ak.is_none(ak.firsts(dijets))], metadata)
                    self.calc_cut_flow(f"{AnType}_dijet_lower_mass_cut_include_or_atleast_one_fatjet", diphotons[(~ak.is_none(ak.firsts(dijets))) | (diphotons["n_fatjets"] > 0)], metadata)
                    dijets = dijets[dijets["mass"] < 190]
                    self.calc_cut_flow(f"{AnType}_dijet_upper_mass_cut", diphotons[~ak.is_none(ak.firsts(dijets))], metadata)
                    self.calc_cut_flow(f"{AnType}_dijet_upper_mass_cut_include_or_atleast_one_fatjet", diphotons[(~ak.is_none(ak.firsts(dijets))) | (diphotons["n_fatjets"] > 0)], metadata)

                lead_bjet_pt = choose_jet(dijets["first_jet"].pt, 0, -999.0)
                lead_bjet_eta = choose_jet(dijets["first_jet"].eta, 0, -999.0)
                lead_bjet_phi = choose_jet(dijets["first_jet"].phi, 0, -999.0)
                lead_bjet_mass = choose_jet(dijets["first_jet"].mass, 0, -999.0)
                lead_bjet_charge = choose_jet(dijets["first_jet"].charge, 0, -999.0)
                lead_bjet_btagPNetB = choose_jet(dijets["first_jet"].btagPNetB, 0, -999.0)
                lead_bjet_PNetRegPtRawCorr = choose_jet(dijets["first_jet"].PNetRegPtRawCorr, 0, -999.0)
                lead_bjet_PNetRegPtRawCorrNeutrino = choose_jet(dijets["first_jet"].PNetRegPtRawCorrNeutrino, 0, -999.0)
                lead_bjet_PNetRegPtRawRes = choose_jet(dijets["first_jet"].PNetRegPtRawRes, 0, -999.0)
                lead_bjet_jet_idx = choose_jet(dijets["first_jet"].index, 0, -999.0)
                lead_bjet_rawFactor = choose_jet(dijets["first_jet"].rawFactor, 0, -999.0)
                lead_bjet_pt_orig = choose_jet(dijets["first_jet"].pt_orig, 0, -999.0)

                sublead_bjet_pt = choose_jet(dijets["second_jet"].pt, 0, -999.0)
                sublead_bjet_eta = choose_jet(dijets["second_jet"].eta, 0, -999.0)
                sublead_bjet_phi = choose_jet(dijets["second_jet"].phi, 0, -999.0)
                sublead_bjet_mass = choose_jet(dijets["second_jet"].mass, 0, -999.0)
                sublead_bjet_charge = choose_jet(dijets["second_jet"].charge, 0, -999.0)
                sublead_bjet_btagPNetB = choose_jet(dijets["second_jet"].btagPNetB, 0, -999.0)
                sublead_bjet_PNetRegPtRawCorr = choose_jet(dijets["second_jet"].PNetRegPtRawCorr, 0, -999.0)
                sublead_bjet_PNetRegPtRawCorrNeutrino = choose_jet(dijets["second_jet"].PNetRegPtRawCorrNeutrino, 0, -999.0)
                sublead_bjet_PNetRegPtRawRes = choose_jet(dijets["second_jet"].PNetRegPtRawRes, 0, -999.0)
                sublead_bjet_jet_idx = choose_jet(dijets["second_jet"].index, 0, -999.0)
                sublead_bjet_rawFactor = choose_jet(dijets["second_jet"].rawFactor, 0, -999.0)
                sublead_bjet_pt_orig = choose_jet(dijets["second_jet"].pt_orig, 0, -999.0)

                MET_2D = ak.Array(
                    {
                        "pt": puppiMET.pt,
                        "phi": puppiMET.phi,
                    },
                    with_name="Momentum2D",
                )
                lead_bjet_2D = ak.Array(
                    {
                        "pt": lead_bjet_pt_orig,
                        "phi": lead_bjet_phi,
                    },
                    with_name="Momentum2D",
                )
                sublead_bjet_2D = ak.Array(
                    {
                        "pt": sublead_bjet_pt_orig,
                        "phi": sublead_bjet_phi,
                    },
                    with_name="Momentum2D",
                )
                lead_bjet_2D_PNet = ak.Array(
                    {
                        "pt": lead_bjet_pt,
                        "phi": lead_bjet_phi,
                    },
                    with_name="Momentum2D",
                )
                sublead_bjet_2D_PNet = ak.Array(
                    {
                        "pt": sublead_bjet_pt,
                        "phi": sublead_bjet_phi,
                    },
                    with_name="Momentum2D",
                )
                # rough type-1 PNet MET
                # derived from removing lead/sublead Puppi bjets
                # and including their PNet-corrected variants
                MET_2D_PNet = (
                    MET_2D
                    + (lead_bjet_2D + sublead_bjet_2D)
                    - (lead_bjet_2D_PNet + sublead_bjet_2D_PNet)
                )

                dijet_pt = choose_jet(dijets.pt, 0, -999.0)
                dijet_eta = choose_jet(dijets.eta, 0, -999.0)
                dijet_phi = choose_jet(dijets.phi, 0, -999.0)
                dijet_mass = choose_jet(dijets.mass, 0, -999.0)
                dijet_charge = choose_jet(dijets.charge, 0, -999.0)

                # Add the bjet matching
                #   - boolean array of matched bjet
                #   - genPartonFlav array of matched genbJet
                if self.data_kind == "mc":
                    for bjet_type, bjet_4mom in {
                        "lead": ak.firsts(dijets["first_jet"]),
                        "sublead": ak.firsts(dijets["second_jet"])
                    }.items():
                        for key, jet_flav in [(f"{bjet_type}_bjet_genMatched", False), (f"{bjet_type}_bjet_genFlav", True)]:
                            value = match_jet(bjet_4mom, genjets, None, -999.0, jet_flav=jet_flav)
                            diphotons[f"{AnType}_{key}"] = value

                # Get the HHbbgg object
                HHbbgg = get_HHbbgg(self, diphotons, dijets)

                # Add the genHiggs matching
                if self.data_kind == "mc":
                    Higgs = ak.zip(
                        {
                            "Higgs_toGG": HHbbgg["obj_diphoton"],
                            "Higgs_tobb": HHbbgg["obj_dijet"],
                        }
                    )
                    if ak.sum(ak.num(events.GenPart[genHiggs_mask])) != 0:
                        for prop in ["Higgs_toGG", "Higgs_tobb"]:
                            key = f"{prop}_genMatched"
                            value = match_jet(Higgs[prop], genHiggs, None, -999.0)
                            diphotons[f"{AnType}_{key}"] = value

                # Write the variables in diphotons
                diphotons[f"{AnType}_HHbbggCandidate_pt"] = ak.fill_none(HHbbgg.obj_HHbbgg.pt, -999.0)
                diphotons[f"{AnType}_HHbbggCandidate_eta"] = ak.fill_none(HHbbgg.obj_HHbbgg.eta, -999.0)
                diphotons[f"{AnType}_HHbbggCandidate_phi"] = ak.fill_none(HHbbgg.obj_HHbbgg.phi, -999.0)
                diphotons[f"{AnType}_HHbbggCandidate_mass"] = ak.fill_none(HHbbgg.obj_HHbbgg.mass, -999.0)

                diphotons[f"{AnType}_M_X"] = ak.fill_none(HHbbgg.obj_HHbbgg.mass - HHbbgg.obj_diphoton.mass - HHbbgg.obj_dijet.mass + (2 * 125), -999.0)

                diphotons[f"{AnType}_lead_bjet_pt"] = lead_bjet_pt
                diphotons[f"{AnType}_lead_bjet_eta"] = lead_bjet_eta
                diphotons[f"{AnType}_lead_bjet_phi"] = lead_bjet_phi
                diphotons[f"{AnType}_lead_bjet_mass"] = lead_bjet_mass
                diphotons[f"{AnType}_lead_bjet_charge"] = lead_bjet_charge
                diphotons[f"{AnType}_lead_bjet_btagPNetB"] = lead_bjet_btagPNetB
                diphotons[f"{AnType}_lead_bjet_PNetRegPtRawCorr"] = lead_bjet_PNetRegPtRawCorr
                diphotons[f"{AnType}_lead_bjet_PNetRegPtRawCorrNeutrino"] = lead_bjet_PNetRegPtRawCorrNeutrino
                diphotons[f"{AnType}_lead_bjet_PNetRegPtRawRes"] = lead_bjet_PNetRegPtRawRes
                diphotons[f"{AnType}_lead_bjet_jet_idx"] = lead_bjet_jet_idx
                diphotons[f"{AnType}_lead_bjet_rawFactor"] = lead_bjet_rawFactor
                diphotons[f"{AnType}_lead_bjet_pt_orig"] = lead_bjet_pt_orig

                diphotons[f"{AnType}_sublead_bjet_pt"] = sublead_bjet_pt
                diphotons[f"{AnType}_sublead_bjet_eta"] = sublead_bjet_eta
                diphotons[f"{AnType}_sublead_bjet_phi"] = sublead_bjet_phi
                diphotons[f"{AnType}_sublead_bjet_mass"] = sublead_bjet_mass
                diphotons[f"{AnType}_sublead_bjet_charge"] = sublead_bjet_charge
                diphotons[f"{AnType}_sublead_bjet_btagPNetB"] = sublead_bjet_btagPNetB
                diphotons[f"{AnType}_sublead_bjet_PNetRegPtRawCorr"] = sublead_bjet_PNetRegPtRawCorr
                diphotons[f"{AnType}_sublead_bjet_PNetRegPtRawCorrNeutrino"] = sublead_bjet_PNetRegPtRawCorrNeutrino
                diphotons[f"{AnType}_sublead_bjet_PNetRegPtRawRes"] = sublead_bjet_PNetRegPtRawRes
                diphotons[f"{AnType}_sublead_bjet_jet_idx"] = sublead_bjet_jet_idx
                diphotons[f"{AnType}_sublead_bjet_rawFactor"] = sublead_bjet_rawFactor
                diphotons[f"{AnType}_sublead_bjet_pt_orig"] = sublead_bjet_pt_orig

                diphotons[f"{AnType}_MET_ptPNetCorr"] = ak.where(lead_bjet_2D.pt != -999.0, MET_2D_PNet.pt, -999.0)
                diphotons[f"{AnType}_MET_phiPNetCorr"] = ak.where(lead_bjet_2D.pt != -999.0, MET_2D_PNet.phi, -999.0)

                diphotons[f"{AnType}_dijet_pt"] = dijet_pt
                diphotons[f"{AnType}_dijet_eta"] = dijet_eta
                diphotons[f"{AnType}_dijet_phi"] = dijet_phi
                diphotons[f"{AnType}_dijet_mass"] = dijet_mass
                diphotons[f"{AnType}_dijet_charge"] = dijet_charge

                diphotons[f"{AnType}_pholead_PtOverM"] = ak.fill_none(HHbbgg.pho_lead.pt / HHbbgg.obj_diphoton.mass, -999.0)
                diphotons[f"{AnType}_phosublead_PtOverM"] = ak.fill_none(HHbbgg.pho_sublead.pt / HHbbgg.obj_diphoton.mass, -999.0)

                diphotons[f"{AnType}_FirstJet_PtOverM"] = ak.fill_none(diphotons[f"{AnType}_lead_bjet_pt"] / diphotons[f"{AnType}_dijet_mass"], -999.0)
                diphotons[f"{AnType}_SecondJet_PtOverM"] = ak.fill_none(diphotons[f"{AnType}_sublead_bjet_pt"] / diphotons[f"{AnType}_dijet_mass"], -999.0)

                diphotons[f"{AnType}_DeltaR_j1g1"] = ak.fill_none(DeltaR(HHbbgg.first_jet, HHbbgg.pho_lead), -999.0)
                diphotons[f"{AnType}_DeltaR_j2g1"] = ak.fill_none(DeltaR(HHbbgg.second_jet, HHbbgg.pho_lead), -999.0)
                diphotons[f"{AnType}_DeltaR_j1g2"] = ak.fill_none(DeltaR(HHbbgg.first_jet, HHbbgg.pho_sublead), -999.0)
                diphotons[f"{AnType}_DeltaR_j2g2"] = ak.fill_none(DeltaR(HHbbgg.second_jet, HHbbgg.pho_sublead), -999.0)

                DeltaR_comb = ak.Array([diphotons[f"{AnType}_DeltaR_j1g1"], diphotons[f"{AnType}_DeltaR_j2g1"], diphotons[f"{AnType}_DeltaR_j1g2"], diphotons[f"{AnType}_DeltaR_j2g2"]])

                diphotons[f"{AnType}_DeltaR_jg_min"] = ak.min(DeltaR_comb, axis=0)

                # ttH Killer vars #
                b_dijets = ak.firsts(dijets)
                chi_t0 = getChi_t0(
                    b_dijets,
                    dijets_for_tth_killer,
                    n_jets,
                    -999.0,
                )
                chi_t1 = getChi_t1(
                    b_dijets,
                    dijets_for_tth_killer,
                    n_jets,
                    -999.0,
                )
                diphotons[f"{AnType}_chi_t0"] = ak.fill_none(chi_t0,-999.0)
                diphotons[f"{AnType}_chi_t1"] = ak.fill_none(chi_t1,-999.0)
                diphotons[f"{AnType}_DeltaPhi_j1MET"] = ak.fill_none(DeltaPhi(HHbbgg.first_jet, puppiMET), -999.0)
                diphotons[f"{AnType}_DeltaPhi_j2MET"] = ak.fill_none(DeltaPhi(HHbbgg.second_jet, puppiMET), -999.0)
                diphotons[f"{AnType}_CosThetaStar_CS"] = ak.fill_none(getCosThetaStar_CS(HHbbgg, 6800), -999.0)
                diphotons[f"{AnType}_CosThetaStar_gg"] = ak.fill_none(getCosThetaStar_gg(HHbbgg), -999.0)
                diphotons[f"{AnType}_CosThetaStar_jj"] = ak.fill_none(getCosThetaStar_jj(HHbbgg), -999.0)

                if AnType == "nonRes":
                    # Add VBF jets information
                    # HHbbgg = ak.with_name(HHbbgg, "PtEtaPhiMCandidate", behavior=candidate.behavior)
                    # jets = ak.with_name(jets, "PtEtaPhiMCandidate", behavior=candidate.behavior)
                    jets["dr_VBFj_b1"] = ak.fill_none(jets.delta_r(HHbbgg.first_jet), -999.0)
                    jets["dr_VBFj_b2"] = ak.fill_none(jets.delta_r(HHbbgg.second_jet), -999.0)
                    jets["dr_VBFj_g1"] = ak.fill_none(jets.delta_r(HHbbgg.pho_lead), -999.0)
                    jets["dr_VBFj_g2"] = ak.fill_none(jets.delta_r(HHbbgg.pho_sublead), -999.0)

                    # VBF jet selection
                    vbf_jets = jets[(jets.pt > 30) & (jets.dr_VBFj_b1 > 0.4) & (jets.dr_VBFj_b2 > 0.4)]
                    vbf_jet_pair = ak.combinations(
                        vbf_jets, 2, fields=("first_jet", "second_jet")
                    )
                    vbf = ak.zip({
                        "first_jet": vbf_jet_pair["0"],
                        "second_jet": vbf_jet_pair["1"],
                        "dijet": vbf_jet_pair["0"] + vbf_jet_pair["1"],
                    })
                    vbf = vbf[vbf.first_jet.pt > 40.]
                    vbf = vbf[ak.argsort(vbf.dijet.mass, ascending=False)]
                    vbf = ak.firsts(vbf)

                    # Store VBF jets properties
                    vbf_jets_properties = ["pt", "eta", "phi", "mass", "charge", "btagPNetB", "PNetRegPtRawCorr", "PNetRegPtRawCorrNeutrino", "PNetRegPtRawRes", "btagPNetQvG", "btagDeepFlav_QG", "rawFactor", "pt_orig"]
                    for i in vbf.fields:
                        vbf_properties = vbf_jets_properties if i != "dijet" else vbf_jets_properties[:5]
                        for prop in vbf_properties:
                            key = f"VBF_{i}_{prop}"
                            value = ak.fill_none(getattr(vbf[i], prop), -999)
                            # Store the value in the diphotons dictionary
                            diphotons[key] = value

                    diphotons["VBF_first_jet_PtOverM"] = ak.where(diphotons.VBF_first_jet_pt != -999, diphotons.VBF_first_jet_pt / diphotons.VBF_dijet_mass, -999)
                    diphotons["VBF_second_jet_PtOverM"] = ak.where(diphotons.VBF_second_jet_pt != -999, diphotons.VBF_second_jet_pt / diphotons.VBF_dijet_mass, -999)
                    diphotons["VBF_first_jet_index"] = ak.fill_none(vbf.first_jet.index, -999)
                    diphotons["VBF_second_jet_index"] = ak.fill_none(vbf.second_jet.index, -999)

                    diphotons["VBF_jet_eta_prod"] = ak.fill_none(vbf.first_jet.eta * vbf.second_jet.eta, -999)
                    diphotons["VBF_jet_eta_diff"] = ak.fill_none(vbf.first_jet.eta - vbf.second_jet.eta, -999)
                    diphotons["VBF_jet_eta_sum"] = ak.fill_none(vbf.first_jet.eta + vbf.second_jet.eta, -999)

                    diphotons["VBF_DeltaR_j1b1"] = ak.fill_none(vbf.first_jet.dr_VBFj_b1, -999)
                    diphotons["VBF_DeltaR_j1b2"] = ak.fill_none(vbf.first_jet.dr_VBFj_b2, -999)
                    diphotons["VBF_DeltaR_j2b1"] = ak.fill_none(vbf.second_jet.dr_VBFj_b1, -999)
                    diphotons["VBF_DeltaR_j2b2"] = ak.fill_none(vbf.second_jet.dr_VBFj_b2, -999)

                    diphotons["VBF_DeltaR_j1g1"] = ak.fill_none(vbf.first_jet.dr_VBFj_g1, -999)
                    diphotons["VBF_DeltaR_j1g2"] = ak.fill_none(vbf.first_jet.dr_VBFj_g2, -999)
                    diphotons["VBF_DeltaR_j2g1"] = ak.fill_none(vbf.second_jet.dr_VBFj_g1, -999)
                    diphotons["VBF_DeltaR_j2g2"] = ak.fill_none(vbf.second_jet.dr_VBFj_g2, -999)

                    DeltaR_jb = ak.Array([diphotons["VBF_DeltaR_j1b1"], diphotons["VBF_DeltaR_j2b1"], diphotons["VBF_DeltaR_j1b2"], diphotons["VBF_DeltaR_j2b2"]])
                    DeltaR_jg = ak.Array([diphotons["VBF_DeltaR_j1g1"], diphotons["VBF_DeltaR_j2g1"], diphotons["VBF_DeltaR_j1g2"], diphotons["VBF_DeltaR_j2g2"]])

                    diphotons["VBF_DeltaR_jb_min"] = ak.min(DeltaR_jb, axis=0)
                    diphotons["VBF_DeltaR_jg_min"] = ak.min(DeltaR_jg, axis=0)

                    diphotons["VBF_Cgg"] = ak.where(diphotons.VBF_jet_eta_diff != -999, Cxx(diphotons.eta, diphotons.VBF_jet_eta_diff, diphotons.VBF_jet_eta_sum), -999)
                    diphotons["VBF_Cbb"] = ak.where(diphotons.VBF_jet_eta_diff != -999, Cxx(diphotons.nonRes_dijet_eta, diphotons.VBF_jet_eta_diff, diphotons.VBF_jet_eta_sum), -999)

                # add flags for the presence of btagged jets for the different analyses
                diphotons[f"{AnType}_has_two_btagged_jets"] = (diphotons[f"{AnType}_sublead_bjet_pt"] > -998)
                diphotons[f"{AnType}_has_atleast_one_fatjet"] = (diphotons["n_fatjets"] > 0)
                diphotons[f"is_{AnType}"] = (diphotons[f"{AnType}_has_two_btagged_jets"]
                                             | diphotons[f"{AnType}_has_atleast_one_fatjet"])

                self.calc_cut_flow(f"{AnType}_two_btagged_jets", diphotons[diphotons[f"{AnType}_has_two_btagged_jets"]], metadata)
                self.calc_cut_flow(f"{AnType}_two_btagged_jets_or_atleast_one_fatjet", diphotons[diphotons[f"is_{AnType}"]], metadata)

            diphotons = diphotons[(diphotons["is_Res"] | diphotons["is_nonRes"])]

            diphotons["nBTight"] = nBTight
            diphotons["nBMedium"] = nBMedium
            diphotons["nBLoose"] = nBLoose
            # Addition of lepton info-> Taken from the top workflow. This part of the code was orignally written by Florain Mausolf
            # Adding a 'generation' field to electrons and muons
            sel_electrons['generation'] = ak.ones_like(sel_electrons.pt)
            sel_muons['generation'] = 2 * ak.ones_like(sel_muons.pt)

            # Combine electrons and muons into a single leptons collection
            leptons = ak.concatenate([sel_electrons, sel_muons], axis=1)
            leptons = ak.with_name(leptons, "PtEtaPhiMCandidate")

            # Sort leptons by pt in descending order
            try:
                leptons = leptons[ak.argsort(leptons.pt, ascending=False)]
            except ValueError as e:
                logger.warning(f"Error sorting leptons: {e}")

            n_leptons = ak.num(leptons)
            diphotons["n_leptons"] = n_leptons

            # Annotate diphotons with selected leptons properties
            lepton_properties = leptons.fields
            for i in range(self.num_leptons_to_store):  # Number of leptons to select
                for prop in lepton_properties:
                    key = f"lepton{i+1}_{prop}"
                    # Retrieve the value using the choose_jet function (which can be used for leptons as well)
                    value = choose_jet(getattr(leptons, prop), i, -999.0)
                    # Store the value in the diphotons dictionary
                    diphotons[key] = value

            # ttH Killer vars cont.
            for jet in range(self.num_jets_to_store):
                for lep in range(self.num_leptons_to_store):
                    diphotons[f"DeltaR_j{jet+1}l{lep+1}"] = ak.fill_none(DeltaR(ak.firsts(jets[ak.local_index(jets) == jet]), ak.firsts(leptons[ak.local_index(leptons) == lep])), -999.0)
            diphotons["DeltaR_b1l1"] = ak.fill_none(DeltaR(HHbbgg.first_jet, ak.firsts(leptons[ak.local_index(leptons) == 0])), -999.0)
            diphotons["DeltaR_b2l1"] = ak.fill_none(DeltaR(HHbbgg.second_jet, ak.firsts(leptons[ak.local_index(leptons) == 0])), -999.0)
            diphotons["DeltaR_b1l2"] = ak.fill_none(DeltaR(HHbbgg.first_jet, ak.firsts(leptons[ak.local_index(leptons) == 1])), -999.0)
            diphotons["DeltaR_b2l2"] = ak.fill_none(DeltaR(HHbbgg.second_jet, ak.firsts(leptons[ak.local_index(leptons) == 1])), -999.0)

            if self.data_kind == 'mc':
                # add in gen lepton info #
                #   - boolean array of matched to electron: same as jet matching
                #   - boolean array of matched to muon: same as jet matching
                genLeptons = events.GenDressedLepton[
                    (events.GenDressedLepton.pdgId == 11)  # electron
                    | (events.GenDressedLepton.pdgId == -11)  # electron
                    | (events.GenDressedLepton.pdgId == 13)  # muon
                    | (events.GenDressedLepton.pdgId == -13)  # muon
                ]
                genLeptons["charge"] = ak.where(genLeptons.pdgId > 0, -1, 0) + ak.where(genLeptons.pdgId < 0, 1, 0)
                genLeptons = ak.with_name(genLeptons, "PtEtaPhiMCandidate")
                genLeptons = genLeptons[ak.argsort(genLeptons.pt, ascending=False)]
                genLepton_abs_pdgId = ak.where(genLeptons.pdgId > 0, genLeptons.pdgId, -genLeptons.pdgId)

                for i in range(self.num_leptons_to_store):  # Number of leptons to select
                    key = f"lepton{i+1}_genMatched"
                    # Retrieve the value using the choose_jet function (which can be used for leptons as well)
                    value = match_jet(
                        leptons,
                        genLeptons[
                            (
                                genLepton_abs_pdgId == ak.where(
                                    ak.firsts(leptons[ak.local_index(leptons) == i])["generation"] == 1, 11, 13
                                )
                            )
                        ],
                        i, -999.0
                    )
                    # Store the value in the diphotons dictionary
                    diphotons[key] = value

            # addition of fatjets and matching subjets, genjet

            fatjet_properties = fatjets.fields
            subjet_properties = subjets.fields
            if self.data_kind == "mc":
                genjetAK8_properties = genjetsAK8.fields
            # Fatjet variables that aren't needed, so they aren't saved to parquet.
            fatjet_drop_vars = {
                "subjet1": {
                    "n2b1",
                    "n3b1",
                    "tau1",
                    "tau2",
                    "tau3",
                    "tau4",
                    "hadronFlavour",
                    "nBHadrons",
                    "nCHadrons",
                    "charge",
                },
                "subjet2": {
                    "n2b1",
                    "n3b1",
                    "tau1",
                    "tau2",
                    "tau3",
                    "tau4",
                    "hadronFlavour",
                    "nBHadrons",
                    "nCHadrons",
                    "charge",
                },
                "genjetAK8": {"charge"},
                "particleNet_XteVsQCD": None,
                "particleNet_XtmVsQCD": None,
                "particleNet_XttVsQCD": None,
                "particleNetWithMass_H4qvsQCD": None,
                "particleNetWithMass_HccvsQCD": None,
                "particleNetWithMass_TvsQCD": None,
                "particleNetWithMass_WvsQCD": None,
                "particleNetWithMass_ZvsQCD": None,
                "electronIdx3SJ": None,
                "muonIdx3SJ": None,
                "lsf3": None,
                "charge": None,
            }

            for i in range(self.num_fatjets_to_store):  # Number of fatjets to select
                for prop in fatjet_properties:
                    if prop[-1] == "G":  # Few of the Idx variables are repeated with name ending with G (eg: 'subJetIdx1G'). Have to figure out why is this the case
                        continue
                    if prop in fatjet_drop_vars.keys():
                        continue
                    key = f"fatjet{i+1}_{prop}"
                    # Retrieve the value using the choose_jet function (which can be used for fatjets as well)
                    value = choose_jet(fatjets[prop], i, -999.0)

                    if prop == "genJetAK8Idx":  # add info of matched GenJetAK8
                        for prop_genJetAK8 in genjetAK8_properties:
                            if prop_genJetAK8 in fatjet_drop_vars["genjetAK8"]:
                                continue
                            key_genJetAK8 = f"fatjet{i+1}_genjetAK8_{prop_genJetAK8}"
                            # Retrieve the value using the choose_jet function (which can also be used here)
                            value_genJetAK8 = choose_jet(genjetsAK8[prop_genJetAK8], value, -999.0)
                            # Store the value in the diphotons dictionary
                            diphotons[key_genJetAK8] = value_genJetAK8
                        continue  # not saving the index values

                    if prop in ["subJetIdx1", "subJetIdx2"]:  # add info of matched SubJets
                        subjet_name = prop.replace("Idx", "").lower()
                        for prop_subjet in subjet_properties:
                            if prop_subjet in fatjet_drop_vars[subjet_name]:
                                continue
                            key_subjet = f"fatjet{i+1}_{subjet_name}_{prop_subjet}"
                            # Retrieve the value using the choose_jet function (which can also be used here)
                            value_subjet = choose_jet(subjets[prop_subjet], value, -999.0)
                            # Store the value in the diphotons dictionary
                            diphotons[key_subjet] = value_subjet
                        continue  # not saving the index values

                    # Store the value in the diphotons dictionary
                    diphotons[key] = value

            ## ----------  End of the HHTobbgg part ----------

            # run taggers on the events list with added diphotons
            # the shape here is ensured to be broadcastable
            for tagger in self.taggers:
                (
                    diphotons["_".join([tagger.name, str(tagger.priority)])],
                    tagger_extra,
                ) = tagger(
                    events, diphotons
                )  # creates new column in diphotons - tagger priority, or 0, also return list of histrograms here?
                histos_etc.update(tagger_extra)

            # if there are taggers to run, arbitrate by them first
            # Deal with order of tagger priorities
            # Turn from diphoton jagged array to whether or not an event was selected
            if len(self.taggers):
                counts = ak.num(diphotons.pt, axis=1)
                flat_tags = numpy.stack(
                    (
                        ak.flatten(
                            diphotons[
                                "_".join([tagger.name, str(tagger.priority)])
                            ]
                        )
                        for tagger in self.taggers
                    ),
                    axis=1,
                )
                tags = ak.from_regular(
                    ak.unflatten(flat_tags, counts), axis=2
                )
                winner = ak.min(tags[tags != 0], axis=2)
                diphotons["best_tag"] = winner

                # lowest priority is most important (ascending sort)
                # leave in order of diphoton pT in case of ties (stable sort)
                sorted = ak.argsort(diphotons.best_tag, stable=True)
                diphotons = diphotons[sorted]

            diphotons = ak.firsts(diphotons)
            # set diphotons as part of the event record
            events[f"diphotons_{do_variation}"] = diphotons
            # annotate diphotons with event information
            diphotons["event"] = events.event
            diphotons["lumi"] = events.luminosityBlock
            diphotons["run"] = events.run
            # nPV just for validation of pileup reweighting
            diphotons["nPV"] = events.PV.npvs
            diphotons["fixedGridRhoAll"] = events.Rho.fixedGridRhoAll
            # annotate diphotons with dZ information (difference between z position of GenVtx and PV) as required by flashggfinalfits
            if self.data_kind == "mc":
                diphotons["genWeight"] = events.genWeight
                diphotons["dZ"] = events.GenVtx.z - events.PV.z
                # Necessary for differential xsec measurements in final fits ("truth" variables)
                diphotons["HTXS_Higgs_pt"] = events.HTXS.Higgs_pt
                diphotons["HTXS_Higgs_y"] = events.HTXS.Higgs_y
                diphotons["HTXS_njets30"] = events.HTXS.njets30  # Need to clarify if this variable is suitable, does it fulfill abs(eta_j) < 2.5? Probably not
                # Preparation for HTXS measurements later, start with stage 0 to disentangle VH into WH and ZH for final fits
                diphotons["HTXS_stage_0"] = events.HTXS.stage_0
            # Fill zeros for data because there is no GenVtx for data, obviously
            else:
                diphotons["dZ"] = ak.zeros_like(events.PV.z)

            # drop events without a preselected diphoton candidate
            # drop events without a tag, if there are tags
            if len(self.taggers):
                selection_mask = ~(
                    ak.is_none(diphotons)
                    | ak.is_none(diphotons.best_tag)
                )
                diphotons = diphotons[selection_mask]
            else:
                selection_mask = ~ak.is_none(diphotons)
                diphotons = diphotons[selection_mask]

            # return if there is no surviving events
            if len(diphotons) == 0:
                logger.info("No surviving events in this run, return now!")
                return histos_etc
            if self.data_kind == "mc":
                # initiate Weight container here, after selection, since event selection cannot easily be applied to weight container afterwards
                event_weights = Weights(size=len(events[selection_mask]), storeIndividual=True)
                # set weights to generator weights
                event_weights._weight = ak.to_numpy(events["genWeight"][selection_mask])

                # corrections to event weights:
                for correction_name in correction_names:
                    if correction_name in available_weight_corrections:
                        logger.info(
                            f"Adding correction {correction_name} to weight collection of dataset {dataset_name}"
                        )
                        varying_function = available_weight_corrections[
                            correction_name
                        ]
                        event_weights = varying_function(
                            events=events[selection_mask],
                            photons=events[f"diphotons_{do_variation}"][selection_mask],
                            # adding muons and electrons because I don't want to introduce a naming obligation like e.g. "sel_muons" in the syst functions
                            muons=events["sel_muons"][selection_mask],
                            electrons=events["sel_electrons"][selection_mask],
                            weights=event_weights,
                            dataset_name=dataset_name,
                            year=self.year[dataset_name][0],
                        )
                diphotons["bTagWeight"] = event_weights.partial_weight(include=["bTagSF"])

                # systematic variations of event weights go to nominal output dataframe:
                if do_variation == "nominal":
                    for systematic_name in systematic_names:
                        if systematic_name in available_weight_systematics:
                            logger.info(
                                f"Adding systematic {systematic_name} to weight collection of dataset {dataset_name}"
                            )
                            if systematic_name == "LHEScale":
                                if hasattr(events, "LHEScaleWeight"):
                                    diphotons["nweight_LHEScale"] = ak.num(
                                        events.LHEScaleWeight[selection_mask],
                                        axis=1,
                                    )
                                    diphotons[
                                        "weight_LHEScale"
                                    ] = events.LHEScaleWeight[selection_mask]
                                else:
                                    logger.info(
                                        f"No {systematic_name} Weights in dataset {dataset_name}"
                                    )
                            elif systematic_name == "LHEPdf":
                                if hasattr(events, "LHEPdfWeight"):
                                    # two AlphaS weights are removed
                                    diphotons["nweight_LHEPdf"] = (
                                        ak.num(
                                            events.LHEPdfWeight[selection_mask],
                                            axis=1,
                                        )
                                        - 2
                                    )
                                    diphotons[
                                        "weight_LHEPdf"
                                    ] = events.LHEPdfWeight[selection_mask][
                                        :, :-2
                                    ]
                                else:
                                    logger.info(
                                        f"No {systematic_name} Weights in dataset {dataset_name}"
                                    )
                            else:
                                varying_function = available_weight_systematics[
                                    systematic_name
                                ]
                                event_weights = varying_function(
                                    events=events[selection_mask],
                                    photons=events[f"diphotons_{do_variation}"][selection_mask],
                                    # adding muons and electrons because I don't want to introduce a naming obligation like e.g. "sel_muons" in the syst functions
                                    muons=events["sel_muons"][selection_mask],
                                    electrons=events["sel_electrons"][selection_mask],
                                    weights=event_weights,
                                    dataset_name=dataset_name,
                                    year=self.year[dataset_name][0],
                                )

                diphotons["weight"] = event_weights.weight()
                diphotons["weight_central"] = event_weights.weight() / events["genWeight"][selection_mask]

                metadata["sum_weight_central"] = str(
                    ak.sum(event_weights.weight())
                )
                metadata["sum_weight_central_wo_bTagSF"] = str(
                    ak.sum(event_weights.weight() / (event_weights.partial_weight(include=["bTagSF"])))
                )

                # Store variations with respect to central weight
                if do_variation == "nominal":
                    if len(event_weights.variations):
                        logger.info(
                            "Adding systematic weight variations to nominal output file."
                        )
                    for modifier in event_weights.variations:
                        diphotons["weight_" + modifier] = event_weights.weight(
                            modifier=modifier
                        )
                        if ("bTagSF" in modifier):
                            metadata["sum_weight_" + modifier] = str(
                                ak.sum(event_weights.weight(modifier=modifier))
                            )

                # Store weights for different width effects
                json_file = os.path.join(os.path.dirname(__file__), "../tools/Weights_interference_HHbbgg.json")
                with open(json_file, "r") as jf:
                    weights_interference = json.load(jf)
                if "RelWidth" in dataset_name:
                    diphotons["weight_interference"] = [weights_interference[str(dataset_name.split('_')[-2] + '_' + dataset_name.split('_')[-1])]] * len(events[selection_mask])
                else:
                    diphotons["weight_interference"] = ak.full_like(diphotons["weight_central"], 0)

            # Add weight variables (=1) for data for consistent datasets
            else:
                diphotons["weight_central"] = ak.ones_like(
                    diphotons["event"]
                )
                diphotons["weight"] = ak.ones_like(diphotons["event"])

            # Compute and store the different variations of sigma_m_over_m
            diphotons = compute_sigma_m(diphotons, processor='base', flow_corrections=self.doFlow_corrections, smear=self.Smear_sigma_m, IsData=(self.data_kind == "data"))

            # Decorrelating the mass resolution - Still need to supress the decorrelator noises
            if self.doDeco:

                # Decorrelate nominal sigma_m_over_m
                diphotons["sigma_m_over_m_nominal_decorr"] = decorrelate_mass_resolution(diphotons, type="nominal", year=self.year[dataset_name][0])

                # decorrelate smeared nominal sigma_m_overm_m
                if (self.Smear_sigma_m):
                    diphotons["sigma_m_over_m_smeared_decorr"] = decorrelate_mass_resolution(diphotons, type="smeared", year=self.year[dataset_name][0])

                # decorrelate flow corrected sigma_m_over_m
                if (self.doFlow_corrections):
                    diphotons["sigma_m_over_m_corr_decorr"] = decorrelate_mass_resolution(diphotons, type="corr", year=self.year[dataset_name][0])

                # decorrelate flow corrected smeared sigma_m_over_m
                if (self.doFlow_corrections and self.Smear_sigma_m):
                    if self.data_kind == "data" and ("Scale_IJazZ" in correction_names or "Scale2G_IJazZ" in correction_names):
                        diphotons["sigma_m_over_m_corr_smeared_decorr"] = decorrelate_mass_resolution(diphotons, type="corr_smeared", year=self.year[dataset_name][0], IsSAS_ET_Dependent=True)
                    elif self.data_kind == "mc" and ("Smearing2G_IJazZ" in correction_names or "Smearing_IJazZ" in correction_names):
                        diphotons["sigma_m_over_m_corr_smeared_decorr"] = decorrelate_mass_resolution(diphotons, type="corr_smeared", year=self.year[dataset_name][0], IsSAS_ET_Dependent=True)
                    else:
                        diphotons["sigma_m_over_m_corr_smeared_decorr"] = decorrelate_mass_resolution(diphotons, type="corr_smeared", year=self.year[dataset_name][0], IsSAS_ET_Dependent=True)

                # Instead of the nominal sigma_m_over_m, we will use the smeared version of it -> (https://indico.cern.ch/event/1319585/#169-update-on-the-run-3-mass-r)
                # else:
                #    warnings.warn("Smeamering need to be applied in order to decorrelate the (Smeared) mass resolution. -- Exiting!")
                #    sys.exit(0)

            if self.output_location is not None:
                if self.output_format == "root":
                    df = diphoton_list_to_pandas(self, diphotons)
                else:
                    akarr = diphoton_ak_array(self, diphotons)

                    # Remove fixedGridRhoAll from photons to avoid having event-level info per photon
                    akarr = akarr[
                        [
                            field
                            for field in akarr.fields
                            if "lead_fixedGridRhoAll" not in field
                        ]
                    ]

                fname = (
                    events.behavior[
                        "__events_factory__"
                    ]._partition_key.replace("/", "_")
                    + ".%s" % self.output_format
                )
                fname = (fname.replace("%2F","")).replace("%3B1","")
                subdirs = []
                if "dataset" in events.metadata:
                    subdirs.append(events.metadata["dataset"])
                subdirs.append(do_variation)
                if self.output_format == "root":
                    dump_pandas(self, df, fname, self.output_location, subdirs)
                else:
                    dump_ak_array(
                        self, akarr, fname, self.output_location, metadata, subdirs,
                    )

        return histos_etc

    def calc_cut_flow(self, cut_name, diphotons, metadata):
        # function takes input array and adds number of events (for data) or sum of genweights (for mc) for a cut_name in metadata
        # for use with calculating cutflow table
        if self.data_kind == "mc":
            counts = ak.sum(diphotons[~ak.is_none(ak.firsts(diphotons))].genWeight)
        else:
            counts = len(diphotons[~ak.is_none(ak.firsts(diphotons))])

        metadata[f"{cut_name}"] = str(counts)

    def postprocess(self, accumulant: Dict[Any, Any]) -> Any:
        pass
