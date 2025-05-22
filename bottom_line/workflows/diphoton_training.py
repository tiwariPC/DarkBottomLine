from bottom_line.workflows.base import bbMETBaseProcessor
from bottom_line.tools.chained_quantile import ChainedQuantileRegression
from bottom_line.tools.xgb_loader import load_bdt
from bottom_line.tools.photonid_mva import load_photonid_mva
from bottom_line.tools.SC_eta import add_photon_SC_eta
from bottom_line.tools.EELeak_region import veto_EEleak_flag
from bottom_line.tools.gen_helpers import get_fiducial_flag, get_higgs_gen_attributes
from bottom_line.tools.sigma_m_tools import compute_sigma_m
from bottom_line.selections.photon_selections import photon_preselection
from bottom_line.selections.diphoton_selections import build_diphoton_candidates, apply_fiducial_cut_det_level
from bottom_line.selections.lepton_selections import select_electrons, select_muons
from bottom_line.selections.jet_selections import select_jets, jetvetomap
from bottom_line.selections.lumi_selections import select_lumis
from bottom_line.utils.dumping_utils import diphoton_ak_array_fields as diphoton_ak_array
from bottom_line.utils.dumping_utils import (
    dump_ak_array,
    diphoton_list_to_pandas,
    dump_pandas,
    get_obj_syst_dict,
)
from bottom_line.utils.misc_utils import choose_jet
from bottom_line.tools.flow_corrections import apply_flow_corrections_to_photons

from bottom_line.tools.mass_decorrelator import decorrelate_mass_resolution

# from bottom_line.utils.dumping_utils import diphoton_list_to_pandas, dump_pandas
from bottom_line.metaconditions import photon_id_mva_weights
from bottom_line.metaconditions import diphoton as diphoton_mva_dir
from bottom_line.systematics import object_systematics as available_object_systematics
from bottom_line.systematics import object_corrections as available_object_corrections
from bottom_line.systematics import weight_systematics as available_weight_systematics
from bottom_line.systematics import weight_corrections as available_weight_corrections

import functools
import os
import warnings
from typing import Any, Dict, List, Optional
import awkward as ak
import numpy
import sys
import vector
from coffea.analysis_tools import Weights
from copy import deepcopy

import logging

logger = logging.getLogger(__name__)

vector.register_awkward()


class DiphoTrainingProcessor(bbMETBaseProcessor):  # type: ignore
    def __init__(
        self,
        metaconditions: Dict[str, Any],
        systematics: Optional[Dict[str, List[str]]] = None,
        corrections: Optional[Dict[str, List[str]]] = None,
        apply_trigger: bool = False,
        nano_version: int = 13,
        output_location: Optional[str] = None,
        taggers: Optional[List[Any]] = None,
        trigger_group: str = ".*DoubleEG.*",
        analysis: str = "mainAnalysis",
        applyCQR: bool = False,
        skipJetVetoMap: bool = False,
        year: Dict[str, List[str]] = None,
        fiducialCuts: str = "simple",
        doDeco: bool = False,
        Smear_sigma_m: bool = False,
        doFlow_corrections: bool = False,
        output_format: str = "parquet",
    ) -> None:
        super().__init__(
            metaconditions,
            systematics=systematics,
            corrections=corrections,
            apply_trigger=apply_trigger,
            nano_version=nano_version,
            output_location=output_location,
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

        # muon selection cuts
        self.muon_pt_threshold = 10
        self.muon_max_eta = 2.4
        self.mu_id_wp = "medium"
        self.global_muon = False

        # electron selection cuts
        self.electron_pt_threshold = 15
        self.electron_max_eta = 2.5
        self.el_id_wp = "WP80"

        # jet selection cuts
        self.jet_jetId = "tight"  # can be "tightLepVeto", "tight" or "loose": https://twiki.cern.ch/twiki/bin/view/CMS/JetID13p6TeV#nanoAOD_Flags
        self.jet_dipho_min_dr = 0.4
        self.jet_pho_min_dr = 0.4
        self.jet_ele_min_dr = 0.4
        self.jet_muo_min_dr = 0.4
        self.jet_pt_threshold = 20
        self.jet_max_eta = 2.5

        self.clean_jet_dipho = True
        self.clean_jet_pho = True
        self.clean_jet_ele = False
        self.clean_jet_muo = False

        # diphoton preselection cuts
        self.min_pt_photon = 25.0
        self.min_pt_lead_photon = 35.0
        self.min_pt_mgg_lead_photon = 1. / 3.
        self.min_pt_mgg_sublead_photon = 0.25
        self.min_mass_range = 100
        self.max_mass_range = 180
        self.min_mvaid = -0.9
        self.max_sc_eta = 2.5
        self.gap_barrel_eta = 1.4442
        self.gap_endcap_eta = 1.566
        self.max_hovere = 0.08
        self.min_full5x5_r9 = 0.8
        self.max_chad_iso = 20.0
        self.max_chad_rel_iso = 0.3

        self.min_full5x5_r9_EB_high_r9 = 0.85
        self.min_full5x5_r9_EE_high_r9 = 0.9
        self.min_full5x5_r9_EB_low_r9 = 0.5
        self.min_full5x5_r9_EE_low_r9 = 0.8
        self.max_trkSumPtHollowConeDR03_EB_low_r9 = 6.0  # for v11, we cut on Photon_pfChargedIsoPFPV
        self.max_trkSumPtHollowConeDR03_EE_low_r9 = 6.0  # Leaving the names of the preselection cut variables the same to change as little as possible
        self.max_sieie_EB_low_r9 = 0.015
        self.max_sieie_EE_low_r9 = 0.035
        self.max_pho_iso_EB_low_r9 = 4.0
        self.max_pho_iso_EE_low_r9 = 4.0

        self.eta_rho_corr = 1.5
        self.low_eta_rho_corr = 0.16544
        self.high_eta_rho_corr = 0.13212
        # EA values for Run3 from Egamma
        self.EA1_EB1 = 0.102056
        self.EA2_EB1 = -0.000398112
        self.EA1_EB2 = 0.0820317
        self.EA2_EB2 = -0.000286224
        self.EA1_EE1 = 0.0564915
        self.EA2_EE1 = -0.000248591
        self.EA1_EE2 = 0.0428606
        self.EA2_EE2 = -0.000171541
        self.EA1_EE3 = 0.0395282
        self.EA2_EE3 = -0.000121398
        self.EA1_EE4 = 0.0369761
        self.EA2_EE4 = -8.10369e-05
        self.EA1_EE5 = 0.0369417
        self.EA2_EE5 = -2.76885e-05
        self.e_veto = 0.5

        logger.debug(f"Setting up processor with metaconditions: {self.meta}")
        logger.debug(f"self.meta = {self.meta}")
        logger.debug(f"self.year = {self.year}")
        logger.debug(f"self.corrections = {self.corrections}")
        logger.debug(f"self.systematics = {self.systematics}")
        logger.debug(f"self.output_location = {self.output_location}")
        logger.debug(f"self.trigger_group = {self.trigger_group}")

        self.taggers = []
        if taggers is not None:
            self.taggers = taggers
            self.taggers.sort(key=lambda x: x.priority)

        self.prefixes = {"pho_lead": "lead", "pho_sublead": "sublead"}

        if not self.doDeco:
            logger.info("Skipping Mass resolution decorrelation as required")
        else:
            logger.info("Performing Mass resolution decorrelation as required")

        # build the chained quantile regressions
        if self.applyCQR:
            try:
                self.chained_quantile: Optional[
                    ChainedQuantileRegression
                ] = ChainedQuantileRegression(**self.meta["PhoIdInputCorrections"])
            except Exception as e:
                warnings.warn(f"Could not instantiate ChainedQuantileRegression: {e}")
                self.chained_quantile = None
        else:
            logger.info("Skipping CQR as required")
            self.chained_quantile = None

        # initialize photonid_mva
        photon_id_mva_dir = os.path.dirname(photon_id_mva_weights.__file__)
        try:
            logger.debug(
                f"Looking for {self.meta['flashggPhotons']['photonIdMVAweightfile_EB']} in {photon_id_mva_dir}"
            )
            self.photonid_mva_EB = load_photonid_mva(
                os.path.join(
                    photon_id_mva_dir,
                    self.meta["flashggPhotons"]["photonIdMVAweightfile_EB"],
                )
            )
            self.photonid_mva_EE = load_photonid_mva(
                os.path.join(
                    photon_id_mva_dir,
                    self.meta["flashggPhotons"]["photonIdMVAweightfile_EE"],
                )
            )
        except Exception as e:
            warnings.warn(f"Could not instantiate PhotonID MVA on the fly: {e}")
            self.photonid_mva_EB = None
            self.photonid_mva_EE = None

        # initialize diphoton mva
        diphoton_weights_dir = os.path.dirname(diphoton_mva_dir.__file__)
        logger.debug(
            f"Base path to look for IDMVA weight files: {diphoton_weights_dir}"
        )

        # this is needed because the new training has a k-fold approach, we have two models, one for
        # even event numbers and one for odd event numbers
        self.diphoton_mva = [None, None]
        try:
            self.diphoton_mva[0] = load_bdt(
                os.path.join(
                    diphoton_weights_dir, self.meta["HiggsDNA_DiPhotonMVA"]["weightFile"][0]
                )
            )
            self.diphoton_mva[1] = load_bdt(
                os.path.join(
                    diphoton_weights_dir, self.meta["HiggsDNA_DiPhotonMVA"]["weightFile"][1]
                )
            )
        except Exception as e:
            warnings.warn(f"Could not instantiate diphoton MVA: {e}")
            self.diphoton_mva = None

    def process_extra(self, events: ak.Array) -> ak.Array:
        raise NotImplementedError

    def process(self, events: ak.Array) -> Dict[Any, Any]:
        dataset_name = events.metadata["dataset"]

        # Filter to remove overlap from bkg samples
        if ("QCD" in dataset_name) or ("GJet" in dataset_name):
            if "QCD" in dataset_name:
                MC_filter = (ak.num(events.Photon.pt[events.Photon.genPartFlav == 1]) == 0)
            else:
                MC_filter = (ak.num(events.Photon.pt[events.Photon.genPartFlav == 1]) <= 1)
            logger.debug("MC filter to remove overlap betwee GG GJ and JJ samples")
            logger.debug(f"Sample: {dataset_name}")
            logger.debug(f"Photons.genPartFlav = {events.Photon.genPartFlav}")
            logger.debug(f"Filter              = {MC_filter}")
            logger.info(f"initial number of events: {len(events)}")
            events = events[MC_filter]
            logger.info(f"number of events after MC filter: {len(events)}")

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
        # apply jetvetomap
        if not self.skipJetVetoMap:
            events = jetvetomap(
                self, events, logger, dataset_name, year=self.year[dataset_name][0]
            )
        # metadata array to append to higgsdna output
        metadata = {}

        if self.data_kind == "mc":
            logger.info("processing MC dataset")
            # Add sum of gen weights before selection for normalisation in postprocessing
            metadata["sum_genw_presel"] = str(ak.sum(events.genWeight))
        else:
            logger.info("processing Data dataset")
            metadata["sum_genw_presel"] = "Data"

        # apply filters and triggers
        events = self.apply_filters_and_triggers(events)

        # we need ScEta for corrections and systematics, it is present in NanoAODv13+ and can be calculated using PV for older versions
        events.Photon = add_photon_SC_eta(events.Photon, events.PV)

        # add veto EE leak branch for photons, could also be used for electrons
        if (
            self.year[dataset_name][0] == "2022EE"
            or self.year[dataset_name][0] == "2022postEE"
        ):
            events.Photon = veto_EEleak_flag(self, events.Photon)

        # read which systematics and corrections to process
        try:
            correction_names = self.corrections[dataset_name]
        except KeyError:
            correction_names = []
        try:
            systematic_names = self.systematics[dataset_name]
        except KeyError:
            systematic_names = []

        # If --Smear-sigma_m == True and no Smearing correction in .json for MC throws an error, since the pt scpectrum need to be smeared in order to properly calculate the smeared sigma_m_m
        if (
            self.data_kind == "mc"
            and self.Smear_sigma_m
            and ("Smearing" not in correction_names and "Et_dependent_Smearing" not in correction_names)
        ):
            warnings.warn(
                "Smearing or Et_dependent_Smearing should be specified in the corrections field in .json in order to smear the mass!"
            )
            sys.exit(0)

        # Since now we are applying Smearing term to the sigma_m_over_m i added this portion of code
        # specially for the estimation of smearing terms for the data events [data pt/energy] are not smeared!
        if self.data_kind == "data" and self.Smear_sigma_m:
            if "Scale" in correction_names:
                correction_name = "Smearing"
            elif "Et_dependent_Scale" in correction_names:
                correction_name = "Et_dependent_Smearing"
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

        # apply jetvetomap: only retain events that without any jets in the veto region
        if not self.skipJetVetoMap:
            events = jetvetomap(
                self, events, logger, dataset_name, year=self.year[dataset_name][0]
            )

        original_photons = events.Photon
        # NOTE: jet jerc systematics are added in the correction functions and handled later
        original_jets = events.Jet

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

        # systematic object variations
        for systematic_name in systematic_names:
            if systematic_name in available_object_systematics.keys():
                systematic_dct = available_object_systematics[systematic_name]
                if systematic_dct["object"] == "Photon":
                    logger.info(
                        f"Adding systematic {systematic_name} to photons collection of dataset {dataset_name}"
                    )
                    original_photons.add_systematic(
                        # passing the arguments here explicitly since I want to pass the events to the varying function. If there is a more elegant / flexible way, just change it!
                        name=systematic_name,
                        kind=systematic_dct["args"]["kind"],
                        what=systematic_dct["args"]["what"],
                        varying_function=functools.partial(
                            systematic_dct["args"]["varying_function"],
                            events=events,
                            year=self.year[dataset_name][0],
                        )
                        # name=systematic_name, **systematic_dct["args"]
                    )
                # to be implemented for other objects here
            elif systematic_name in available_weight_systematics:
                # event weight systematics will be applied after photon preselection / application of further taggers
                continue
            else:
                # may want to throw an error instead, needs to be discussed
                warnings.warn(
                    f"Could not process systematic variation {systematic_name}."
                )
                continue

        # Applying systematic variations
        photons_dct = {}
        photons_dct["nominal"] = original_photons
        logger.debug(original_photons.systematics.fields)
        for systematic in original_photons.systematics.fields:
            for variation in original_photons.systematics[systematic].fields:
                # deepcopy to allow for independent calculations on photon variables with CQR
                photons_dct[f"{systematic}_{variation}"] = deepcopy(
                    original_photons.systematics[systematic][variation]
                )

        # NOTE: jet jerc systematics are added in the corrections, now extract those variations and create the dictionary
        jerc_syst_list, jets_dct = get_obj_syst_dict(original_jets, ["pt", "mass"])
        # object systematics dictionary
        logger.debug(f"[ jerc systematics ] {jerc_syst_list}")

        # Build the flattened array of all possible variations
        variations_combined = []
        variations_combined.append(original_photons.systematics.fields)
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
            photons, jets = photons_dct["nominal"], events.Jet
            if variation == "nominal":
                pass  # Do nothing since we already get the unvaried, but nominally corrected objets above
            elif variation in [*photons_dct]:  # [*dict] gets the keys of the dict since Python >= 3.5
                photons = photons_dct[variation]
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

            # Apply the fiducial cut at detector level with helper function
            diphotons = apply_fiducial_cut_det_level(self, diphotons)

            # preselection, require at least one candidate
            dipho_presel_cut = ak.num(diphotons["mass"]) >= 1
            dipho_num = ak.num(diphotons["mass"])

            if self.data_kind == "mc":

                # Add the fiducial flags for particle level
                diphotons['fiducialClassicalFlag'] = get_fiducial_flag(events, flavour='Classical')
                diphotons['fiducialGeometricFlag'] = get_fiducial_flag(events, flavour='Geometric')

                GenPTH, GenYH, GenPhiH = get_higgs_gen_attributes(events)
                GenPTH = ak.fill_none(GenPTH, -999.0)

            diphotons = ak.firsts(diphotons)
            diphotons["n_dipho_cand"] = dipho_num

            # select only events with a candidate
            diphotons = diphotons[dipho_presel_cut]
            dipho_events = events[dipho_presel_cut]
            jets = jets[dipho_presel_cut]

            # baseline modifications to diphotons
            if self.diphoton_mva is not None:
                diphotons, dipho_events = self.add_diphoton_mva(diphotons, dipho_events)

            logger.info(f"selected number of events  after diphoton preselection {len(dipho_events)}")

            # jet_variables
            jets = ak.zip(
                {
                    "pt": jets.pt,
                    "eta": jets.eta,
                    "phi": jets.phi,
                    "mass": jets.mass,
                    "charge": jets.chEmEF,
                    "hFlav": jets.hadronFlavour if self.data_kind == "mc" else ak.ones_like(jets.pt) * -1.,
                    "particleNetAK4_CvsL": jets.particleNetAK4_CvsL if hasattr(jets, "particleNetAK4_CvsL") else ak.ones_like(jets.pt) * -1.,
                    "particleNetAK4_CvsB": jets.particleNetAK4_CvsB if hasattr(jets, "particleNetAK4_CvsB") else ak.ones_like(jets.pt) * -1.,
                    "particleNetAK4_B": jets.particleNetAK4_B if hasattr(jets, "particleNetAK4_B") else ak.ones_like(jets.pt) * -1.,
                    "btagDeepFlav_B": jets.btagDeepFlavB,
                    "btagDeepFlav_CvB": jets.btagDeepFlavCvB,
                    "btagDeepFlav_CvL": jets.btagDeepFlavCvL,
                    "btagDeepFlav_QG": jets.btagDeepFlavQG,
                    "jetId": jets.jetId,
                    "n_sv": jets.nSVs if hasattr(jets, "nSVs") else ak.ones_like(jets.pt) * -1.,
                    "n_muons": jets.nMuons if hasattr(jets, "nMuons") else ak.ones_like(jets.pt) * -1.,
                    "n_electrons": jets.nElectrons if hasattr(jets, "nElectrons") else ak.ones_like(jets.pt) * -1.,
                    "nConst": jets.nConstituents if hasattr(jets, "nConstituents") else ak.ones_like(jets.pt) * -1.,
                    "neHEF": jets.neHEF if hasattr(jets, "neHEF") else ak.ones_like(jets.pt) * -1.,
                    "neEmEF": jets.neEmEF if hasattr(jets, "neEmEF") else ak.ones_like(jets.pt) * -1.,
                    "chHEF": jets.chHEF if hasattr(jets, "chHEF") else ak.ones_like(jets.pt) * -1.,
                    "chEmEF": jets.neHEF if hasattr(jets, "chEmEF") else ak.ones_like(jets.pt) * -1.,
                }
            )
            jets = ak.with_name(jets, "PtEtaPhiMCandidate")
            if not hasattr(jets, "particleNetAK4_CvsL"):
                logger.info("your jet collection doesn't have the PNet score fields")

            electrons = ak.zip(
                {
                    "pt": dipho_events.Electron.pt,
                    "eta": dipho_events.Electron.eta,
                    "phi": dipho_events.Electron.phi,
                    "mass": dipho_events.Electron.mass,
                    "charge": dipho_events.Electron.charge,
                    "mvaIso_WP90": dipho_events.Electron.mvaIso_Fall17V2_WP90 if hasattr(dipho_events.Electron, "mvaIso_Fall17V2_WP90") else dipho_events.Electron.mvaIso_WP90,
                    "mvaIso_WP80": dipho_events.Electron.mvaIso_Fall17V2_WP80 if hasattr(dipho_events.Electron, "mvaIso_Fall17V2_WP80") else dipho_events.Electron.mvaIso_WP80,
                    "mvaIso_WPL": dipho_events.Electron.mvaIso_Fall17V2_WPL if hasattr(dipho_events.Electron, "mvaIso_Fall17V2_WPL") else dipho_events.Electron.mvaIso_WP80,
                }
            )
            electrons = ak.with_name(electrons, "PtEtaPhiMCandidate")

            muons = ak.zip(
                {
                    "pt": dipho_events.Muon.pt,
                    "eta": dipho_events.Muon.eta,
                    "phi": dipho_events.Muon.phi,
                    "mass": dipho_events.Muon.mass,
                    "charge": dipho_events.Muon.charge,
                    "tightId": dipho_events.Muon.tightId,
                    "mediumId": dipho_events.Muon.mediumId,
                    "looseId": dipho_events.Muon.looseId,
                    "isGlobal": dipho_events.Muon.isGlobal,
                    "pfIsoId": dipho_events.Muon.pfIsoId
                }
            )
            muons = ak.with_name(muons, "PtEtaPhiMCandidate")

            sv = ak.zip(
                {
                    "pt": dipho_events.SV.pt,
                    "eta": dipho_events.SV.eta,
                    "phi": dipho_events.SV.phi,
                    "mass": dipho_events.SV.mass,
                    "charge": dipho_events.SV.charge,
                    "dlen": dipho_events.SV.dlen,
                    "dlenSig": dipho_events.SV.dlenSig,
                    "dxy": dipho_events.SV.dxy,
                    "dxySig": dipho_events.SV.dxySig,
                    "pAngle": dipho_events.SV.pAngle,
                    "chi2": dipho_events.SV.chi2,
                    "x": dipho_events.SV.x,
                    "y": dipho_events.SV.y,
                    "z": dipho_events.SV.z,
                    "ndof": dipho_events.SV.ndof,
                    "ntracks": dipho_events.SV.ntracks,
                }
            )
            sv = ak.with_name(sv, "PtEtaPhiMCandidate")

            # lepton cleaning
            sel_electrons = electrons[
                select_electrons(self, electrons, diphotons)
            ]
            sel_muons = muons[select_muons(self, muons, diphotons)]

            # jet selection and pt ordering
            jets = jets[
                select_jets(self, jets, diphotons, sel_muons, sel_electrons)
            ]

            # sort diphotons by pT
            jets = jets[ak.argsort(jets.pt, ascending=False, axis=1)]

            n_jets = ak.num(jets)

            first_jet_pt = choose_jet(jets.pt, 0, -999.0)
            first_jet_eta = choose_jet(jets.eta, 0, -999.0)
            first_jet_phi = choose_jet(jets.phi, 0, -999.0)
            first_jet_mass = choose_jet(jets.mass, 0, -999.0)
            first_jet_charge = choose_jet(jets.charge, 0, -999.0)
            first_jet_hFlav = choose_jet(jets.hFlav, 0, -999.0)
            first_jet_DeepFlavour_CvsL = choose_jet(jets.btagDeepFlav_CvL, 0, -999.0)
            first_jet_DeepFlavour_CvsB = choose_jet(jets.btagDeepFlav_CvB, 0, -999.0)
            first_jet_DeepFlavour_B = choose_jet(jets.btagDeepFlav_B, 0, -999.0)
            first_jet_particleNetAK4_CvsL = choose_jet(jets.particleNetAK4_CvsL, 0, -999.0)
            first_jet_particleNetAK4_CvsB = choose_jet(jets.particleNetAK4_CvsB, 0, -999.0)
            first_jet_particleNetAK4_B = choose_jet(jets.particleNetAK4_B, 0, -999.0)
            first_jet_n_sv = choose_jet(jets.n_sv, 0, -999.0)
            first_jet_n_muons = choose_jet(jets.n_muons, 0, -999.0)
            first_jet_n_electrons = choose_jet(jets.n_electrons, 0, -999.0)

            second_jet_pt = choose_jet(jets.pt, 1, -999.0)
            second_jet_eta = choose_jet(jets.eta, 1, -999.0)
            second_jet_phi = choose_jet(jets.phi, 1, -999.0)
            second_jet_mass = choose_jet(jets.mass, 1, -999.0)
            second_jet_charge = choose_jet(jets.charge, 1, -999.0)
            second_jet_hFlav = choose_jet(jets.hFlav, 1, -999.0)
            second_jet_DeepFlavour_CvsL = choose_jet(jets.btagDeepFlav_CvL, 1, -999.0)
            second_jet_DeepFlavour_CvsB = choose_jet(jets.btagDeepFlav_CvB, 1, -999.0)
            second_jet_DeepFlavour_B = choose_jet(jets.btagDeepFlav_B, 1, -999.0)
            second_jet_particleNetAK4_CvsL = choose_jet(jets.particleNetAK4_CvsL, 1, -999.0)
            second_jet_particleNetAK4_CvsB = choose_jet(jets.particleNetAK4_CvsB, 1, -999.0)
            second_jet_particleNetAK4_B = choose_jet(jets.particleNetAK4_B, 1, -999.0)
            second_jet_n_sv = choose_jet(jets.n_sv, 1, -999.0)

            diphotons["first_jet_pt"] = first_jet_pt
            diphotons["first_jet_eta"] = first_jet_eta
            diphotons["first_jet_phi"] = first_jet_phi
            diphotons["first_jet_mass"] = first_jet_mass
            diphotons["first_jet_charge"] = first_jet_charge
            diphotons["first_jet_hFlav"] = first_jet_hFlav
            diphotons["first_jet_DeepFlavour_CvsL"] = first_jet_DeepFlavour_CvsL
            diphotons["first_jet_DeepFlavour_CvsB"] = first_jet_DeepFlavour_CvsB
            diphotons["first_jet_DeepFlavour_B"] = first_jet_DeepFlavour_B
            diphotons["first_jet_particleNetAK4_CvsL"] = first_jet_particleNetAK4_CvsL
            diphotons["first_jet_particleNetAK4_CvsB"] = first_jet_particleNetAK4_CvsB
            diphotons["first_jet_particleNetAK4_B"] = first_jet_particleNetAK4_B
            diphotons["first_jet_n_sv"] = first_jet_n_sv
            diphotons["first_jet_n_muons"] = first_jet_n_muons
            diphotons["first_jet_n_electrons"] = first_jet_n_electrons

            diphotons["second_jet_pt"] = second_jet_pt
            diphotons["second_jet_eta"] = second_jet_eta
            diphotons["second_jet_phi"] = second_jet_phi
            diphotons["second_jet_mass"] = second_jet_mass
            diphotons["second_jet_charge"] = second_jet_charge
            diphotons["second_jet_hFlav"] = second_jet_hFlav
            diphotons["second_jet_DeepFlavour_CvsL"] = second_jet_DeepFlavour_CvsL
            diphotons["second_jet_DeepFlavour_CvsB"] = second_jet_DeepFlavour_CvsB
            diphotons["second_jet_DeepFlavour_B"] = second_jet_DeepFlavour_B
            diphotons["second_jet_particleNetAK4_CvsL"] = second_jet_particleNetAK4_CvsL
            diphotons["second_jet_particleNetAK4_CvsB"] = second_jet_particleNetAK4_CvsB
            diphotons["second_jet_particleNetAK4_B"] = second_jet_particleNetAK4_B
            diphotons["second_jet_n_sv"] = second_jet_n_sv

            diphotons["n_jets"] = n_jets

            diphotons["LeadPhoton_pt"] = diphotons.pho_lead.pt
            diphotons["SubleadPhoton_pt"] = diphotons.pho_sublead.pt
            diphotons["LeadPhoton_energy"] = diphotons.pho_lead.energy
            diphotons["SubleadPhoton_energy"] = diphotons.pho_sublead.energy
            diphotons["LeadPhoton_r9"] = diphotons.pho_lead.r9
            diphotons["LeadPhoton_s4"] = diphotons.pho_lead.s4
            diphotons["LeadPhoton_sieie"] = diphotons.pho_lead.sieie
            diphotons["LeadPhoton_sieip"] = diphotons.pho_lead.sieip
            diphotons["LeadPhoton_etaWidth"] = diphotons.pho_lead.etaWidth
            diphotons["LeadPhoton_phiWidth"] = diphotons.pho_lead.phiWidth
            diphotons["LeadPhoton_pfChargedIsoPFPV"] = diphotons.pho_lead.pfChargedIsoPFPV
            diphotons["LeadPhoton_pfChargedIsoWorstVtx"] = diphotons.pho_lead.pfChargedIsoWorstVtx
            diphotons["LeadPhoton_pfPhoIso03"] = diphotons.pho_lead.pfPhoIso03

            # run taggers on the events list with added diphotons
            # the shape here is ensured to be broadcastable
            for tagger in self.taggers:
                (
                    diphotons["_".join([tagger.name, str(tagger.priority)])],
                    tagger_extra,
                ) = tagger(
                    dipho_events, diphotons
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

            # set diphotons as part of the event record
            dipho_events[f"diphotons_{do_variation}"] = diphotons
            # annotate diphotons with event information
            diphotons["event"] = dipho_events.event
            diphotons["lumi"] = dipho_events.luminosityBlock
            diphotons["run"] = dipho_events.run
            # nPV just for validation of pileup reweighting
            diphotons["nPV"] = dipho_events.PV.npvs if not self.data_kind else ak.ones_like(dipho_events.event)
            diphotons["nPU"] = dipho_events.Pileup.nPU if not self.data_kind else ak.ones_like(dipho_events.event)
            diphotons["rho"] = dipho_events.Rho.fixedGridRhoAll

            # here I add lead jet to event because I need it for cTagSF evaluation
            # this may be not needed if one change the cTagSF function in event_weight_systematics.py to use the jet collection
            dipho_events["first_jet_pt"] = first_jet_pt
            dipho_events["first_jet_eta"] = first_jet_eta
            dipho_events["first_jet_phi"] = first_jet_phi
            dipho_events["first_jet_mass"] = first_jet_mass
            dipho_events["first_jet_charge"] = first_jet_charge
            dipho_events["first_jet_hFlav"] = ak.values_astype(first_jet_hFlav, numpy.int)
            dipho_events["first_jet_DeepFlavour_CvsL"] = first_jet_DeepFlavour_CvsL
            dipho_events["first_jet_DeepFlavour_CvsB"] = first_jet_DeepFlavour_CvsB

            dipho_events["sel_jets"] = jets
            dipho_events["n_jets"] = n_jets

            # annotate diphotons with dZ information (difference between z position of GenVtx and PV) as required by flashggfinalfits
            if self.data_kind == "mc":
                diphotons["genWeight"] = dipho_events.genWeight
                diphotons["dZ"] = dipho_events.GenVtx.z - dipho_events.PV.z
                # Necessary for differential xsec measurements in final fits ("truth" variables)
                diphotons["HTXS_Higgs_pt"] = dipho_events.HTXS.Higgs_pt
                diphotons["HTXS_Higgs_y"] = dipho_events.HTXS.Higgs_y
                diphotons["HTXS_njets30"] = dipho_events.HTXS.njets30  # Need to clarify if this variable is suitable, does it fulfill abs(eta_j) < 2.5? Probably not
                # Preparation for HTXS measurements later, start with stage 0 to disentangle VH into WH and ZH for final fits
                diphotons["HTXS_stage_0"] = dipho_events.HTXS.stage_0
            # Fill zeros for data because there is no GenVtx for data, obviously
            else:
                diphotons["dZ"] = ak.zeros_like(dipho_events.PV.z)

            # list of field to dump to pandas
            fields = []
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
                event_weights = Weights(size=len(dipho_events[selection_mask]), storeIndividual=True)
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
                            events=dipho_events[selection_mask],
                            photons=dipho_events[f"diphotons_{do_variation}"][selection_mask],
                            weights=event_weights,
                            dataset_name=dataset_name,
                            year=self.year[dataset_name][0],
                        )
                # systematic variations of event weights go to nominal output dataframe:
                if do_variation == "nominal":
                    for systematic_name in systematic_names:
                        if systematic_name in available_weight_systematics:
                            logger.info(
                                f"Adding systematic {systematic_name} to weight collection of dataset {dataset_name}"
                            )
                            if systematic_name == "LHEScale":
                                if hasattr(dipho_events, "LHEScaleWeight"):
                                    diphotons["nweight_LHEScale"] = ak.num(
                                        dipho_events.LHEScaleWeight[selection_mask],
                                        axis=1,
                                    )
                                    diphotons[
                                        "weight_LHEScale"
                                    ] = dipho_events.LHEScaleWeight[selection_mask]
                                else:
                                    logger.info(
                                        f"No {systematic_name} Weights in dataset {dataset_name}"
                                    )
                            elif systematic_name == "LHEPdf":
                                if hasattr(dipho_events, "LHEPdfWeight"):
                                    # two AlphaS weights are removed
                                    diphotons["nweight_LHEPdf"] = (
                                        ak.num(
                                            dipho_events.LHEPdfWeight[selection_mask],
                                            axis=1,
                                        )
                                        - 2
                                    )
                                    diphotons[
                                        "weight_LHEPdf"
                                    ] = dipho_events.LHEPdfWeight[selection_mask][
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
                                    events=dipho_events[selection_mask],
                                    photons=dipho_events[f"diphotons_{do_variation}"][
                                        selection_mask
                                    ],
                                    weights=event_weights,
                                    dataset_name=dataset_name,
                                    year=self.year[dataset_name][0],
                                )

                diphotons["weight"] = event_weights.weight()
                diphotons["weight_central"] = event_weights.weight() / dipho_events[selection_mask].genWeight
                diphotons["genWeight"] = dipho_events[selection_mask].genWeight

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

            # Add weight variables (=1) for data for consistent datasets
            else:
                diphotons["weight_central"] = ak.ones_like(
                    diphotons["event"]
                )
                diphotons["weight"] = ak.ones_like(diphotons["event"])
                diphotons["genWeight"] = ak.ones_like(diphotons["event"])

            # Compute and store the different variations of sigma_m_over_m
            diphotons = compute_sigma_m(diphotons, processor='base', flow_corrections=self.doFlow_corrections, smear=self.Smear_sigma_m, IsData=(self.data_kind == "data"))

            diphotons["CMS_hgg_mass"] = diphotons.mass
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
                    if self.data_kind == "data" and "Et_dependent_Scale" in correction_names:
                        diphotons["sigma_m_over_m_corr_smeared_decorr"] = decorrelate_mass_resolution(diphotons, type="corr_smeared", year=self.year[dataset_name][0], IsSAS_ET_Dependent=True)
                    elif self.data_kind == "mc" and "Et_dependent_Smearing" in correction_names:
                        diphotons["sigma_m_over_m_corr_smeared_decorr"] = decorrelate_mass_resolution(diphotons, type="corr_smeared", year=self.year[dataset_name][0], IsSAS_ET_Dependent=True)
                    else:
                        diphotons["sigma_m_over_m_corr_smeared_decorr"] = decorrelate_mass_resolution(diphotons, type="corr_smeared", year=self.year[dataset_name][0])

                # Instead of the nominal sigma_m_over_m, we will use the smeared version of it -> (https://indico.cern.ch/event/1319585/#169-update-on-the-run-3-mass-r)
                # else:
                #    warnings.warn("Smeamering need to be applied in order to decorrelate the (Smeared) mass resolution. -- Exiting!")
                #    sys.exit(0)

            if self.output_location is not None:
                if self.output_format == "root":
                    df = diphoton_list_to_pandas(self, diphotons)
                else:
                    fields = fields + [
                        "weight",
                        "weight_central",
                        "genWeight",
                        "bdt_score",
                        "dZ",
                        "dZ_1",
                        "dZ_2",
                        "CMS_hgg_mass",
                        "event",
                        "pt",
                        "eta",
                        "phi",
                        "LeadPhoton_pt_mgg",
                        "LeadPhoton_eta",
                        "LeadPhoton_mvaID",
                        "SubleadPhoton_pt_mgg",
                        "SubleadPhoton_eta",
                        "SubleadPhoton_mvaID",
                        "LeadPhoton_pt",
                        "SubleadPhoton_pt",
                        "LeadPhoton_energy",
                        "SubleadPhoton_energy",
                        "Diphoton_cos_dPhi",
                        "sigmaMrv",
                        # "sigmaMwv",
                        "PV_score",
                        # "PV_chi2",
                        "nPV",
                        "nPU",
                        "rho",
                        "n_jets",
                        "first_jet_pt",
                        "first_jet_eta",
                        "first_jet_phi",
                        "first_jet_mass",
                        "first_jet_hFlav",
                        "first_jet_DeepFlavour_CvsL",
                        "first_jet_DeepFlavour_CvsB",
                        "first_jet_DeepFlavour_B",
                        "first_jet_particleNetAK4_CvsL",
                        "first_jet_particleNetAK4_CvsB",
                        "first_jet_particleNetAK4_B",
                        "first_jet_n_sv",
                        "first_jet_n_muons",
                        "first_jet_n_electrons",
                        "second_jet_pt",
                        "second_jet_eta",
                        "second_jet_phi",
                        "second_jet_mass",
                        "second_jet_hFlav",
                        "second_jet_DeepFlavour_CvsL",
                        "second_jet_DeepFlavour_CvsB",
                        "second_jet_DeepFlavour_B",
                        "second_jet_particleNetAK4_CvsL",
                        "second_jet_particleNetAK4_CvsB",
                        "second_jet_particleNetAK4_B",
                        "second_jet_n_sv",
                    ]
                    for f in diphotons.fields:
                        if "weight" in f:
                            fields = fields + [f]
                    # df = diphoton_list_to_pandas(self, diphotons, fields, logger)
                    akarr = diphoton_ak_array(self, diphotons, fields, logger)
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

    def postprocess(self, accumulant: Dict[Any, Any]) -> Any:
        pass
