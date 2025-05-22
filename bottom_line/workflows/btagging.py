###############################################################################
#                                                                             #
#  This is just an EXAMPLE btagging file made for the phase space of the      #
#  intermediate Run3 analysis of the cross sections.                          #
#                                                                             #
#  In case you want to compute the btagging efficiencies for your analysis,   #
#  insert all your selections on the jets before the calling of select_jets.  #
#  Then run the btagging processor. Do NOT change the MANDATORY PART.         #
#                                                                             #
###############################################################################

from bottom_line.workflows.base import bbMETBaseProcessor
from bottom_line.tools.SC_eta import add_photon_SC_eta
from bottom_line.tools.EELeak_region import veto_EEleak_flag
from bottom_line.tools.EcalBadCalibCrystal_events import remove_EcalBadCalibCrystal_events
from bottom_line.tools.gen_helpers import get_fiducial_flag, get_genJets, get_higgs_gen_attributes
from bottom_line.selections.photon_selections import photon_preselection
from bottom_line.selections.diphoton_selections import build_diphoton_candidates, apply_fiducial_cut_det_level
from bottom_line.selections.lepton_selections import select_electrons, select_muons
from bottom_line.selections.jet_selections import select_jets, jetvetomap, getBTagMVACut
from bottom_line.selections.lumi_selections import select_lumis
from bottom_line.utils.dumping_utils import (
    get_obj_syst_dict,
)
from bottom_line.utils.misc_utils import choose_jet
from bottom_line.tools.flow_corrections import apply_flow_corrections_to_photons

# from bottom_line.utils.dumping_utils import diphoton_list_to_pandas, dump_pandas
from bottom_line.systematics import object_systematics as available_object_systematics
from bottom_line.systematics import object_corrections as available_object_corrections
from bottom_line.systematics import weight_systematics as available_weight_systematics
from bottom_line.systematics import weight_corrections as available_weight_corrections
from bottom_line.systematics import apply_systematic_variations_object_level

import os
import warnings
from typing import Any, Dict, List, Optional
import awkward as ak
import numpy
import sys
import vector
from copy import deepcopy
import pathlib
import pickle


import logging

logger = logging.getLogger(__name__)

vector.register_awkward()


class BTaggingEfficienciesProcessor(bbMETBaseProcessor):
    def __init__(
        self,
        metaconditions: Dict[str, Any],
        systematics: Dict[str, List[Any]] = None,
        corrections: Dict[str, List[Any]] = None,
        apply_trigger: bool = False,
        output_location: Optional[str] = None,
        taggers: Optional[List[Any]] = None,
        nano_version: int = None,
        bTagEffFileName: Optional[str] = None,
        trigger_group: str = ".*DoubleEG.*",
        analysis: str = "mainAnalysis",
        applyCQR: bool = False,
        skipJetVetoMap: bool = False,
        year: Dict[str, List[str]] = None,
        fiducialCuts: str = "classical",
        doDeco: bool = False,
        Smear_sigma_m: bool = False,
        doFlow_corrections: bool = False,
        output_format: str = "parquet"
    ) -> None:
        super().__init__(
            metaconditions,
            systematics=systematics,
            corrections=corrections,
            apply_trigger=apply_trigger,
            nano_version=nano_version,
            bTagEffFileName=bTagEffFileName,
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

    def process(self, events: ak.Array) -> Dict[Any, Any]:
        dataset_name = events.metadata["dataset"]

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

        # If --Smear-sigma_m == True and no Smearing correction in .json for MC throws an error, since the pt spectrum need to be smeared in order to properly calculate the smeared sigma_m_m
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
                Applying correction {correction_name} to dataset {dataset_name}\n
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

        # Add additional collections if object systematics should be applied
        collections = {
            "Photon": original_photons,
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

            if self.data_kind == "mc":
                # Add the fiducial flags for particle level
                diphotons['fiducialClassicalFlag'] = get_fiducial_flag(events, flavour='Classical')
                diphotons['fiducialGeometricFlag'] = get_fiducial_flag(events, flavour='Geometric')

                GenPTH, GenYH, GenPhiH = get_higgs_gen_attributes(events)

                GenPTH = ak.fill_none(GenPTH, -999.0)
                diphotons['GenPTH'] = GenPTH

                genJets = get_genJets(self, events, pt_cut=30., eta_cut=2.5)
                diphotons['GenNJ'] = ak.num(genJets)
                GenPTJ0 = choose_jet(genJets.pt, 0, -999.0)  # Choose zero (leading) jet and pad with -999 if none
                diphotons['GenPTJ0'] = GenPTJ0

                gen_first_jet_eta = choose_jet(genJets.eta, 0, -999.0)
                gen_first_jet_mass = choose_jet(genJets.mass, 0, -999.0)
                gen_first_jet_phi = choose_jet(genJets.phi, 0, -999.0)

                gen_first_jet_pz = GenPTJ0 * numpy.sinh(gen_first_jet_eta)
                gen_first_jet_energy = numpy.sqrt((GenPTJ0**2 * numpy.cosh(gen_first_jet_eta)**2) + gen_first_jet_mass**2)

                # B-Jets
                # Following the recommendations of https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideBTagMCTools for hadronFlavour
                # and the Run 2 recommendations for the bjets
                genJetCondition = (genJets.pt > 30) & (numpy.abs(genJets.eta) < 2.5)
                genBJetCondition = genJetCondition & (genJets.hadronFlavour == 5)
                genJets = ak.with_field(genJets, genBJetCondition, "GenIsBJet")
                genCJetCondition = genJetCondition & (genJets.hadronFlavour == 4)
                genJets = ak.with_field(genJets, genCJetCondition, "GenIsCJet")
                genLJetCondition = genJetCondition & (genJets.hadronFlavour == 0)
                genJets = ak.with_field(genJets, genLJetCondition, "GenIsLJet")
                num_bjets = ak.sum(genJets["GenIsBJet"], axis=-1)
                diphotons["GenNBJet"] = num_bjets

                gen_first_bjet_pt = choose_jet(genJets[genJets["GenIsBJet"] == True].pt, 0, -999.0)
                diphotons["GenPTbJ0"] = gen_first_bjet_pt

                gen_first_jet_hFlav = choose_jet(genJets.hadronFlavour, 0, -999.0)
                diphotons["GenJ0hFlav"] = gen_first_jet_hFlav

                with numpy.errstate(divide='ignore', invalid='ignore'):
                    GenYJ0 = 0.5 * numpy.log((gen_first_jet_energy + gen_first_jet_pz) / (gen_first_jet_energy - gen_first_jet_pz))

                GenYJ0 = ak.fill_none(GenYJ0, -999)
                GenYJ0 = ak.where(numpy.isnan(GenYJ0), -999, GenYJ0)
                diphotons['GenYJ0'] = GenYJ0

                GenYH = ak.fill_none(GenYH, -999)
                GenYH = ak.where(numpy.isnan(GenYH), -999, GenYH)
                diphotons['GenYH'] = GenYH

                GenAbsPhiHJ0 = numpy.abs(gen_first_jet_phi - GenPhiH)

                # Set all entries above 2*pi to -999
                GenAbsPhiHJ0 = ak.where(
                    GenAbsPhiHJ0 > 2 * numpy.pi,
                    -999,
                    GenAbsPhiHJ0
                )
                GenAbsPhiHJ0_pi_array = ak.full_like(GenAbsPhiHJ0, 2 * numpy.pi)

                # Select the smallest angle
                GenAbsPhiHJ0 = ak.where(
                    GenAbsPhiHJ0 > numpy.pi,
                    GenAbsPhiHJ0_pi_array - GenAbsPhiHJ0,
                    GenAbsPhiHJ0
                )
                GenAbsPhiHJ0 = ak.fill_none(GenAbsPhiHJ0, -999.0)

                diphotons["GenDPhiHJ0"] = GenAbsPhiHJ0

                GenAbsYHJ0 = numpy.abs(GenYJ0 - GenYH)

                # Set all entries above 500 to -999
                GenAbsYHJ0 = ak.where(
                    GenAbsYHJ0 > 500,
                    -999,
                    GenAbsYHJ0
                )

                diphotons["GenDYHJ0"] = GenAbsYHJ0

            # baseline modifications to diphotons
            if self.diphoton_mva is not None:
                diphotons = self.add_diphoton_mva(diphotons, events)

            # workflow specific processing
            events, process_extra = self.process_extra(events)
            histos_etc.update(process_extra)

            btagMVA_selection = {
                "deepJet": {"btagDeepFlavB": jets.btagDeepFlavB},  # Always available
                "particleNet": {"btagPNetB": jets.btagPNetB} if self.nano_version >= 12 else {},
                "robustParticleTransformer": {"btagRobustParTAK4B": jets.btagRobustParTAK4B} if self.nano_version in [12, 13] else {},
            }

            # jet_variables
            jets = ak.zip(
                {
                    "pt": jets.pt,
                    "eta": jets.eta,
                    "phi": jets.phi,
                    "mass": jets.mass,
                    "charge": ak.zeros_like(
                        jets.pt
                    ),
                    **btagMVA_selection.get(self.bjet_mva, {}),
                    "hFlav": jets.hadronFlavour if self.data_kind == "mc" else ak.zeros_like(jets.pt),
                    "btagDeepFlav_CvB": jets.btagDeepFlavCvB,
                    "btagDeepFlav_CvL": jets.btagDeepFlavCvL,
                    "btagDeepFlav_QG": jets.btagDeepFlavQG,
                    "jetId": jets.jetId,
                    **(
                        {"neHEF": jets.neHEF, "neEmEF": jets.neEmEF, "chEmEF": jets.chEmEF, "muEF": jets.muEF} if self.nano_version == 12 else {}
                    ),
                    **(
                        {"neHEF": jets.neHEF, "neEmEF": jets.neEmEF, "chMultiplicity": jets.chMultiplicity, "neMultiplicity": jets.neMultiplicity, "chEmEF": jets.chEmEF, "chHEF": jets.chHEF, "muEF": jets.muEF} if self.nano_version == 13 else {}
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
                }
            )
            electrons = ak.with_name(electrons, "PtEtaPhiMCandidate")

            # Special cut for base workflow to replicate iso cut for electrons also for muons
            events['Muon'] = events.Muon[events.Muon.pfRelIso03_all < 0.2]

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
                    "pfIsoId": events.Muon.pfIsoId
                }
            )
            muons = ak.with_name(muons, "PtEtaPhiMCandidate")

            # lepton cleaning
            sel_electrons = electrons[
                select_electrons(self, electrons, diphotons)
            ]
            sel_muons = muons[select_muons(self, muons, diphotons)]

            # jet selection and pt ordering
            jets = jets[
                select_jets(self, jets, diphotons, sel_muons, sel_electrons)
            ]
            jets = jets[ak.argsort(jets.pt, ascending=False)]

            # adding selected jets to events to be used in ctagging SF calculation
            events["sel_jets"] = jets

            #################################################################################
            #################################################################################
            #################################################################################
            #                                                                               #
            #  The full jet selection for your analysis has to be made up to this point.    #
            #                                                                               #
            #                  B E G I N   O F   M A N D A T O R Y   P A R T                #
            #                                                                               #
            #################################################################################
            #################################################################################
            #################################################################################

            # Based on recommendations for the tight QCD WP seen here: https://btv-wiki.docs.cern.ch/PerformanceCalibration/#working-points
            btag_WP = getBTagMVACut(mva_name=self.bjet_mva,
                                    mva_wp=self.bjet_wp,
                                    year=self.year[dataset_name][0])

            btag_mva_column = list(btagMVA_selection[self.bjet_mva].keys())[0]

            # B-Jets
            bJetCondition = (jets.pt > 30) & (abs(jets.eta) < 2.5) & (jets[btag_mva_column] >= btag_WP)
            jets = ak.with_field(jets, bJetCondition, "IsBJet")
            num_bjets = ak.sum(jets["IsBJet"], axis=-1)
            diphotons["NBJet"] = num_bjets

            # Efficiency variables
            selected_bjets = jets[jets.hFlav == 5]

            bJetCondition_bjet = (selected_bjets.pt > 30) & (numpy.abs(selected_bjets.eta) < 2.5) & (selected_bjets[btag_mva_column] >= btag_WP)
            jetCondition_bjet = (selected_bjets.pt > 30) & (numpy.abs(selected_bjets.eta) < 2.5)
            num_bjets_bjet = ak.num(selected_bjets[bJetCondition_bjet])
            diphotons["NBJet_GenBJet"] = num_bjets_bjet
            num_jets_bjet = ak.num(selected_bjets[jetCondition_bjet])
            diphotons["NJet_GenBJet"] = num_jets_bjet

            genBJets_selected_bjets_pt_list = ak.flatten(selected_bjets[bJetCondition_bjet].pt.to_list())
            genJets_selected_bjets_pt_list = ak.flatten(selected_bjets[jetCondition_bjet].pt.to_list())

            selected_cjets = jets[jets.hFlav == 4]
            bJetCondition_cjet = (selected_cjets.pt > 30) & (numpy.abs(selected_cjets.eta) < 2.5) & (selected_cjets[btag_mva_column] >= btag_WP)
            jetCondition_cjet = (selected_cjets.pt > 30) & (numpy.abs(selected_cjets.eta) < 2.5)
            num_bjets_cjet = ak.num(selected_cjets[bJetCondition_cjet])
            diphotons["NBJet_GenCJet"] = num_bjets_cjet
            num_jets_cjet = ak.num(selected_cjets[jetCondition_cjet])
            diphotons["NJet_GenCJet"] = num_jets_cjet

            genCJets_selected_cjets_pt_list = ak.flatten(selected_cjets[bJetCondition_cjet].pt.to_list())
            genJets_selected_cjets_pt_list = ak.flatten(selected_cjets[jetCondition_cjet].pt.to_list())

            selected_ljets = jets[jets.hFlav == 0]
            bJetCondition_ljet = (selected_ljets.pt > 30) & (numpy.abs(selected_ljets.eta) < 2.5) & (selected_ljets[btag_mva_column] >= btag_WP)
            jetCondition_ljet = (selected_ljets.pt > 30) & (numpy.abs(selected_ljets.eta) < 2.5)
            num_bjets_ljet = ak.num(selected_ljets[bJetCondition_ljet])
            diphotons["NBJet_GenLJet"] = num_bjets_ljet
            num_jets_ljet = ak.num(selected_ljets[jetCondition_ljet])
            diphotons["NJet_GenLJet"] = num_jets_ljet

            genLJets_selected_ljets_pt_list = ak.flatten(selected_ljets[bJetCondition_ljet].pt.to_list())
            genJets_selected_ljets_pt_list = ak.flatten(selected_ljets[jetCondition_ljet].pt.to_list())

            selected_jets_pt_dict = {
                "genBJet_recoBJet": genBJets_selected_bjets_pt_list,
                "genBJet_recoJet": genJets_selected_bjets_pt_list,
                "genCJet_recoBJet": genCJets_selected_cjets_pt_list,
                "genCJet_recoJet": genJets_selected_cjets_pt_list,
                "genLJet_recoBJet": genLJets_selected_ljets_pt_list,
                "genLJet_recoJet": genJets_selected_ljets_pt_list,
            }

            if self.output_location is not None:
                fname = (
                    events.behavior[
                        "__events_factory__"
                    ]._partition_key.replace("/", "_")
                    + ".%s" % self.output_format
                )
                subdirs = []
                if "dataset" in events.metadata:
                    subdirs.append(events.metadata["dataset"])
                subdirs.append(do_variation)

                # Create output directory
                output_dir = os.path.join(self.output_location, os.path.sep.join(subdirs))
                pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

                # Output pt lists
                output_pkl = os.path.join(self.output_location, os.path.sep.join(subdirs), fname).replace("parquet", "pkl")
                with open(output_pkl, 'wb') as pickle_file:
                    pickle.dump(selected_jets_pt_dict, pickle_file)

            #################################################################################
            #################################################################################
            #################################################################################
            #                                                                               #
            #                   E N D    O F    M A N D A T O R Y    P A R T.               #
            #                                                                               #
            #################################################################################
            #################################################################################
            #################################################################################

        return histos_etc

    def process_extra(self, events: ak.Array) -> ak.Array:
        return events, {}

    def postprocess(self, accumulant: Dict[Any, Any]) -> Any:
        pass
