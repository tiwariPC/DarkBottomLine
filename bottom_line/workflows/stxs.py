from bottom_line.workflows.base import bbMETBaseProcessor
from bottom_line.tools.SC_eta import add_photon_SC_eta
from bottom_line.tools.EELeak_region import veto_EEleak_flag
from bottom_line.tools.EcalBadCalibCrystal_events import remove_EcalBadCalibCrystal_events
from bottom_line.tools.gen_helpers import get_fiducial_flag, get_genJets, get_higgs_gen_attributes
from bottom_line.tools.sigma_m_tools import compute_sigma_m
from bottom_line.selections.photon_selections import photon_preselection
from bottom_line.selections.diphoton_selections import build_diphoton_candidates, apply_fiducial_cut_det_level
from bottom_line.selections.lepton_selections import select_electrons, select_muons, select_taus
from bottom_line.selections.jet_selections import select_jets, jetvetomap
from bottom_line.selections.lumi_selections import select_lumis
from bottom_line.utils.dumping_utils import (
    diphoton_ak_array,
    dump_ak_array,
    diphoton_list_to_pandas,
    dump_pandas,
    get_obj_syst_dict,
    apply_naming_convention,
)
from bottom_line.utils.misc_utils import choose_jet
from bottom_line.tools.flow_corrections import apply_flow_corrections_to_photons

from bottom_line.tools.mass_decorrelator import decorrelate_mass_resolution

from bottom_line.systematics import object_systematics as available_object_systematics
from bottom_line.systematics import object_corrections as available_object_corrections
from bottom_line.systematics import weight_systematics as available_weight_systematics
from bottom_line.systematics import weight_corrections as available_weight_corrections
from bottom_line.systematics import apply_systematic_variations_object_level

import warnings
from typing import Any, Dict, List, Optional
import awkward as ak
import numpy
import pandas as pd
import sys
import vector
from coffea.analysis_tools import Weights
from copy import deepcopy

import logging

logger = logging.getLogger(__name__)

vector.register_awkward()


class STXSProcessor(bbMETBaseProcessor):
    def __init__(
        self,
        metaconditions: Dict[str, Any],
        systematics: Optional[Dict[str, List[str]]] = None,
        corrections: Optional[Dict[str, List[str]]] = None,
        apply_trigger: bool = False,
        output_location: Optional[str] = None,
        taggers: Optional[List[Any]] = None,
        nano_version: int = None,
        bTagEffFileName: Optional[str] = None,
        trigger_group: str = ".*DoubleEG.*",
        analysis: str = "mainAnalysis",
        applyCQR: bool = False,
        skipJetVetoMap: bool = False,
        year: Optional[Dict[str, List[str]]] = None,
        fiducialCuts: str = "classical",
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

        self.name_convention = "DAS"

        # tau selection cuts (for the moment these are just the same as ditau)
        self.tau_pt_threshold = 18
        self.tau_max_eta = 2.3
        self.tau_max_dz = 0.2

        self.tau_photon_min_dr = 0.2

        self.jet_tau_min_dr = 0.4
        self.clean_jet_tau = True

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

            # Add sum of gen weights before selection for each HTXS.stage_0 bin
            genWeight_sums = pd.DataFrame({
                "HTXS_stage_0": events.HTXS.stage_0,
                "genWeight": events.genWeight,
            }).groupby("HTXS_stage_0")["genWeight"].sum()
            custom_accumulator = {
                f"sum_genw_presel_HTXS_Stage_0:{bin_val}": genWeight_sums[bin_val]
                for bin_val in genWeight_sums.index
            }
            metadata["custom_accumulator"] = str(dict(custom_accumulator))
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
                    "hFlav": jets.hadronFlavour if self.data_kind == "mc" else ak.zeros_like(jets.pt),
                    "btagDeepFlav_B": jets.btagDeepFlavB,
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
                    "leptonFlavour": ak.full_like(events.Electron.pt, 0),
                    "leptonID": events.Electron.mvaIso
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
                    "pfIsoId": events.Muon.pfIsoId,
                    "leptonFlavour": ak.full_like(events.Muon.pt, 1),
                    "leptonID": events.Muon.mvaMuID
                }
            )
            muons = ak.with_name(muons, "PtEtaPhiMCandidate")

            taus = ak.zip(
                {
                    "pt": events.Tau.pt,
                    "eta": events.Tau.eta,
                    "phi": events.Tau.phi,
                    "mass": events.Tau.mass,
                    "charge": events.Tau.charge,
                    "decayMode": events.Tau.decayMode,
                    "dz": events.Tau.dz,
                    "idDeepTau2018v2p5VSe": events.Tau.idDeepTau2018v2p5VSe,
                    "idDeepTau2018v2p5VSmu": events.Tau.idDeepTau2018v2p5VSmu,
                    "idDeepTau2018v2p5VSjet": events.Tau.idDeepTau2018v2p5VSjet,
                    "leptonFlavour": ak.full_like(events.Tau.pt, 2),
                    "leptonID": ak.full_like(events.Tau.pt, -999.0)  # TODO: we don't have a score, only WPs
                }
            )
            taus = ak.with_name(taus, "PtEtaPhiMCandidate")

            # lepton cleaning
            sel_electrons = electrons[select_electrons(self, electrons, diphotons)]
            sel_muons = muons[select_muons(self, muons, diphotons)]
            sel_taus = taus[select_taus(self, taus, diphotons)]

            # Build pt-ordered lepton collection
            sel_leptons = ak.concatenate([sel_electrons, sel_muons, sel_taus], axis=1)
            sel_leptons = sel_leptons[ak.argsort(sel_leptons.pt, ascending=False)]

            # Add lepton variables to diphotons
            lepton_indices = [0, 1]
            choose_lepton = choose_jet  # TODO this should be renamed to e.g. choose_object
            diphotons["n_leptons"] = ak.num(sel_leptons)
            diphotons["n_electrons"] = ak.num(sel_electrons)
            diphotons["n_muons"] = ak.num(sel_muons)
            diphotons["n_taus"] = ak.num(sel_taus)
            for i in lepton_indices:
                # Add 'merged' leptons
                diphotons[f"Lep{i}_pt"] = choose_lepton(sel_leptons.pt, i, -999.0)
                diphotons[f"Lep{i}_eta"] = choose_lepton(sel_leptons.eta, i, -999.0)
                diphotons[f"Lep{i}_phi"] = choose_lepton(sel_leptons.phi, i, -999.0)
                diphotons[f"Lep{i}_mass"] = choose_lepton(sel_leptons.mass, i, -999.0)
                diphotons[f"Lep{i}_charge"] = choose_lepton(sel_leptons.charge, i, -999.0)
                diphotons[f"Lep{i}_leptonFlavour"] = choose_lepton(sel_leptons.leptonFlavour, i, -999.0)
                diphotons[f"Lep{i}_id"] = choose_lepton(sel_leptons.leptonID, i, -999.0)

                # Add individual leptons
                diphotons[f"Ele{i}_pt"] = choose_lepton(sel_electrons.pt, i, -999.0)
                diphotons[f"Ele{i}_eta"] = choose_lepton(sel_electrons.eta, i, -999.0)
                diphotons[f"Ele{i}_phi"] = choose_lepton(sel_electrons.phi, i, -999.0)
                diphotons[f"Ele{i}_mass"] = choose_lepton(sel_electrons.mass, i, -999.0)
                diphotons[f"Ele{i}_charge"] = choose_lepton(sel_electrons.charge, i, -999.0)
                diphotons[f"Ele{i}_leptonFlavour"] = choose_lepton(sel_electrons.leptonFlavour, i, -999.0)
                diphotons[f"Ele{i}_id"] = choose_lepton(sel_electrons.leptonID, i, -999.0)
                diphotons[f"Muo{i}_pt"] = choose_lepton(sel_muons.pt, i, -999.0)
                diphotons[f"Muo{i}_eta"] = choose_lepton(sel_muons.eta, i, -999.0)
                diphotons[f"Muo{i}_phi"] = choose_lepton(sel_muons.phi, i, -999.0)
                diphotons[f"Muo{i}_mass"] = choose_lepton(sel_muons.mass, i, -999.0)
                diphotons[f"Muo{i}_charge"] = choose_lepton(sel_muons.charge, i, -999.0)
                diphotons[f"Muo{i}_leptonFlavour"] = choose_lepton(sel_muons.leptonFlavour, i, -999.0)
                diphotons[f"Muo{i}_id"] = choose_lepton(sel_muons.leptonID, i, -999.0)
                diphotons[f"Tau{i}_pt"] = choose_lepton(sel_taus.pt, i, -999.0)
                diphotons[f"Tau{i}_eta"] = choose_lepton(sel_taus.eta, i, -999.0)
                diphotons[f"Tau{i}_phi"] = choose_lepton(sel_taus.phi, i, -999.0)
                diphotons[f"Tau{i}_mass"] = choose_lepton(sel_taus.mass, i, -999.0)
                diphotons[f"Tau{i}_charge"] = choose_lepton(sel_taus.charge, i, -999.0)
                diphotons[f"Tau{i}_leptonFlavour"] = choose_lepton(sel_taus.leptonFlavour, i, -999.0)
                diphotons[f"Tau{i}_id"] = choose_lepton(sel_taus.leptonID, i, -999.0)

            # jet selection and pt ordering
            jets = jets[
                select_jets(self, jets, diphotons, sel_muons, sel_electrons, sel_taus)
            ]
            jets = jets[ak.argsort(jets.pt, ascending=False)]

            # adding selected jets to events to be used in ctagging SF calculation
            events["sel_jets"] = jets
            n_jets = ak.num(jets)
            Njets2p5 = ak.num(jets[(jets.pt > 30) & (numpy.abs(jets.eta) < 2.5)])

            # Add jets
            jet_indices = [0, 1, 2, 3]
            jet_collection = {}
            for i in jet_indices:
                jet_collection[f"J{i}_pt"] = choose_jet(jets.pt, i, -999.0)
                jet_collection[f"J{i}_eta"] = choose_jet(jets.eta, i, -999.0)
                jet_collection[f"J{i}_phi"] = choose_jet(jets.phi, i, -999.0)
                jet_collection[f"J{i}_mass"] = choose_jet(jets.mass, i, -999.0)
                jet_collection[f"J{i}_charge"] = choose_jet(jets.charge, i, -999.0)
                jet_collection[f"J{i}_btagDeepFlavB"] = choose_jet(jets.btagDeepFlav_B, i, -999.0)
                jet_collection[f"J{i}_btagDeepFlavCvB"] = choose_jet(jets.btagDeepFlav_CvB, i, -999.0)
                jet_collection[f"J{i}_btagDeepFlavCvL"] = choose_jet(jets.btagDeepFlav_CvL, i, -999.0)
                jet_collection[f"J{i}_btagDeepFlavQG"] = choose_jet(jets.btagDeepFlav_QG, i, -999.0)

            # Add MET
            met = events.PuppiMET
            met_pt = met.pt
            met_pt = ak.fill_none(met_pt, -999.0)
            met_phi = met.phi
            met_phi = ak.fill_none(met_phi, -999.0)
            met_sumEt = met.sumEt
            met_sumEt = ak.fill_none(met_sumEt, -999.0)
            diphotons["MET_pt"] = met_pt
            diphotons["MET_phi"] = met_phi
            diphotons["MET_sumEt"] = met_sumEt

            # Add Ht (scalar sum of jet Et)
            jet_Et = numpy.sqrt(jets.pt**2 + jets.mass**2)
            jet_Ht = ak.sum(jet_Et, axis=1)
            jet_Ht = ak.fill_none(jet_Ht, -999.0)
            diphotons["HT"] = jet_Ht

            # Add jet variables to diphotons
            for i in jet_indices:
                diphotons[f"J{i}_pt"] = jet_collection[f"J{i}_pt"]
                diphotons[f"J{i}_eta"] = jet_collection[f"J{i}_eta"]
                diphotons[f"J{i}_phi"] = jet_collection[f"J{i}_phi"]
                diphotons[f"J{i}_mass"] = jet_collection[f"J{i}_mass"]
                diphotons[f"J{i}_charge"] = jet_collection[f"J{i}_charge"]
                diphotons[f"J{i}_btagDeepFlavB"] = jet_collection[f"J{i}_btagDeepFlavB"]
                diphotons[f"J{i}_btagDeepFlavCvB"] = jet_collection[f"J{i}_btagDeepFlavCvB"]
                diphotons[f"J{i}_btagDeepFlavCvL"] = jet_collection[f"J{i}_btagDeepFlavCvL"]
                diphotons[f"J{i}_btagDeepFlavQG"] = jet_collection[f"J{i}_btagDeepFlavQG"]
            diphotons["n_jets"] = n_jets
            diphotons["NJ"] = Njets2p5

            first_jet_pz = jet_collection["J0_pt"] * numpy.sinh(jet_collection["J0_eta"])
            first_jet_energy = numpy.sqrt((jet_collection["J0_pt"]**2 * numpy.cosh(jet_collection["J0_eta"])**2) + jet_collection["J0_mass"]**2)

            first_jet_y = 0.5 * numpy.log((first_jet_energy + first_jet_pz) / (first_jet_energy - first_jet_pz))
            first_jet_y = ak.fill_none(first_jet_y, -999)
            first_jet_y = ak.where(numpy.isnan(first_jet_y), -999, first_jet_y)
            diphotons["YJ0"] = first_jet_y

            AbsPhiHJ0 = numpy.abs(jet_collection["J0_phi"] - diphotons["phi"])

            AbsPhiHJ0_pi_array = ak.full_like(AbsPhiHJ0, 2 * numpy.pi)

            # Select the smallest angle
            AbsPhiHJ0 = ak.where(
                AbsPhiHJ0 > numpy.pi,
                AbsPhiHJ0_pi_array - AbsPhiHJ0,
                AbsPhiHJ0
            )
            AbsPhiHJ0 = ak.where(
                AbsPhiHJ0 > 2 * numpy.pi,
                -999,
                ak.where(
                    AbsPhiHJ0 < 0,
                    -999,
                    AbsPhiHJ0
                )
            )
            diphotons["DPhiHJ0"] = AbsPhiHJ0

            AbsYHJ0 = numpy.abs(first_jet_y - diphotons["rapidity"])

            # Set all entries above 500 to -999
            AbsYHJ0 = ak.where(
                AbsYHJ0 > 500,
                -999,
                AbsYHJ0
            )

            diphotons["DYHJ0"] = AbsYHJ0

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
            diphotons["PVScore"] = events.PV.score
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
                            photons=events[f"diphotons_{do_variation}"][
                                selection_mask
                            ],
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
                                    photons=events[f"diphotons_{do_variation}"][
                                        selection_mask
                                    ],
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
                    if self.data_kind == "data" and "Scale_IJazZ" in correction_names:
                        diphotons["sigma_m_over_m_corr_smeared_decorr"] = decorrelate_mass_resolution(diphotons, type="corr_smeared", year=self.year[dataset_name][0], IsSAS_ET_Dependent=True)
                    elif self.data_kind == "mc" and "Smearing_IJazZ" in correction_names:
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
                    akarr = diphoton_ak_array(self, diphotons)

                    # Remove fixedGridRhoAll from photons to avoid having event-level info per photon
                    akarr = akarr[
                        [
                            field
                            for field in akarr.fields
                            if "lead_fixedGridRhoAll" not in field
                        ]
                    ]

                fname = apply_naming_convention(self, events)
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

    def process_extra(self, events: ak.Array) -> ak.Array:
        return events, {}

    def postprocess(self, accumulant: Dict[Any, Any]) -> Any:
        pass
