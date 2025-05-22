from bottom_line.workflows.base import bbMETBaseProcessor
from bottom_line.tools.SC_eta import add_photon_SC_eta
from bottom_line.tools.EcalBadCalibCrystal_events import remove_EcalBadCalibCrystal_events
from bottom_line.selections.lepton_selections_Zmmy import (
    select_muons_zmmy,
    select_photons_zmmy,
    get_zmmy,
)
from bottom_line.selections.lumi_selections import select_lumis
from bottom_line.utils.dumping_utils import (
    dump_ak_array,
    dress_branches,
)

# from bottom_line.utils.dumping_utils import diphoton_list_to_pandas, dump_pandas
from bottom_line.systematics import object_corrections as available_object_corrections
from bottom_line.systematics import weight_systematics as available_weight_systematics
from bottom_line.systematics import weight_corrections as available_weight_corrections
from bottom_line.tools.flow_corrections import calculate_flow_corrections

from typing import Any, Dict, List, Optional
import awkward as ak
import numpy as np
import warnings
import vector
import logging
import functools
import operator
from collections import defaultdict
from coffea.analysis_tools import Weights
from coffea.analysis_tools import PackedSelection
import hist

logger = logging.getLogger(__name__)
vector.register_awkward()


class ZmmyProcessor(bbMETBaseProcessor):
    def __init__(
        self,
        metaconditions: Dict[str, Any],
        systematics: Dict[str, List[Any]] = None,
        corrections: Dict[str, List[Any]] = None,
        nano_version: int = None,
        apply_trigger: bool = False,
        bTagEffFileName: Optional[str] = None,
        output_location: Optional[str] = None,
        taggers: Optional[List[Any]] = None,
        trigger_group=".*DoubleMuon.*",
        analysis="ZmmyAnalysis",
        applyCQR: bool = False,
        skipJetVetoMap: bool = False,
        year: Dict[str, List[str]] = None,
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
            output_format=output_format,
        )
        self.trigger_group = ".*DoubleMuon.*"
        self.analysis = "ZmmyAnalysis"
        # muon selection cuts
        self.muon_pt_threshold = 10
        self.muon_max_eta = 2.4
        self.muon_wp = "tightId"
        self.muon_max_pfRelIso03_chg = 0.2
        self.muon_global = False
        self.min_farmuon_pt = 20
        self.min_dimuon_mass = 35
        # photon selection cuts
        self.photon_pt_threshold = 20

        # mumugamma selection cuts
        self.Zmass = 91.2
        self.min_mmy_mass = 60
        self.max_mmy_mass = 120
        self.max_mm_mmy_mass = 180
        self.max_fsr_photon_dR = 0.8

    def apply_metfilters(self, events: ak.Array) -> ak.Array:
        # met filters
        met_filters = self.meta["flashggMetFilters"][self.data_kind]
        filtered = functools.reduce(
            operator.and_,
            (events.Flag[metfilter.replace("Flag_", "")] for metfilter in met_filters),
        )

        return filtered

    def apply_triggers(self, events: ak.Array) -> ak.Array:
        trigger_names = []
        triggers = self.meta["TriggerPaths"][self.trigger_group][self.analysis]
        hlt = events.HLT
        for trigger in triggers:
            actual_trigger = trigger.replace("HLT_", "").replace("*", "")
            for field in hlt.fields:
                if field == actual_trigger:
                    trigger_names.append(field)
        triggered = functools.reduce(
            operator.or_, (hlt[trigger_name] for trigger_name in trigger_names)
        )

        return triggered

    def process(self, events: ak.Array) -> Dict[Any, Any]:
        dataset = events.metadata["dataset"]
        eve_sel = PackedSelection()

        # data or monte carlo?
        self.data_kind = "mc" if hasattr(events, "GenPart") else "data"

        run_summary = defaultdict(float)
        run_summary[dataset] = {}
        if self.data_kind == "mc":
            run_summary[dataset]["nTot"] = int(ak.num(events.genWeight, axis=0))
            run_summary[dataset]["nPos"] = int(ak.sum(events.genWeight > 0))
            run_summary[dataset]["nNeg"] = int(ak.sum(events.genWeight < 0))
            run_summary[dataset]["nEff"] = int(
                run_summary[dataset]["nPos"] - run_summary[dataset]["nNeg"]
            )
            run_summary[dataset]["genWeightSum"] = float(ak.sum(events.genWeight))
        else:
            run_summary[dataset]["nTot"] = int(len(events))
            run_summary[dataset]["nPos"] = int(run_summary[dataset]["nTot"])
            run_summary[dataset]["nNeg"] = int(0)
            run_summary[dataset]["nEff"] = int(run_summary[dataset]["nTot"])
            run_summary[dataset]["genWeightSum"] = float(len(events))

        # lumi mask
        if self.data_kind == "data":
            lumimask = select_lumis(self.year[dataset][0], events, logger)
            events = events[lumimask]
            # try:
            #     lumimask = select_lumis(self.year[dataset][0], events, logger)
            #     events = events[lumimask]
            # except:
            #     logger.info(
            #         f"[ lumimask ] Skip now! Unable to find year info of dataset: {dataset}"
            #     )

        # metadata array to append to higgsdna output
        metadata = {}

        if self.data_kind == "mc":
            # Add sum of gen weights before selection for normalisation in postprocessing
            metadata["sum_genw_presel"] = str(ak.sum(events.genWeight))
        else:
            metadata["sum_genw_presel"] = "Data"

        # apply filters and triggers
        # events = events[self.apply_metfilters(events)]
        if self.apply_trigger:
            trig_flag = self.apply_triggers(events)
            events = events[trig_flag]
            # events = events[self.apply_triggers(events)]

        # remove events affected by EcalBadCalibCrystal
        if self.data_kind == "data":
            events = remove_EcalBadCalibCrystal_events(events)

        # we need ScEta for corrections and systematics, it is present in NanoAODv13+ and can be calculated using PV for older versions
        events.Photon = add_photon_SC_eta(events.Photon, events.PV)

        # read which systematics and corrections to process
        try:
            correction_names = self.corrections[dataset]
        except KeyError:
            correction_names = []
        try:
            systematic_names = self.systematics[dataset]
        except KeyError:
            systematic_names = []

        # save raw pt if we use scale/smearing corrections
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

        # object corrections:
        for correction_name in correction_names:
            if correction_name in available_object_corrections.keys():
                varying_function = available_object_corrections[correction_name]
                events = varying_function(events=events, year=self.year[dataset][0])
            elif correction_name in available_weight_corrections:
                # event weight corrections will be applied after photon preselection / application of further taggers
                continue
            else:
                # may want to throw an error instead, needs to be discussed
                warnings.warn(f"Could not process correction {correction_name}.")
                continue

        # select muons
        muons = events.Muon
        good_muons = muons[select_muons_zmmy(self, muons)]
        dimuons = ak.combinations(good_muons, 2, fields=["lead", "sublead"])
        sel_dimuons = (
            (abs(dimuons["lead"].pdgId) == 13)
            & (abs(dimuons["sublead"].pdgId) == 13)
            & (dimuons["lead"].pdgId + dimuons["sublead"].pdgId == 0)
            & (dimuons["lead"].pt > 20)
        )
        good_dimuons = dimuons[sel_dimuons]
        n_good_dimuon = ak.sum(sel_dimuons, axis=1)
        eve_sel.add("n_dimuon", n_good_dimuon > 0)
        # select photons
        photons = events.Photon
        good_photons = photons[select_photons_zmmy(self, photons)]
        n_good_photon = ak.sum(ak.ones_like(good_photons.pt) > 0, axis=1)
        eve_sel.add("n_photon", n_good_photon > 0)
        events = events.mask[eve_sel.all("n_dimuon", "n_photon")]
        # get mmy obj
        events["mmy"] = get_zmmy(self, good_dimuons, good_photons)
        sel_mmy = ak.fill_none(ak.ones_like(events["mmy"].dimuon.lead.pt) > 0, False)
        eve_sel.add("mmy_finder", sel_mmy)

        # after all cuts
        events = events[eve_sel.all(*(eve_sel.names))]
        if len(events) == 0:
            logger.info("No surviving events in this run, return now!")
            return run_summary

        # fill ntuple
        if self.output_location is not None:
            ntuple = {}
            ntuple["event"] = events.event
            ntuple["lumi"] = events.luminosityBlock
            ntuple["run"] = events.run
            ntuple = dress_branches(ntuple, events.PV, "PV")
            ntuple = dress_branches(ntuple, events.Rho, "Rho")
            if self.data_kind == "mc":
                ntuple = dress_branches(ntuple, events.Pileup, "Pileup")
            # near muon
            ntuple["muon_near_pt"] = events.mmy.muon_near.pt
            ntuple["muon_near_eta"] = events.mmy.muon_near.eta
            ntuple["muon_near_phi"] = events.mmy.muon_near.phi
            ntuple["muon_near_mass"] = events.mmy.muon_near.mass
            ntuple["muon_near_tunepRelPt"] = events.mmy.muon_near.tunepRelPt
            ntuple["muon_near_tunep_pt"] = (
                events.mmy.muon_near.pt * events.mmy.muon_near.tunepRelPt
            )
            ntuple["muon_near_track_ptErr"] = events.mmy.muon_near.ptErr
            # far muon
            ntuple["muon_far_pt"] = events.mmy.muon_far.pt
            ntuple["muon_far_eta"] = events.mmy.muon_far.eta
            ntuple["muon_far_phi"] = events.mmy.muon_far.phi
            ntuple["muon_far_mass"] = events.mmy.muon_far.mass
            ntuple["muon_far_tunepRelPt"] = events.mmy.muon_far.tunepRelPt
            ntuple["muon_far_tunep_pt"] = (
                events.mmy.muon_far.pt * events.mmy.muon_far.tunepRelPt
            )
            ntuple["muon_far_track_ptErr"] = events.mmy.muon_far.ptErr
            # photon
            ## get photon in mmy system
            photon_in_mmy = ak.copy(events.mmy.photon)
            ## Store deltaR first, the other variables should be Normalizing flowed, then be stored
            photon_in_mmy["muon_near_dR"] = photon_in_mmy.delta_r(events.mmy.muon_near)
            photon_in_mmy["muon_far_dR"] = photon_in_mmy.delta_r(events.mmy.muon_far)
            ## dr with SC eta
            vec_muon_near = ak.Array(
                {
                    "rho": events.mmy.muon_near.pt,
                    "phi": events.mmy.muon_near.phi,
                    "eta": events.mmy.muon_near.eta,
                },
                with_name="Vector3D",
            )
            vec_muon_far = ak.Array(
                {
                    "rho": events.mmy.muon_far.pt,
                    "phi": events.mmy.muon_far.phi,
                    "eta": events.mmy.muon_far.eta,
                },
                with_name="Vector3D",
            )
            vec_photon = ak.Array(
                {
                    "rho": photon_in_mmy.pt,
                    "phi": photon_in_mmy.phi,
                    "eta": photon_in_mmy.ScEta,
                },
                with_name="Vector3D",
            )
            photon_in_mmy["muon_near_dR_SC"] = vec_muon_near.deltaR(vec_photon)
            photon_in_mmy["muon_far_dR_SC"] = vec_muon_far.deltaR(vec_photon)
            ## update traker iso
            ### backup traker isolation
            photon_in_mmy["trkSumPtHollowConeDR03_nano"] = photon_in_mmy[
                "trkSumPtHollowConeDR03"
            ]
            photon_in_mmy["trkSumPtSolidConeDR04_nano"] = photon_in_mmy[
                "trkSumPtSolidConeDR04"
            ]
            ### modification refer to: https://indico.cern.ch/event/1319573/contributions/5694074/attachments/2769362/4824835/202312_Zmmg_Hgg.pdf#page=5
            #### photon_trkSumPtHollowConeDR03
            ##### only near muon in the cone, and iso/pt1 > 0.998
            sel_one_incone_mu_trkIso03 = (
                (photon_in_mmy["muon_near_dR_SC"] < 0.3)
                & (photon_in_mmy["muon_far_dR_SC"] > 0.3)
                & (
                    photon_in_mmy["trkSumPtHollowConeDR03_nano"]
                    / ntuple["muon_near_pt"]
                    > 0.998
                )
            )
            ##### both near and far muon are in the cone, and iso/(pt1+pt2) > 0.95
            sel_two_incone_mu_trkIso03 = (
                (photon_in_mmy["muon_near_dR_SC"] < 0.3)
                & (photon_in_mmy["muon_far_dR_SC"] < 0.3)
                & (
                    (photon_in_mmy["trkSumPtHollowConeDR03_nano"])
                    / (ntuple["muon_near_pt"] + ntuple["muon_far_pt"])
                    > 0.95
                )
            )
            ##### subtract muon pt
            tmp_pho_trkIso03 = (
                photon_in_mmy["trkSumPtHollowConeDR03_nano"]
                - sel_one_incone_mu_trkIso03 * ntuple["muon_near_pt"]
                - sel_two_incone_mu_trkIso03
                * (ntuple["muon_near_pt"] + ntuple["muon_far_pt"])
            )
            ##### update photon_trkSumPtHollowConeDR03
            photon_in_mmy["trkSumPtHollowConeDR03"] = ak.where(
                tmp_pho_trkIso03 > 0, tmp_pho_trkIso03, 0
            )
            #### photon_trkSumPtSolidConeDR04
            ##### only near muon in the cone, and iso/pt1 > 0.998
            sel_one_incone_mu_trkIso04 = (
                (photon_in_mmy["muon_near_dR_SC"] < 0.4)
                & (photon_in_mmy["muon_far_dR_SC"] > 0.4)
                & (
                    photon_in_mmy["trkSumPtSolidConeDR04_nano"] / ntuple["muon_near_pt"]
                    > 0.998
                )
            )
            ##### both near and far muon are in the cone, and iso/(pt1+pt2) > 0.95
            sel_two_incone_mu_trkIso04 = (
                (photon_in_mmy["muon_near_dR_SC"] < 0.4)
                & (photon_in_mmy["muon_far_dR_SC"] < 0.4)
                & (
                    (photon_in_mmy["trkSumPtSolidConeDR04_nano"])
                    / (ntuple["muon_near_pt"] + ntuple["muon_far_pt"])
                    > 0.95
                )
            )
            ##### subtract muon pt
            tmp_pho_trkIso04 = (
                photon_in_mmy["trkSumPtSolidConeDR04_nano"]
                - sel_one_incone_mu_trkIso04 * ntuple["muon_near_pt"]
                - sel_two_incone_mu_trkIso04
                * (ntuple["muon_near_pt"] + ntuple["muon_far_pt"])
            )
            ##### update photon_trkSumPtSolidConeDR04
            photon_in_mmy["trkSumPtSolidConeDR04"] = ak.where(
                tmp_pho_trkIso04 > 0, tmp_pho_trkIso04, 0
            )
            ###### Redo store of the photons
            photon_in_mmy_collection = ak.singletons(photon_in_mmy)
            counts = ak.num(photon_in_mmy_collection)

            # Keeping the nano value
            photon_in_mmy_collection["mvaID_nano"] = photon_in_mmy_collection["mvaID"]

            # Recalculate photon mvaID after the muon pt subtractions
            photon_in_mmy_collection["mvaID"] = ak.unflatten(
                self.add_photonid_mva_run3(photon_in_mmy_collection, events), counts
            )

            ## Performing photon corrections using normalizing flows
            if self.data_kind == "mc" and self.doFlow_corrections:
                # Applyting the Flow corrections to all photons before pre-selection
                corrected_inputs, var_list = calculate_flow_corrections(
                    photon_in_mmy_collection,
                    events,
                    self.meta["flashggPhotons"]["flow_inputs"],
                    self.meta["flashggPhotons"]["Isolation_transform_order"],
                    year=self.year[dataset][0],
                )

                # adding the corrected values to the tnp_candidates
                for i in range(len(var_list)):

                    photon_in_mmy_collection["raw_" + str(var_list[i])] = photon_in_mmy_collection[str(var_list[i])]
                    photon_in_mmy_collection[str(var_list[i])] = ak.unflatten(
                        np.ascontiguousarray(corrected_inputs[:, i]), counts
                    )

                # Keeping the mvaID after the subtraction as "mvaID_raw"
                photon_in_mmy_collection["mvaID_raw"] = photon_in_mmy_collection["mvaID"]

                photon_in_mmy_collection["mvaID"] = ak.unflatten(
                    self.add_photonid_mva_run3(photon_in_mmy_collection, events),
                    counts,
                )

            photon_in_mmy = ak.flatten(photon_in_mmy_collection)
            ntuple = dress_branches(ntuple, photon_in_mmy, "photon")
            # dimuon
            ntuple["dimuon_pt"] = events.mmy.obj_dimuon.pt
            ntuple["dimuon_eta"] = events.mmy.obj_dimuon.eta
            ntuple["dimuon_phi"] = events.mmy.obj_dimuon.phi
            ntuple["dimuon_mass"] = events.mmy.obj_dimuon.mass
            # mmy
            ntuple["mmy_pt"] = events.mmy.obj_mmy.pt
            ntuple["mmy_eta"] = events.mmy.obj_mmy.eta
            ntuple["mmy_phi"] = events.mmy.obj_mmy.phi
            ntuple["mmy_mass"] = events.mmy.obj_mmy.mass

        # Making the photon selection
        photons = photons[eve_sel.all(*(eve_sel.names))]

        photons["trkSumPtSolidConeDR04"] = ntuple["photon_trkSumPtSolidConeDR04"]
        photons["trkSumPtHollowConeDR03"] = ntuple["photon_trkSumPtHollowConeDR03"]

        if self.data_kind == "mc":
            # annotate diphotons with dZ information (difference between z position of GenVtx and PV) as required by flashggfinalfits
            ntuple["dZ"] = events.GenVtx.z - events.PV.z
            ntuple["genWeight"] = events.genWeight
            ntuple["genWeight_sign"] = np.sign(events.genWeight)
        # Fill zeros for data because there is no GenVtx for data, obviously
        else:
            ntuple["dZ"] = ak.zeros_like(events.PV.z)

        # return if there is no surviving events
        if len(ntuple) == 0:
            logger.info("No surviving events in this run, return now!")
            return run_summary

        if self.data_kind == "mc":
            # initiate Weight container here, after selection, since event selection cannot easily be applied to weight container afterwards
            event_weights = Weights(size=len(events))
            # _weight will correspond to the product of genWeight and the scale factors
            event_weights._weight = events["genWeight"]

            # corrections to event weights:
            for correction_name in correction_names:
                if correction_name in available_weight_corrections:
                    logger.info(
                        f"Adding correction {correction_name} to weight collection of dataset {dataset}"
                    )
                    varying_function = available_weight_corrections[correction_name]
                    event_weights = varying_function(
                        events=events,
                        weights=event_weights,
                        dataset=dataset,
                        logger=logger,
                        year=self.year[dataset][0],
                    )
            ntuple["weight"] = event_weights.weight()
            ntuple["weight_central"] = event_weights.weight() / events["genWeight"]

            # systematic variations of event weights go to nominal output dataframe:
            for systematic_name in systematic_names:
                if systematic_name in available_weight_systematics:
                    logger.info(
                        f"Adding systematic {systematic_name} to weight collection of dataset {dataset}"
                    )
                    if systematic_name == "LHEScale":
                        if hasattr(events, "LHEScaleWeight"):
                            ntuple["nLHEScaleWeight"] = ak.num(
                                events.LHEScaleWeight,
                                axis=1,
                            )
                            ntuple["LHEScaleWeight"] = events.LHEScaleWeight
                        else:
                            logger.info(
                                f"No {systematic_name} Weights in dataset {dataset}"
                            )
                    elif systematic_name == "LHEPdf":
                        if hasattr(events, "LHEPdfWeight"):
                            # two AlphaS weights are removed
                            ntuple["nLHEPdfWeight"] = (
                                ak.num(
                                    events.LHEPdfWeight,
                                    axis=1,
                                )
                                - 2
                            )
                            ntuple["LHEPdfWeight"] = events.LHEPdfWeight[:, :-2]
                        else:
                            logger.info(
                                f"No {systematic_name} Weights in dataset {dataset}"
                            )
                    else:
                        varying_function = available_weight_systematics[systematic_name]
                        event_weights = varying_function(
                            events=events,
                            weights=event_weights,
                            logger=logger,
                            dataset=dataset,
                            year=self.year[dataset][0],
                        )

                # Store variations with respect to central weight
                if len(event_weights.variations):
                    logger.info(
                        "Adding systematic weight variations to nominal output file."
                    )
                    for modifier in event_weights.variations:
                        ntuple["weight_" + modifier] = event_weights.weight(
                            modifier=modifier
                        )
        # Add weight variables (=1) for data for consistent datasets
        else:
            ntuple["weight_central"] = ak.ones_like(ntuple["event"])
        # to Awkward array: this is necessary, or the saved parquet file is not correct
        ak_ntuple = ak.Array(ntuple)
        if self.output_location is not None:
            fname = (
                events.behavior["__events_factory__"]._partition_key.replace("/", "_")
                + ".parquet"
            )
            subdirs = []
            if "dataset" in events.metadata:
                subdirs.append(events.metadata["dataset"])
            dump_ak_array(self, ak_ntuple, fname, self.output_location, None, subdirs)

        return run_summary

    def process_extra(self, events: ak.Array) -> ak.Array:
        return events, {}

    def postprocess(self, accumulant: Dict[Any, Any]) -> Any:
        pass


class ZmmyHist(bbMETBaseProcessor):
    def __init__(
        self,
        metaconditions: Dict[str, Any],
        systematics: Dict[str, List[Any]] = None,
        corrections: Dict[str, List[Any]] = None,
        apply_trigger: bool = False,
        nano_version: str = None,
        bTagEffFileName: Optional[str] = None,
        output_location: Optional[str] = None,
        taggers: Optional[List[Any]] = None,
        applyCQR: bool = False,
        skipJetVetoMap: bool = False,
        year: Dict[str, List[str]] = None,
        fiducialCuts: str = "classical",
        doDeco: bool = False,
        output_format: str = "parquet",
        trigger_group: str = ".*DoubleMuon.*",
        analysis: str = "ZmmyHist",
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
            output_format=output_format,
        )
        self.analysis = "ZmmyHist"
        # muon selection cuts

    def process(self, events: ak.Array) -> Dict[Any, Any]:
        dataset = events.metadata["dataset"]
        # data or monte carlo?
        self.data_kind = "mc" if hasattr(events, "genWeight") else "data"

        # read which systematics and corrections to process
        try:
            correction_names = self.corrections[dataset]
        except KeyError:
            correction_names = []

        nbins = 300
        axis_dataset = hist.axis.StrCategory(
            [], growth=True, name="dataset", label="Primary dataset"
        )
        axis_mmy_mass = hist.axis.Regular(
            nbins, 80, 100, name="x", label=r"m$_{\mu\mu\gamma}$ [GeV]"
        )
        axis_mmy_pt = hist.axis.Regular(
            nbins, 0, 100, name="x", label=r"p$_{T}^{\mu\mu\gamma}$ [GeV]"
        )
        axis_far_mu_pt = hist.axis.Regular(
            nbins, 10, 110, name="x", label=r"p$_{T}^{far}$ [GeV]"
        )
        axis_near_mu_pt = hist.axis.Regular(
            nbins, 10, 110, name="x", label=r"p$_{T}^{near}$ [GeV]"
        )
        axis_far_mu_eta = hist.axis.Regular(
            nbins, -2.5, 2.5, name="x", label=r"$\eta_{far \mu}$"
        )
        axis_near_mu_eta = hist.axis.Regular(
            nbins, -2.5, 2.5, name="x", label=r"$\eta_{near \mu}$"
        )
        axis_mm_mass = hist.axis.Regular(
            nbins, 30, 90, name="x", label=r"m$_{\mu\mu}$ [GeV]"
        )
        axis_dR_near = hist.axis.Regular(
            nbins, 0, 1, name="x", label=r"$\Delta R(\gamma,\mu_{near})$"
        )
        axis_y_pt = hist.axis.Regular(
            nbins, 20, 110, name="x", label=r"p$_{T}^{\gamma}$ [GeV]"
        )
        axis_y_eta = hist.axis.Regular(
            nbins, -2.5, 2.5, name="x", label=r"$\eta_{\gamma}$"
        )
        axis_y_phi = hist.axis.Regular(
            nbins, -np.pi, np.pi, name="y", label=r"$\phi_{\gamma}$"
        )
        axis_y_r9 = hist.axis.Regular(
            nbins, 0.5, 1.1, name="x", label=r"R$_{9} (\gamma)$"
        )
        axis_y_r9_2 = hist.axis.Regular(
            nbins, 0.8, 1.1, name="x", label=r"R$_{9} (\gamma)$"
        )
        axis_y_s4 = hist.axis.Regular(
            nbins, 0.5, 1, name="x", label=r"E$_{2 \times 2}$/E$_{5 \times 5}$"
        )
        axis_y_sieie_EB = hist.axis.Regular(
            nbins, 0.005, 0.015, name="x", label=r"$\sigma _{i\eta i\eta} (\gamma)$"
        )
        axis_y_sieie_EE = hist.axis.Regular(
            nbins, 0.005, 0.05, name="x", label=r"$\sigma _{i\eta i\eta} (\gamma)$"
        )
        axis_y_sieip_EB = hist.axis.Regular(
            nbins, -0.0001, 0.0001, name="x", label=r"$\sigma _{i\eta i\phi} (\gamma)$"
        )
        axis_y_sieip_EE = hist.axis.Regular(
            nbins, -0.0005, 0.0005, name="x", label=r"$\sigma _{i\eta i\phi} (\gamma)$"
        )
        axis_y_sipip_EB = hist.axis.Regular(
            nbins, 0.005, 0.025, name="x", label=r"$\sigma _{i\phi i\phi} (\gamma)$"
        )
        axis_y_sipip_EE = hist.axis.Regular(
            nbins, 0.01, 0.06, name="x", label=r"$\sigma _{i\phi i\phi} (\gamma)$"
        )
        axis_y_hoe = hist.axis.Regular(nbins, 0, 0.2, name="x", label=r"H/E $(\gamma)$")
        axis_y_eta_width = hist.axis.Regular(
            nbins, 0, 0.025, name="x", label=r"$\eta_{width} (\gamma)$"
        )
        axis_y_phi_width = hist.axis.Regular(
            nbins, 0, 0.2, name="x", label=r"$\phi_{width} (\gamma)$"
        )
        axis_y_sigmaRR = hist.axis.Regular(
            nbins, -1, 14, name="x", label=r"$\sigma_{RR}$"
        )
        axis_y_es_over_raw = hist.axis.Regular(
            nbins, 0, 0.4, name="x", label=r"E$_{es}$/E$_{raw}$"
        )
        axis_y_phoiso = hist.axis.Regular(
            nbins,
            0,
            20,
            name="x",
            label=r"iso$_{PF(\Delta R=0.3)}^{\gamma} (\gamma)$ [GeV]",
        )
        axis_y_chaiso_PV = hist.axis.Regular(
            nbins,
            0,
            20,
            name="x",
            label=r"iso$_{PF(\Delta R=0.3)}^{Charged wrt PV} (\gamma)$ [GeV]",
        )
        axis_y_chaiso_worst_vtx = hist.axis.Regular(
            nbins,
            0,
            20,
            name="x",
            label=r"iso$_{PF(\Delta R=0.3)}^{Charged wrt worst vtx} (\gamma)$ [GeV]",
        )
        axis_y_mvaid = hist.axis.Regular(
            nbins, -1, 1, name="x", label=r"Photon ID MVA score"
        )
        # axis_y_mvaid_fall17 = hist.axis.Regular(
        #     nbins, -1, 1, name="x", label=r"Photon ID MVA score (Fall17v2)"
        # )
        axis_nPU = hist.axis.Regular(nbins, 0, 100, name="x", label=r"N$_{PU}$")
        axis_rho = hist.axis.Regular(nbins, 0, 60, name="x", label=r"$\rho$")

        hdict = {}
        hdict["h_nPU"] = hist.Hist(
            axis_dataset, axis_nPU, storage="weight", label="Counts"
        )
        hdict["h_rho"] = hist.Hist(
            axis_dataset, axis_rho, storage="weight", label="Counts"
        )
        hdict["h_mmy_mass"] = hist.Hist(
            axis_dataset, axis_mmy_mass, storage="weight", label="Counts"
        )
        hdict["h_mmy_mass_EB"] = hist.Hist(
            axis_dataset, axis_mmy_mass, storage="weight", label="Counts"
        )
        hdict["h_mmy_mass_EE"] = hist.Hist(
            axis_dataset, axis_mmy_mass, storage="weight", label="Counts"
        )
        hdict["h_mmy_pt"] = hist.Hist(
            axis_dataset, axis_mmy_pt, storage="weight", label="Counts"
        )
        hdict["h_far_mu_pt"] = hist.Hist(
            axis_dataset, axis_far_mu_pt, storage="weight", label="Counts"
        )
        hdict["h_near_mu_pt"] = hist.Hist(
            axis_dataset, axis_near_mu_pt, storage="weight", label="Counts"
        )
        hdict["h_far_mu_eta"] = hist.Hist(
            axis_dataset, axis_far_mu_eta, storage="weight", label="Counts"
        )
        hdict["h_near_mu_eta"] = hist.Hist(
            axis_dataset, axis_near_mu_eta, storage="weight", label="Counts"
        )
        hdict["h_mm_mass"] = hist.Hist(
            axis_dataset, axis_mm_mass, storage="weight", label="Counts"
        )
        hdict["h_mm_mass_EB"] = hist.Hist(
            axis_dataset, axis_mm_mass, storage="weight", label="Counts"
        )
        hdict["h_mm_mass_EE"] = hist.Hist(
            axis_dataset, axis_mm_mass, storage="weight", label="Counts"
        )
        hdict["h_dR_near"] = hist.Hist(
            axis_dataset, axis_dR_near, storage="weight", label="Counts"
        )
        hdict["h_y_pt"] = hist.Hist(
            axis_dataset, axis_y_pt, storage="weight", label="Counts"
        )
        hdict["h_y_eta"] = hist.Hist(
            axis_dataset, axis_y_eta, storage="weight", label="Counts"
        )
        hdict["h_y_eta_phi"] = hist.Hist(
            axis_dataset, axis_y_eta, axis_y_phi, storage="weight", label="Counts"
        )

        ## EB bin
        hdict["h_y_pt_EB"] = hist.Hist(
            axis_dataset, axis_y_pt, storage="weight", label="Counts"
        )
        hdict["h_y_r9_EB"] = hist.Hist(
            axis_dataset, axis_y_r9, storage="weight", label="Counts"
        )
        hdict["h_y_r9_2_EB"] = hist.Hist(
            axis_dataset, axis_y_r9_2, storage="weight", label="Counts"
        )
        hdict["h_y_s4_EB"] = hist.Hist(
            axis_dataset, axis_y_s4, storage="weight", label="Counts"
        )
        hdict["h_y_sieie_EB"] = hist.Hist(
            axis_dataset, axis_y_sieie_EB, storage="weight", label="Counts"
        )
        hdict["h_y_sieip_EB"] = hist.Hist(
            axis_dataset, axis_y_sieip_EB, storage="weight", label="Counts"
        )
        hdict["h_y_sipip_EB"] = hist.Hist(
            axis_dataset, axis_y_sipip_EB, storage="weight", label="Counts"
        )
        hdict["h_y_hoe_EB"] = hist.Hist(
            axis_dataset, axis_y_hoe, storage="weight", label="Counts"
        )
        hdict["h_y_eta_width_EB"] = hist.Hist(
            axis_dataset, axis_y_eta_width, storage="weight", label="Counts"
        )
        hdict["h_y_phi_width_EB"] = hist.Hist(
            axis_dataset, axis_y_phi_width, storage="weight", label="Counts"
        )
        hdict["h_y_sigmaRR_EB"] = hist.Hist(
            axis_dataset, axis_y_sigmaRR, storage="weight", label="Counts"
        )
        hdict["h_y_es_over_raw_EB"] = hist.Hist(
            axis_dataset, axis_y_es_over_raw, storage="weight", label="Counts"
        )
        hdict["h_y_phoiso_EB"] = hist.Hist(
            axis_dataset, axis_y_phoiso, storage="weight", label="Counts"
        )
        hdict["h_y_chaiso_PV_EB"] = hist.Hist(
            axis_dataset, axis_y_chaiso_PV, storage="weight", label="Counts"
        )
        hdict["h_y_chaiso_worst_vtx_EB"] = hist.Hist(
            axis_dataset, axis_y_chaiso_worst_vtx, storage="weight", label="Counts"
        )
        hdict["h_y_mvaid_EB"] = hist.Hist(
            axis_dataset, axis_y_mvaid, storage="weight", label="Counts"
        )
        # hdict["h_y_mvaid_fall17_EB"] = hist.Hist(
        #     axis_dataset, axis_y_mvaid_fall17, storage="weight", label="Counts"
        # )

        hdict["h_y_pt_EE"] = hist.Hist(
            axis_dataset, axis_y_pt, storage="weight", label="Counts"
        )
        hdict["h_y_r9_EE"] = hist.Hist(
            axis_dataset, axis_y_r9, storage="weight", label="Counts"
        )
        hdict["h_y_r9_2_EE"] = hist.Hist(
            axis_dataset, axis_y_r9_2, storage="weight", label="Counts"
        )
        hdict["h_y_s4_EE"] = hist.Hist(
            axis_dataset, axis_y_s4, storage="weight", label="Counts"
        )
        hdict["h_y_sieie_EE"] = hist.Hist(
            axis_dataset, axis_y_sieie_EE, storage="weight", label="Counts"
        )
        hdict["h_y_sieip_EE"] = hist.Hist(
            axis_dataset, axis_y_sieip_EE, storage="weight", label="Counts"
        )
        hdict["h_y_sipip_EE"] = hist.Hist(
            axis_dataset, axis_y_sipip_EE, storage="weight", label="Counts"
        )
        hdict["h_y_hoe_EE"] = hist.Hist(
            axis_dataset, axis_y_hoe, storage="weight", label="Counts"
        )
        hdict["h_y_eta_width_EE"] = hist.Hist(
            axis_dataset, axis_y_eta_width, storage="weight", label="Counts"
        )
        hdict["h_y_phi_width_EE"] = hist.Hist(
            axis_dataset, axis_y_phi_width, storage="weight", label="Counts"
        )
        hdict["h_y_sigmaRR_EE"] = hist.Hist(
            axis_dataset, axis_y_sigmaRR, storage="weight", label="Counts"
        )
        hdict["h_y_es_over_raw_EE"] = hist.Hist(
            axis_dataset, axis_y_es_over_raw, storage="weight", label="Counts"
        )
        hdict["h_y_phoiso_EE"] = hist.Hist(
            axis_dataset, axis_y_phoiso, storage="weight", label="Counts"
        )
        hdict["h_y_chaiso_PV_EE"] = hist.Hist(
            axis_dataset, axis_y_chaiso_PV, storage="weight", label="Counts"
        )
        hdict["h_y_chaiso_worst_vtx_EE"] = hist.Hist(
            axis_dataset, axis_y_chaiso_worst_vtx, storage="weight", label="Counts"
        )
        hdict["h_y_mvaid_EE"] = hist.Hist(
            axis_dataset, axis_y_mvaid, storage="weight", label="Counts"
        )
        # hdict["h_y_mvaid_fall17_EE"] = hist.Hist(
        #     axis_dataset, axis_y_mvaid_fall17, storage="weight", label="Counts"
        # )

        if self.data_kind == "mc":
            # initiate Weight container here, after selection, since event selection cannot easily be applied to weight container afterwards
            event_weights = Weights(size=len(events))
            # _weight will correspond to the product of genWeight and the scale factors
            event_weights._weight = events.genWeight * events.weight_central
            # event_weights._weight = events.genWeight
            # corrections to event weights:
            for correction_name in correction_names:
                if correction_name in available_weight_corrections:
                    logger.debug(
                        f"Adding correction {correction_name} to weight collection of dataset {dataset}"
                    )
                    varying_function = available_weight_corrections[correction_name]
                    event_weights = varying_function(
                        events=events,
                        photons=events,
                        weights=event_weights,
                        dataset_name=dataset,
                        logger=logger,
                        year=self.year[dataset][0],
                    )

            wgt = event_weights.weight()
        else:
            wgt = ak.ones_like(events.event)

        sel_photon_EB = np.abs(events.photon_ScEta) < 1.4442
        events_EB = events[sel_photon_EB]
        wgt_EB = wgt[sel_photon_EB]
        sel_photon_EE = np.abs(events.photon_ScEta) > 1.566
        events_EE = events[sel_photon_EE]
        wgt_EE = wgt[sel_photon_EE]

        # hdict["h_nPU"].fill(dataset=dataset, x=events.Pileup_nPU, weight=wgt)
        hdict["h_nPU"].fill(dataset=dataset, x=events.PV_npvs, weight=wgt)
        hdict["h_rho"].fill(dataset=dataset, x=events.Rho_fixedGridRhoAll, weight=wgt)
        hdict["h_mmy_mass"].fill(dataset=dataset, x=events.mmy_mass, weight=wgt)
        hdict["h_mmy_pt"].fill(dataset=dataset, x=events.mmy_pt, weight=wgt)
        hdict["h_far_mu_pt"].fill(dataset=dataset, x=events.muon_far_pt, weight=wgt)
        hdict["h_near_mu_pt"].fill(dataset=dataset, x=events.muon_near_pt, weight=wgt)
        hdict["h_far_mu_eta"].fill(dataset=dataset, x=events.muon_far_eta, weight=wgt)
        hdict["h_near_mu_eta"].fill(dataset=dataset, x=events.muon_near_eta, weight=wgt)
        hdict["h_mm_mass"].fill(dataset=dataset, x=events.dimuon_mass, weight=wgt)
        hdict["h_dR_near"].fill(
            dataset=dataset, x=events.photon_muon_near_dR, weight=wgt
        )
        hdict["h_y_pt"].fill(dataset=dataset, x=events.photon_pt, weight=wgt)
        hdict["h_y_eta"].fill(dataset=dataset, x=events.photon_eta, weight=wgt)
        hdict["h_y_eta_phi"].fill(
            dataset=dataset, x=events.photon_eta, y=events.photon_phi, weight=wgt
        )

        # EE and EB binned
        # EB
        hdict["h_mm_mass_EB"].fill(
            dataset=dataset, x=events_EB.dimuon_mass, weight=wgt_EB
        )
        hdict["h_mmy_mass_EB"].fill(
            dataset=dataset, x=events_EB.mmy_mass, weight=wgt_EB
        )
        hdict["h_y_pt_EB"].fill(dataset=dataset, x=events_EB.photon_pt, weight=wgt_EB)
        hdict["h_y_r9_EB"].fill(dataset=dataset, x=events_EB.photon_r9, weight=wgt_EB)
        hdict["h_y_r9_2_EB"].fill(dataset=dataset, x=events_EB.photon_r9, weight=wgt_EB)
        hdict["h_y_s4_EB"].fill(dataset=dataset, x=events_EB.photon_s4, weight=wgt_EB)
        hdict["h_y_sieie_EB"].fill(
            dataset=dataset, x=events_EB.photon_sieie, weight=wgt_EB
        )
        hdict["h_y_sieip_EB"].fill(
            dataset=dataset, x=events_EB.photon_sieip, weight=wgt_EB
        )
        hdict["h_y_sipip_EB"].fill(
            dataset=dataset, x=events_EB.photon_sipip, weight=wgt_EB
        )
        hdict["h_y_hoe_EB"].fill(dataset=dataset, x=events_EB.photon_hoe, weight=wgt_EB)
        hdict["h_y_eta_width_EB"].fill(
            dataset=dataset, x=events_EB.photon_etaWidth, weight=wgt_EB
        )
        hdict["h_y_phi_width_EB"].fill(
            dataset=dataset, x=events_EB.photon_phiWidth, weight=wgt_EB
        )
        hdict["h_y_sigmaRR_EB"].fill(
            dataset=dataset, x=events_EB.photon_esEffSigmaRR, weight=wgt_EB
        )
        hdict["h_y_es_over_raw_EB"].fill(
            dataset=dataset, x=events_EB.photon_esEnergyOverRawE, weight=wgt_EB
        )
        hdict["h_y_phoiso_EB"].fill(
            dataset=dataset, x=events_EB.photon_pfPhoIso03, weight=wgt_EB
        )
        hdict["h_y_chaiso_PV_EB"].fill(
            dataset=dataset, x=events_EB.photon_pfChargedIsoPFPV, weight=wgt_EB
        )
        hdict["h_y_chaiso_worst_vtx_EB"].fill(
            dataset=dataset, x=events_EB.photon_pfChargedIsoWorstVtx, weight=wgt_EB
        )
        hdict["h_y_mvaid_EB"].fill(
            dataset=dataset, x=events_EB.photon_mvaID, weight=wgt_EB
        )
        # hdict["h_y_mvaid_fall17_EB"].fill(
        #     dataset=dataset, x=events_EB.photon_mvaID_Fall17V2, weight=wgt_EB
        # )
        # EE
        hdict["h_mm_mass_EE"].fill(
            dataset=dataset, x=events_EE.dimuon_mass, weight=wgt_EE
        )
        hdict["h_mmy_mass_EE"].fill(
            dataset=dataset, x=events_EE.mmy_mass, weight=wgt_EE
        )
        hdict["h_y_pt_EE"].fill(dataset=dataset, x=events_EE.photon_pt, weight=wgt_EE)
        hdict["h_y_r9_EE"].fill(dataset=dataset, x=events_EE.photon_r9, weight=wgt_EE)
        hdict["h_y_r9_2_EE"].fill(dataset=dataset, x=events_EE.photon_r9, weight=wgt_EE)
        hdict["h_y_s4_EE"].fill(dataset=dataset, x=events_EE.photon_s4, weight=wgt_EE)
        hdict["h_y_sieie_EE"].fill(
            dataset=dataset, x=events_EE.photon_sieie, weight=wgt_EE
        )
        hdict["h_y_sieip_EE"].fill(
            dataset=dataset, x=events_EE.photon_sieip, weight=wgt_EE
        )
        hdict["h_y_sipip_EE"].fill(
            dataset=dataset, x=events_EE.photon_sipip, weight=wgt_EE
        )
        hdict["h_y_hoe_EE"].fill(dataset=dataset, x=events_EE.photon_hoe, weight=wgt_EE)
        hdict["h_y_eta_width_EE"].fill(
            dataset=dataset, x=events_EE.photon_etaWidth, weight=wgt_EE
        )
        hdict["h_y_phi_width_EE"].fill(
            dataset=dataset, x=events_EE.photon_phiWidth, weight=wgt_EE
        )
        hdict["h_y_sigmaRR_EE"].fill(
            dataset=dataset, x=events_EE.photon_esEffSigmaRR, weight=wgt_EE
        )
        hdict["h_y_es_over_raw_EE"].fill(
            dataset=dataset, x=events_EE.photon_esEnergyOverRawE, weight=wgt_EE
        )
        hdict["h_y_phoiso_EE"].fill(
            dataset=dataset, x=events_EE.photon_pfPhoIso03, weight=wgt_EE
        )
        hdict["h_y_chaiso_PV_EE"].fill(
            dataset=dataset, x=events_EE.photon_pfChargedIsoPFPV, weight=wgt_EE
        )
        hdict["h_y_chaiso_worst_vtx_EE"].fill(
            dataset=dataset, x=events_EE.photon_pfChargedIsoWorstVtx, weight=wgt_EE
        )
        hdict["h_y_mvaid_EE"].fill(
            dataset=dataset, x=events_EE.photon_mvaID, weight=wgt_EE
        )
        # hdict["h_y_mvaid_fall17_EE"].fill(
        #     dataset=dataset, x=events_EE.photon_mvaID_Fall17V2, weight=wgt_EE
        # )

        return hdict

    def process_extra(self, events: ak.Array) -> ak.Array:
        return events, {}

    def postprocess(self, accumulant: Dict[Any, Any]) -> Any:
        pass


class ZmmyZptHist(ZmmyHist):
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
        applyCQR: bool = False,
        skipJetVetoMap: bool = False,
        year: Dict[str, List[str]] = None,
        fiducialCuts: str = "classical",
        doDeco: bool = False,
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
            trigger_group=".*DoubleMuon.*",
            analysis="ZmmyZptHist",
            applyCQR=applyCQR,
            year=year,
            fiducialCuts=fiducialCuts,
            doDeco=doDeco,
            output_format=output_format,
        )
        self.analysis = "ZmmyZptHist"
        # muon selection cuts

    def process(self, events: ak.Array) -> Dict[Any, Any]:
        dataset = events.metadata["dataset"]

        nbins = 100
        axis_dataset = hist.axis.StrCategory(
            [], growth=True, name="dataset", label="Primary dataset"
        )
        axis_mmy_pt = hist.axis.Regular(
            nbins, 0, 100, name="x", label=r"p$_{T}^{\mu\mu\gamma}$ [GeV]"
        )

        hdict = {}
        hdict["h_mmy_pt"] = hist.Hist(
            axis_dataset, axis_mmy_pt, storage="weight", label="Counts"
        )
        if hasattr(events, "genWeight"):
            # wgt = events.genWeight
            wgt = events.genWeight * events.weight_central
            # wgt = events.genWeight_sign
        else:
            wgt = ak.ones_like(events.event)

        hdict["h_mmy_pt"].fill(dataset=dataset, x=events.mmy_pt, weight=wgt)

        return hdict
