from bottom_line.workflows.base import bbMETBaseProcessor
from bottom_line.systematics import object_systematics as available_object_systematics
from bottom_line.systematics import object_corrections as available_object_corrections
from bottom_line.systematics import weight_systematics as available_weight_systematics
from bottom_line.systematics import weight_corrections as available_weight_corrections
from bottom_line.systematics import apply_systematic_variations_object_level
from bottom_line.selections.photon_selections import photon_preselection
from bottom_line.selections.lumi_selections import select_lumis
from bottom_line.utils.dumping_utils import diphoton_list_to_pandas, dump_pandas
from bottom_line.utils.misc_utils import trigger_match, delta_r_with_ScEta
from bottom_line.tools.SC_eta import add_photon_SC_eta
from bottom_line.tools.flow_corrections import apply_flow_corrections_to_photons
from bottom_line.tools.EcalBadCalibCrystal_events import remove_EcalBadCalibCrystal_events
from bottom_line.tools.sigma_m_tools import compute_sigma_m
from typing import Any, Dict, List, Optional
import awkward as ak
import logging
import warnings
import numpy
import sys
from coffea.analysis_tools import Weights

logger = logging.getLogger(__name__)


class DYStudiesProcessor(bbMETBaseProcessor):
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
            nano_version=nano_version,
            bTagEffFileName=bTagEffFileName,
            apply_trigger=apply_trigger,
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

    def process_extra(self, events: ak.Array) -> ak.Array:
        return events, {}

    def postprocess(self, accumulant: Dict[Any, Any]) -> Any:
        pass


class TagAndProbeProcessor(bbMETBaseProcessor):
    def __init__(
        self,
        metaconditions: Dict[str, Any],
        systematics: Dict[str, List[Any]] = None,
        corrections: Optional[Dict[str, List[str]]] = None,
        apply_trigger: bool = False,
        output_location: Optional[str] = None,
        taggers: Optional[List[Any]] = None,
        nano_version: int = None,
        bTagEffFileName: Optional[str] = None,
        trigger_group: str = ".*SingleEle.*",
        analysis: str = "tagAndProbe",
        applyCQR: bool = False,
        skipJetVetoMap: bool = False,
        year: Optional[Dict[str, List[str]]] = None,
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
            trigger_group=".*SingleEle.*",
            analysis="tagAndProbe",
            applyCQR=applyCQR,
            skipJetVetoMap=False,
            year=year if year is not None else {},
            fiducialCuts=fiducialCuts,
            doDeco=doDeco,
            Smear_sigma_m=Smear_sigma_m,
            doFlow_corrections=doFlow_corrections,
            output_format=output_format
        )

        self.prefixes = {"tag": "tag", "probe": "probe"}

    def process(self, events: ak.Array) -> Dict[Any, Any]:

        dataset_name = events.metadata["dataset"]

        # data or mc?
        self.data_kind = "mc" if "GenPart" in ak.fields(events) else "data"

        # lumi mask
        if self.data_kind == "data":
            try:
                lumimask = select_lumis(self.year[dataset_name][0], events, logger)
                events = events[lumimask]
            except:
                logger.info(
                    f"[ lumimask ] Skip now! Unable to find year info of {dataset_name}"
                )
        # apply filters and triggers
        events = self.apply_filters_and_triggers(events)

        # remove events affected by EcalBadCalibCrystal
        if self.data_kind == "data":
            events = remove_EcalBadCalibCrystal_events(events)

        # we need ScEta for corrections and systematics, it is present in NanoAODv13+ and can be calculated using PV for older versions
        events.Photon = add_photon_SC_eta(events.Photon, events.PV)

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
            and ("Smearing_Trad" not in correction_names and "Smearing_IJazZ" not in correction_names and "Smearing2G_IJazZ" not in correction_names)
        ):
            warnings.warn(
                "Smearing_Trad or  Smearing_IJazZ or Smearing2G_IJazZ should be specified in the corrections field in .json in order to smear the mass!"
            )
            sys.exit(0)

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
                f"\nApplying correction {correction_name} to dataset {dataset_name}\n"
            )
            varying_function = available_object_corrections[correction_name]
            events = varying_function(events=events,year=self.year[dataset_name][0])

        for correction_name in correction_names:
            if correction_name in available_object_corrections.keys():
                logger.info(
                    f"\nApplying correction {correction_name} to dataset {dataset_name}\n"
                )
                varying_function = available_object_corrections[correction_name]
                events = varying_function(events=events, year=self.year[dataset_name][0])
            elif correction_name in available_weight_corrections:
                # event weight corrections will be applied after photon preselection / application of further taggers
                continue
            else:
                # may want to throw an error instead, needs to be discussed
                warnings.warn(f"Could not process correction {correction_name}.")
                continue

        original_photons = events.Photon

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
                photons_dct[f"{systematic}_{variation}"] = original_photons.systematics[
                    systematic
                ][variation]

        for variation, photons in photons_dct.items():
            logger.debug(f"Variation: {variation}")

            if variation == "nominal":
                do_variation = "nominal"

            if self.chained_quantile is not None:
                photons = self.chained_quantile.apply(photons, events)

            # recompute photonid_mva on the fly
            if self.photonid_mva_EB and self.photonid_mva_EE:
                photons = self.add_photonid_mva(photons, events)

            # photon preselection
            photons = photon_preselection(
                self, photons, events, electron_veto=False, revert_electron_veto=True, year=self.year[dataset_name][0]
            )

            if self.data_kind == "mc":
                # TODO: add weight systs and corrections! (if needed)
                # need to annotate the photons already here with a weight since later, each photon can be tag and probe and this changes the length of the array
                photons["weight"] = events["genWeight"]
                # keep only photons matched to gen e+ or e-
                photons = photons[photons.genPartFlav == 11]

            # other event related variables need to be added before the tag&probe combination
            # nPV just for validation of pileup reweighting
            photons["nPV"] = events.PV.npvs
            photons["fixedGridRhoAll"] = events.Rho.fixedGridRhoAll

            # TODO: HLT matching for data

            # double the number of diphoton candidates (each item in the pair can be both a tag and a probe)
            tnp = ak.combinations(photons, 2, fields=["tag", "probe"])
            pnt = ak.combinations(photons, 2, fields=["probe", "tag"])
            tnp_candidates = ak.concatenate([tnp, pnt], axis=1)

            # Ensure gen matching in MC for origin from gamma star Z (equivalent when it comes to PDG ID assigment in CMS MC)
            # This means a) parent of tag and probe is the same b) parent is a Z
            if self.data_kind == "mc":
                logger.info("Matching to Z boson")
                mask = (tnp_candidates.tag.matched_gen.distinctParentIdxG == tnp_candidates.probe.matched_gen.distinctParentIdxG) & (tnp_candidates.tag.matched_gen.distinctParent.pdgId == 23) & (tnp_candidates.probe.matched_gen.distinctParent.pdgId == 23)
                tnp_candidates = tnp_candidates[ak.fill_none(mask, False)]

            # add ScEta to the matched electrons of the tags
            matched_electrons_tags = tnp_candidates.tag.matched_electron
            matched_electrons_tags["ScEta"] = matched_electrons_tags.eta + matched_electrons_tags.deltaEtaSC

            # check that the e+/e- matched to tag and probe are not the same particle
            if self.data_kind == "mc":
                tnp_candidates = tnp_candidates[
                    tnp_candidates.tag.genPartIdx != tnp_candidates.probe.genPartIdx
                ]

            # imply trigger threshold from year
            year = self.year[dataset_name][0]
            if "2016" in year:
                trigger_pt = 27
            elif "2017" in year or "2018" in year:
                trigger_pt = 32
            else:
                trigger_pt = 30

            # find out if we're running on EGMNano samples. If so, filterbit for Ele*_WPTight_Gsf is 12, otherwise 1
            try:
                eledoc = [x for x in events.TrigObj.filterBits.__doc__.split(";") if "for Electron" in x][0]
            except IndexError:
                eledoc = ""

            filterbit = 12 if "1e WPTight L1T match" in eledoc else 1

            # tag selections
            tag_mask = (
                (tnp_candidates.tag.pt > 40)
                & (tnp_candidates.tag.electronIdx != -1)
                & (tnp_candidates.tag.pixelSeed)
                & (
                    tnp_candidates.tag.pfChargedIsoPFPV < 20
                )  # was: (tnp_candidates.tag.chargedHadronIso < 20)
                & (
                    tnp_candidates.tag.pfChargedIsoPFPV / tnp_candidates.tag.pt < 0.3
                )  # was: (tnp_candidates.tag.chargedHadronIso / tnp_candidates.tag.pt < 0.3)
                & (
                    trigger_match(
                        matched_electrons_tags, events.TrigObj, pdgid=11, pt=trigger_pt, filterbit=filterbit, metric=delta_r_with_ScEta, dr=0.1
                    )
                )  # match tag with an HLT Ele30 object
            )

            # No selection on the probe to not bias it!

            # apply selections
            tnp_candidates = tnp_candidates[tag_mask]

            # Since the Weights object accepts only flat masks, the tag and probe mask is flattened
            flat_tag_and_probe_mask = ak.any(tag_mask, axis=1)

            """
            This n_event_tnp_cand array is created to keep track of how many tag and probe candidates we have at each event
            Since the pileup rw is calculated at a event level, we will have only one weight for event
            But since we are saving ak.flatten(tnp_candidates) , we need the n_event_tnp_cand to unroll the weights to each tnp candidate at the event
            """
            n_event_tnp_cand = [numpy.ones(n_tnp_candidates) for n_tnp_candidates in ak.num(tnp_candidates[flat_tag_and_probe_mask])]

            # candidates need to be flattened since we have each photon as a tag and probe, otherwise it can't be exported to numpy
            tnp_candidates = ak.flatten(tnp_candidates)

            # performing the weight corrections after the preselctions
            if self.data_kind == "mc":

                event_weights = Weights(size=len(events[flat_tag_and_probe_mask]))
                event_weights._weight = numpy.array(events[flat_tag_and_probe_mask].genWeight)

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
                            events=events[flat_tag_and_probe_mask],
                            photons=events.Photon[flat_tag_and_probe_mask],
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
                            varying_function = available_weight_systematics[
                                systematic_name
                            ]
                            event_weights = varying_function(
                                events=events[flat_tag_and_probe_mask],
                                photons=events.Photon[flat_tag_and_probe_mask],
                                weights=event_weights,
                                dataset_name=dataset_name,
                                year=self.year[dataset_name][0],
                            )

            # Compute and store the different variations of sigma_m_over_m
            tnp_candidates = compute_sigma_m(tnp_candidates, processor='tnp', flow_corrections=self.doFlow_corrections, smear=self.Smear_sigma_m, IsData=(self.data_kind == "data"))

            # Adding the tagandprobe pair mass. Based on the expression provided here for a massless pair of particles -> (https://en.wikipedia.org/wiki/Invariant_mass)
            tnp_candidates["mass"] = numpy.sqrt(2 * tnp_candidates["tag"].pt * tnp_candidates["probe"].pt * (numpy.cosh(tnp_candidates["tag"].eta - tnp_candidates["probe"].eta) - numpy.cos(tnp_candidates["tag"].phi - tnp_candidates["probe"].phi)))

            if self.output_location is not None:
                df = diphoton_list_to_pandas(self, tnp_candidates)

                # since we annotated the photons with event variables, these exist now for tag and probe. This concerns weights as well as nPV and fixedGridRhoAll Remove:
                if self.data_kind == "mc":

                    # Store variations with respect to central weight
                    if do_variation == "nominal":
                        if len(event_weights.variations):
                            logger.info(
                                "Adding systematic weight variations to nominal output file."
                            )
                        for modifier in event_weights.variations:
                            df["weight_" + modifier] = numpy.hstack(event_weights.weight(
                                modifier=modifier
                            ) * n_event_tnp_cand)

                    # storing the central weights
                    df["weight_central"] = numpy.hstack(
                        (event_weights.weight() / numpy.array(events[flat_tag_and_probe_mask].genWeight)) * n_event_tnp_cand
                    )
                    # generated weights * other weights (pile up, SF, etc ...)
                    df["weight"] = numpy.hstack(event_weights.weight() * n_event_tnp_cand)
                    df["weight_no_pu"] = df["tag_weight"]

                    # dropping the nominal and varitation weights
                    df.drop(["tag_weight", "probe_weight"], axis=1, inplace=True)

                df["nPV"] = df["tag_nPV"]
                df.drop(["tag_nPV", "probe_nPV"], axis=1, inplace=True)
                df["fixedGridRhoAll"] = df["tag_fixedGridRhoAll"]
                df.drop(
                    ["tag_fixedGridRhoAll", "probe_fixedGridRhoAll"],
                    axis=1,
                    inplace=True,
                )

                fname = (
                    events.behavior["__events_factory__"]._partition_key.replace(
                        "/", "_"
                    )
                    + ".%s" % self.output_format
                )
                subdirs = []
                if "dataset" in events.metadata:
                    subdirs.append(events.metadata["dataset"])
                subdirs.append(variation)
                dump_pandas(self, df, fname, self.output_location, subdirs)

        return {}

    def process_extra(self, events: ak.Array) -> ak.Array:
        return events, {}

    def postprocess(self, accumulant: Dict[Any, Any]) -> Any:
        pass
