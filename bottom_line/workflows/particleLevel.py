from bottom_line.workflows.base import bbMETBaseProcessor
from bottom_line.systematics import object_corrections as available_object_corrections
from bottom_line.systematics import weight_corrections as available_weight_corrections
from bottom_line.utils.dumping_utils import diphoton_ak_array, dump_ak_array, diphoton_list_to_pandas, dump_pandas

from bottom_line.tools.gen_helpers import get_fiducial_flag, get_genJets, get_higgs_gen_attributes
from bottom_line.utils.misc_utils import choose_jet

from typing import Any, Dict, List, Optional
import awkward as ak
import logging
import warnings
import numpy
import sys
from coffea.analysis_tools import Weights

logger = logging.getLogger(__name__)


class ParticleLevelProcessor(bbMETBaseProcessor):
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
        fiducialCuts: str = "none",
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

    def process_extra(self, events: ak.Array) -> ak.Array:
        return events, {}

    def process(self, events: ak.Array) -> Dict[Any, Any]:
        dataset_name = events.metadata["dataset"]

        # metadata array to append to higgsdna output
        metadata = {}

        # data or monte carlo?
        self.data_kind = "mc" if hasattr(events, "GenPart") else "data"

        if self.data_kind == "data":
            logger.info("The 'particleLevel' processor can only be run on MC. Aborting now...")
            sys.exit(0)

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

        # Add sum of gen weights before selection for normalisation in postprocessing
        metadata["sum_genw_presel"] = str(ak.sum(events.genWeight))

        # read which systematics and corrections to process
        try:
            correction_names = self.corrections[dataset_name]
        except KeyError:
            correction_names = []

        for correction_name in correction_names:
            if correction_name in available_object_corrections.keys():
                logger.info(
                    f"Applying correction {correction_name} to dataset {dataset_name}"
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

        # Filling with some dummy values
        diphotons = ak.Array({"pt": numpy.ones(len(events))})

        events["GenIsolatedPhoton"] = ak.pad_none(events["GenIsolatedPhoton"], 2)

        diphotons["leadingGenIsolatedPhoton_pt"] = events.GenIsolatedPhoton[:,0].pt
        diphotons["leadingGenIsolatedPhoton_eta"] = events.GenIsolatedPhoton[:,0].eta
        diphotons["leadingGenIsolatedPhoton_phi"] = events.GenIsolatedPhoton[:,0].phi

        diphotons["subleadingGenIsolatedPhoton_pt"] = events.GenIsolatedPhoton[:,1].pt
        diphotons["subleadingGenIsolatedPhoton_eta"] = events.GenIsolatedPhoton[:,1].eta
        diphotons["subleadingGenIsolatedPhoton_phi"] = events.GenIsolatedPhoton[:,1].phi

        diphotons['fiducialClassicalFlag'] = get_fiducial_flag(events, flavour='Classical')
        diphotons['fiducialGeometricFlag'] = get_fiducial_flag(events, flavour='Geometric')
        diphotons['GenPTH'], diphotons['GenYH'], diphotons['GenPhiH'] = get_higgs_gen_attributes(events)

        genJets = get_genJets(self, events, pt_cut=30., eta_cut=2.5)
        diphotons['GenNJ'] = ak.num(genJets)
        diphotons['GenPTJ0'] = choose_jet(genJets.pt, 0, -999.0)  # Choose zero (leading) jet and pad with -999 if none

        # workflow specific processing
        events, process_extra = self.process_extra(events)
        histos_etc.update(process_extra)

        # set diphotons as part of the event record
        events["diphotons"] = diphotons
        # annotate diphotons with event information
        diphotons["event"] = events.event
        diphotons["lumi"] = events.luminosityBlock
        diphotons["run"] = events.run
        # annotate diphotons with dZ information (difference between z position of GenVtx and PV) as required by flashggfinalfits
        diphotons["genWeight"] = events.genWeight
        diphotons["dZ"] = events.GenVtx.z - events.PV.z

        for i in range(9):
            diphotons["LHEScaleWeight_" + str(i)] = events.LHEScaleWeight[:,i]
        for i in range(103):
            diphotons["LHEPdfWeight_" + str(i)] = events.LHEPdfWeight[:,i]

        # return if there is no surviving events
        if len(diphotons) == 0:
            logger.info("No surviving events in this run, return now!")
            return histos_etc

        # Retain all events
        selection_mask = numpy.ones(len(diphotons), dtype=bool)
        # initiate Weight container here, after selection, since event selection cannot easily be applied to weight container afterwards
        event_weights = Weights(size=len(events[selection_mask]))
        # set weights to generator weights
        event_weights._weight = events["genWeight"][selection_mask]
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
                    photons=events["diphotons"][
                        selection_mask
                    ],
                    weights=event_weights,
                    dataset_name=dataset_name,
                    year=self.year[dataset_name][0],
                )
        diphotons["weight_central"] = event_weights.weight() / events["genWeight"][selection_mask]  # Here, if diphotons none, then also the weight is None.
        diphotons["weight"] = event_weights.weight()

        if self.output_location is not None:
            if self.output_format == "root":
                df = diphoton_list_to_pandas(self, diphotons)
            else:
                akarr = diphoton_ak_array(self, diphotons)
            fname = (
                events.behavior[
                    "__events_factory__"
                ]._partition_key.replace("/", "_")
                + ".%s" % self.output_format
            )

            subdirs = []
            if "dataset" in events.metadata:
                subdirs.append(events.metadata["dataset"])
            if self.output_format == "root":
                dump_pandas(self, df, fname, self.output_location, subdirs)
            else:
                dump_ak_array(
                    self, akarr, fname, self.output_location, metadata, subdirs,
                )

        return histos_etc

    def postprocess(self, accumulant: Dict[Any, Any]) -> Any:
        pass
