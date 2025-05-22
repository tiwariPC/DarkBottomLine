from coffea.jetmet_tools import JetCorrectionUncertainty

# from coffea.jetmet_tools import FactorizedJetCorrector
from coffea.jetmet_tools import JECStack, CorrectedJetsFactory
from coffea.lookup_tools import extractor
import awkward as ak
import numpy as np
import os

# from coffea.nanoevents import NanoEventsFactory, NanoAODSchema


# jet dummy
def jet_pt_scale_dummy(pt, events, year, is_correction=True):
    _pt = events.Jet.pt
    if is_correction:
        events.Jet['pt'] = 1.1 * _pt[:, None]
        return events
    else:
        _pt = ak.flatten(_pt)
        up_variation = 1.1 * np.ones(len(_pt))
        down_variation = 0.9 * np.ones(len(_pt))
        return np.concatenate((up_variation.reshape(-1,1), down_variation.reshape(-1,1)), axis=1) * _pt[:, None]


def JERC_jet(pt, events, year="2017", skip_JER=False, skip_JEC=False, is_correction=True):
    ext = extractor()
    metadata_dict = {
        "2022postEE": {
            "filenames": [
                "data/Winter22Run3_MC/JEC/Winter22Run3_V2_MC_L1FastJet_AK4PFPuppi.txt",
                "data/Winter22Run3_MC/JEC/Winter22Run3_V2_MC_L2L3Residual_AK4PFPuppi.txt",
                "data/Winter22Run3_MC/JEC/Winter22Run3_V2_MC_L2Relative_AK4PFPuppi.txt",
                "data/Winter22Run3_MC/JEC/Winter22Run3_V2_MC_L2Residual_AK4PFPuppi.txt",
                "data/Winter22Run3_MC/JEC/Winter22Run3_V2_MC_L3Absolute_AK4PFPuppi.txt",
                "data/Winter22Run3_MC/JEC/Winter22Run3_V2_MC_Uncertainty_AK4PFPuppi.junc.txt"
            ],
            "jec_stack_names": [
                "Winter22Run3_V2_MC_L1FastJet_AK4PFPuppi",
                "Winter22Run3_V2_MC_L2L3Residual_AK4PFPuppi",
                "Winter22Run3_V2_MC_L2Relative_AK4PFPuppi",
                "Winter22Run3_V2_MC_L2Residual_AK4PFPuppi",
                "Winter22Run3_V2_MC_L3Absolute_AK4PFPuppi",
                "Winter22Run3_V2_MC_Uncertainty_AK4PFPuppi",
            ],
            "variables":{
                "pt": "JetPt",
                "mass": "JetMass",
                "eta": "JetEta",
                "area": "JetA",
                "pt_gen": "ptGenJet",
                "pt_raw": "ptRaw",
                "mass_raw": "massRaw",
                "rho": "Rho",
            }
        },
        "2017": {
            "filenames": [
                "data/Summer19UL17_MC/JEC/Summer19UL17_V5_MC_L1FastJet_AK4PFchs.txt",
                "data/Summer19UL17_MC/JEC/Summer19UL17_V5_MC_L1RC_AK4PFchs.txt",
                "data/Summer19UL17_MC/JEC/Summer19UL17_V5_MC_L2L3Residual_AK4PFchs.txt",
                "data/Summer19UL17_MC/JEC/Summer19UL17_V5_MC_L2Relative_AK4PFchs.txt",
                "data/Summer19UL17_MC/JEC/Summer19UL17_V5_MC_L2Residual_AK4PFchs.txt",
                "data/Summer19UL17_MC/JEC/Summer19UL17_V5_MC_L3Absolute_AK4PFchs.txt",
                "data/Summer19UL17_MC/JEC/RegroupedV2_Summer19UL17_V5_MC_UncertaintySources_AK4PFchs.junc.txt",
                "data/Summer19UL17_MC/JEC/Summer19UL17_V5_MC_Uncertainty_AK4PFchs.junc.txt",
                "data/Summer19UL17_MC/JER/Summer19UL17_JRV2_MC_PtResolution_AK4PFchs.jr.txt",
                "data/Summer19UL17_MC/JER/Summer19UL17_JRV2_MC_SF_AK4PFchs.jersf.txt"
            ],
            "jec_stack_names": [
                "Summer19UL17_V5_MC_L1FastJet_AK4PFchs",
                "Summer19UL17_V5_MC_L2L3Residual_AK4PFchs",
                "Summer19UL17_V5_MC_L2Relative_AK4PFchs",
                "Summer19UL17_V5_MC_L3Absolute_AK4PFchs",
                # "Summer19UL17_V5_MC_L1RC_AK4PFchs",  these are not supposed to be applied to MC https://github.com/cms-sw/cmssw/blob/5e1e09cbe7366545e2679ddd6c6f2a8aca21953e/PhysicsTools/NanoAOD/python/jetsAK4_CHS_cff.py#L10
                # "Summer19UL17_V5_MC_L2Residual_AK4PFchs",
                "Summer19UL17_V5_MC_Uncertainty_AK4PFchs",
                "Summer19UL17_JRV2_MC_PtResolution_AK4PFchs",
                "Summer19UL17_JRV2_MC_SF_AK4PFchs"
            ],
            "variables":{
                "pt": "JetPt",
                "mass": "JetMass",
                "eta": "JetEta",
                "area": "JetA",
                "pt_gen": "ptGenJet",
                "pt_raw": "ptRaw",
                "mass_raw": "massRaw",
                "rho": "Rho",
            }
        }
    }

    # era/year defined as parameter of the function, only 2017 and 2022 is implemented up to now
    avail_years = [*metadata_dict]
    if year not in avail_years:
        print(f"\n WARNING: only scale corrections for the year strings {avail_years} are already implemented! \n Exiting. \n")
        exit()

    # removing JER files and correction names if we don't want them
    if skip_JER:
        print("Skipping JER")
        metadata_dict[year]["filenames"] = [corr_file for corr_file in metadata_dict[year]["filenames"] if "JR" not in corr_file]
        metadata_dict[year]["jec_stack_names"] = [corr_name for corr_name in metadata_dict[year]["jec_stack_names"] if "JR" not in corr_name]
    # removing JEC files and correction names if we don't want them
    if skip_JEC:
        print("Skipping JEC")
        metadata_dict[year]["filenames"] = [corr_file for corr_file in metadata_dict[year]["filenames"] if (("JR" in corr_file) or ("Uncertainty" in corr_file))]
        metadata_dict[year]["jec_stack_names"] = [corr_name for corr_name in metadata_dict[year]["jec_stack_names"] if (("JR" in corr_name) or ("Uncertainty" in corr_name))]

    # adding weight files to the set of weights to be used
    metadata_dict[year]["filenames"] = ["* * {}".format(os.path.join(os.path.dirname(__file__), corr_file)) for corr_file in metadata_dict[year]["filenames"]]

    ext.add_weight_sets(metadata_dict[year]["filenames"])
    ext.finalize()

    evaluator = ext.make_evaluator()

    # adding factorised sources of uncertainty
    for name in dir(evaluator):
        if "UncertaintySources_AK4PFchs" in name:
            metadata_dict[year]["jec_stack_names"].append(name)

    jec_inputs = {name: evaluator[name] for name in metadata_dict[year]["jec_stack_names"]}
    jec_stack = JECStack(jec_inputs)

    # preparing input variables
    name_map = jec_stack.blank_name_map
    for var in metadata_dict[year]["variables"]:
        name_map[metadata_dict[year]["variables"][var]] = var

    jets = events.Jet

    jets["pt_raw"] = (1 - jets["rawFactor"]) * jets["pt"]
    jets["mass_raw"] = (1 - jets["rawFactor"]) * jets["mass"]
    jets["pt_gen"] = ak.values_astype(ak.fill_none(jets.matched_gen.pt, 0), np.float32)
    jets["rho"] = ak.broadcast_arrays(events.Rho.fixedGridRhoFastjetAll, jets.pt)[0]

    events_cache = events.caches[0]
    if year == "2022postEE":
        uncertainties = JetCorrectionUncertainty(
            Winter22Run3_V2_MC_Uncertainty_AK4PFPuppi=evaluator[
                "Winter22Run3_V2_MC_Uncertainty_AK4PFPuppi"
            ]
        )
    elif year == "2017":
        sources = [unc_name for unc_name in metadata_dict[year]["jec_stack_names"] if "Uncertainty" in unc_name]
        uncertainties = {}
        for source in sources:
            parts = source.rsplit('_', 2)
            key = parts[-1] if parts[-1] != "2017" else f"{parts[-2]}_{parts[-1]}"
            uncertainties[key] = JetCorrectionUncertainty(Summer19UL17_V5_MC_Uncertainty_AK4PFchs=evaluator[source])

    jet_factory = CorrectedJetsFactory(name_map, jec_stack)
    if is_correction:
        corrected_jets = jet_factory.build(jets, lazy_cache=events_cache)
        events.Jet = corrected_jets

        return events

    else:
        sys = list(uncertainties["Total"].getUncertainty(JetEta=jets.eta, JetPt=jets.pt))
        level, corrs = sys[0]
        uncertainty = ak.flatten(corrs)

        return uncertainty * (ak.flatten(jets.pt)[:, None])
