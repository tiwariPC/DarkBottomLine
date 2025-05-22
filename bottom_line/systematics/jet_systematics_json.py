from copy import deepcopy
import gzip
import awkward as ak
import numpy as np
import correctionlib
from correctionlib.schemav2 import Correction, CorrectionSet

import os
import logging

logger = logging.getLogger(__name__)


def get_jer_correction_set(jer_json, jer_ptres_tag, jer_sf_tag):
    # learned from: https://github.com/cms-nanoAOD/correctionlib/issues/130
    with gzip.open(jer_json) as fin:
        cset = CorrectionSet.parse_raw(fin.read())

    cset.corrections = [
        c
        for c in cset.corrections
        if c.name
        in (
            jer_ptres_tag,
            jer_sf_tag,
        )
    ]
    cset.compound_corrections = []

    res = Correction.parse_obj(
        {
            "name": "JERSmear",
            "description": "Jet smearing tool",
            "inputs": [
                {"name": "JetPt", "type": "real"},
                {"name": "JetEta", "type": "real"},
                {
                    "name": "GenPt",
                    "type": "real",
                    "description": "matched GenJet pt, or -1 if no match",
                },
                {"name": "Rho", "type": "real", "description": "entropy source"},
                {"name": "EventID", "type": "int", "description": "entropy source"},
                {
                    "name": "JER",
                    "type": "real",
                    "description": "Jet energy resolution",
                },
                {
                    "name": "JERsf",
                    "type": "real",
                    "description": "Jet energy resolution scale factor",
                },
            ],
            "output": {"name": "smear", "type": "real"},
            "version": 1,
            "data": {
                "nodetype": "binning",
                "input": "GenPt",
                "edges": [-1, 0, 1],
                "flow": "clamp",
                "content": [
                    # stochastic
                    {
                        # rewrite gen_pt with a random gaussian
                        "nodetype": "transform",
                        "input": "GenPt",
                        "rule": {
                            "nodetype": "hashprng",
                            "inputs": ["JetPt", "JetEta", "Rho", "EventID"],
                            "distribution": "normal",
                        },
                        "content": {
                            "nodetype": "formula",
                            # TODO min jet pt?
                            "expression": "1+sqrt(max(x*x - 1, 0)) * y * z",
                            "parser": "TFormula",
                            # now gen_pt is actually the output of hashprng
                            "variables": ["JERsf", "JER", "GenPt"],
                        },
                    },
                    # deterministic
                    {
                        "nodetype": "formula",
                        # TODO min jet pt?
                        "expression": "1+(x-1)*(y-z)/y",
                        "parser": "TFormula",
                        "variables": ["JERsf", "JetPt", "GenPt"],
                    },
                ],
            },
        }
    )
    cset.corrections.append(res)
    ceval = cset.to_evaluator()
    return ceval


def get_jersmear(_eval_dict, _ceval, _jer_sf_tag, _syst="nom"):
    _eval_dict.update({"systematic": _syst})
    _inputs_jer_sf = [_eval_dict[input.name] for input in _ceval[_jer_sf_tag].inputs]
    _jer_sf = _ceval[_jer_sf_tag].evaluate(*_inputs_jer_sf)
    _eval_dict.update({"JERsf": _jer_sf})
    _inputs = [_eval_dict[input.name] for input in _ceval["JERSmear"].inputs]
    _jersmear = _ceval["JERSmear"].evaluate(*_inputs)
    return _eval_dict, _jersmear


def jerc_jet(
    pt,
    events,
    year="2022postEE",
    era="MC",
    level="L1L2L3Res",
    apply_jec=True,
    jec_syst=False,
    split_jec_syst=False,
    apply_jer=False,
    jer_syst=False,
    pnet="",
):
    # first, check if it's data or MC
    if era == "MC" and hasattr(events, "GenPart"):
        logger.debug(
            f"[ jerc_jet ] - JERC for simulation - Year: {year} - Era: {era} - JEC: {apply_jec}, systematics: {jec_syst}, Regrouped: {split_jec_syst} - JER: {apply_jer}, systematics: {jer_syst}"
        )
    elif "Run" in era and (not hasattr(events, "GenPart")):
        apply_jec = True
        jec_syst = False
        split_jec_syst = False
        apply_jer = False
        jer_syst = False
        logger.debug(
            f"[ jerc_jet ] - JERC for Data - Year: {year} - Era: {era} - Only JEC to be applied - JEC: {apply_jec}, systematics: {jec_syst}, Regrouped: {split_jec_syst} - JER: {apply_jer}, systematics: {jer_syst}"
        )
    else:
        logger.error(f"[ jerc_jet ] - Era: {era} doesn't match the input dataset")
        exit(-1)
    # run2: AK4PFchs - run3: AK4PFPuppi
    # If some PNet regression is used, the proper corrections will be picked up
    if int(year[:4]) > 2018:
        algo = "AK4PFPuppi" + pnet
    else:
        algo = "AK4PFchs"

    pnetFlag = ""
    if pnet != "":
        pnetFlag = "_PNet"
    # jec json file
    jerc_json = {
        "2016preVFP": os.path.join(
            os.path.dirname(__file__),
            "../systematics/JSONs/POG/JME/2016preVFP_UL/jet_jerc.json.gz",
        ),
        "2016postVFP": os.path.join(
            os.path.dirname(__file__),
            "../systematics/JSONs/POG/JME/2016postVFP_UL/jet_jerc.json.gz",
        ),
        "2017": os.path.join(
            os.path.dirname(__file__),
            "../systematics/JSONs/POG/JME/2017_UL/jet_jerc.json.gz",
        ),
        "2018": os.path.join(
            os.path.dirname(__file__),
            "../systematics/JSONs/POG/JME/2018_UL/jet_jerc.json.gz",
        ),
        "2022preEE": os.path.join(
            os.path.dirname(__file__),
            "../systematics/JSONs/POG/JME/2022_Summer22/jet_jerc" + pnetFlag + ".json.gz",
        ),
        "2022postEE": os.path.join(
            os.path.dirname(__file__),
            "../systematics/JSONs/POG/JME/2022_Summer22EE/jet_jerc" + pnetFlag + ".json.gz",
        ),
        "2023preBPix": os.path.join(
            os.path.dirname(__file__),
            "../systematics/JSONs/POG/JME/2023_Summer23/jet_jerc" + pnetFlag + ".json.gz",
        ),
        "2023postBPix": os.path.join(
            os.path.dirname(__file__),
            "../systematics/JSONs/POG/JME/2023_Summer23BPix/jet_jerc" + pnetFlag + ".json.gz",
        ),
    }
    jec_version = {
        "2016preVFP": {
            "RunB": "Summer19UL16APV_RunBCD_V7_DATA",
            "RunC": "Summer19UL16APV_RunBCD_V7_DATA",
            "RunD": "Summer19UL16APV_RunBCD_V7_DATA",
            "RunE": "Summer19UL16APV_RunEF_V7_DATA",
            "RunF": "Summer19UL16APV_RunEF_V7_DATA",
            "MC": "Summer19UL16APV_V7_MC",
        },
        "2016postVFP": {
            "RunF": "Summer19UL16_RunFGH_V7_DATA",
            "RunG": "Summer19UL16_RunFGH_V7_DATA",
            "RunH": "Summer19UL16_RunFGH_V7_DATA",
            "MC": "Summer19UL16_V7_MC",
        },
        "2017": {
            "RunB": "Summer19UL17_RunB_V5_DATA",
            "RunC": "Summer19UL17_RunC_V5_DATA",
            "RunD": "Summer19UL17_RunD_V5_DATA",
            "RunE": "Summer19UL17_RunE_V5_DATA",
            "RunF": "Summer19UL17_RunF_V5_DATA",
            "MC": "Summer19UL17_V5_MC",
        },
        "2018": {
            "RunA": "Summer19UL18_RunA_V5_DATA",
            "RunB": "Summer19UL18_RunB_V5_DATA",
            "RunC": "Summer19UL18_RunC_V5_DATA",
            "RunD": "Summer19UL18_RunD_V5_DATA",
            "MC": "Summer19UL18_V5_MC",
        },
        "2022preEE": {
            "RunC": "Summer22_22Sep2023_RunCD_V2_DATA",
            "RunD": "Summer22_22Sep2023_RunCD_V2_DATA",
            "MC": "Summer22_22Sep2023_V2_MC",
        },
        "2022postEE": {
            "RunE": "Summer22EE_22Sep2023_RunE_V2_DATA",
            "RunF": "Summer22EE_22Sep2023_RunF_V2_DATA",
            "RunG": "Summer22EE_22Sep2023_RunG_V2_DATA",
            "MC": "Summer22EE_22Sep2023_V2_MC",
        },
        # For 2023 era C, different version of the datasets have different JECs
        # Details: https://gitlab.cern.ch/cms-analysis/general/HiggsDNA/-/issues/220#note_9180675
        "2023preBPix": {
            "RunCv123": "Summer23Prompt23_RunCv123_V1_DATA",
            "RunCv4": "Summer23Prompt23_RunCv4_V1_DATA",
            "MC": "Summer23Prompt23_V1_MC",
        },
        "2023postBPix": {
            "RunD": "Summer23BPixPrompt23_RunD_V1_DATA",
            "MC": "Summer23BPixPrompt23_V1_MC",
        },
    }
    jec = jec_version[year][era]
    tag_jec = "_".join([jec, level, algo])

    # get the correction sets
    cset = correctionlib.CorrectionSet.from_file(jerc_json[year])

    # prepare inputs
    jets_jagged = deepcopy(events.Jet)
    counts = ak.num(jets_jagged)
    # backup of the original nanoaod jet pt, only for once
    if "pt_nano" not in jets_jagged.fields:
        jets_jagged["pt_nano"] = jets_jagged.pt
        jets_jagged["mass_nano"] = jets_jagged.mass
    # store the raw jet pt, only for once
    if "pt_raw" not in jets_jagged.fields:
        pnetFactor = 1.0
        if pnet == "PNetRegression":
            pnetFactor = jets_jagged.PNetRegPtRawCorr
        if pnet == "PNetRegressionPlusNeutrino":
            pnetFactor = jets_jagged.PNetRegPtRawCorr * jets_jagged.PNetRegPtRawCorrNeutrino
        jets_jagged["pt_raw"] = jets_jagged.pt * (1 - jets_jagged.rawFactor) * pnetFactor
        jets_jagged["mass_raw"] = jets_jagged.mass * (1 - jets_jagged.rawFactor)
    # avoid using hasattr(jets_jagged, "rho"). Same name as the coffea vector property of rho: https://github.com/CoffeaTeam/coffea/blob/0e43daf8e40ccec44efb2622777354ebd0424b84/src/coffea/nanoevents/methods/vector.py#L482
    if "rho_value" not in jets_jagged.fields:
        try:
            jets_jagged["rho_value"] = (
                ak.ones_like(jets_jagged.pt) * events.Rho.fixedGridRhoFastjetAll
            )
        except:
            # UL datasets have different naming convention
            jets_jagged["rho_value"] = (
                ak.ones_like(jets_jagged.pt) * events.fixedGridRhoFastjetAll
            )
    # create the gen_matched pt, only for once
    if ("pt_gen" not in jets_jagged.fields) and (apply_jer or jer_syst):
        # TODO: finalize the gen-matching algorithms
        # current follow coffea example: https://github.com/CoffeaTeam/coffea/blob/16db8f663e40dafd2399d32862c20e3faa5542be/binder/applying_corrections.ipynb#L423
        jets_jagged["pt_gen"] = ak.fill_none(jets_jagged.matched_gen.pt, -99999)
    # create the eventid, only for once
    if ("event_id" not in jets_jagged.fields) and (apply_jer or jer_syst):
        jets_jagged["event_id"] = ak.ones_like(jets_jagged.pt) * events.event

    # flatten
    jets = ak.flatten(jets_jagged)
    # evaluate dictionary
    eval_dict = {
        "JetPt": jets.pt_raw,
        "JetEta": jets.eta,
        "JetPhi": jets.phi,
        "Rho": jets.rho_value,
        "JetA": jets.area,
    }

    # jec central
    if apply_jec:
        # get the correction
        if tag_jec in list(cset.compound.keys()):
            sf = cset.compound[tag_jec]
        elif tag_jec in list(cset.keys()):
            sf = cset[tag_jec]
        else:
            logger.error(
                f"[ jerc_jet ] No JEC correction: {tag_jec} - Year: {year} - Era: {era} - Level: {level}"
            )
            exit(-1)
        inputs = [eval_dict[input.name] for input in sf.inputs]
        sf_value = sf.evaluate(*inputs)
        jets["pt_jec"] = sf_value * jets["pt_raw"]
        jets["mass_jec"] = sf_value * jets["mass_raw"]
        # update the nominal pt and mass
        jets["pt"] = jets["pt_jec"]
        jets["mass"] = jets["mass_jec"]

    # jer central and systematics
    if apply_jer or jer_syst:
        # learned from: https://github.com/cms-nanoAOD/correctionlib/issues/130
        jer_version = {
            "2016preVFP": "Summer20UL16APV_JRV3_MC",
            "2016postVFP": "Summer20UL16_JRV3_MC",
            "2017": "Summer19UL17_JRV2_MC",
            "2018": "Summer19UL18_JRV2_MC",
            "2022preEE": "Summer22_22Sep2023_JRV1_MC",
            "2022postEE": "Summer22EE_22Sep2023_JRV1_MC",
            "2023preBPix": "Summer23Prompt23_RunCv1234_JRV1_MC",
            "2023postBPix": "Summer23BPixPrompt23_RunD_JRV1_MC",
        }
        jer = jer_version[year]
        jer_ptres_tag = f"{jer}_PtResolution_{algo}"
        jer_sf_tag = f"{jer}_ScaleFactor_{algo}"

        ceval_jer = get_jer_correction_set(jerc_json[year], jer_ptres_tag, jer_sf_tag)
        # update evaluate dictionary
        eval_dict.update(
            {
                "JetPt": jets.pt if pnet == "" else jets.pt_nano,  # JER SFs for PNet run on stadandard jets
                "GenPt": jets.pt_gen,
                "EventID": jets.event_id,
            }
        )
        # get jer pt resolution
        inputs_jer_ptres = [
            eval_dict[input.name] for input in ceval_jer[jer_ptres_tag].inputs
        ]
        jer_ptres = ceval_jer[jer_ptres_tag].evaluate(*inputs_jer_ptres)
        # update evaluate dictionary
        eval_dict.update({"JER": jer_ptres})
        # addjust pt gen
        eval_dict.update(
            {
                "GenPt": np.where(
                    np.abs(eval_dict["JetPt"] - eval_dict["GenPt"])
                    < 3 * eval_dict["JetPt"] * eval_dict["JER"],
                    eval_dict["GenPt"],
                    -1.0,
                ),
            }
        )
        if apply_jer:
            eval_dict, jersmear = get_jersmear(eval_dict, ceval_jer, jer_sf_tag, "nom")
            jets["pt_jer"] = jets.pt * jersmear
            jets["mass_jer"] = jets.mass * jersmear
        if jer_syst:
            # jer up
            eval_dict, jersmear = get_jersmear(eval_dict, ceval_jer, jer_sf_tag, "up")
            jets["pt_jer_syst_up"] = jets.pt * jersmear
            jets["mass_jer_syst_up"] = jets.mass * jersmear
            # jer down
            eval_dict, jersmear = get_jersmear(eval_dict, ceval_jer, jer_sf_tag, "down")
            jets["pt_jer_syst_down"] = jets.pt * jersmear
            jets["mass_jer_syst_down"] = jets.mass * jersmear
        if apply_jer:
            # to avoid the sf: jer*jer_up or jer*jer_down, update the jer pt/mass after calculation of the jer up/down
            jets["pt"] = jets["pt_jer"]
            jets["mass"] = jets["mass_jer"]

    # jec systematics
    if jec_syst:
        # update evaluate dictionary
        eval_dict.update({"JetPt": jets.pt})
        if not split_jec_syst:
            # get the total uncertainty
            tag_jec_syst = "_".join([jec, "Total", algo])
            try:
                sf = cset[tag_jec_syst]
            except:
                logger.error(
                    f"[ jerc_jet ] No JEC systematic: {tag_jec_syst} - Year: {year} - Era: {era}"
                )
                exit(-1)
            # systematics
            inputs = [eval_dict[input.name] for input in sf.inputs]
            sf_delta = sf.evaluate(*inputs)

            # divide by correction since it is already applied before
            corr_up_variation = 1 + sf_delta
            corr_down_variation = 1 - sf_delta

            jets["pt_jec_syst_Total_up"] = jets.pt * corr_up_variation
            jets["pt_jec_syst_Total_down"] = jets.pt * corr_down_variation
            jets["mass_jec_syst_Total_up"] = jets.mass * corr_up_variation
            jets["mass_jec_syst_Total_down"] = jets.mass * corr_down_variation
        else:
            jec_syst_regrouped = {
                "2016preVFP": {
                    # regrouped jec uncertainty
                    "jec_syst_Absolute_2016": "Regrouped_Absolute_2016",
                    "jec_syst_Absolute": "Regrouped_Absolute",
                    "jec_syst_BBEC1_2016": "Regrouped_BBEC1_2016",
                    "jec_syst_BBEC1": "Regrouped_BBEC1",
                    "jec_syst_EC2_2016": "Regrouped_EC2_2016",
                    "jec_syst_EC2": "Regrouped_EC2",
                    "jec_syst_FlavorQCD": "Regrouped_FlavorQCD",
                    "jec_syst_HF_2016": "Regrouped_HF_2016",
                    "jec_syst_HF": "Regrouped_HF",
                    "jec_syst_RelativeBal": "Regrouped_Absolute",
                    "jec_syst_RelativeSample_2016": "Regrouped_RelativeSample_2016",
                    # total regrouped jec uncertainty
                    "jec_syst_Regrouped_Total": "Regrouped_Total",
                },
                "2016postVFP": {
                    # regrouped jec uncertainty
                    "jec_syst_Absolute_2016": "Regrouped_Absolute_2016",
                    "jec_syst_Absolute": "Regrouped_Absolute",
                    "jec_syst_BBEC1_2016": "Regrouped_BBEC1_2016",
                    "jec_syst_BBEC1": "Regrouped_BBEC1",
                    "jec_syst_EC2_2016": "Regrouped_EC2_2016",
                    "jec_syst_EC2": "Regrouped_EC2",
                    "jec_syst_FlavorQCD": "Regrouped_FlavorQCD",
                    "jec_syst_HF_2016": "Regrouped_HF_2016",
                    "jec_syst_HF": "Regrouped_HF",
                    "jec_syst_RelativeBal": "Regrouped_Absolute",
                    "jec_syst_RelativeSample_2016": "Regrouped_RelativeSample_2016",
                    # total regrouped jec uncertainty
                    "jec_syst_Regrouped_Total": "Regrouped_Total",
                },
                "2017": {
                    # regrouped jec uncertainty
                    "jec_syst_Absolute_2017": "Regrouped_Absolute_2017",
                    "jec_syst_Absolute": "Regrouped_Absolute",
                    "jec_syst_BBEC1_2017": "Regrouped_BBEC1_2017",
                    "jec_syst_BBEC1": "Regrouped_BBEC1",
                    "jec_syst_EC2_2017": "Regrouped_EC2_2017",
                    "jec_syst_EC2": "Regrouped_EC2",
                    "jec_syst_FlavorQCD": "Regrouped_FlavorQCD",
                    "jec_syst_HF_2017": "Regrouped_HF_2017",
                    "jec_syst_HF": "Regrouped_HF",
                    "jec_syst_RelativeBal": "Regrouped_Absolute",
                    "jec_syst_RelativeSample_2017": "Regrouped_RelativeSample_2017",
                    # total regrouped jec uncertainty
                    "jec_syst_Regrouped_Total": "Regrouped_Total",
                },
                "2018": {
                    # regrouped jec uncertainty
                    "jec_syst_Absolute_2018": "Regrouped_Absolute_2018",
                    "jec_syst_Absolute": "Regrouped_Absolute",
                    "jec_syst_BBEC1_2018": "Regrouped_BBEC1_2018",
                    "jec_syst_BBEC1": "Regrouped_BBEC1",
                    "jec_syst_EC2_2018": "Regrouped_EC2_2018",
                    "jec_syst_EC2": "Regrouped_EC2",
                    "jec_syst_FlavorQCD": "Regrouped_FlavorQCD",
                    "jec_syst_HF_2018": "Regrouped_HF_2018",
                    "jec_syst_HF": "Regrouped_HF",
                    "jec_syst_RelativeBal": "Regrouped_Absolute",
                    "jec_syst_RelativeSample_2018": "Regrouped_RelativeSample_2018",
                    # total regrouped jec uncertainty
                    "jec_syst_Regrouped_Total": "Regrouped_Total",
                },
                "2022preEE": {
                    "jec_syst_AbsoluteMPFBias": "AbsoluteMPFBias",
                    "jec_syst_AbsoluteScale": "AbsoluteScale",
                    "jec_syst_AbsoluteStat": "AbsoluteStat",
                    "jec_syst_FlavorQCD": "FlavorQCD",
                    "jec_syst_Fragmentation": "Fragmentation",
                    "jec_syst_PileUpDataMC": "PileUpDataMC",
                    "jec_syst_PileUpPtBB": "PileUpPtBB",
                    "jec_syst_PileUpPtEC1": "PileUpPtEC1",
                    "jec_syst_PileUpPtEC2": "PileUpPtEC2",
                    "jec_syst_PileUpPtHF": "PileUpPtHF",
                    "jec_syst_PileUpPtRef": "PileUpPtRef",
                    "jec_syst_RelativeFSR": "RelativeFSR",
                    "jec_syst_RelativeJEREC1": "RelativeJEREC1",
                    "jec_syst_RelativeJEREC2": "RelativeJEREC2",
                    "jec_syst_RelativeJERHF": "RelativeJERHF",
                    "jec_syst_RelativePtBB": "RelativePtBB",
                    "jec_syst_RelativePtEC1": "RelativePtEC1",
                    "jec_syst_RelativePtEC2": "RelativePtEC2",
                    "jec_syst_RelativePtHF": "RelativePtHF",
                    "jec_syst_RelativeBal": "RelativeBal",
                    "jec_syst_RelativeSample": "RelativeSample",
                    "jec_syst_RelativeStatEC": "RelativeStatEC",
                    "jec_syst_RelativeStatFSR": "RelativeStatFSR",
                    "jec_syst_RelativeStatHF": "RelativeStatHF",
                    "jec_syst_SinglePionECAL": "SinglePionECAL",
                    "jec_syst_SinglePionHCAL": "SinglePionHCAL",
                    "jec_syst_TimePtEta": "TimePtEta",
                    "jec_syst_Total": "Total",
                },
                "2022postEE": {
                    "jec_syst_AbsoluteMPFBias": "AbsoluteMPFBias",
                    "jec_syst_AbsoluteScale": "AbsoluteScale",
                    "jec_syst_AbsoluteStat": "AbsoluteStat",
                    "jec_syst_FlavorQCD": "FlavorQCD",
                    "jec_syst_Fragmentation": "Fragmentation",
                    "jec_syst_PileUpDataMC": "PileUpDataMC",
                    "jec_syst_PileUpPtBB": "PileUpPtBB",
                    "jec_syst_PileUpPtEC1": "PileUpPtEC1",
                    "jec_syst_PileUpPtEC2": "PileUpPtEC2",
                    "jec_syst_PileUpPtHF": "PileUpPtHF",
                    "jec_syst_PileUpPtRef": "PileUpPtRef",
                    "jec_syst_RelativeFSR": "RelativeFSR",
                    "jec_syst_RelativeJEREC1": "RelativeJEREC1",
                    "jec_syst_RelativeJEREC2": "RelativeJEREC2",
                    "jec_syst_RelativeJERHF": "RelativeJERHF",
                    "jec_syst_RelativePtBB": "RelativePtBB",
                    "jec_syst_RelativePtEC1": "RelativePtEC1",
                    "jec_syst_RelativePtEC2": "RelativePtEC2",
                    "jec_syst_RelativePtHF": "RelativePtHF",
                    "jec_syst_RelativeBal": "RelativeBal",
                    "jec_syst_RelativeSample": "RelativeSample",
                    "jec_syst_RelativeStatEC": "RelativeStatEC",
                    "jec_syst_RelativeStatFSR": "RelativeStatFSR",
                    "jec_syst_RelativeStatHF": "RelativeStatHF",
                    "jec_syst_SinglePionECAL": "SinglePionECAL",
                    "jec_syst_SinglePionHCAL": "SinglePionHCAL",
                    "jec_syst_TimePtEta": "TimePtEta",
                    "jec_syst_Total": "Total",
                },
                "2023preBPix": {
                    "jec_syst_AbsoluteMPFBias": "AbsoluteMPFBias",
                    "jec_syst_AbsoluteScale": "AbsoluteScale",
                    "jec_syst_AbsoluteStat": "AbsoluteStat",
                    "jec_syst_FlavorQCD": "FlavorQCD",
                    "jec_syst_Fragmentation": "Fragmentation",
                    "jec_syst_PileUpDataMC": "PileUpDataMC",
                    "jec_syst_PileUpPtBB": "PileUpPtBB",
                    "jec_syst_PileUpPtEC1": "PileUpPtEC1",
                    "jec_syst_PileUpPtEC2": "PileUpPtEC2",
                    "jec_syst_PileUpPtHF": "PileUpPtHF",
                    "jec_syst_PileUpPtRef": "PileUpPtRef",
                    "jec_syst_RelativeFSR": "RelativeFSR",
                    "jec_syst_RelativeJEREC1": "RelativeJEREC1",
                    "jec_syst_RelativeJEREC2": "RelativeJEREC2",
                    "jec_syst_RelativeJERHF": "RelativeJERHF",
                    "jec_syst_RelativePtBB": "RelativePtBB",
                    "jec_syst_RelativePtEC1": "RelativePtEC1",
                    "jec_syst_RelativePtEC2": "RelativePtEC2",
                    "jec_syst_RelativePtHF": "RelativePtHF",
                    "jec_syst_RelativeBal": "RelativeBal",
                    "jec_syst_RelativeSample": "RelativeSample",
                    "jec_syst_RelativeStatEC": "RelativeStatEC",
                    "jec_syst_RelativeStatFSR": "RelativeStatFSR",
                    "jec_syst_RelativeStatHF": "RelativeStatHF",
                    "jec_syst_SinglePionECAL": "SinglePionECAL",
                    "jec_syst_SinglePionHCAL": "SinglePionHCAL",
                    "jec_syst_TimePtEta": "TimePtEta",
                    "jec_syst_Total": "Total",
                },
                "2023postBPix": {
                    "jec_syst_AbsoluteMPFBias": "AbsoluteMPFBias",
                    "jec_syst_AbsoluteScale": "AbsoluteScale",
                    "jec_syst_AbsoluteStat": "AbsoluteStat",
                    "jec_syst_FlavorQCD": "FlavorQCD",
                    "jec_syst_Fragmentation": "Fragmentation",
                    "jec_syst_PileUpDataMC": "PileUpDataMC",
                    "jec_syst_PileUpPtBB": "PileUpPtBB",
                    "jec_syst_PileUpPtEC1": "PileUpPtEC1",
                    "jec_syst_PileUpPtEC2": "PileUpPtEC2",
                    "jec_syst_PileUpPtHF": "PileUpPtHF",
                    "jec_syst_PileUpPtRef": "PileUpPtRef",
                    "jec_syst_RelativeFSR": "RelativeFSR",
                    "jec_syst_RelativeJEREC1": "RelativeJEREC1",
                    "jec_syst_RelativeJEREC2": "RelativeJEREC2",
                    "jec_syst_RelativeJERHF": "RelativeJERHF",
                    "jec_syst_RelativePtBB": "RelativePtBB",
                    "jec_syst_RelativePtEC1": "RelativePtEC1",
                    "jec_syst_RelativePtEC2": "RelativePtEC2",
                    "jec_syst_RelativePtHF": "RelativePtHF",
                    "jec_syst_RelativeBal": "RelativeBal",
                    "jec_syst_RelativeSample": "RelativeSample",
                    "jec_syst_RelativeStatEC": "RelativeStatEC",
                    "jec_syst_RelativeStatFSR": "RelativeStatFSR",
                    "jec_syst_RelativeStatHF": "RelativeStatHF",
                    "jec_syst_SinglePionECAL": "SinglePionECAL",
                    "jec_syst_SinglePionHCAL": "SinglePionHCAL",
                    "jec_syst_TimePtEta": "TimePtEta",
                    "jec_syst_Total": "Total",
                },
            }
            for i in jec_syst_regrouped[year]:
                # get the total uncertainty
                tag_jec_syst = "_".join([jec, jec_syst_regrouped[year][i], algo])
                try:
                    sf = cset[tag_jec_syst]
                except:
                    logger.error(
                        f"[ jerc_jet ] No JEC systematic: {tag_jec_syst} - Year: {year} - Era: {era}"
                    )
                    exit(-1)
                # systematics
                inputs = [eval_dict[input.name] for input in sf.inputs]
                sf_delta = sf.evaluate(*inputs)

                # divide by correction since it is already applied before
                corr_up_variation = 1 + sf_delta
                corr_down_variation = 1 - sf_delta

                jets[f"pt_{i}_up"] = jets.pt * corr_up_variation
                jets[f"pt_{i}_down"] = jets.pt * corr_down_variation
                jets[f"mass_{i}_up"] = jets.mass * corr_up_variation
                jets[f"mass_{i}_down"] = jets.mass * corr_down_variation
    jets_jagged = ak.unflatten(jets, counts)
    events.Jet = jets_jagged
    return events
