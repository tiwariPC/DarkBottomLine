import numpy as np
import awkward as ak
import correctionlib
import os
import sys
from copy import deepcopy
import logging

logger = logging.getLogger(__name__)


# Not nice but working: if the functions are called in the base processor by Photon.add_systematic(... "what"="pt"...), the pt is passed to the function as first argument.
# I need the full events here, so I pass in addition the events. Seems to only work if it is explicitly a function of pt, but I might be missing something. Open for better solutions.
def EGM_Scale_Trad(pt, events, year="2022postEE", is_correction=True, restriction=None, is_electron=False):
    """
    Applies the photon pt scale corrections (use on data!) and corresponding uncertainties (on MC!).
    JSONs need to be pulled first with scripts/pull_files.py
    """
    if year in ["2016preVFP", "2016postVFP", "2017", "2018"]:
        if is_electron:
            logger.info("WARNING: there are only electrons scale corrections for the year strings [\"2022preEE\", \"2022postEE\", \"2023preBPix\", \"2023postBPix\"]! \n Exiting. \n")
            exit()

    elif year in ["2022preEE", "2022postEE", "2023preBPix", "2023postBPix"]:
        if is_electron:
            object_type = "_Electron"
            egm_object = events.Electron
        else:
            object_type = ""
            egm_object = events.Photon

    else:
        logger.error("The correction for the selected year is not implemented yet! Valid year tags are [\"2022preEE\", \"2022postEE\", \"2023preBPix\", \"2023postBPix\"] \n Exiting. \n")
        sys.exit(1)

    # for later unflattening:
    counts = ak.num(egm_object.pt)

    run = ak.flatten(ak.broadcast_arrays(events.run, egm_object.pt)[0])
    gain = ak.flatten(egm_object.seedGain)
    eta = ak.flatten(egm_object.ScEta)
    r9 = ak.flatten(egm_object.r9)
    _pt = ak.flatten(egm_object.pt)

    if year == "2016preVFP":
        path_json = os.path.join(os.path.dirname(__file__), 'JSONs/scaleAndSmearing/EGM_ScaleUnc_2016preVFP.json')
        evaluator = correctionlib.CorrectionSet.from_file(path_json)["UL-EGM_ScaleUnc"]
    elif year == "2016postVFP":
        path_json = os.path.join(os.path.dirname(__file__), 'JSONs/scaleAndSmearing/EGM_ScaleUnc_2016postVFP.json')
        evaluator = correctionlib.CorrectionSet.from_file(path_json)["UL-EGM_ScaleUnc"]
    elif year == "2017":
        path_json = os.path.join(os.path.dirname(__file__), 'JSONs/scaleAndSmearing/EGM_ScaleUnc_2017.json')
        evaluator = correctionlib.CorrectionSet.from_file(path_json)["UL-EGM_ScaleUnc"]
    elif year == "2018":
        path_json = os.path.join(os.path.dirname(__file__), 'JSONs/scaleAndSmearing/EGM_ScaleUnc_2018.json')
        evaluator = correctionlib.CorrectionSet.from_file(path_json)["UL-EGM_ScaleUnc"]
    elif year == "2022preEE":
        path_json = os.path.join(os.path.dirname(__file__), f'JSONs/scaleAndSmearing/SS{object_type}_Rereco2022BCD.json')
        evaluator = correctionlib.CorrectionSet.from_file(path_json)["2022Re-recoBCD_ScaleJSON"]
    elif year == "2022postEE":
        path_json = os.path.join(os.path.dirname(__file__), f'JSONs/scaleAndSmearing/SS{object_type}_RerecoE_PromptFG_2022.json')
        evaluator = correctionlib.CorrectionSet.from_file(path_json)["2022Re-recoE+PromptFG_ScaleJSON"]
    elif year == "2023preBPix":
        path_json = os.path.join(os.path.dirname(__file__), f'JSONs/scaleAndSmearing/SS{object_type}_Prompt23C.json')
        evaluator = correctionlib.CorrectionSet.from_file(path_json)["2023PromptC_ScaleJSON"]
    elif year == "2023postBPix":
        path_json = os.path.join(os.path.dirname(__file__), f'JSONs/scaleAndSmearing/SS{object_type}_Prompt23D.json')
        evaluator = correctionlib.CorrectionSet.from_file(path_json)["2023PromptD_ScaleJSON"]
    else:
        logger.error("There are only scale corrections for the year strings [\"2016preVFP\", \"2016postVFP\", \"2017\", \"2018\", \"2022preEE\", \"2022postEE\", \"2023preBPix\", \"2023postBPix\"]! \n Exiting. \n")
        sys.exit(1)

    if is_correction:
        # scale is a residual correction on data to match MC calibration. Check if is MC, throw error in this case.
        if hasattr(events, "GenPart"):
            raise ValueError("Scale corrections should only be applied to data!")

        if year in ["2016preVFP", "2016postVFP", "2017", "2018"]:
            # the correction is already applied for Run 2
            logger.info("the scale correction for Run 2  MC is already applied in nAOD, nothing to be done")
        else:
            correction = evaluator.evaluate("total_correction", gain, run, eta, r9, _pt)
            pt_corr = _pt * correction

            corrected_egm_object = deepcopy(egm_object)
            pt_corr = ak.unflatten(pt_corr, counts)
            corrected_egm_object["pt"] = pt_corr

            if is_electron:
                events.Electron = corrected_egm_object
            else:
                events.Photon = corrected_egm_object

        return events

    else:
        if not hasattr(events, "GenPart"):
            raise ValueError("Scale uncertainties should only be applied to MC!")

        if year in ["2016preVFP", "2016postVFP", "2017", "2018"]:
            # the uncertainty is applied in reverse because the correction is meant for data as I understand fro EGM instructions here: https://cms-talk.web.cern.ch/t/pnoton-energy-corrections-in-nanoaod-v11/34327/2
            uncertainty_up = evaluator.evaluate(year, "scaledown", eta, gain)
            uncertainty_down = evaluator.evaluate(year, "scaleup", eta, gain)

            corr_up_variation = uncertainty_up
            corr_down_variation = uncertainty_down

        else:
            correction = evaluator.evaluate("total_correction", gain, run, eta, r9, _pt)
            uncertainty = evaluator.evaluate("total_uncertainty", gain, run, eta, r9, _pt)

            if restriction is not None:
                if restriction == "EB":
                    uncMask = ak.to_numpy(ak.flatten(egm_object.isScEtaEB))

                elif restriction == "EE":
                    uncMask = ak.to_numpy(ak.flatten(egm_object.isScEtaEE))
                    if year == "2022preEE":
                        rescaleFactor = 1.5
                        logger.info(f"Increasing EB scale uncertainty by factor {rescaleFactor}.")
                        uncertainty *= rescaleFactor
                    elif year == "2022postEE":
                        rescaleFactor = 2.
                        logger.info(f"Increasing EE scale uncertainty by factor {rescaleFactor}.")
                        uncertainty *= rescaleFactor

                uncertainty = np.where(
                    uncMask, uncertainty, np.zeros_like(uncertainty)
                )

            # divide by correction since it is already applied before
            corr_up_variation = (correction + uncertainty) / correction
            corr_down_variation = (correction - uncertainty) / correction

        # coffea does the unflattenning step itself and sets this value as pt of the up/down variations
        return np.concatenate((corr_up_variation.reshape(-1,1), corr_down_variation.reshape(-1,1)), axis=1) * _pt[:, None]


def EGM_Smearing_Trad(pt, events, year="2022postEE", is_correction=True, is_electron=False):
    """
    Applies the photon smearing corrections and corresponding uncertainties (on MC!).
    JSON needs to be pulled first with scripts/pull_files.py
    """
    if year in ["2016preVFP", "2016postVFP", "2017", "2018"]:
        if is_electron:
            logger.info("WARNING: there are only electrons smearing corrections for the year strings [\"2022preEE\", \"2022postEE\", \"2023preBPix\", \"2023postBPix\"]! \n Exiting. \n")
            exit()

    elif year in ["2022preEE", "2022postEE", "2023preBPix", "2023postBPix"]:
        if is_electron:
            object_type = "_Electron"
            egm_object = events.Electron
        else:
            object_type = ""
            egm_object = events.Photon

    else:
        logger.error("The correction for the selected year is not implemented yet! Valid year tags are [\"2022preEE\", \"2022postEE\", \"2023preBPix\", \"2023postBPix\"] \n Exiting. \n")
        sys.exit(1)

    # for later unflattening:
    counts = ak.num(egm_object.pt)

    eta = ak.flatten(egm_object.ScEta)
    r9 = ak.flatten(egm_object.r9)
    _pt = ak.flatten(egm_object.pt)

    # we need reproducible random numbers since in the systematics call, the previous correction needs to be cancelled out
    rng = np.random.default_rng(seed=125)

    if year == "2022preEE":
        path_json = os.path.join(os.path.dirname(__file__), f'JSONs/scaleAndSmearing/SS{object_type}_Rereco2022BCD.json')
        evaluator = correctionlib.CorrectionSet.from_file(path_json)["2022Re-recoBCD_SmearingJSON"]
    elif year == "2022postEE":
        path_json = os.path.join(os.path.dirname(__file__), f'JSONs/scaleAndSmearing/SS{object_type}_RerecoE_PromptFG_2022.json')
        evaluator = correctionlib.CorrectionSet.from_file(path_json)["2022Re-recoE+PromptFG_SmearingJSON"]
    elif year == "2023preBPix":
        path_json = os.path.join(os.path.dirname(__file__), f'JSONs/scaleAndSmearing/SS{object_type}_Prompt23C.json')
        evaluator = correctionlib.CorrectionSet.from_file(path_json)["2023PromptC_SmearingJSON"]
    elif year == "2023postBPix":
        path_json = os.path.join(os.path.dirname(__file__), f'JSONs/scaleAndSmearing/SS{object_type}_Prompt23D.json')
        evaluator = correctionlib.CorrectionSet.from_file(path_json)["2023PromptD_SmearingJSON"]
    elif year in ["2016preVFP", "2016postVFP", "2017", "2018"]:
        logger.info("the systematic variations are taken directly from the dedicated nAOD branches Photon.dEsigmaUp and Photon.dEsigmaDown")
    else:
        logger.error("The correction for the selected year is not implemented yet! Valid year tags are [\"2016preVFP\", \"2016postVFP\", \"2017\", \"2018\", \"2022preEE\", \"2022postEE\", \"2023preBPix\", \"2023postBPix\"] \n Exiting. \n")
        sys.exit(1)

    if is_correction:

        if year in ["2016", "2016preVFP", "2016postVFP", "2017", "2018"]:
            logger.info("the smearing correction for Run 2 MC is already applied in nAOD")
        else:
            # In theory, the energy should be smeared and not the pT, see: https://mattermost.web.cern.ch/cmseg/channels/egm-ss/6mmucnn8rjdgt8x9k5zaxbzqyh
            # However, there is a linear proportionality between pT and E: E = pT * cosh(eta)
            # Because of that, applying the correction to pT and E is equivalent (since eta does not change)
            # Energy is provided as a LorentzVector mixin, so we choose to correct pT
            # Also holds true for the scale part
            rho = evaluator.evaluate("rho", eta, r9)
            smearing = rng.normal(loc=1., scale=rho)
            pt_corr = _pt * smearing
            corrected_egm_object = deepcopy(egm_object)
            pt_corr = ak.unflatten(pt_corr, counts)
            rho_corr = ak.unflatten(rho, counts)

            # If it is data, dont perform the pt smearing, only save the std of the gaussian for each event!
            try:
                events.GenIsolatedPhoton  # this operation is here because if there is no "events.GenIsolatedPhoton" field on data, an error will be thrown and we go to the except - so we dont smear the data pt spectrum
                corrected_egm_object["pt"] = pt_corr
            except:
                pass

            corrected_egm_object["rho_smear"] = rho_corr

            if is_electron:
                events.Electron = corrected_egm_object
            else:
                events.Photon = corrected_egm_object
        return events

    else:

        if year in ["2016", "2016preVFP", "2016postVFP", "2017", "2018"]:
            # the correction is already applied for Run 2
            dEsigmaUp = ak.flatten(egm_object.dEsigmaUp)
            dEsigmaDown = ak.flatten(egm_object.dEsigmaDown)
            logger.info(f"{dEsigmaUp}, {egm_object.dEsigmaUp}")

            # the correction is given as additive factor for the Energy (Et_smear_up = Et + abs(dEsigmaUp)) so we have to convert it before applying it to the Pt
            # for EGM instruction on how to calculate uncertainties I link to this CMSTalk post https://cms-talk.web.cern.ch/t/pnoton-energy-corrections-in-nanoaod-v11/34327/2
            smearing_up = (_pt + abs(dEsigmaUp) / np.cosh(eta))
            smearing_down = (_pt - abs(dEsigmaDown) / np.cosh(eta))

            # divide by correction since it is already applied before we also divide for the Pt because it is later multipled when passing it to coffea, to be compatible with Run 3 calculation from json.
            # I convert it to numpy because ak.Arrays don't have the .reshape method needed further on.
            corr_up_variation = (smearing_up / _pt).to_numpy()
            corr_down_variation = (smearing_down / _pt).to_numpy()

        else:
            rho = evaluator.evaluate("rho", eta, r9)
            # produce the same numbers as in correction step
            smearing = rng.normal(loc=1., scale=rho)

            err_rho = evaluator.evaluate("err_rho", eta, r9)
            rho_up = rho + err_rho
            rho_down = rho - err_rho
            smearing_up = rng.normal(loc=1., scale=rho_up)
            smearing_down = rng.normal(loc=1., scale=rho_down)

            # divide by correction since it is already applied before
            corr_up_variation = smearing_up / smearing
            corr_down_variation = smearing_down / smearing

        # coffea does the unflattenning step itself and sets this value as pt of the up/down variations
        return np.concatenate((corr_up_variation.reshape(-1,1), corr_down_variation.reshape(-1,1)), axis=1) * _pt[:, None]


def EGM_Scale_IJazZ(pt, events, year="2022postEE", is_correction=True, gaussians="1G", restriction=None, is_electron=False):
    """
    Applies the IJazZ photon pt scale corrections (use on data!) and corresponding uncertainties (on MC!).
    JSONs need to be pulled first with scripts/pull_files.py.
    The IJazZ corrections are independent and detached from the Egamma corrections.
    """
    if is_electron:
        object_type = "Ele"
        egm_object = events.Electron
    else:
        object_type = "Pho"
        egm_object = events.Photon

    # for later unflattening:
    counts = ak.num(egm_object.pt)

    run = ak.flatten(ak.broadcast_arrays(events.run, egm_object.pt)[0])
    gain = ak.flatten(egm_object.seedGain)
    eta = ak.flatten(egm_object.ScEta)
    AbsScEta = abs(eta)
    r9 = ak.flatten(egm_object.r9)
    # scale uncertainties are applied on the smeared pt but computed from the raw pt
    pt_raw = ak.flatten(egm_object.pt_raw)
    _pt = ak.flatten(egm_object.pt)

    if gaussians == "1G":
        gaussian_postfix = ""
    elif gaussians == "2G":
        gaussian_postfix = "2G"
    else:
        logger.error("The selected number of gaussians is not implemented yet! Valid options are [\"1G\", \"2G\"] \n Exiting. \n")
        sys.exit(1)

    valid_years_paths = {
        "2022preEE": f"EGMScalesSmearing_{object_type}_2022preEE",
        "2022postEE": f"EGMScalesSmearing_{object_type}_2022postEE",
        "2023preBPix": f"EGMScalesSmearing_{object_type}_2023preBPIX",
        "2023postBPix": f"EGMScalesSmearing_{object_type}_2023postBPIX"
    }

    ending = ".v1.json"

    if year not in valid_years_paths:
        logger.error("The correction for the selected year is not implemented yet! Valid year tags are [\"2022preEE\", \"2022postEE\", \"2023preBPix\", \"2023postBPix\"] \n Exiting. \n")
        sys.exit(1)

    else:
        path_json = os.path.join(os.path.dirname(__file__), 'JSONs/scaleAndSmearing', valid_years_paths[year] + gaussian_postfix + ending)
        try:
            cset = correctionlib.CorrectionSet.from_file(path_json)
        except:
            logger.error(f"WARNING: the JSON file {path_json} could not be found! \n Check if the file has been pulled \n pull_files.py -t SS-IJazZ \n")
            sys.exit(1)
        # Convention of Fabrice and Paul: Capitalise IX (for some reason)
        if "BPix" in year:
            year = year.replace("BPix", "BPIX")
        scale_evaluator = cset.compound[f"EGMScale_Compound_{object_type}_{year}{gaussian_postfix}"]
        smear_and_syst_evaluator = cset[f"EGMSmearAndSyst_{object_type}PTsplit_{year}{gaussian_postfix}"]

    if is_correction:
        # scale is a residual correction on data to match MC calibration. Check if is MC, throw error in this case.
        if hasattr(events, "GenPart"):
            raise ValueError("Scale corrections should only be applied to data!")

        correction = scale_evaluator.evaluate("scale", run, eta, r9, AbsScEta, pt_raw, gain)
        pt_corr = pt_raw * correction
        corrected_egm_object = deepcopy(egm_object)
        pt_corr = ak.unflatten(pt_corr, counts)
        corrected_egm_object["pt"] = pt_corr

        if is_electron:
            events.Electron = corrected_egm_object
        else:
            events.Photon = corrected_egm_object
        return events

    else:
        # Note the conventions in the JSON, both `scale_up`/`scale_down` and `escale` are available.
        # scale_up = 1 + escale
        if not hasattr(events, "GenPart"):
            raise ValueError("Scale uncertainties should only be applied to MC!")

        corr_up_variation = smear_and_syst_evaluator.evaluate('scale_up', pt_raw, r9, AbsScEta)
        corr_down_variation = smear_and_syst_evaluator.evaluate('scale_down', pt_raw, r9, AbsScEta)

        if restriction == "EB":
            corr_up_variation[ak.to_numpy(ak.flatten(egm_object.isScEtaEE))] = 1.
            corr_up_variation[ak.to_numpy(ak.flatten(egm_object.isScEtaEE))] = 1.
        elif restriction == "EE":
            corr_up_variation[ak.to_numpy(ak.flatten(egm_object.isScEtaEB))] = 1.
            corr_up_variation[ak.to_numpy(ak.flatten(egm_object.isScEtaEB))] = 1.
        elif restriction is not None:
            logger.error("The restriction is not implemented yet! Valid options are [\"EB\", \"EE\"] \n Exiting. \n")
            sys.exit(1)

        # Coffea does the unflattenning step itself and sets this value as pt of the up/down variations
        # scale uncertainties are applied on the smeared pt
        return np.concatenate((corr_up_variation.reshape(-1,1), corr_down_variation.reshape(-1,1)), axis=1) * _pt[:, None]


def double_smearing(std_normal, std_flat, mu, sigma1, sigma2, frac):
    # compute the two smearing scales from the gaussian draws
    scales = np.array([1 + sigma1 * std_normal, mu * (1 + sigma2 * std_normal)])
    # select the gaussian based on the relative fraction and the flat draw
    binom = (std_flat > frac).astype(int)
    return scales[binom, np.arange(len(mu))]


def EGM_Smearing_IJazZ(pt, events, year="2022postEE", is_correction=True, gaussians="1G", is_electron=False):
    """
    Applies the photon smearing corrections and corresponding uncertainties (on MC!).
    JSON needs to be pulled first with scripts/pull_files.py
    """

    if is_electron:
        object_type = "Ele"
        egm_object = events.Electron
    else:
        object_type = "Pho"
        egm_object = events.Photon

    # for later unflattening:
    counts = ak.num(egm_object.pt)

    eta = ak.flatten(egm_object.ScEta)
    AbsScEta = abs(eta)
    r9 = ak.flatten(egm_object.r9)
    pt_raw = ak.flatten(egm_object.pt_raw)
    # Need some broadcasting to make the event numbers match
    event_number = ak.flatten(ak.broadcast_arrays(events.event, egm_object.pt)[0])

    if gaussians == "1G":
        gaussian_postfix = ""
    elif gaussians == "2G":
        gaussian_postfix = "2G"
    else:
        logger.error("The selected number of gaussians is not implemented yet! Valid options are [\"1G\", \"2G\"] \n Exiting. \n")
        sys.exit(1)

    valid_years_paths = {
        "2022preEE": f"EGMScalesSmearing_{object_type}_2022preEE",
        "2022postEE": f"EGMScalesSmearing_{object_type}_2022postEE",
        "2023preBPix": f"EGMScalesSmearing_{object_type}_2023preBPIX",
        "2023postBPix": f"EGMScalesSmearing_{object_type}_2023postBPIX"
    }

    ending = ".v1.json"

    if year not in valid_years_paths:
        logger.error("The correction for the selected year is not implemented yet! Valid year tags are [\"2022preEE\", \"2022postEE\", \"2023preBPix\", \"2023postBPix\"] \n Exiting. \n")
        sys.exit(1)

    else:
        path_json = os.path.join(os.path.dirname(__file__), 'JSONs/scaleAndSmearing', valid_years_paths[year] + gaussian_postfix + ending)
        try:
            cset = correctionlib.CorrectionSet.from_file(path_json)
        except:
            logger.error(f"The JSON file {path_json} could not be found! \n Check if the file has been pulled \n pull_files.py -t SS-IJazZ \n")
            sys.exit(1)
        # Convention of Fabrice and Paul: Capitalise IX (for some reason)
        if "BPix" in year:
            year_ = year.replace("BPix", "BPIX")
        else:
            year_ = year
        smear_and_syst_evaluator = cset[f"EGMSmearAndSyst_{object_type}PTsplit_{year_}{gaussian_postfix}"]
        random_generator = cset['EGMRandomGenerator']

    # In theory, the energy should be smeared and not the pT, see: https://mattermost.web.cern.ch/cmseg/channels/egm-ss/6mmucnn8rjdgt8x9k5zaxbzqyh
    # However, there is a linear proportionality between pT and E: E = pT * cosh(eta)
    # Because of that, applying the correction to pT and E is equivalent (since eta does not change)
    # Energy is provided as a LorentzVector mixin, so we choose to correct pT
    # Also holds true for the scale part

    # Calculate upfront since it is needed for both correction and uncertainty
    smearing = smear_and_syst_evaluator.evaluate('smear', pt_raw, r9, AbsScEta)
    random_numbers = random_generator.evaluate('stdnormal', pt_raw, r9, AbsScEta, event_number)

    if gaussians == "1G":
        correction = (1 + smearing * random_numbers)
    # Else can only be "2G" due to the checks above
    # Have to use else here to satisfy that correction is always defined in all possible branches of the code
    else:
        correction = double_smearing(
            random_numbers,
            random_generator.evaluate('stdflat', pt_raw, r9, AbsScEta, event_number),
            smear_and_syst_evaluator.evaluate('mu', pt_raw, r9, AbsScEta),
            smearing,
            smear_and_syst_evaluator.evaluate('reso2', pt_raw, r9, AbsScEta),
            smear_and_syst_evaluator.evaluate('frac', pt_raw, r9, AbsScEta)
        )

    if is_correction:
        pt_corr = pt_raw * correction
        corrected_egm_object = deepcopy(egm_object)
        pt_corr = ak.unflatten(pt_corr, counts)
        # For the 2G case, also take the rho_corr from the 1G case as advised by Fabrice
        # Otherwise, the sigma_m/m will be lower on average, new CDFs will be needed etc. not worth the hassle
        if gaussians == "2G":
            path_json = os.path.join(os.path.dirname(__file__), 'JSONs/scaleAndSmearing', valid_years_paths[year] + ending)
            try:
                cset = correctionlib.CorrectionSet.from_file(path_json)
            except:
                logger.error(f"The JSON file {path_json} could not be found! \n Check if the file has been pulled \n pull_files.py -t SS-IJazZ \n")
                sys.exit(1)
            if "BPix" in year:
                year = year.replace("BPix", "BPIX")
            smear_and_syst_evaluator_for_rho_corr = cset[f"EGMSmearAndSyst_{object_type}PTsplit_{year}"]
            rho_corr = ak.unflatten(smear_and_syst_evaluator_for_rho_corr.evaluate('smear', pt_raw, r9, AbsScEta), counts)
        else:
            rho_corr = ak.unflatten(smearing, counts)

        # If it is data, dont perform the pt smearing, only save the std of the gaussian for each event!
        try:
            events.GenIsolatedPhoton  # this operation is here because if there is no "events.GenIsolatedPhoton" field on data, an error will be thrown and we go to the except - so we dont smear the data pt spectrum
            corrected_egm_object["pt"] = pt_corr
        except:
            pass

        corrected_egm_object["rho_smear"] = rho_corr

        if is_electron:
            events.Electron = corrected_egm_object
        else:
            events.Photon = corrected_egm_object

        return events

    else:
        # Note the conventions in the JSON, both `smear_up`/`smear_down` and `esmear` are available.
        # smear_up = smear + esmear
        if gaussians == "1G":
            corr_up_variation = 1 + smear_and_syst_evaluator.evaluate('smear_up', pt_raw, r9, AbsScEta) * random_numbers
            corr_down_variation = 1 + smear_and_syst_evaluator.evaluate('smear_down', pt_raw, r9, AbsScEta) * random_numbers

        else:
            corr_up_variation = double_smearing(
                random_numbers,
                random_generator.evaluate('stdflat', pt_raw, r9, AbsScEta, event_number),
                smear_and_syst_evaluator.evaluate('mu', pt_raw, r9, AbsScEta),
                smear_and_syst_evaluator.evaluate('smear_up', pt_raw, r9, AbsScEta),
                smear_and_syst_evaluator.evaluate('reso2', pt_raw, r9, AbsScEta),
                smear_and_syst_evaluator.evaluate('frac', pt_raw, r9, AbsScEta)
            )

            corr_down_variation = double_smearing(
                random_numbers,
                random_generator.evaluate('stdflat', pt_raw, r9, AbsScEta, event_number),
                smear_and_syst_evaluator.evaluate('mu', pt_raw, r9, AbsScEta),
                smear_and_syst_evaluator.evaluate('smear_down', pt_raw, r9, AbsScEta),
                smear_and_syst_evaluator.evaluate('reso2', pt_raw, r9, AbsScEta),
                smear_and_syst_evaluator.evaluate('frac', pt_raw, r9, AbsScEta)
            )

        # coffea does the unflattenning step itself and sets this value as pt of the up/down variations
        # smearing uncertainties are applied on the raw pt because the smearing is redone from scratch
        return np.concatenate((corr_up_variation.reshape(-1,1), corr_down_variation.reshape(-1,1)), axis=1) * pt_raw[:, None]
