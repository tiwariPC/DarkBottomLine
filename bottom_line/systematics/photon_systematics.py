import numpy as np
import awkward as ak
import correctionlib
import os
import sys
from copy import deepcopy
import logging
from bottom_line.systematics.EGM_SS_systematics import EGM_Scale_Trad, EGM_Smearing_Trad, EGM_Scale_IJazZ, EGM_Smearing_IJazZ

logger = logging.getLogger(__name__)


# first dummy, keeping it at this point as reference for even simpler implementations
def photon_pt_scale_dummy(pt, **kwargs):
    return (1.0 + np.array([0.05, -0.05], dtype=np.float32)) * pt[:, None]


# Not nice but working: if the functions are called in the base processor by Photon.add_systematic(... "what"="pt"...), the pt is passed to the function as first argument.
# I need the full events here, so I pass in addition the events. Seems to only work if it is explicitly a function of pt, but I might be missing something. Open for better solutions.
def Scale_Trad(pt, events, year="2022postEE", is_correction=True, restriction=None):
    """
    Applies the photon pt scale corrections (use on data!) and corresponding uncertainties (on MC!).
    JSONs need to be pulled first with scripts/pull_files.py
    """

    return EGM_Scale_Trad(pt, events, year, is_correction, restriction, is_electron=False)


def Smearing_Trad(pt, events, year="2022postEE", is_correction=True):
    """
    Applies the photon smearing corrections and corresponding uncertainties (on MC!).
    JSON needs to be pulled first with scripts/pull_files.py
    """

    return EGM_Smearing_Trad(pt, events, year, is_correction, is_electron=False)


def Scale_IJazZ(pt, events, year="2022postEE", is_correction=True, gaussians="1G", restriction=None):
    """
    Applies the IJazZ photon pt scale corrections (use on data!) and corresponding uncertainties (on MC!).
    JSONs need to be pulled first with scripts/pull_files.py.
    The IJazZ corrections are independent and detached from the Egamma corrections.
    """

    return EGM_Scale_IJazZ(pt, events, year, is_correction, gaussians, restriction, is_electron=False)


def Smearing_IJazZ(pt, events, year="2022postEE", is_correction=True, gaussians="1G"):
    """
    Applies the photon smearing corrections and corresponding uncertainties (on MC!).
    JSON needs to be pulled first with scripts/pull_files.py
    """

    return EGM_Smearing_IJazZ(pt, events, year, is_correction, gaussians, is_electron=False)


def energyErrShift(energyErr, events, year="2022postEE", is_correction=True):
    # See also https://indico.cern.ch/event/1131803/contributions/4758593/attachments/2398621/4111806/Hgg_Differentials_Approval_080322.pdf#page=47
    # 2% with flows justified by https://indico.cern.ch/event/1495536/#20-study-of-the-sigma_mm-mismo
    if is_correction:
        return events
    else:
        _energyErr = ak.flatten(events.Photon.energyErr)
        uncertainty_up = np.ones(len(_energyErr)) * 1.02
        uncertainty_dn = np.ones(len(_energyErr)) * 0.98
        return (
            np.concatenate(
                (uncertainty_up.reshape(-1, 1), uncertainty_dn.reshape(-1, 1)), axis=1
            )
            * _energyErr[:, None]
        )


# Not nice but working: if the functions are called in the base processor by Photon.add_systematic(... "what"="pt"...), the pt is passed to the function as first argument.
# I need the full events here, so I pass in addition the events. Seems to only work if it is explicitly a function of pt, but I might be missing something. Open for better solutions.
def FNUF(pt, events, year="2017", is_correction=True):
    """
    ---This is an implementation of the FNUF uncertainty copied from flashgg,
    --- Preliminary JSON (run2 I don't know if this needs to be changed) file created with correctionlib starting from flashgg: https://github.com/cms-analysis/flashgg/blob/2677dfea2f0f40980993cade55144636656a8a4f/Systematics/python/flashggDiPhotonSystematics2017_Legacy_cfi.py
    Applies the photon pt and energy scale corrections and corresponding uncertainties.
    To be checked by experts
    """

    # for later unflattening:
    counts = ak.num(events.Photon.pt)
    eta = ak.flatten(events.Photon.ScEta)
    r9 = ak.flatten(events.Photon.r9)
    _energy = ak.flatten(events.Photon.energy)
    _pt = ak.flatten(events.Photon.pt)

    # era/year defined as parameter of the function
    avail_years = ["2016", "2016preVFP", "2016postVFP", "2017", "2018", "2022preEE", "2022postEE", "2023preBPix", "2023postBPix"]
    if year not in avail_years:
        logger.error(f"Only FNUF corrections for the year strings {avail_years} are already implemented! \n Exiting. \n")
        sys.exit(1)
    elif "2022" or "2023" in year:
        logger.warning(f"""You selected the year_string {year}, which is a 2022 era.
                        FNUF was not re-derived for Run 3 yet, but we fall back to the Run 2 2018 values.
                        These values only constitute up/down variations, no correction is applied.
                        The values are the averaged corrections from Run 2, turned into a systematic and inflated by 25%.
                        Please make sure that this is what you want. You have been warned.""")
        # The values have been provided by Badder for HIG-23-014 and Fabrice suggested to increase uncertainty a bit.
        year = "2022"
    elif "2016" in year:
        year = "2016"

    jsonpog_file = os.path.join(os.path.dirname(__file__), f"JSONs/FNUF/{year}/FNUF_{year}.json")
    evaluator = correctionlib.CorrectionSet.from_file(jsonpog_file)["FNUF"]

    if is_correction:
        correction = evaluator.evaluate("nominal", eta, r9)
        corr_energy = _energy * correction
        corr_pt = _pt * correction

        corrected_photons = deepcopy(events.Photon)
        corr_energy = ak.unflatten(corr_energy, counts)
        corr_pt = ak.unflatten(corr_pt, counts)
        corrected_photons["energy"] = corr_energy
        corrected_photons["pt"] = corr_pt
        events.Photon = corrected_photons

        return events

    else:
        correction = evaluator.evaluate("nominal", eta, r9)
        # When creating the JSON I already included added the variation to the returned value
        uncertainty_up = evaluator.evaluate("up", eta, r9) / correction
        uncertainty_dn = evaluator.evaluate("down", eta, r9) / correction
        # coffea does the unflattenning step itself and sets this value as pt of the up/down variations
        return (
            np.concatenate(
                (uncertainty_up.reshape(-1, 1), uncertainty_dn.reshape(-1, 1)), axis=1
            )
            * _pt[:, None]
        )


# Same old same old, just reiterated: if the functions are called in the base processor by Photon.add_systematic(... "what"="pt"...), the pt is passed to the function as first argument.
# Open for better solutions.
def ShowerShape(pt, events, year="2017", is_correction=True):
    """
    ---This is an implementation of the ShowerShape uncertainty copied from flashgg,
    --- Preliminary JSON (run2 I don't know if this needs to be changed) file created with correctionlib starting from flashgg: https://github.com/cms-analysis/flashgg/blob/2677dfea2f0f40980993cade55144636656a8a4f/Systematics/python/flashggDiPhotonSystematics2017_Legacy_cfi.py
    Applies the photon pt and energy scale corrections and corresponding uncertainties (only on the pt because it is what is used in selection).
    To be checked by experts
    """

    # for later unflattening:
    counts = ak.num(events.Photon.pt)
    eta = ak.flatten(events.Photon.ScEta)
    r9 = ak.flatten(events.Photon.r9)
    _energy = ak.flatten(events.Photon.energy)
    _pt = ak.flatten(events.Photon.pt)

    # era/year defined as parameter of the function
    avail_years = ["2016", "2016preVFP", "2016postVFP", "2017", "2018"]
    if year not in avail_years:
        logger.error(f"Only ShowerShape corrections for the year strings {avail_years} are already implemented! \n Exiting. \n")
        sys.exit(1)
    elif "2016" in year:
        year = "2016"

    jsonpog_file = os.path.join(os.path.dirname(__file__), f"JSONs/ShowerShape/{year}/ShowerShape_{year}.json")
    evaluator = correctionlib.CorrectionSet.from_file(jsonpog_file)["ShowerShape"]

    if is_correction:
        correction = evaluator.evaluate("nominal", eta, r9)
        corr_energy = _energy * correction
        corr_pt = _pt * correction

        corrected_photons = deepcopy(events.Photon)
        corr_energy = ak.unflatten(corr_energy, counts)
        corr_pt = ak.unflatten(corr_pt, counts)
        corrected_photons["energy"] = corr_energy
        corrected_photons["pt"] = corr_pt
        events.Photon = corrected_photons

        return events

    else:
        correction = evaluator.evaluate("nominal", eta, r9)
        # When creating the JSON I already included added the variation to the returned value
        uncertainty_up = evaluator.evaluate("up", eta, r9) / correction
        uncertainty_dn = evaluator.evaluate("down", eta, r9) / correction
        # coffea does the unflattenning step itself and sets this value as pt of the up/down variations
        return (
            np.concatenate(
                (uncertainty_up.reshape(-1, 1), uncertainty_dn.reshape(-1, 1)), axis=1
            )
            * _pt[:, None]
        )


def Material(pt, events, year="2017", is_correction=True):
    """
    ---This is an implementation of the Material uncertainty copied from flashgg,
    --- JSON file for run2 created with correctionlib starting from flashgg: https://github.com/cms-analysis/flashgg/blob/2677dfea2f0f40980993cade55144636656a8a4f/Systematics/python/flashggDiPhotonSystematics2017_Legacy_cfi.py
    Applies the photon pt and energy scale corrections and corresponding uncertainties.
    To be checked by experts
    """

    # for later unflattening:
    counts = ak.num(events.Photon.pt)
    eta = ak.flatten(abs(events.Photon.ScEta))
    r9 = ak.flatten(events.Photon.r9)
    _energy = ak.flatten(events.Photon.energy)
    _pt = ak.flatten(events.Photon.pt)

    # era/year defined as parameter of the function, only 2017 is implemented up to now
    avail_years = ["2016", "2016preVFP", "2016postVFP", "2017", "2018", "2022preEE", "2022postEE", "2023preBPix", "2023postBPix"]
    if year not in avail_years:
        logger.error(f"Only eVetoSF corrections for the year strings {avail_years} are already implemented! \n Exiting. \n")
        sys.exit(1)
    elif "2016" in year:
        year = "2016"
    # use Run 2 files also for Run 3, preliminary
    elif year in ["2022preEE", "2022postEE", "2023preBPix", "2023postBPix"]:
        logger.warning(f"""You selected the year_string {year}, which is a Run 3 era.
                  Material was not rederived for Run 3 yet, but we fall back to the Run 2 2018 values.
                  Please make sure that this is what you want. You have been warned.""")
        year = "2018"

    jsonpog_file = os.path.join(os.path.dirname(__file__), f"JSONs/Material/{year}/Material_{year}.json")
    evaluator = correctionlib.CorrectionSet.from_file(jsonpog_file)["Material"]

    if is_correction:
        correction = evaluator.evaluate("nominal", eta, r9)
        corr_energy = _energy * correction
        corr_pt = _pt * correction

        corrected_photons = deepcopy(events.Photon)
        corr_energy = ak.unflatten(corr_energy, counts)
        corr_pt = ak.unflatten(corr_pt, counts)
        corrected_photons["energy"] = corr_energy
        corrected_photons["pt"] = corr_pt
        events.Photon = corrected_photons

        return events

    else:
        correction = evaluator.evaluate("nominal", eta, r9)
        # When creating the JSON I already included added the variation to the returned value
        uncertainty_up = evaluator.evaluate("up", eta, r9) / correction
        uncertainty_dn = evaluator.evaluate("down", eta, r9) / correction
        # coffea does the unflattenning step itself and sets this value as pt of the up/down variations
        return (
            np.concatenate(
                (uncertainty_up.reshape(-1, 1), uncertainty_dn.reshape(-1, 1)), axis=1
            )
            * _pt[:, None]
        )
