# Ref: https://gitlab.cern.ch/cms-muonPOG/muonscarekit/-/blob/master/scripts/MuonScaRe.py

import numpy as np
import awkward as ak
import correctionlib
import os
from copy import deepcopy
from bottom_line.tools.doublecrystalball import doublecrystalball
import logging

logger = logging.getLogger(__name__)


def get_rndm(eta, nL, cset):
    # obtain parameters from correctionlib
    eta_f, nL_f = eta, nL

    mean_f = cset.get("cb_params").evaluate(abs(eta_f), nL_f, 0)
    sigma_f = cset.get("cb_params").evaluate(abs(eta_f), nL_f, 1)
    n_f = cset.get("cb_params").evaluate(abs(eta_f), nL_f, 2)
    alpha_f = cset.get("cb_params").evaluate(abs(eta_f), nL_f, 3)

    # get random number following the CB
    # we need reproducible random numbers since in the systematics call, the previous correction needs to be cancelled out
    rng = np.random.default_rng(seed=125)
    rndm_f = rng.random(len(eta))

    dcb_f = doublecrystalball(alpha_f, alpha_f, n_f, n_f, mean_f, sigma_f)

    return dcb_f.ppf(rndm_f)


def get_std(pt, eta, nL, cset):
    eta_f, nL_f, pt_f = eta, nL, pt

    # obtain parameters from correctionlib
    param0_f = cset.get("poly_params").evaluate(abs(eta_f), nL_f, 0)
    param1_f = cset.get("poly_params").evaluate(abs(eta_f), nL_f, 1)
    param2_f = cset.get("poly_params").evaluate(abs(eta_f), nL_f, 2)

    # calculate value and return max(0, val)
    sigma_f = param0_f + param1_f * pt_f + param2_f * pt_f * pt_f

    return ak.where(sigma_f < 0, 0, sigma_f)


def get_k(eta, var, cset):
    # obtain parameters from correctionlib
    k_data_f = cset.get("k_data").evaluate(abs(eta), var)
    k_mc_f = cset.get("k_mc").evaluate(abs(eta), var)

    # obtain parameters from correctionlib
    k_data_f = cset.get("k_data").evaluate(abs(eta), var)
    k_mc_f = cset.get("k_mc").evaluate(abs(eta), var)

    # calculate residual smearing factor
    # return 0 if smearing in MC already larger than in data
    k_f = ak.where(k_mc_f < k_data_f, (k_data_f**2 - k_mc_f**2) ** 0.5, 0)

    return k_f


def filter_boundaries(pt_corr, pt):
    # Check for pt values outside the range of [26, 200]
    outside_bounds = (pt < 26) | (pt > 200)

    n_pt_outside = ak.sum(outside_bounds)

    if n_pt_outside > 0:
        logger.debug(
            f"[ Muon S&S ] There are {n_pt_outside} events with muon pt outside of [26,200] GeV. Setting those entries to their initial value."
        )
        pt_corr = ak.where(pt > 200, pt, pt_corr)
        pt_corr = ak.where(pt < 26, pt, pt_corr)

    # Check for NaN entries in pt_corr
    nan_entries = np.isnan(pt_corr)

    n_nan = ak.sum(nan_entries)

    if n_nan > 0:
        logger.debug(
            f"[ Muon S&S ] There are {n_nan} nan entries in the corrected pt. "
            "This might be due to the number of tracker layers hitting boundaries. "
            "Setting those entries to their initial value."
        )
        pt_corr = ak.where(np.isnan(pt_corr), pt, pt_corr)

    return pt_corr


def pt_resol(pt, eta, nL, cset):
    """ "
    Function for the calculation of the resolution correction
    Input:
    pt - muon transverse momentum
    eta - muon pseudorapidity
    nL - muon number of tracker layers
    cset - correctionlib object

    This function should only be applied to reco muons in MC!
    """
    rndm = get_rndm(eta, nL, cset)
    std = get_std(pt, eta, nL, cset)
    k = get_k(eta, "nom", cset)

    pt_corr = pt * (1 + k * std * rndm)

    pt_corr = filter_boundaries(pt_corr, pt)

    return pt_corr


def pt_resol_var(pt_woresol, pt_wresol, eta, updn, cset):
    """
    Function for the calculation of the resolution uncertainty
    Input:
    pt_woresol - muon transverse momentum without resolution correction
    pt_wresol - muon transverse momentum with resolution correction
    eta - muon pseudorapidity
    updn - uncertainty variation (up or dn)
    cset - correctionlib object

    This function should only be applied to reco muons in MC!
    """

    pt_wresol_f, pt_woresol_f = pt_wresol, pt_woresol

    k_unc_f = cset.get("k_mc").evaluate(abs(eta), "stat")
    k_f = cset.get("k_mc").evaluate(abs(eta), "nom")

    pt_var_f = pt_wresol_f

    # Define condition and standard correction
    condition = k_f > 0
    std_x_cb = (pt_wresol_f / pt_woresol_f - 1) / k_f

    # Apply up or down variation using ak.where
    if updn == "up":
        pt_var_f = ak.where(
            condition,
            pt_woresol_f * (1 + (k_f + k_unc_f) * std_x_cb),
            pt_var_f,
        )
    elif updn == "dn":
        pt_var_f = ak.where(
            condition,
            pt_woresol_f * (1 + (k_f - k_unc_f) * std_x_cb),
            pt_var_f,
        )
    else:
        logger.info("[ Muon Scale ] ERROR: updn must be 'up' or 'dn'")

    return pt_var_f


def pt_scale(is_data, pt, eta, phi, charge, cset):
    """
    Function for the calculation of the scale correction
    Input:
    is_data - flag that is True if dealing with data and False if MC
    pt - muon transverse momentum
    eta - muon pseudorapidity
    phi - muon angle
    charge - muon charge
    var - variation (standard is "nom")
    cset - correctionlib object

    This function should be applied to reco muons in data and MC
    """
    if is_data:
        dtmc = "data"
    else:
        dtmc = "mc"

    a_f = cset.get("a_" + dtmc).evaluate(eta, phi, "nom")
    m_f = cset.get("m_" + dtmc).evaluate(eta, phi, "nom")

    pt_corr = 1.0 / (m_f / pt + charge * a_f)

    pt_corr = filter_boundaries(pt_corr, pt)

    return pt_corr


def pt_scale_var(pt, eta, phi, charge, updn, cset):
    """
    Function for the calculation of the scale uncertainty
    Input:
    pt - muon transverse momentum
    eta - muon pseudorapidity
    phi - muon angle
    charge - muon charge
    updn - uncertainty variation (up or dn)
    cset - correctionlib object

    This function should be applied to reco muons in MC!
    """

    stat_a_f = cset.get("a_mc").evaluate(eta, phi, "stat")
    stat_m_f = cset.get("m_mc").evaluate(eta, phi, "stat")
    stat_rho_f = cset.get("m_mc").evaluate(eta, phi, "rho_stat")

    unc = (
        pt
        * pt
        * (
            stat_m_f * stat_m_f / (pt * pt)
            + stat_a_f * stat_a_f
            + 2 * charge * stat_rho_f * stat_m_f / pt * stat_a_f
        )
        ** 0.5
    )

    pt_var = pt

    if updn == "up":
        pt_var = pt_var + unc
    elif updn == "dn":
        pt_var = pt_var - unc

    return pt_var


# Reference: https://gitlab.cern.ch/cms-muonPOG/muonscarekit
def muon_pt_scare(pt, events, year="2022postEE", unc_type=None, is_correction=True):
    """
    Applies the photon pt scale corrections (use on data!) and corresponding uncertainties (on MC!).
    JSONs need to be pulled first with scripts/pull_files.py
    """

    # for later unflattening:
    counts = ak.num(events.Muon.pt)
    # decide if the process data
    is_data = False if hasattr(events, "genWeight") else True

    muons_jagged = deepcopy(events.Muon)
    muons = ak.flatten(muons_jagged)

    if year == "2022preEE":
        path_json = os.path.join(
            os.path.dirname(__file__), "JSONs/MuonScaRe/2022_Summer22.json"
        )
    elif year == "2022postEE":
        path_json = os.path.join(
            os.path.dirname(__file__), "JSONs/MuonScaRe/2022_Summer22EE.json"
        )
    elif year == "2023preBPix":
        path_json = os.path.join(
            os.path.dirname(__file__), "JSONs/MuonScaRe/2023_Summer23.json"
        )
    elif year == "2023postBPix":
        path_json = os.path.join(
            os.path.dirname(__file__), "JSONs/MuonScaRe/2023_Summer23BPix.json"
        )
    else:
        logger.info(
            'WARNING: there are only scale corrections for the year strings ["2022preEE", "2022postEE", "2023preBPix", "2023postBPix"]! \n Exiting. \n'
        )
        exit()

    evaluator = correctionlib.CorrectionSet.from_file(path_json)

    if is_correction:
        muons["pt_nanoaod"] = muons.pt
        # * Data only need scale correction
        muons_pt_scalecorr = pt_scale(
            1 if is_data else 0,  # 1 for data, 0 for mc
            muons.pt,
            muons.eta,
            muons.phi,
            muons.charge,
            evaluator,
        )

        muons["pt_scale_factor"] = muons_pt_scalecorr / muons.pt_nanoaod
        muons["pt_scalecorr"] = muons_pt_scalecorr
        logger.debug("[ Muon Scale ] Muon pt scale correction applied")

        if is_data:
            logger.debug("[ Muon Scale ] Data only need muon pt scale correction")
            muons["pt"] = muons["pt_scalecorr"]
            muons_jagged = ak.unflatten(muons, counts)
            events.Muon = muons_jagged
            return events
        else:
            # * MC needs both scale and resolution corrections
            muons_pt_scarecorr = pt_resol(
                muons.pt_scalecorr, muons.eta, muons.nTrackerLayers, evaluator
            )
            muons["pt_scare_factor"] = muons_pt_scarecorr / muons.pt_nanoaod
            muons["pt_scarecorr"] = muons_pt_scarecorr
            logger.debug(
                "[ Muon SCARE ] MC need both pt scale and resolution corrections"
            )

            muons["pt"] = muons["pt_scarecorr"]
            muons_jagged = ak.unflatten(muons, counts)
            events.Muon = muons_jagged
            return events
    else:
        if not hasattr(events, "genWeight"):
            raise ValueError("Scale uncertainties should only be applied to MC!")

        if unc_type:
            if unc_type == "Scale":
                if not hasattr(muons, "pt_scalecorr"):
                    logger.info(
                        "[ Muon Scale ] WARNING: muons.pt_scalecorr is not defined! \n Exiting. \n"
                    )
                    exit()
                muons_pt_scalecorr_up = pt_scale_var(
                    muons.pt_scarecorr,
                    muons.eta,
                    muons.phi,
                    muons.charge,
                    "up",
                    evaluator,
                )

                muons_pt_scalecorr_down = pt_scale_var(
                    muons.pt_scarecorr,
                    muons.eta,
                    muons.phi,
                    muons.charge,
                    "dn",
                    evaluator,
                )
                # coffea does the unflattenning step itself and sets this value as pt of the up/down variations
                return np.concatenate(
                    (
                        (muons_pt_scalecorr_up.to_numpy()).reshape(-1, 1),
                        (muons_pt_scalecorr_down.to_numpy()).reshape(-1, 1),
                    ),
                    axis=1,
                ) * (ak.ones_like(muons.pt_nanoaod)[:, None])

            elif unc_type == "Resolution":
                if not hasattr(muons, "pt_scalecorr") or not hasattr(
                    muons, "pt_scarecorr"
                ):
                    logger.info(
                        "[ Muon S&S ] WARNING: muons.pt_scalecorr or muons.pt_scarecorr is not defined! \n Exiting. \n"
                    )
                    exit()
                muons_pt_rescorr_up = pt_resol_var(
                    muons.pt_scalecorr, muons.pt_scarecorr, muons.eta, "up", evaluator
                )
                muons_pt_rescorr_down = pt_resol_var(
                    muons.pt_scalecorr, muons.pt_scarecorr, muons.eta, "dn", evaluator
                )
                # coffea does the unflattenning step itself and sets this value as pt of the up/down variations
                return np.concatenate(
                    (
                        (muons_pt_rescorr_up.to_numpy()).reshape(-1, 1),
                        (muons_pt_rescorr_down.to_numpy()).reshape(-1, 1),
                    ),
                    axis=1,
                ) * (ak.ones_like(muons.pt_nanoaod)[:, None])
            else:
                logger.info(
                    '[ Muon S&S ] WARNING: there are only unc_type strings ["Scale", "Resolution"]! \n Exiting. \n'
                )
                exit()
