from functools import partial
import logging
from .jet_systematics_json import jerc_jet

logger = logging.getLogger(__name__)


def add_jme_corr_syst(corrections_dict, systematics_dict, logger):
    corrections_dict.update(
        {
            # jerc for MC
            "jec_jet": partial(jerc_jet, pt=None, apply_jec=True),
            "jec_jet_syst": partial(jerc_jet, pt=None, apply_jec=True, jec_syst=True),
            "jec_jet_pnet": partial(jerc_jet, pt=None, apply_jec=True, pnet="PNetRegression"),
            "jec_jet_pnetNu": partial(jerc_jet, pt=None, apply_jec=True, pnet="PNetRegressionPlusNeutrino"),
            "jec_jet_pnet_syst": partial(jerc_jet, pt=None, apply_jec=True, jec_syst=True, pnet="PNetRegression"),
            "jec_jet_pnetNu_syst": partial(jerc_jet, pt=None, apply_jec=True, jec_syst=True, pnet="PNetRegressionPlusNeutrino"),
            "jec_jet_regrouped_syst": partial(
                jerc_jet, pt=None, apply_jec=True, jec_syst=True, split_jec_syst=True
            ),
            "jerc_jet": partial(jerc_jet, pt=None, apply_jec=True, apply_jer=True),
            "jerc_jet_pnet": partial(jerc_jet, pt=None, apply_jec=True, apply_jer=True, pnet="PNetRegression"),
            "jerc_jet_pnetNu": partial(jerc_jet, pt=None, apply_jec=True, apply_jer=True, pnet="PNetRegressionPlusNeutrino"),
            "jerc_jet_syst": partial(
                jerc_jet,
                pt=None,
                apply_jec=True,
                jec_syst=True,
                apply_jer=True,
                jer_syst=True,
            ),
            "jerc_jet_pnet_syst": partial(
                jerc_jet,
                pt=None,
                apply_jec=True,
                jec_syst=True,
                apply_jer=True,
                jer_syst=True,
                pnet="PNetRegression",
            ),
            "jerc_jet_pnetNu_syst": partial(
                jerc_jet,
                pt=None,
                apply_jec=True,
                jec_syst=True,
                apply_jer=True,
                jer_syst=True,
                pnet="PNetRegressionPlusNeutrino",
            ),
            "jerc_jet_regrouped_syst": partial(
                jerc_jet,
                pt=None,
                apply_jec=True,
                jec_syst=True,
                split_jec_syst=True,
                apply_jer=True,
                jer_syst=True,
            ),
            # jec for data: Usually corrections on Data innecessary
            # Use jec corrections with Era to re-do jec for data
            "jec_RunA": partial(jerc_jet, pt=None, era="RunA", level="L1L2L3Res"),
            "jec_pnet_RunA": partial(jerc_jet, pt=None, era="RunA", level="L1L2L3Res", pnet="PNetRegression"),
            "jec_pnetNu_RunA": partial(jerc_jet, pt=None, era="RunA", level="L1L2L3Res", pnet="PNetRegressionPlusNeutrino"),
            "jec_RunB": partial(jerc_jet, pt=None, era="RunB", level="L1L2L3Res"),
            "jec_pnet_RunB": partial(jerc_jet, pt=None, era="RunB", level="L1L2L3Res", pnet="PNetRegression"),
            "jec_pnetNu_RunB": partial(jerc_jet, pt=None, era="RunB", level="L1L2L3Res", pnet="PNetRegressionPlusNeutrino"),
            "jec_RunC": partial(jerc_jet, pt=None, era="RunC", level="L1L2L3Res"),
            "jec_pnet_RunC": partial(jerc_jet, pt=None, era="RunC", level="L1L2L3Res", pnet="PNetRegression"),
            "jec_pnetNu_RunC": partial(jerc_jet, pt=None, era="RunC", level="L1L2L3Res", pnet="PNetRegressionPlusNeutrino"),
            "jec_RunD": partial(jerc_jet, pt=None, era="RunD", level="L1L2L3Res"),
            "jec_pnet_RunD": partial(jerc_jet, pt=None, era="RunD", level="L1L2L3Res", pnet="PNetRegression"),
            "jec_pnetNu_RunD": partial(jerc_jet, pt=None, era="RunD", level="L1L2L3Res", pnet="PNetRegressionPlusNeutrino"),
            "jec_RunE": partial(jerc_jet, pt=None, era="RunE", level="L1L2L3Res"),
            "jec_pnet_RunE": partial(jerc_jet, pt=None, era="RunE", level="L1L2L3Res", pnet="PNetRegression"),
            "jec_pnetNu_RunE": partial(jerc_jet, pt=None, era="RunE", level="L1L2L3Res", pnet="PNetRegressionPlusNeutrino"),
            "jec_RunF": partial(jerc_jet, pt=None, era="RunF", level="L1L2L3Res"),
            "jec_pnet_RunF": partial(jerc_jet, pt=None, era="RunF", level="L1L2L3Res", pnet="PNetRegression"),
            "jec_pnetNu_RunF": partial(jerc_jet, pt=None, era="RunF", level="L1L2L3Res", pnet="PNetRegressionPlusNeutrino"),
            "jec_RunG": partial(jerc_jet, pt=None, era="RunG", level="L1L2L3Res"),
            "jec_pnet_RunG": partial(jerc_jet, pt=None, era="RunG", level="L1L2L3Res", pnet="PNetRegression"),
            "jec_pnetNu_RunG": partial(jerc_jet, pt=None, era="RunG", level="L1L2L3Res", pnet="PNetRegressionPlusNeutrino"),
            "jec_RunH": partial(jerc_jet, pt=None, era="RunH", level="L1L2L3Res"),
            "jec_pnet_RunH": partial(jerc_jet, pt=None, era="RunH", level="L1L2L3Res", pnet="PNetRegression"),
            "jec_pnetNu_RunH": partial(jerc_jet, pt=None, era="RunH", level="L1L2L3Res", pnet="PNetRegressionPlusNeutrino"),
            # For 2023 era C, different version of the datasets have different JECs
            # Details: https://gitlab.cern.ch/cms-analysis/general/HiggsDNA/-/issues/220#note_9180675
            "jec_RunCv123": partial(jerc_jet, pt=None, era="RunCv123", level="L1L2L3Res"),
            "jec_pnet_RunCv123": partial(jerc_jet, pt=None, era="RunCv123", level="L1L2L3Res", pnet="PNetRegression"),
            "jec_pnetNu_RunCv123": partial(jerc_jet, pt=None, era="RunCv123", level="L1L2L3Res", pnet="PNetRegressionPlusNeutrino"),
            "jec_RunCv4": partial(jerc_jet, pt=None, era="RunCv4", level="L1L2L3Res"),
            "jec_pnet_RunCv4": partial(jerc_jet, pt=None, era="RunCv4", level="L1L2L3Res", pnet="PNetRegression"),
            "jec_pnetNu_RunCv4": partial(jerc_jet, pt=None, era="RunCv4", level="L1L2L3Res", pnet="PNetRegressionPlusNeutrino"),
        }
    )
    logger.info(
        f"""_summary_

    Available correction keys:
    {corrections_dict.keys()}
    Available systematic keys:
    {systematics_dict.keys()}
    """
    )
    return corrections_dict, systematics_dict
