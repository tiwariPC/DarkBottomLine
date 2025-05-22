import warnings
from typing import List, Optional, Tuple

import awkward as ak
import numpy
import xgboost


def load_photonid_mva(fname: str) -> Optional[xgboost.Booster]:
    try:
        photonid_mva = xgboost.Booster()
        photonid_mva.load_model(fname)
    except xgboost.core.XGBoostError:
        warnings.warn(f"SKIPPING photonid_mva, could not find: {fname}")
        photonid_mva = None
    return photonid_mva


def calculate_photonid_mva(
    mva: Tuple[Optional[xgboost.Booster], List[str]],
    photon: ak.Array,
) -> ak.Array:
    """Recompute PhotonIDMVA on-the-fly. This step is necessary considering that the inputs have to be corrected
    with the QRC process. Following is the list of features (barrel has 12, endcap two more):
    EB:
        events.Photon.energyRaw
        events.Photon.r9
        events.Photon.sieie
        events.Photon.etaWidth
        events.Photon.phiWidth
        events.Photon.sieip
        events.Photon.s4
        events.Photon.pfPhoIso03
        events.Photon.pfChargedIsoPFPV
        events.Photon.pfChargedIsoWorstVtx
        events.Photon.eta
        events.fixedGridRhoAll

    EE: EB +
        events.Photon.esEffSigmaRR
        events.Photon.esEnergyOverRawE
    """
    photonid_mva, var_order = mva

    if photonid_mva is None:
        return ak.ones_like(photon.pt)

    bdt_inputs = {}
    bdt_inputs = numpy.column_stack(
        [ak.to_numpy(photon[name]) for name in var_order]
    )
    tempmatrix = xgboost.DMatrix(bdt_inputs, feature_names=var_order)

    mvaID = photonid_mva.predict(tempmatrix)

    # Only needed to compare to TMVA
    # mvaID = 1.0 - 2.0 / (1.0 + numpy.exp(2.0 * mvaID))

    # the previous transformation was not working correctly, peakin at about 0.7
    # since we can't really remember why that functional form was picked in the first place we decided
    # to switch to a simpler stretch of the output that works better, even though not perfectly.
    # Open for changes/ideas
    mvaID = -1 + 2 * mvaID

    return mvaID


def load_photonid_mva_run3(fname: str) -> Optional[xgboost.Booster]:

    """ Reads and returns both the EB and EE Xgboost run3 mvaID models """

    photonid_mva_EB = xgboost.Booster()
    photonid_mva_EB.load_model(fname + 'Egamma_Run3_photonIDMVA_EB.json')

    photonid_mva_EE = xgboost.Booster()
    photonid_mva_EE.load_model(fname + 'Egamma_Run3_photonIDMVA_EE.json')

    return photonid_mva_EB, photonid_mva_EE


def calculate_photonid_mva_run3(
    mva: Tuple[Optional[xgboost.Booster], List[str]],
    photon: ak.Array, rho,
) -> ak.Array:

    """Recompute PhotonIDMVA on-the-fly. This step is necessary considering that the inputs have to be corrected
    with the QRC process. Following is the list of features (barrel has 12, endcap two more):
    EB:
        events.Photon.energyRaw
        events.Photon.r9
        events.Photon.sieie
        events.Photon.etaWidth
        events.Photon.phiWidth
        events.Photon.sieip
        events.Photon.s4
        events.photon.hoe
        probe_ecalPFClusterIso
        probe_trkSumPtHollowConeDR03
        probe_trkSumPtSolidConeDR04
        probe_pfChargedIso
        probe_pfChargedIsoWorstVtx
        events.Photon.ScEta
        events.fixedGridRhoAll

    EE: +
        events.Photon.energyRaw
        events.Photon.r9
        events.Photon.sieie
        events.Photon.etaWidth
        events.Photon.phiWidth
        events.Photon.sieip
        events.Photon.s4
        events.photon.hoe
        probe_ecalPFClusterIso
        probe_hcalPFClusterIso
        probe_trkSumPtHollowConeDR03
        probe_trkSumPtSolidConeDR04
        probe_pfChargedIso
        probe_pfChargedIsoWorstVtx
        events.Photon.ScEta
        events.fixedGridRhoAll
        events.Photon.esEffSigmaRR
        events.Photon.esEnergyOverRawE
    """

    photon["fixedGridRhoAll"] = numpy.nan_to_num(rho)
    photon = numpy.nan_to_num(photon)

    photonid_mva, var_order = mva

    if photonid_mva is None:
        return ak.ones_like(photon.pt)

    bdt_inputs = {}
    bdt_inputs = numpy.column_stack(
        [ak.to_numpy(photon[name]) for name in var_order]
    )

    tempmatrix = xgboost.DMatrix(bdt_inputs)

    mvaID = photonid_mva.predict(tempmatrix)

    # Only needed to compare to TMVA
    mvaID = 1.0 - 2.0 / (1.0 + numpy.exp(2.0 * mvaID))

    return mvaID
