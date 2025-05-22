import awkward as ak
import numpy as np


def delta_r_mask(
    first: ak.highlevel.Array, second: ak.highlevel.Array, threshold: float
) -> ak.highlevel.Array:
    """
    Select objects from first which are at least threshold away from all objects in second.
    The result is a mask (i.e., a boolean array) of the same shape as first.

    :param first: objects which are required to be at least threshold away from all objects in second
    :type first: coffea.nanoevents.methods.candidate.PtEtaPhiMCandidate
    :param second: objects which are all objects in first must be at leats threshold away from
    :type second: coffea.nanoevents.methods.candidate.PtEtaPhiMCandidate
    :param threshold: minimum delta R between objects
    :type threshold: float
    :return: boolean array of objects in objects1 which pass delta_R requirement
    :rtype: coffea.nanoevents.methods.candidate.PtEtaPhiMCandidate
    """
    mval = first.metric_table(second)
    return ak.all(mval > threshold, axis=-1)


def delta_phi_mask(
        Phi1: ak.highlevel.Array,
        Phi2: ak.highlevel.Array,
        threshold: float
) -> ak.highlevel.Array:
    # Select objects that are at least threshold away in Phi space

    # calculate delta_phi
    dPhi = abs(Phi1 - Phi2) % (2 * np.pi)
    dPhi = ak.where(dPhi > np.pi, 2 * np.pi - dPhi, dPhi)

    return dPhi > threshold
