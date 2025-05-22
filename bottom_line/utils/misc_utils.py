import awkward as ak
import numba
import numpy as np


def choose_jet(jets_variable, n, fill_value):
    """
    this helper function is used to create flat jets from a jagged collection,
    parameters:
    * jet_variable: (ak array) selected variable from the jet collection
    * n: (int) nth jet to be selected
    * fill_value: (float) value with wich to fill the padded none.
    """
    leading_jets_variable = jets_variable[
        ak.local_index(jets_variable) == n
    ]
    leading_jets_variable = ak.pad_none(
        leading_jets_variable, 1
    )
    leading_jets_variable = ak.flatten(
        ak.fill_none(leading_jets_variable, fill_value)
    )
    return leading_jets_variable


def add_pnet_prob(
    self,
    jets: ak.highlevel.Array
):
    """
    this helper function is used to add to the jets from the probability of PNet
    calculated starting from the standard scores contained in the JetMET nAODs
    """

    jet_pn_b = jets.particleNetAK4_B

    jet_pn_c = jets.particleNetAK4_B * jets.particleNetAK4_CvsB / (ak.ones_like(jets.particleNetAK4_B) - jets.particleNetAK4_CvsB)
    jet_pn_c = ak.where(
        (jets.particleNetAK4_CvsB >= 0) & (jets.particleNetAK4_CvsB < 1),
        jet_pn_c,
        -1
    )

    # Use ak.where to constrain the values within [0, 1]
    pn_uds_base = ak.ones_like(jet_pn_b) - jet_pn_b - jet_pn_c
    pn_uds_clipped = ak.where(pn_uds_base < 0, 0, ak.where(pn_uds_base > 1, 1, pn_uds_base))
    jet_pn_uds = pn_uds_clipped * jets.particleNetAK4_QvsG
    jet_pn_uds = ak.where(
        (jets.particleNetAK4_QvsG >= 0) & (jets.particleNetAK4_QvsG < 1),
        jet_pn_uds,
        -1
    )

    jet_pn_g_base = ak.ones_like(jet_pn_b) - jet_pn_b - jet_pn_c - jet_pn_uds
    jet_pn_g = ak.where(jet_pn_g_base < 0, 0, ak.where(jet_pn_g_base > 1, 1, jet_pn_g_base))
    jet_pn_g = ak.where(
        (jets.particleNetAK4_QvsG >= 0) & (jets.particleNetAK4_QvsG < 1),
        jet_pn_g,
        -1
    )

    jet_pn_b_plus_c = jet_pn_b + jet_pn_c
    jet_pn_b_vs_c = jet_pn_b / jet_pn_b_plus_c

    jets["pn_b"] = jet_pn_b
    jets["pn_c"] = jet_pn_c
    jets["pn_uds"] = jet_pn_uds
    jets["pn_g"] = jet_pn_g
    jets["pn_b_plus_c"] = jet_pn_b_plus_c
    jets["pn_b_vs_c"] = jet_pn_b_vs_c

    return jets


@numba.vectorize(
    [
        numba.float32(numba.float32, numba.float32),
        numba.float64(numba.float64, numba.float64),
    ]
)
def delta_phi(a, b):
    """Compute difference in angle given two angles a and b

    Returns a value within [-pi, pi)
    """
    return (a - b + np.pi) % (2 * np.pi) - np.pi


@numba.vectorize(
    [
        numba.float32(numba.float32, numba.float32, numba.float32, numba.float32),
        numba.float64(numba.float64, numba.float64, numba.float64, numba.float64),
    ]
)
def delta_r(eta1, phi1, eta2, phi2):
    r"""Distance in (eta,phi) plane given two pairs of (eta,phi)

    :math:`\sqrt{\Delta\eta^2 + \Delta\phi^2}`
    """
    deta = eta1 - eta2
    dphi = delta_phi(phi1, phi2)
    return np.hypot(deta, dphi)


def delta_r_with_ScEta(a, b):
    """Distance in (eta,phi) plane between two objects using `ScEta` insetad of `eta`"""
    return delta_r(a.ScEta, a.phi, b.eta, b.phi)


def trigger_match(
        offline_objects, trigobjs, pdgid, pt, filterbit, metric=lambda a, b: a.delta_r(b), dr=0.1
):
    """
    Matches offline objects  with online trigger objects using dR < dr criterion
    The filterbit corresponds to the trigger we want our offline objects to have fired
    """
    pass_pt = trigobjs.pt > pt
    pass_id = abs(trigobjs.id) == pdgid
    pass_filterbit = (trigobjs.filterBits & (0x1 << filterbit)) != 0
    trigger_cands = trigobjs[pass_pt & pass_id & pass_filterbit]
    delta_r = offline_objects.metric_table(trigger_cands, metric=metric)
    pass_delta_r = delta_r < dr
    n_of_trigger_matches = ak.sum(pass_delta_r, axis=2)
    trig_matched_locs = n_of_trigger_matches >= 1

    return trig_matched_locs
