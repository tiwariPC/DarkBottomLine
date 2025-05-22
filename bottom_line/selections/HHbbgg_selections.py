import awkward as ak
import numpy as np
import vector

vector.register_awkward()


def get_HHbbgg(
    self, diphotons: ak.highlevel.Array, dijets: ak.highlevel.Array
) -> ak.highlevel.Array:
    # Script adapted from the Zmumug analysis
    # combine dijet & diphoton
    dijets["charge"] = ak.zeros_like(dijets.pt, dtype=int)
    HHbbgg_jagged = ak.cartesian({"diphoton": diphotons, "dijet": dijets}, axis=1)
    # flatten HHbbgg, selection only accept flatten arrays
    count = ak.num(HHbbgg_jagged)
    HHbbgg = ak.flatten(HHbbgg_jagged)

    # diphoton and dijet obj
    diphoton_obj = HHbbgg.diphoton.pho_lead + HHbbgg.diphoton.pho_sublead
    dijet_obj = HHbbgg.dijet.first_jet + HHbbgg.dijet.second_jet

    HHbbgg_obj = (
        HHbbgg.diphoton.pho_lead
        + HHbbgg.diphoton.pho_sublead
        + HHbbgg.dijet.first_jet
        + HHbbgg.dijet.second_jet
    )

    # dress other variables
    HHbbgg["obj_diphoton"] = diphoton_obj
    HHbbgg["pho_lead"] = HHbbgg.diphoton.pho_lead
    HHbbgg["pho_sublead"] = HHbbgg.diphoton.pho_sublead
    HHbbgg["obj_dijet"] = dijet_obj
    HHbbgg["first_jet"] = HHbbgg.dijet.first_jet
    HHbbgg["second_jet"] = HHbbgg.dijet.second_jet
    HHbbgg["obj_HHbbgg"] = HHbbgg_obj
    HHbbgg_jagged = ak.unflatten(HHbbgg, count)

    # get best matched HHbbgg for each event
    best_HHbbgg = ak.firsts(HHbbgg_jagged)

    return best_HHbbgg


def getCosThetaStar_CS(HHbbgg, ebeam):
    """
    cos theta star angle in the Collins Soper frame
    Copied directly from here: https://github.com/ResonantHbbHgg/Selection/blob/master/selection.h#L3367-L3385
    """
    p1 = ak.zip(
        {
            "px": 0,
            "py": 0,
            "pz": ebeam,
            "E": ebeam,
        },
        with_name="Momentum4D",
    )

    p2 = ak.zip(
        {
            "px": 0,
            "py": 0,
            "pz": -ebeam,
            "E": ebeam,
        },
        with_name="Momentum4D",
    )

    diphoton = ak.with_name(HHbbgg.obj_diphoton, "Momentum4D")
    # dijet=ak.with_name(HHbbgg.obj_dijet,"Momentum4D")
    HH = ak.with_name(HHbbgg.obj_HHbbgg, "Momentum4D")

    # Check if there are any events; if not, return an empty array
    if len(HH) == 0:
        return ak.Array([])

    hhforboost = ak.zip({"px": -HH.px, "py": -HH.py, "pz": -HH.pz, "E": HH.E})
    hhforboost = ak.with_name(hhforboost, "Momentum4D")

    p1 = p1.boost(hhforboost)
    p2 = p2.boost(hhforboost)
    diphotonBoosted = diphoton.boost(hhforboost)

    CSaxis = p1 - p2

    return np.cos(CSaxis.deltaangle(diphotonBoosted))


def getCosThetaStar_gg(HHbbgg):

    Hgg = ak.with_name(HHbbgg.obj_diphoton, "Momentum4D")

    # If there are no diphoton events, return an empty array
    if len(Hgg) == 0:
        return ak.Array([])

    hggforboost = ak.zip({"px": -Hgg.px, "py": -Hgg.py, "pz": -Hgg.pz, "E": Hgg.E})
    hggforboost = ak.with_name(hggforboost, "Momentum4D")

    photon = ak.zip(
        {
            "px": HHbbgg.pho_lead.px,
            "py": HHbbgg.pho_lead.py,
            "pz": HHbbgg.pho_lead.pz,
            "mass": 0,
        }
    )
    Hgg_photon = ak.with_name(photon, "Momentum4D")

    Hgg_photon_boosted = Hgg_photon.boost(hggforboost)

    return Hgg_photon_boosted.costheta


def getCosThetaStar_jj(HHbbgg):

    Hjj = ak.with_name(HHbbgg.obj_dijet, "Momentum4D")

    # If there are no dijet events, return an empty array
    if len(Hjj) == 0:
        return ak.Array([])

    hjjforboost = ak.zip({"px": -Hjj.px, "py": -Hjj.py, "pz": -Hjj.pz, "E": Hjj.E})
    hjjforboost = ak.with_name(hjjforboost, "Momentum4D")

    Hjj_jet = ak.with_name(HHbbgg.first_jet, "Momentum4D")

    Hjj_jet_boosted = Hjj_jet.boost(hjjforboost)

    return Hjj_jet_boosted.costheta


def disjoint_dijets(dijet1, dijet2):

    mask = (
        (DeltaR(dijet1["first_jet"], dijet2["first_jet"]) > 0.4)
        & (DeltaR(dijet1["first_jet"], dijet2["second_jet"]) > 0.4)
        & (DeltaR(dijet1["second_jet"], dijet2["first_jet"]) > 0.4)
        & (DeltaR(dijet1["second_jet"], dijet2["second_jet"]) > 0.4)
    )
    return mask


def getChi_t0(b_dijets, dijets, n_jets, fill_value):

    # Combinations of non-b_dijets
    non_b_dijets = dijets[disjoint_dijets(dijets, b_dijets)]
    non_b_dijets = non_b_dijets[ak.argsort(non_b_dijets["DeltaR_jj"], axis=1)]

    # Find w_jets by minimizing DeltaR btwn two jets
    w_jets = ak.firsts(non_b_dijets)
    w_jets_4mom = w_jets["first_jet"] + w_jets["second_jet"]

    # Choose associated bjet with w_dijet by minimizing DeltaR
    b_jets_mask = DeltaR(w_jets_4mom, b_dijets["first_jet"]) < DeltaR(
        w_jets_4mom, b_dijets["second_jet"]
    )
    top_jets = ak.where(
        b_jets_mask,
        w_jets_4mom + b_dijets["first_jet"],
        w_jets_4mom + b_dijets["second_jet"],
    )

    # Combine terms to obtain chi^2
    w_mass = 80.377
    top_mass = 172.76
    term1 = ((w_mass - w_jets.mass) / (0.1 * w_mass)) ** 2
    term2 = ((top_mass - top_jets.mass) / (0.1 * top_mass)) ** 2

    # Mask for events with minimum 4 jets (2 bjets + 2 additional jets)
    #   -> mask out events with >=6 jets because those are covered by chi_t1
    t0_mask = (n_jets >= 4) & (n_jets < 6)
    chi_t0 = ak.mask(term1 + term2, t0_mask)

    return ak.fill_none(chi_t0, fill_value)


def getChi_t1(b_dijets, dijets, n_jets, fill_value):

    # Combinations of non-b_dijets
    non_b_dijets = dijets[disjoint_dijets(dijets, b_dijets)]
    non_b_dijets = non_b_dijets[ak.argsort(non_b_dijets["DeltaR_jj"], axis=1)]

    # Find w_jets by minimizing DeltaR btwn two jets
    w1_jets = ak.firsts(non_b_dijets)
    w1_jets_4mom = w1_jets["first_jet"] + w1_jets["second_jet"]

    # Find 2nd w_jets by minimizing DeltaR btwn two jets of the remaining jets
    w2_jets = ak.firsts(non_b_dijets[disjoint_dijets(non_b_dijets, w1_jets)])
    w2_jets_4mom = w2_jets["first_jet"] + w2_jets["second_jet"]

    # Choose associated bjet with w1_dijet by minimizing DeltaR
    #   -> choose other associated bjet by picking the one NOT associated with w1_dijet
    b1_jets_mask = DeltaR(w1_jets_4mom, b_dijets["first_jet"]) < DeltaR(
        w1_jets_4mom, b_dijets["second_jet"]
    )
    top1_jets = ak.where(
        b1_jets_mask,
        w1_jets_4mom + b_dijets["first_jet"],
        w1_jets_4mom + b_dijets["second_jet"],
    )
    top2_jets = ak.where(
        ~b1_jets_mask,
        w2_jets_4mom + b_dijets["first_jet"],
        w2_jets_4mom + b_dijets["second_jet"],
    )

    # Combine terms to obtain chi^2
    w_mass = 80.377
    top_mass = 172.76
    term1 = ((w_mass - w1_jets.mass) / (0.1 * w_mass)) ** 2
    term2 = ((top_mass - top1_jets.mass) / (0.1 * top_mass)) ** 2
    term3 = ((w_mass - w2_jets.mass) / (0.1 * w_mass)) ** 2
    term4 = ((top_mass - top2_jets.mass) / (0.1 * top_mass)) ** 2

    # Mask for events with minimum 6 jets (2 bjets + 4 additional jets)
    t1_mask = n_jets >= 6
    chi_t1 = ak.mask(term1 + term2 + term3 + term4, t1_mask)

    return ak.fill_none(chi_t1, fill_value)


def DeltaR(photon, jet):
    pho_obj = ak.with_name(photon, "Momentum4D")
    jet_obj = ak.with_name(jet, "Momentum4D")
    return vector.Spatial.deltaR(pho_obj, jet_obj)


def DeltaPhi(jet, MET):
    jet_obj = ak.with_name(jet, "Momentum4D")
    MET_obj = ak.with_name(MET, "Momentum4D")
    return vector.Spatial.deltaphi(jet_obj, MET_obj)


def Cxx(higgs_eta, VBFjet_eta_diff, VBFjet_eta_sum):
    # Centrality variable
    return np.exp(-4 / (VBFjet_eta_diff) ** 2 * (higgs_eta - (VBFjet_eta_sum) / 2) ** 2)
