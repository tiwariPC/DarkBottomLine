### functions for Z to mumugamma
import awkward as ak
from coffea.analysis_tools import PackedSelection
import vector

vector.register_awkward()


def select_muons_zmmy(self, muons_jagged: ak.highlevel.Array) -> ak.highlevel.Array:
    selection = PackedSelection()
    # flatten
    count = ak.num(muons_jagged)
    muons = ak.flatten(muons_jagged)

    zero_cut = ak.ones_like(muons.pt) > 0

    selection.add("muon_pt", muons.pt > self.muon_pt_threshold)
    selection.add("muon_eta", abs(muons.eta) < self.muon_max_eta)
    selection.add("muon_iso", muons.pfRelIso03_chg < self.muon_max_pfRelIso03_chg)
    selection.add(
        "muon_id",
        muons[self.muon_wp]
        if self.muon_wp in ["tightId", "mediumId", "looseId"]
        else zero_cut,
    )
    selection.add("muon_global", muons.isGlobal if self.global_muon else zero_cut)

    # unflatten selection
    final_sel = ak.unflatten(
        selection.all("muon_pt", "muon_eta", "muon_iso", "muon_id", "muon_global"),
        count,
    )
    return final_sel


def select_photons_zmmy(self, photons_jagged: ak.highlevel.Array) -> ak.highlevel.Array:
    selection = PackedSelection()
    # flatten
    count = ak.num(photons_jagged)
    photons = ak.flatten(photons_jagged)
    count = ak.num(photons_jagged)
    photons = ak.flatten(photons_jagged)

    selection.add("photon_pt", photons.pt > self.photon_pt_threshold)
    selection.add("photon_eta", (photons.isScEtaEB) | (photons.isScEtaEE))

    # unflatten selection
    final_sel = ak.unflatten(selection.all("photon_pt", "photon_eta"), count)
    return final_sel


def get_zmmy(
    self, dimuons: ak.highlevel.Array, photons: ak.highlevel.Array
) -> ak.highlevel.Array:
    sel_obj = PackedSelection()
    # combine dimuon & photon
    photons["charge"] = ak.zeros_like(photons.pt, dtype=int)
    mmy_jagged = ak.cartesian({"dimuon": dimuons, "photon": photons}, axis=1)
    # flatten mmy, selection only accept flatten arrays
    count = ak.num(mmy_jagged)
    mmy = ak.flatten(mmy_jagged)

    # selection.add("n_mmy", ak.num(mmy) > 0)
    # mmy = mmy.mask(selection.all("n_mmy"))
    # fsr selections
    # use photon SC eta to calculate dr
    vec_muon1 = ak.Array(
        {
            "rho": mmy.dimuon.lead.pt,
            "phi": mmy.dimuon.lead.phi,
            "eta": mmy.dimuon.lead.eta,
        },
        with_name="Vector3D",
    )
    vec_muon2 = ak.Array(
        {
            "rho": mmy.dimuon.sublead.pt,
            "phi": mmy.dimuon.sublead.phi,
            "eta": mmy.dimuon.sublead.eta,
        },
        with_name="Vector3D",
    )
    vec_photon = ak.Array(
        {"rho": mmy.photon.pt, "phi": mmy.photon.phi, "eta": mmy.photon.ScEta},
        with_name="Vector3D",
    )
    dR_muon1_photon = vec_muon1.deltaR(vec_photon)
    dR_muon2_photon = vec_muon2.deltaR(vec_photon)
    # dR_muon1_photon = mmy.dimuon.lead.delta_r(mmy.photon)
    # dR_muon2_photon = mmy.dimuon.sublead.delta_r(mmy.photon)
    sel_obj.add(
        "deltaR",
        ak.where(dR_muon1_photon < dR_muon2_photon, dR_muon1_photon, dR_muon2_photon)
        < self.max_fsr_photon_dR,
    )
    # far muon pt
    muon_far = ak.where(
        dR_muon1_photon > dR_muon2_photon, mmy.dimuon.lead, mmy.dimuon.sublead
    )
    muon_near = ak.where(
        ~(dR_muon1_photon > dR_muon2_photon), mmy.dimuon.lead, mmy.dimuon.sublead
    )
    sel_obj.add("farmuon_pt", muon_far.pt > self.min_farmuon_pt)
    # dimuon obj
    dimuon_obj = mmy.dimuon.lead + mmy.dimuon.sublead
    sel_obj.add("dimuon_mass", dimuon_obj.mass > self.min_dimuon_mass)
    # mmy obj
    mmy_obj = mmy.dimuon.lead + mmy.dimuon.sublead + mmy.photon
    sel_obj.add(
        "mmy_mass",
        (mmy_obj.mass > self.min_mmy_mass) & (mmy_obj.mass < self.max_mmy_mass),
    )
    sel_obj.add(
        "dimuon_mmy_mass", (dimuon_obj.mass + mmy_obj.mass) < self.max_mm_mmy_mass
    )
    final_sel_obj = sel_obj.all(*(sel_obj.names))

    # unflatten
    final_sel_obj = ak.unflatten(final_sel_obj, count)
    # dress other variables
    mmy["muon_far"] = muon_far
    mmy["muon_near"] = muon_near
    mmy["obj_dimuon"] = dimuon_obj
    mmy["obj_mmy"] = mmy_obj
    mmy_jagged = ak.unflatten(mmy, count)

    mmy_jagged = mmy_jagged[final_sel_obj]

    # event selection
    zlike_idx = ak.argmin(abs(mmy_jagged.obj_mmy.mass - self.Zmass), axis=1)
    # get best matched mmy for each event
    best_mmy = ak.firsts(mmy_jagged[ak.singletons(zlike_idx)])

    return best_mmy
