import awkward as ak

def select_photons(
    self,
    photons: ak.highlevel.Array,
) -> ak.highlevel.Array:
    pt_cut = photons.pt > self.photon_pt_threshold

    eta_cut = abs(photons.eta) < self.photon_max_eta

    if self.pho_id_wp == "WP90":
        id_cut = photons.mvaIso_WP90
    elif self.pho_id_wp == "WP80":
        id_cut = photons.mvaIso_WP80
    elif self.pho_id_wp == "loose":
        id_cut = photons.cutBased >= 2
    else:
        id_cut = photons.pt > 0.

    return pt_cut & eta_cut & id_cut