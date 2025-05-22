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



def select_electrons(
    self,
    electrons: ak.highlevel.Array,
) -> ak.highlevel.Array:
    pt_cut = electrons.pt > self.electron_pt_threshold

    eta_cut = abs(electrons.eta) < self.electron_max_eta

    if self.el_id_wp == "WP90":
        id_cut = electrons.mvaIso_WP90
    elif self.el_id_wp == "WP80":
        id_cut = electrons.mvaIso_WP80
    elif self.el_id_wp == "loose":
        id_cut = electrons.cutBased >= 2
    # WPL is not supported anymore with the Run 3 electron ID, CMSSW 130X v12 nanoAODs only have WP80 and WP90 options
    # elif self.el_id_wp == "WPL":
    #    id_cut = electrons.mvaIso_WPL
    else:
        id_cut = electrons.pt > 0.

    dxy_cut = abs(electrons.dxy) < self.electron_max_dxy if self.electron_max_dxy is not None else electrons.pt > 0
    dz_cut = abs(electrons.dz) < self.electron_max_dz if self.electron_max_dz is not None else electrons.pt > 0

    return pt_cut & eta_cut & id_cut  & dxy_cut & dz_cut


def select_muons(
    self,
    muons: ak.highlevel.Array,
) -> ak.highlevel.Array:
    pt_cut = muons.pt > self.muon_pt_threshold

    eta_cut = abs(muons.eta) < self.muon_max_eta

    if self.mu_id_wp == "tight":
        id_cut = muons.tightId
    elif self.mu_id_wp == "medium":
        id_cut = muons.mediumId
    elif self.mu_id_wp == "loose":
        id_cut = muons.looseId
    else:
        id_cut = muons.pt > 0

    # if I understand https://twiki.cern.ch/twiki/bin/view/CMS/MuonRun32022#Medium_pT_15_GeV_to_200_GeV correctly, only loose and tight are PF isos are supported with a SF (so far?)
    # also very loose, very tight and very very tight WPs are available: 1=PFIsoVeryLoose, 2=PFIsoLoose, 3=PFIsoMedium, 4=PFIsoTight, 5=PFIsoVeryTight, 6=PFIsoVeryVeryTight
    if self.mu_iso_wp == "tight":
        iso_cut = muons.pfIsoId >= 4
    elif self.mu_iso_wp == "medium":
        iso_cut = muons.pfIsoId >= 3
    elif self.mu_iso_wp == "loose":
        iso_cut = muons.pfIsoId >= 2
    else:
        iso_cut = muons.pt > 0

    if self.global_muon:
        global_cut = muons.isGlobal
    else:
        global_cut = muons.pt > 0


    dxy_cut = abs(muons.dxy) < self.muon_max_dxy if self.muon_max_dxy is not None else muons.pt > 0
    dz_cut = abs(muons.dz) < self.muon_max_dz if self.muon_max_dz is not None else muons.pt > 0

    return pt_cut & eta_cut & id_cut & iso_cut & global_cut & dxy_cut & dz_cut


def select_taus(
    self,
    taus: ak.highlevel.Array,
) -> ak.highlevel.Array:
    # Kinematic cuts
    pt_cut = taus.pt > self.tau_pt_threshold
    eta_cut = abs(taus.eta) < abs(self.tau_max_eta)
    dz_cut = abs(taus.dz) < self.tau_max_dz

    # apply the loosest working points for jet and muon discriminators, and the second loosest electron discriminator
    # as we are using the DeepTau ID we also veto dm 5 and 6, when the code is updated to use particleNet we may want to remove this requirement
    tau_id = "DeepTau2018v2p5"
    id_cut = (taus[f"id{tau_id}VSjet"] > 0) & (taus[f"id{tau_id}VSmu"] > 0) & (taus[f"id{tau_id}VSe"] > 1) & (taus.decayMode != 5) & (taus.decayMode != 6)


    return pt_cut & eta_cut & dz_cut & id_cut
