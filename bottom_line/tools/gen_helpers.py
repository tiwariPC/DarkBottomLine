import awkward as ak
import numpy as np
from bottom_line.selections.HHbbgg_selections import DeltaR
from bottom_line.selections.object_selections import delta_r_mask
import logging

logger = logging.getLogger(__name__)


def get_fiducial_flag(events: ak.Array, flavour: str = "Geometric") -> ak.Array:
    """
    Calculate the fiducial flag for events based on photon kinematics and geometric criteria at particle level.

    The function processes the events, identifying those that meet
    specific criteria based on the properties of the leading and subleading photons.
    The fiducial flag is determined based on transverse momentum (pt), mass, and pseudorapidity (eta)
    of the photon pairs, applying either 'Geometric' or 'Classical' selection criteria.

    Parameters:
    - events (ak.Array): An Awkward Array containing event data with GenIsolatedPhoton fields.
    - flavour (str, optional): The selection criterion to apply. Defaults to "Geometric".
      Can be "Geometric" for geometric mean based selection (https://arxiv.org/abs/2106.08329) or "Classical" for classical CMS pt/mass scaled cuts.

    Returns:
    - ak.Array: An Awkward Array of boolean flags, where True indicates an event meets the fiducial
      selection criteria.

    Note:
    - The function pads GenIsolatedPhoton fields to ensure at least two photons are present per event,
      filling missing values with None.
    - If the GenPart_iso branch is not included in the NanoAOD, the GenIsolatedPhotons collection is used
    """
    if 'iso' in events.GenPart.fields:
        sel_pho = (events.GenPart.pdgId == 22) & (events.GenPart.status == 1) & (events.GenPart.iso * events.GenPart.pt < 10)
        photons = events.GenPart[sel_pho]
        photons = photons[ak.argsort(photons.pt, ascending=False)]
        GenIsolatedPhotons = ak.pad_none(photons, 2)
    else:
        # Extract and pad the gen isolated photons
        GenIsolatedPhotons = events.GenIsolatedPhoton
        GenIsolatedPhotons = ak.pad_none(GenIsolatedPhotons, 2)

    # Separate leading and subleading photons
    lead_pho = GenIsolatedPhotons[:, 0]
    sublead_pho = GenIsolatedPhotons[:, 1]

    # Calculate diphoton system four vector
    diphoton = lead_pho + sublead_pho

    # Apply selection criteria based on the specified flavour
    if flavour == 'Geometric':
        # Geometric mean of pt criterion
        lead_mask = np.sqrt(lead_pho.pt * sublead_pho.pt) / diphoton.mass > 1 / 3
    elif flavour == 'Classical':
        # Classical pt/mass ratio criterion
        lead_mask = lead_pho.pt / diphoton.mass > 1 / 3

    # Subleading photon criterion always the same
    sublead_mask = sublead_pho.pt / diphoton.mass > 1 / 4

    # Pseudorapidity criteria for leading and subleading photons
    # Within tracker acceptance and remove the gap region
    # Note: Based on classical eta, not SC eta here
    lead_eta_mask = (np.abs(lead_pho.eta) < 1.4442) | ((np.abs(lead_pho.eta) < 2.5) & (np.abs(lead_pho.eta) > 1.566))
    sublead_eta_mask = (np.abs(sublead_pho.eta) < 1.4442) | ((np.abs(sublead_pho.eta) < 2.5) & (np.abs(sublead_pho.eta) > 1.566))

    # Combine all selection masks to form the fiducial flag
    fiducial_flag = lead_mask & sublead_mask & lead_eta_mask & sublead_eta_mask
    # Fill None values with False
    # Note: These values result from the padding
    # Only occurs for events that did not have two GenIsolatedPhoton origin
    fiducial_flag = ak.fill_none(fiducial_flag, False)

    return fiducial_flag


def get_genJets(self, events: ak.Array, pt_cut, eta_cut) -> ak.Array:
    GenJets = events.GenJet
    # We decide to clean based on dR criteria and not use partonFlavour as this is easier to reproduce
    # The commented option below is also interesting, removing jets that have not been matched to a coloured parton...
    # GenJets = GenJets[GenJets.partonFlavour != 0]

    if 'iso' in events.GenPart.fields:
        # Note: iso is a relative quantity
        sel_pho = (events.GenPart.pdgId == 22) & (events.GenPart.status == 1) & (events.GenPart.iso * events.GenPart.pt < 10)
        photons = events.GenPart[sel_pho]
        photons = photons[ak.argsort(photons.pt, ascending=False)]
        GenIsolatedPhotons = ak.pad_none(photons, 2)
    else:
        # Extract and pad the gen isolated photons
        GenIsolatedPhotons = events.GenIsolatedPhoton
        GenIsolatedPhotons = ak.pad_none(GenIsolatedPhotons, 2)

    # Separate leading and subleading photons
    lead_pho = GenIsolatedPhotons[:, 0]
    sublead_pho = GenIsolatedPhotons[:, 1]

    diphotons = lead_pho + sublead_pho

    if (ak.num(diphotons.pt, axis=0) > 0):
        lead = ak.zip(
            {
                "pt": lead_pho.pt,
                "eta": lead_pho.eta,
                "phi": lead_pho.phi,
                "mass": lead_pho.mass,
            }
        )
        lead = ak.with_name(lead, "PtEtaPhiMCandidate")
        sublead = ak.zip(
            {
                "pt": sublead_pho.pt,
                "eta": sublead_pho.eta,
                "phi": sublead_pho.phi,
                "mass": sublead_pho.mass,
            }
        )
        sublead = ak.with_name(sublead, "PtEtaPhiMCandidate")
        dr_pho_lead_cut = delta_r_mask(GenJets, lead, self.jet_pho_min_dr)
        dr_pho_sublead_cut = delta_r_mask(GenJets, sublead, self.jet_pho_min_dr)
    else:
        dr_pho_lead_cut = GenJets.pt > -1
        dr_pho_sublead_cut = GenJets.pt > -1

    # Lepton selection for overlap removal
    GenLeptons = events.GenPart[(abs(events.GenPart.pdgId) == 11) | (abs(events.GenPart.pdgId) == 13) & (events.GenPart.status == 1)]
    # # 11: Electron, 13: Muon
    if 'iso' in events.GenPart.fields:
        SelGenElectrons = GenLeptons[(abs(GenLeptons.pdgId) == 11) & (GenLeptons.pt > self.electron_pt_threshold) & (abs(GenLeptons.eta) < self.electron_max_eta) & (GenLeptons.iso < 0.2)]
        SelGenMuons = GenLeptons[(abs(GenLeptons.pdgId) == 13) & (GenLeptons.pt > self.muon_pt_threshold) & (abs(GenLeptons.eta) < self.muon_max_eta) & (GenLeptons.iso < 0.2)]
        dr_electrons_mask = delta_r_mask(GenJets, SelGenElectrons, self.jet_ele_min_dr)
        dr_muons_mask = delta_r_mask(GenJets, SelGenMuons, self.jet_muo_min_dr)
    else:
        logger.info("Careful: You are running over a sample that is nanoAOD v13 or older where the genPart collection does not contain the iso field")
        logger.info("Overlap removal for counting GenJets wrt to leptons will not be performed.")
        dr_electrons_mask = GenJets.pt > -1
        dr_muons_mask = GenJets.pt > -1

    GenJets = GenJets[(dr_pho_lead_cut) & (dr_pho_sublead_cut) & (dr_electrons_mask) & (dr_muons_mask)]

    # This is targeted primarly at photons from Higgs decay but also prompt electrons and muons in VH, TTH
    GenJets = GenJets[GenJets.pt > pt_cut]
    GenJets = GenJets[np.abs(GenJets.eta) < eta_cut]

    return GenJets


def get_higgs_gen_attributes(events: ak.Array) -> ak.Array:
    """
    Calculate the Higgs pt and y based on photon kinematics at particle level.

    Note:
    - The function pads GenIsolatedPhoton fields to ensure at least two photons are present per event,
      filling missing values with None.
    - If the GenPart_iso branch is not included in the NanoAOD, the GenIsolatedPhotons collection is used, to be consistent with get_fiducial_flag() above.
    """
    if 'iso' in events.GenPart.fields:
        sel_pho = (events.GenPart.pdgId == 22) & (events.GenPart.status == 1) & (events.GenPart.iso * events.GenPart.pt < 10)
        gen_photons = events.GenPart[sel_pho]
        gen_photons = gen_photons[ak.argsort(gen_photons.pt, ascending=False)]
        gen_photons = ak.pad_none(gen_photons, 2)
    else:
        # Extract and pad the gen isolated photons
        gen_photons = events.GenIsolatedPhoton
        gen_photons = ak.pad_none(gen_photons, 2)

    # Separate leading and subleading photons
    lead_pho = gen_photons[:, 0]
    sublead_pho = gen_photons[:, 1]
    gen_diphoton = lead_pho + sublead_pho

    pt = gen_diphoton.pt
    y = 0.5 * np.log((gen_diphoton.energy + gen_diphoton.pz) / (gen_diphoton.energy - gen_diphoton.pz))
    phi = gen_diphoton.phi

    return (pt, y, phi)


def match_jet(reco_jets, gen_jets, n, fill_value, jet_size=0.4, jet_flav=False):
    """
    this helper function is used to identify if a reco jet (or lepton) has a matching gen jet (lepton) for MC,
    -> Returns an array with 3 possible values:
        0 if reco not genMatched,
        1 if reco genMatched,
        -999 if reco doesn't exist
    parameters:
    * reco_jets: (ak array) reco_jet from the jets collection.
    * gen_jets: (ak array) gen_jet from the events.GenJet (or equivalent, e.g. events.Electron) collection.
    * n: (int) nth jet to be selected.
    * fill_value: (float) value with wich to fill the padded none if nth jet doesnt exist in the event.
    """
    if n is not None:
        reco_jets_i = reco_jets[ak.local_index(reco_jets, axis=1) == n]
    else:
        # This is for arrays already split into separate event slices, used for Higgs an bjet matching.
        reco_jets_i = ak.singletons(reco_jets)
    reco_jets_i = ak.pad_none(reco_jets_i, 1, clip=True)

    candidate_jet_matches = ak.cartesian({"reco": reco_jets_i, "gen": gen_jets}, axis=1)
    candidate_jet_matches["deltaR_jj"] = DeltaR(
        candidate_jet_matches["reco"], candidate_jet_matches["gen"]
    )

    matched_jets = ak.firsts(
        candidate_jet_matches[
            ak.argmin(candidate_jet_matches["deltaR_jj"], axis=1, keepdims=True)
        ], axis=1
    )
    matched_jets_bool = matched_jets["deltaR_jj"] < jet_size

    if jet_flav:
        matched_gen_flav = ak.where(
            matched_jets_bool, matched_jets["gen"].partonFlavour, fill_value
        )
        matched_gen_flav = ak.where(
            ~ak.is_none(ak.firsts(reco_jets_i)), matched_gen_flav, fill_value
        )
        return matched_gen_flav
    else:
        matched_jets_bool = ak.where(
            ~ak.is_none(ak.firsts(reco_jets_i)), matched_jets_bool, fill_value
        )
        return matched_jets_bool
