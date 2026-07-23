"""
Unit tests for objects module.

Events are built with FLAT NanoAOD branches (e.g. ``Muon_pt``), matching how the
processor reads them. Config keys mirror ``configs/{year}.yaml`` (loud KeyError on
missing keys — no ``.get`` fallbacks in the module). The select_* mask functions
take a ``wp`` ("loose" | "tight") working point.
"""

import pytest
import awkward as ak
import numpy as np
from darkbottomline.objects import (
    select_muons, select_electrons, select_taus, select_photons, select_jets,
    clean_jets_from_leptons, get_bjet_mask, build_objects,
)


MU_CFG = {
    "pt_min": 30.0, "pt_min_loose": 10.0, "eta_max": 2.4,
    "iso_wp_loose": 1, "iso_wp_tight": 3,
}
EL_CFG = {
    "pt_min": 32.0, "pt_min_loose": 10.0, "eta_max": 2.5,
    "id_wp_loose": 2, "id_wp_tight": 4,
}
TAU_CFG = {
    "pt_min": 20.0, "eta_max": 2.3,
    "id_wp_vsjet": 3, "id_wp_vse": 1, "id_wp_vsmu": 1,
    "decay_modes": [0, 1, 2, 10, 11],
}
PHO_CFG = {"pt_min": 15.0, "eta_max": 2.5, "id_wp_loose": 1}
JET_CFG = {"pt_min": 30.0, "eta_max": 2.5}


class TestObjectSelection:
    """Test object selection mask functions (flat-branch input, wp working point)."""

    def test_select_muons_loose(self):
        # Event 0: [pass, fail-lowpt, pass]; ev1: [pass, fail-eta]; ev2: [pass]
        events = ak.Array({
            "Muon_pt":     [[25.0, 8.0, 40.0], [15.0, 20.0], [35.0]],
            "Muon_eta":    [[1.0, 2.0, 0.5],   [1.5, 3.0],    [0.8]],
            "Muon_looseId":[[1, 1, 1],         [1, 1],        [1]],
            "Muon_tightId":[[1, 1, 1],         [1, 1],        [1]],
            "Muon_pfIsoId":[[3, 3, 3],         [3, 3],        [3]],
        })
        mask = select_muons(events, MU_CFG, wp="loose")
        # loose pt_min=10, eta_max=2.4, iso>=1, looseId==1
        assert ak.to_list(ak.sum(mask, axis=1)) == [2, 1, 1]

    def test_select_muons_tight_pt_and_iso(self):
        # tight raises pt_min to 30 and iso to >=3
        events = ak.Array({
            "Muon_pt":     [[25.0, 40.0]],   # 25 fails tight pt(30), 40 passes
            "Muon_eta":    [[1.0, 0.5]],
            "Muon_looseId":[[1, 1]],
            "Muon_tightId":[[1, 1]],
            "Muon_pfIsoId":[[3, 3]],
        })
        mask = select_muons(events, MU_CFG, wp="tight")
        assert ak.to_list(mask) == [[False, True]]

    def test_select_muons_id_branch_by_wp(self):
        # tightId=0 but looseId=1: passes loose, fails tight
        events = ak.Array({
            "Muon_pt":     [[40.0]],
            "Muon_eta":    [[0.5]],
            "Muon_looseId":[[1]],
            "Muon_tightId":[[0]],
            "Muon_pfIsoId":[[3]],
        })
        assert ak.to_list(select_muons(events, MU_CFG, wp="loose")) == [[True]]
        assert ak.to_list(select_muons(events, MU_CFG, wp="tight")) == [[False]]

    def test_select_electrons_loose(self):
        events = ak.Array({
            "Electron_pt":          [[25.0, 8.0, 40.0], [15.0, 20.0], [35.0]],
            "Electron_eta":         [[1.0, 2.0, 0.5],   [1.0, 3.0],   [0.8]],
            "Electron_cutBased":    [[2, 2, 2],         [2, 2],       [2]],
            "Electron_mvaIso_WP80": [[1, 1, 1],         [1, 1],       [1]],
            "Electron_mvaIso_WP90": [[1, 1, 1],         [1, 1],       [1]],
        })
        mask = select_electrons(events, EL_CFG, wp="loose")
        # loose pt_min=10, eta_max=2.5, cutBased>=2, WP90 (ev1: eta 3.0 fails)
        assert ak.to_list(ak.sum(mask, axis=1)) == [2, 1, 1]

    def test_select_electrons_gap_veto(self):
        # |eta| in (1.4442, 1.566) is vetoed
        events = ak.Array({
            "Electron_pt":          [[40.0, 40.0]],
            "Electron_eta":         [[1.5, 0.5]],   # 1.5 in gap → veto
            "Electron_cutBased":    [[4, 4]],
            "Electron_mvaIso_WP80": [[1, 1]],
            "Electron_mvaIso_WP90": [[1, 1]],
        })
        assert ak.to_list(select_electrons(events, EL_CFG, wp="tight")) == [[False, True]]

    def test_select_electrons_tight_id_iso(self):
        # tight needs cutBased>=4 and WP80
        events = ak.Array({
            "Electron_pt":          [[40.0, 40.0]],
            "Electron_eta":         [[0.5, 0.7]],
            "Electron_cutBased":    [[2, 4]],   # first fails tight id
            "Electron_mvaIso_WP80": [[1, 1]],
            "Electron_mvaIso_WP90": [[1, 1]],
        })
        assert ak.to_list(select_electrons(events, EL_CFG, wp="tight")) == [[False, True]]

    def test_select_taus(self):
        events = ak.Array({
            "Tau_pt":                        [[25.0, 15.0, 30.0], [22.0, 10.0], [35.0]],
            "Tau_eta":                       [[1.0, 2.0, 0.5],    [1.5, 3.0],   [0.8]],
            "Tau_idDeepTau2018v2p5VSjet":    [[3, 3, 3],          [3, 3],       [3]],
            "Tau_idDeepTau2018v2p5VSe":      [[1, 1, 1],          [1, 1],       [1]],
            "Tau_idDeepTau2018v2p5VSmu":     [[1, 1, 1],          [1, 1],       [1]],
            "Tau_decayMode":                 [[0, 1, 2],          [0, 10],      [1]],
        })
        mask = select_taus(events, TAU_CFG, wp="loose")
        # pt_min=20 (ev0: 25,30 pass; 15 fails), eta_max=2.3 (ev1: 3.0 fails)
        assert ak.to_list(ak.sum(mask, axis=1)) == [2, 1, 1]

    def test_select_taus_decay_mode_and_id(self):
        # decayMode not in allowed list → fail; VSjet below WP → fail
        events = ak.Array({
            "Tau_pt":                        [[40.0, 40.0, 40.0]],
            "Tau_eta":                       [[0.5, 0.5, 0.5]],
            "Tau_idDeepTau2018v2p5VSjet":    [[3, 2, 3]],   # middle below vsjet WP
            "Tau_idDeepTau2018v2p5VSe":      [[1, 1, 1]],
            "Tau_idDeepTau2018v2p5VSmu":     [[1, 1, 1]],
            "Tau_decayMode":                 [[0, 0, 5]],   # last mode 5 not allowed
        })
        assert ak.to_list(select_taus(events, TAU_CFG, wp="loose")) == [[True, False, False]]

    def test_select_photons(self):
        events = ak.Array({
            "Photon_pt":           [[20.0, 10.0, 30.0], [16.0, 8.0], [40.0]],
            "Photon_eta":          [[1.0, 2.0, 0.5],    [1.5, 3.0],  [0.8]],
            "Photon_cutBased":     [[1, 1, 1],          [1, 1],      [1]],
            "Photon_electronVeto": [[1, 1, 0],          [1, 1],      [1]],
        })
        mask = select_photons(events, PHO_CFG)
        # pt_min=15, eta_max=2.5, cutBased>=1, electronVeto==1
        # ev0: 20 passes, 10 fails pt, 30 fails electronVeto → 1
        assert ak.to_list(ak.sum(mask, axis=1)) == [1, 1, 1]

    def test_select_jets(self):
        events = ak.Array({
            "Jet_pt":  [[35.0, 25.0, 40.0], [30.0, 20.0], [45.0]],  # pt_min=30 strict >
            "Jet_eta": [[1.0, 2.0, 0.5],    [1.5, 3.0],   [0.8]],
        })
        mask = select_jets(events, JET_CFG)
        # pt>30 (ev0: 35,40 → 25 fails; ev1: 30 fails strict, 20 fails → 0)
        assert ak.to_list(ak.sum(mask, axis=1)) == [2, 0, 1]


class TestJetCleaningAndBtag:

    def test_clean_jets_from_leptons(self):
        # jets: 3 per event; lepton overlaps jet 0 (dR~0)
        # Jets and leptons are jagged list-of-records (var * {field}), as produced
        # by build_*_collection / the cleaning-objects concatenate in build_objects.
        jets = ak.zip({
            "pt":  [[30.0, 35.0, 40.0]],
            "eta": [[1.0, 1.5, 2.0]],
            "phi": [[0.0, 1.0, 2.0]],
        })
        leptons = ak.zip({
            "eta": [[1.0]],
            "phi": [[0.0]],   # overlaps jet 0
        })
        mask = clean_jets_from_leptons(jets, leptons, dr_min=0.4)
        assert ak.to_list(mask) == [[False, True, True]]

    def test_clean_jets_empty_leptons(self):
        jets = ak.zip({"pt": [[30.0, 40.0]], "eta": [[1.0, 2.0]], "phi": [[0.0, 1.0]]})
        leptons = ak.zip({"eta": [[]], "phi": [[]]})
        mask = clean_jets_from_leptons(jets, leptons, dr_min=0.4)
        assert ak.to_list(mask) == [[True, True]]

    def test_get_bjet_mask(self):
        jets = ak.Array({"btagScore": [[0.1, 0.5, 0.8]]})
        mask = get_bjet_mask(jets, {"score": 0.2605})
        assert ak.to_list(mask) == [[False, True, True]]


class TestBuildObjects:
    """End-to-end object building on flat-branch events with a realistic config."""

    def _events(self):
        return ak.Array({
            "Muon_pt":     [[35.0, 12.0], [40.0], [8.0]],
            "Muon_eta":    [[1.0, 0.5],   [0.8],  [0.3]],
            "Muon_phi":    [[0.0, 1.0],   [0.5],  [1.5]],
            "Muon_charge": [[1, -1],      [1],    [-1]],
            "Muon_looseId":[[1, 1],       [1],    [1]],
            "Muon_tightId":[[1, 0],       [1],    [1]],
            "Muon_pfIsoId":[[3, 3],       [3],    [3]],
            "Electron_pt":          [[40.0], [20.0], [45.0]],
            "Electron_eta":         [[0.9],  [1.8],  [0.4]],
            "Electron_phi":         [[0.2],  [0.7],  [1.7]],
            "Electron_charge":      [[1],    [-1],   [1]],
            "Electron_cutBased":    [[4],    [2],    [4]],
            "Electron_mvaIso_WP80": [[1],    [1],    [1]],
            "Electron_mvaIso_WP90": [[1],    [1],    [1]],
            "Tau_pt":                     [[35.0], [30.0], [45.0]],
            "Tau_eta":                    [[1.1],  [1.6],  [0.9]],
            "Tau_phi":                    [[0.1],  [0.6],  [1.6]],
            "Tau_idDeepTau2018v2p5VSjet": [[3],    [3],    [3]],
            "Tau_idDeepTau2018v2p5VSe":   [[1],    [1],    [1]],
            "Tau_idDeepTau2018v2p5VSmu":  [[1],    [1],    [1]],
            "Tau_decayMode":              [[0],    [1],    [0]],
            "Jet_pt":          [[80.0, 45.0], [60.0], [90.0]],
            "Jet_eta":         [[1.0, 2.0],   [1.5],  [0.8]],
            "Jet_phi":         [[2.5, 1.8],   [2.0],  [2.9]],
            "Jet_mass":        [[10.0, 8.0],  [9.0],  [12.0]],
            "Jet_btagPNetB":   [[0.9, 0.1],   [0.5],  [0.8]],
            "Jet_hadronFlavour":[[5, 0],      [5],    [5]],
            "Photon_pt":           [[10.0], [20.0], [8.0]],
            "Photon_eta":          [[1.0],  [1.5],  [0.8]],
            "Photon_phi":          [[0.0],  [0.5],  [1.5]],
            "Photon_cutBased":     [[1],    [1],    [1]],
            "Photon_electronVeto": [[1],    [1],    [1]],
            "PuppiMET_pt":  [50.0, 60.0, 70.0],
            "PuppiMET_phi": [0.5, 1.0, 1.5],
        })

    def _config(self):
        return {
            "objects": {
                "muons": MU_CFG, "electrons": EL_CFG, "taus": TAU_CFG,
                "photons": PHO_CFG, "jets": JET_CFG,
            },
            "btagging": {"branch": "Jet_btagPNetB", "score": 0.2605},
            "cleaning": {"dr_jet": 0.4, "dr_photon_lep": 0.4},
        }

    def test_build_objects_keys_and_shapes(self):
        objects = build_objects(self._events(), self._config())
        # Collections present
        for key in ("muons", "electrons", "taus", "photons", "jets", "bjets",
                    "tight_muons", "tight_electrons", "tight_taus"):
            assert key in objects, f"missing {key}"
        # Per-event scalar/derived arrays present with correct length
        n_ev = 3
        for key in ("recoil", "recoil_phi", "costheta_star",
                    "n_z_muons", "n_z_electrons", "mll_mu", "mll_el",
                    "z_pt_mu", "z_pt_el"):
            assert key in objects, f"missing {key}"
            assert len(objects[key]) == n_ev

    def test_build_objects_selection_counts(self):
        objects = build_objects(self._events(), self._config())
        # Loose muons: ev0 both pass (35,12 > pt_min_loose 10), ev1 pass, ev2 8<10 fails
        assert ak.to_list(ak.num(objects["muons"], axis=1)) == [2, 1, 0]
        # Tight muons: pt>30 & tightId==1 & iso>=3 → ev0 only muon0 (35, tight);
        # ev1 (40, tight); ev2 none
        assert ak.to_list(ak.num(objects["tight_muons"], axis=1)) == [1, 1, 0]
        # is_tight flag carried on loose muons
        assert "is_tight" in objects["muons"].fields

    def test_build_objects_jet_cleaning_and_btag(self):
        objects = build_objects(self._events(), self._config())
        # Jets pass pt>30; none overlap leptons (phi separated), so all survive cleaning
        assert ak.to_list(ak.num(objects["jets"], axis=1)) == [2, 1, 1]
        # b-jets: btagScore > 0.2605 → ev0 jet0(0.9), ev1(0.5), ev2(0.8)
        assert ak.to_list(ak.num(objects["bjets"], axis=1)) == [1, 1, 1]


if __name__ == "__main__":
    pytest.main([__file__])
