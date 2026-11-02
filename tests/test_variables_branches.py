"""Event-selection branches added for the W CR studies: Jet{1,2,3}HadronFlavour and CaloMET_pt/phi."""
import awkward as ak
import numpy as np

from darkbottomline import variables as v


def _jets():
    return ak.zip({
        "pt": ak.Array([[100.0, 50.0], [200.0], []]),
        "eta": ak.Array([[0.1, 0.2], [0.0], []]),
        "phi": ak.Array([[0.1, 0.2], [0.0], []]),
        "mass": ak.Array([[5.0, 5.0], [5.0], []]),
        "btagScore": ak.Array([[0.9, 0.1], [0.2], []]),
        "hadronFlavour": ak.Array([[5, 4], [0], []]),
    })


def test_jet_hadron_flavour_branches_follow_jet_order():
    out = v._jet_lead_variables({"jets": _jets()})
    assert out["Jet1HadronFlavour"].dtype == np.int32
    assert out["Jet1HadronFlavour"].tolist() == [5, 0, 0]
    assert out["Jet2HadronFlavour"].tolist() == [4, 0, 0]   # missing jets default to 0
    assert out["Jet3HadronFlavour"].tolist() == [0, 0, 0]


def test_jet_hadron_flavour_is_zero_without_the_field():
    jets = ak.without_field(_jets(), "hadronFlavour")       # data has no truth flavour
    out = v._jet_lead_variables({"jets": jets})
    assert out["Jet1HadronFlavour"].tolist() == [0, 0, 0]


def test_calo_met_branches_come_from_the_nanoaod_fields():
    ev = ak.Array({"MET_pt": [1.0, 2.0], "MET_phi": [0.0, 0.0], "CaloMET_pt": [10.0, 20.0], "CaloMET_phi": [1.0, 2.0]})
    out = v._met_variables(ev, {})
    assert out["CaloMET_pt"].tolist() == [10.0, 20.0]
    assert out["CaloMET_phi"].tolist() == [1.0, 2.0]


def test_calo_met_defaults_to_sentinel_when_the_field_is_missing():
    ev = ak.Array({"MET_pt": [1.0], "MET_phi": [0.0]})
    assert v._met_variables(ev, {})["CaloMET_pt"].tolist() == [v.SENTINEL]


def test_new_branches_are_in_the_scalar_schema():
    for name in ("CaloMET_pt", "CaloMET_phi", "Jet1HadronFlavour", "Jet2HadronFlavour", "Jet3HadronFlavour"):
        assert name in v._SCALAR_BRANCHES
    assert v.get_empty_branch_types()["Jet1HadronFlavour"] == np.dtype(np.int32)
