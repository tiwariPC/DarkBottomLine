import awkward as ak
import numpy as np

from darkbottomline.selections import met_quality


def test_met_quality_definition():
    met = ak.Array([300.0, 300.0, 300.0, 100.0])
    calo = ak.Array([280.0, 100.0, 300.0, 0.0])
    recoil = ak.Array([300.0, 300.0, 300.0, 0.0])
    q = ak.to_numpy(met_quality(met, calo, recoil))
    assert np.allclose(q, [20 / 300, 200 / 300, 0.0, 0.0])  # recoil == 0 -> 0, never a division by zero


def test_met_quality_cut_is_strict():
    q = ak.to_numpy(met_quality(ak.Array([150.0, 149.0]), ak.Array([0.0, 0.0]), ak.Array([300.0, 300.0])))
    assert list(q < 0.5) == [False, True]
