"""Per-mass-point weighted_total_events for multi-point signal samples."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import awkward as ak
import numpy as np
import pytest
import uproot

from darkbottomline.corrections import CorrectionManager
from darkbottomline.cli import _read_weighted_total_events_per_genmodel
from darkbottomline.plotting import _mp_total_events, read_weighted_total_events_per_genmodel

PTS = ["GenModel_MH3_600_MH4_10_Mchi_1", "GenModel_MH3_600_MH4_150_Mchi_1"]


def _events():
    return ak.Array({
        "genWeight": [1.0, -2.0, 3.0, 0.5, -1.0],
        PTS[0]: [1, 1, 0, 0, 0],
        PTS[1]: [0, 0, 1, 1, 1],
    })


def test_counts_use_sign_of_genweight():
    stub = SimpleNamespace(config={"data": {"is_data": False}})
    out = CorrectionManager.get_weighted_total_events_per_genmodel(stub, _events())
    assert out[PTS[0]] == 0.0 and out[PTS[1]] == 1.0


def test_every_grid_point_has_a_counter():
    """A file only has GenModel_ branches for the points it contains; the counters still cover the
    whole grid (0 for absent points), like the flags in the output tree."""
    from darkbottomline.variables import _signal_grid_branches
    stub = SimpleNamespace(config={"data": {"is_data": False}})
    out = CorrectionManager.get_weighted_total_events_per_genmodel(stub, _events())
    grid = set(_signal_grid_branches())
    assert len(grid) == 29 and set(out) == grid | set(PTS)
    absent = [k for k in out if k not in PTS]
    assert absent and all(out[k] == 0.0 for k in absent)


def test_empty_for_data_and_single_point():
    data = SimpleNamespace(config={"data": {"is_data": True}})
    assert CorrectionManager.get_weighted_total_events_per_genmodel(data, _events()) == {}
    mc = SimpleNamespace(config={"data": {"is_data": False}})
    assert CorrectionManager.get_weighted_total_events_per_genmodel(mc, ak.Array({"genWeight": [1.0]})) == {}


def test_root_round_trip_and_readers(tmp_path):
    p = tmp_path / "s.root"
    edges = np.array([0.0, 1.0])
    with uproot.recreate(p) as f:
        f["weighted_total_events"] = (np.array([600.0]), edges)
        for k in PTS:
            f["weighted_total_events_" + k] = (np.array([300.0]), edges)
    with uproot.open(p) as f:
        assert read_weighted_total_events_per_genmodel(f) == {k: 300.0 for k in PTS}
        assert _read_weighted_total_events_per_genmodel(f) == {k: 300.0 for k in PTS}


def test_mp_total_events_no_fallback():
    wte_gm = {PTS[0]: 300.0}
    assert _mp_total_events(600.0, wte_gm, "MH3_600_MH4_10_Mchi_1") == 300.0
    with pytest.raises(ValueError):
        _mp_total_events(600.0, wte_gm, "MH3_600_MH4_150_Mchi_1")
    with pytest.raises(ValueError):
        _mp_total_events(600.0, None, "MH3_600_MH4_10_Mchi_1")
    # counted but no generated events: present, the point's flag selects nothing
    assert _mp_total_events(600.0, {PTS[1]: 0.0}, "MH3_600_MH4_150_Mchi_1") == 0.0


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "add_genmodel_counts", Path(__file__).resolve().parents[1] / "scripts" / "add_genmodel_counts.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_script_count_file(tmp_path):
    p = tmp_path / "nano.root"
    with uproot.recreate(p) as f:
        f["Events"] = {k: v for k, v in zip(
            ["genWeight"] + PTS,
            [np.array([1.0, -2.0, 3.0]), np.array([1, 1, 0], dtype=np.int8), np.array([0, 0, 1], dtype=np.int8)])}
    n, total, per = _load_script().count_file(str(p))
    assert (n, total) == (3, 1.0)
    assert per == {PTS[0]: 0.0, PTS[1]: 1.0}
