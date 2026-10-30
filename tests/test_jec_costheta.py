"""JEC for the 2b fit variable: the shifted-recoil variations are translated to |cos theta*| as a migration through
`recoil > recoil_min` (JEC changes jet pT/MET, not eta, so each event keeps its nominal cos theta* value)."""
import pickle
import textwrap

import numpy as np
import pytest
import uproot
import yaml

from darkbottomline.plotting import PlotManager

SAMPLE = "TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8_EVENTSELECTION"
EDGES = [0.0, 0.25, 0.5, 0.75, 1.0]
CTS = np.array([0.1, 0.1, 0.3, 0.3, -9999., 0.9, 0.9, 0.9])   # event 4: one-jet event, cos theta* is a sentinel
RECOIL = np.array([300., 260., 400., 255., 500., 700., 252., 600.])
UP = RECOIL * 1.05                     # JEC up: every event moves up, nothing drops below 250
DOWN = np.array([300., 240., 400., 249., 500., 700., 240., 600.])   # events 1, 3, 6 migrate below 250
W = np.array([1., 2., 1., 4., 1., 1., 8., 1.])


def _run(tmp_path, jec_source="AbsoluteUp"):
    cfg = yaml.safe_load(textwrap.dedent(f"""\
        process_groups:
          ttbar:
            type: background
            patterns: ["TTto2L2Nu"]
        common_variables: [costheta_star]
        systematic_variables: [costheta_star]
        variable_bins:
          costheta_star:
            edges: {EDGES}
        """))
    pm = PlotManager(config=cfg)
    pm.recoil_min = 250.0
    values = {"costheta_star": CTS, "Recoil": RECOIL, "Recoil_JECAbsoluteUp": UP, "Recoil_JECAbsoluteDown": DOWN,
              "Recoil_JERUp": UP}
    pkl = {"region_histograms": {"2b:SR": {"weight": W, "values": values}},
           "metadata": {"weighted_total_events": 1.0, "sample": SAMPLE}}
    with open(tmp_path / f"{SAMPLE}.pkl", "wb") as fh:
        pickle.dump(pkl, fh)
    pm._create_region_stacked_plots(
        input_folder=str(tmp_path), process_groups=pm.process_groups, signal_groups={}, data_groups={},
        output_dir=str(tmp_path / "out"), luminosity=1.0, year="2024", version="v",
        cross_sections={"TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8": 1.0e-3},
        variables=["costheta_star"], regions=["2b:SR"], save_root=True)
    return tmp_path / "out" / "v" / "root"


def _hist(root_dir, name):
    with uproot.open(root_dir / name) as f:
        return f["TotalBkg"].values()


def test_costheta_jec_is_migration_through_the_recoil_cut(tmp_path):
    root = _run(tmp_path)
    nom = _hist(root, "hist_2b_SR_costheta_star_log.root")
    up = _hist(root, "hist_2b_SR_costheta_star_JECAbsoluteUP_log.root")
    down = _hist(root, "hist_2b_SR_costheta_star_JECAbsoluteDOWN_log.root")
    np.testing.assert_allclose(up, nom)                           # nothing left the region: same events, same values
    # events 1 (w=2, cts 0.1), 3 (w=4, cts 0.3), 6 (w=8, cts 0.9) fall below 250 in the down variation
    np.testing.assert_allclose(nom, [3., 5., 0., 10.])           # the sentinel event is not filled
    expected = nom - np.array([2., 4., 0., 8.])
    np.testing.assert_allclose(down, expected)


def test_costheta_jer_is_not_written(tmp_path):
    root = _run(tmp_path)
    assert not (root / "hist_2b_SR_costheta_star_JERUP_log.root").exists()
