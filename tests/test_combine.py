"""
Unit tests for Combine integration functionality.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import uproot
import yaml

from darkbottomline.combine_inputs import (
    load_region_bin_edges,
    load_region_histogram,
    load_region_syst_histogram,
    load_signal_grid,
    merge_emu_histograms,
    merged_region_dir_for_role,
    normalize_pdf_histograms,
    passthrough_region_histogram,
    region_dir_from_role,
    region_syst_hist_path,
    resolve_active_eras,
    resolve_version_dir,
    systematic_applies_to_region,
)
from darkbottomline.combine_tools import CombineDatacardWriter, CombineRunner

REPO_ROOT = Path(__file__).resolve().parent.parent
REGIONS_CONFIG = str(REPO_ROOT / "configs" / "regions.yaml")


def _write_region_histogram(path: Path, histograms: dict, edges=None):
    """Write a small fixture region ROOT file with known bin values."""
    if edges is None:
        edges = np.array([0.0, 50.0, 100.0, 150.0], dtype=float)
    path.parent.mkdir(parents=True, exist_ok=True)
    with uproot.recreate(str(path)) as f:
        for key, values in histograms.items():
            f[key] = (np.asarray(values, dtype=float), edges)


@pytest.fixture
def minimal_combine_config():
    return {
        "eras": [{"year": 2024, "year_config": "configs/2024.yaml", "active": True}],
        "regions_config": REGIONS_CONFIG,
        "blind": {"SR": True, "CR": False},
        "merge": {"merge_categories": True, "merge_eras": True, "merged_category_label": "C"},
        "categories": ["1b"],
        "regions": {
            "1b": {"signal_region": "SR", "control_regions": ["CR_Zmumu"]},
        },
        "fit_variable": {"1b": "Recoil"},
        "combine_emu": True,
        "pdf_normalize": True,
        "rateparam_mode": "perBinChannel",
        "input": {"region_root_dir_template": "outputs/region_plots_{year}/plots/{version}/root"},
        "datacard": {
            "processes": {
                "signal": {"name": "signal", "is_signal": True, "plot_group_label": "Signal_2HDMa"},
                "ttbar": {"name": "ttbar", "plot_group_label": "ttbar"},
                "zjets": {"name": "zjets", "plot_group_label": "DYto2LJets"},
            },
            "data": {"name": "data_obs"},
            "systematics": {
                "lumi": {"type": "lnN", "value": 1.014, "processes": ["signal", "ttbar", "zjets"]},
                "btagSF": {
                    "type": "shape",
                    "syst_suffix": "weight_btag",
                    "processes": ["signal", "ttbar", "zjets"],
                    "gated_by_cut": ["Bjet1bCond", "Bjet2bCond"],
                },
            },
            "rate_parameters": {
                "zjets_1b_Zmumu": {
                    "type": "rateParam", "regions": ["1b:CR_Zmumu"],
                    "processes": ["zjets"], "value": 1.0, "range": [0.05, 1.95],
                },
                "zjets_1b_Zee": {
                    "type": "rateParam", "regions": ["1b:CR_Zee"],
                    "processes": ["zjets"], "value": 1.0, "range": [0.05, 1.95],
                },
                "zjets_1b_Zll": {
                    "type": "rateParam", "regions": ["1b:CR_Zmumu"],
                    "processes": ["zjets"], "value": 1.0, "range": [0.05, 1.95],
                },
            },
            "sr_rate_parameters": {
                "1b": [
                    {"process": "zjets", "unmerged": ["zjets_1b_Zmumu", "zjets_1b_Zee"],
                     "merged": "zjets_1b_Zll"},
                ],
            },
        },
        "signal_grid": {
            "xsection_json": "data/cross-section/xsection_signal.json",
            "model_key": "2HDMa",
            "points": None,
        },
        "fit": {
            "strategy": "robustHesse",
            "options": {
                "asymptotic_limits": {"r_min": 0.0, "r_max": 10.0},
                "fit_diagnostics": {"save_shapes": True},
                "goodness_of_fit": {"algorithm": "saturated", "toys": 500, "seed": 12345},
            },
        },
        "output": {
            "datacard_file": "datacard_{year}_{category}_{region}_{mass_point}.txt",
            "shapes_file": "shapes_{year}_{category}_{region}_{mass_point}.root",
            "workspace_file": "workspace.root",
            "fit_results": {
                "asymptotic_limits": "asymptotic_limits.root",
                "fit_diagnostics": "fitDiagnostics.root",
                "goodness_of_fit": "gof.root",
                "goodness_of_fit_json": "gof.json",
                "impacts": "impacts.json",
            },
        },
        "advanced": {
            "combine_commands": {
                "text2workspace": "text2workspace.py",
                "combine": "combine",
                "combine_tool": "combineTool.py",
            },
            "workspace_options": {"args": ["--channel-masks"]},
            "commands": {
                "general_args": ["--cminDefaultMinimizerStrategy=0"],
                "AsymptoticLimits": {
                    "steps": [{"binary": "combine",
                               "args": ["-M", "AsymptoticLimits", "{workspace}", "{blind_args}",
                                        "--run={run_mode}"]}],
                },
                "FitDiagnostics": {
                    "steps": [{"binary": "combine",
                               "args": ["-M", "FitDiagnostics", "{workspace}", "{blind_args}"]}],
                },
                "GoodnessOfFit": {
                    "steps": [
                        {"binary": "combine",
                         "args": ["-M", "GoodnessOfFit", "{workspace}", "--algo={gof_algo}",
                                  "-n", "Observed",
                                  "--setParametersForFit", "{mask_sr_on}",
                                  "--setParametersForEval", "{mask_sr_off}",
                                  "--freezeParameters", "r", "--setParameters", "r=0"]},
                        {"binary": "combine",
                         "args": ["-M", "GoodnessOfFit", "{workspace}", "--algo={gof_algo}",
                                  "-n", "Toys",
                                  "--setParametersForFit", "{mask_sr_on}",
                                  "--setParametersForEval", "{mask_sr_off}",
                                  "--freezeParameters", "r", "--setParameters", "r=0,{mask_sr_on}",
                                  "-t", "{toys}", "--toysFrequentist"]},
                    ],
                },
                "Impacts": {
                    "steps": [
                        {"binary": "combineTool.py",
                         "args": ["-M", "Impacts", "-d", "{workspace}", "{blind_args}", "--doInitialFit"]},
                        {"binary": "combineTool.py",
                         "args": ["-M", "Impacts", "-d", "{workspace}", "{blind_args}", "--doFits"]},
                        {"binary": "combineTool.py",
                         "args": ["-M", "Impacts", "-d", "{workspace}", "-o", "{impacts_filename}"]},
                    ],
                },
                "PullsCRonly": {
                    "steps": [{"binary": "combine",
                               "args": ["-M", "FitDiagnostics", "{workspace}", "-n", "_CRonly",
                                        "--setParameters", "{mask_sr_on}"]}],
                },
                "PullsAsimovT0": {
                    "steps": [{"binary": "combine",
                               "args": ["-M", "FitDiagnostics", "{workspace}", "-n", "_AsimovT0",
                                        "-t", "-1", "--expectSignal", "0"]}],
                },
                "PullsSbT0": {
                    "steps": [{"binary": "combine",
                               "args": ["-M", "FitDiagnostics", "{workspace}", "-n", "_SbT0",
                                        "--expectSignal", "0"]}],
                },
                "PullsSbT1": {
                    "steps": [{"binary": "combine",
                               "args": ["-M", "FitDiagnostics", "{workspace}", "-n", "_SbT1",
                                        "--expectSignal", "1"]}],
                },
            },
            "parallel": {"n_workers": 2},
            "pulls_tooling": {
                "diff_nuisances_args": ["--abs", "--all"],
                "plot_pulls_macro": "condorJobs/combine/PlotPulls.C",
            },
        },
    }


class TestCombineInputs:
    """Test the combine_inputs.py helper functions."""

    def test_region_dir_from_role(self):
        assert region_dir_from_role("CR_Wmunu") == "Wmunu"
        assert region_dir_from_role("SR") == "SR"

    def test_resolve_version_dir_explicit_version(self, tmp_path):
        (tmp_path / "20260101_abc1234" / "root").mkdir(parents=True)
        template = str(tmp_path / "{version}" / "root")
        resolved = resolve_version_dir(template, "20260101_abc1234")
        assert resolved == tmp_path / "20260101_abc1234" / "root"

    def test_resolve_version_dir_explicit_version_missing_raises(self, tmp_path):
        template = str(tmp_path / "{version}" / "root")
        with pytest.raises(FileNotFoundError):
            resolve_version_dir(template, "does_not_exist")

    def test_resolve_version_dir_picks_latest_by_mtime(self, tmp_path):
        import os
        import time

        older = tmp_path / "20260101_aaa1111"
        newer = tmp_path / "20260102_bbb2222"
        (older / "root").mkdir(parents=True)
        (newer / "root").mkdir(parents=True)
        now = time.time()
        os.utime(older / "root", (now - 100, now - 100))
        os.utime(newer / "root", (now, now))

        template = str(tmp_path / "{version}" / "root")
        resolved = resolve_version_dir(template, None)
        assert resolved == newer / "root"

    def test_resolve_version_dir_no_candidates_raises(self, tmp_path):
        (tmp_path / "empty_dir").mkdir()
        template = str(tmp_path / "{version}" / "root")
        with pytest.raises(FileNotFoundError):
            resolve_version_dir(template, None)

    def test_load_region_histogram(self, tmp_path):
        root_file = tmp_path / "hist_1b_SR_Recoil_log.root"
        _write_region_histogram(root_file, {
            "TotalBkg": [10.0, 20.0, 15.0],
            "ttbar": [5.0, 10.0, 8.0],
        })

        values = load_region_histogram(str(tmp_path), "1b", "SR", "Recoil", "TotalBkg")
        np.testing.assert_array_equal(values, [10.0, 20.0, 15.0])

    def test_load_region_histogram_missing_key_raises(self, tmp_path):
        root_file = tmp_path / "hist_1b_SR_Recoil_log.root"
        _write_region_histogram(root_file, {"TotalBkg": [10.0, 20.0, 15.0]})

        with pytest.raises(KeyError):
            load_region_histogram(str(tmp_path), "1b", "SR", "Recoil", "nonexistent")

    def test_load_region_bin_edges(self, tmp_path):
        root_file = tmp_path / "hist_1b_SR_Recoil_log.root"
        edges = np.array([0.0, 50.0, 100.0, 150.0])
        _write_region_histogram(root_file, {"TotalBkg": [10.0, 20.0, 15.0]}, edges=edges)

        result = load_region_bin_edges(str(tmp_path), "1b", "SR", "Recoil")
        np.testing.assert_array_almost_equal(result, edges)

    def test_systematic_applies_to_region_sr_has_btag(self):
        assert systematic_applies_to_region(
            REGIONS_CONFIG, "1b:SR", ["Bjet1bCond", "Bjet2bCond"]
        ) is True

    def test_systematic_applies_to_region_zcr_has_no_btag(self):
        """Z CR has no b-tag cut (verified against configs/regions.yaml) —
        btagSF must not apply there."""
        assert systematic_applies_to_region(
            REGIONS_CONFIG, "1b:CR_Zmumu", ["Bjet1bCond", "Bjet2bCond"]
        ) is False

    def test_systematic_applies_to_region_wcr_has_btag(self):
        assert systematic_applies_to_region(
            REGIONS_CONFIG, "1b:CR_Wmunu", ["Bjet1bCond", "Bjet2bCond"]
        ) is True

    def test_systematic_applies_everywhere_when_not_gated(self):
        assert systematic_applies_to_region(REGIONS_CONFIG, "1b:CR_Zmumu", None) is True

    def test_load_signal_grid(self):
        grid = load_signal_grid(
            str(REPO_ROOT / "data" / "cross-section" / "xsection_signal.json"),
            "2HDMa",
        )
        assert len(grid) == 29
        assert all(not k.startswith("_") for k in grid)

    def test_load_signal_grid_subset(self):
        full_grid = load_signal_grid(
            str(REPO_ROOT / "data" / "cross-section" / "xsection_signal.json"),
            "2HDMa",
        )
        one_point = list(full_grid.keys())[:1]
        grid = load_signal_grid(
            str(REPO_ROOT / "data" / "cross-section" / "xsection_signal.json"),
            "2HDMa", points=one_point,
        )
        assert list(grid.keys()) == one_point

    def test_resolve_active_eras(self, minimal_combine_config):
        active = resolve_active_eras(minimal_combine_config)
        assert len(active) == 1
        assert active[0]["year"] == 2024

    def test_merge_emu_histograms(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _write_region_histogram(input_dir / "hist_1b_Wmunu_Recoil_log.root", {"ttbar": [1.0, 2.0, 3.0]})
        _write_region_histogram(input_dir / "hist_1b_Wenu_Recoil_log.root", {"ttbar": [4.0, 5.0, 6.0]})

        out_path = merge_emu_histograms(str(input_dir), str(output_dir), "1b", "Recoil",
                                         "CR_Wmunu", "CR_Wenu", "Wlnu")

        merged = load_region_histogram(str(output_dir), "1b", "Wlnu", "Recoil", "ttbar")
        np.testing.assert_array_equal(merged, [5.0, 7.0, 9.0])

    def test_merge_emu_histograms_merges_systematic_variants(self, tmp_path):
        """Regression: merge_emu_histograms previously only merged the nominal
        file, never the shape-systematic UP/DOWN variant files — so every
        shape systematic silently vanished ("-" in the datacard) for
        combine_emu-merged CRs even when both channels had real per-systematic
        histograms upstream. Reproduced against real Wmunu/Wenu region-plot
        output this session."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _write_region_histogram(input_dir / "hist_1b_Wmunu_Recoil_log.root", {"ttbar": [1.0, 2.0, 3.0]})
        _write_region_histogram(input_dir / "hist_1b_Wenu_Recoil_log.root", {"ttbar": [4.0, 5.0, 6.0]})
        # btagSF present for both channels -> should be summed
        _write_region_histogram(
            input_dir / "hist_1b_Wmunu_Recoil_weight_btagUP_log.root", {"ttbar": [1.1, 2.1, 3.1]})
        _write_region_histogram(
            input_dir / "hist_1b_Wenu_Recoil_weight_btagUP_log.root", {"ttbar": [4.1, 5.1, 6.1]})
        # JES present only for the muon channel -> should pass through unchanged
        _write_region_histogram(
            input_dir / "hist_1b_Wmunu_Recoil_JESUP_log.root", {"ttbar": [1.5, 2.5, 3.5]})

        merge_emu_histograms(str(input_dir), str(output_dir), "1b", "Recoil",
                              "CR_Wmunu", "CR_Wenu", "Wlnu",
                              syst_suffixes=["weight_btag", "JES"])

        btag_merged = load_region_syst_histogram(str(output_dir), "1b", "Wlnu", "Recoil",
                                                   "weight_btag", "UP", "ttbar")
        np.testing.assert_allclose(btag_merged, [5.2, 7.2, 9.2])

        jes_passthrough = load_region_syst_histogram(str(output_dir), "1b", "Wlnu", "Recoil",
                                                       "JES", "UP", "ttbar")
        np.testing.assert_allclose(jes_passthrough, [1.5, 2.5, 3.5])

        # No DOWN variant provided for either channel -> no output file, and
        # no crash (must not assume every direction exists for every syst).
        assert not (output_dir / "hist_1b_Wlnu_Recoil_weight_btagDOWN_log.root").exists()

    def test_normalize_pdf_histograms_preserves_integral(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _write_region_histogram(input_dir / "hist_1b_SR_Recoil_log.root", {
            "ttbar": [10.0, 10.0, 10.0],
        })
        _write_region_histogram(
            region_syst_hist_path(str(input_dir), "1b", "SR", "Recoil", "weight_pdf", "UP"),
            {"ttbar": [12.0, 12.0, 12.0]},
        )

        normalize_pdf_histograms(str(input_dir), str(output_dir), "1b", "SR", "Recoil",
                                  syst_suffixes=["weight_pdf"], pdf_syst_suffix="weight_pdf")

        nominal = load_region_histogram(str(output_dir), "1b", "SR", "Recoil", "ttbar")
        variant_path = region_syst_hist_path(str(output_dir), "1b", "SR", "Recoil", "weight_pdf", "UP")
        import uproot as _uproot
        variant_values, _ = _uproot.open(str(variant_path))["ttbar"].to_numpy()
        assert nominal.sum() == pytest.approx(variant_values.sum(), rel=1e-6)

    def test_normalize_pdf_histograms_copies_other_systematics_unchanged(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _write_region_histogram(input_dir / "hist_1b_SR_Recoil_log.root", {"ttbar": [10.0, 10.0, 10.0]})
        _write_region_histogram(
            region_syst_hist_path(str(input_dir), "1b", "SR", "Recoil", "JES", "UP"),
            {"ttbar": [11.0, 11.0, 11.0]},
        )

        normalize_pdf_histograms(str(input_dir), str(output_dir), "1b", "SR", "Recoil",
                                  syst_suffixes=["JES"], pdf_syst_suffix="weight_pdf")

        out_path = region_syst_hist_path(str(output_dir), "1b", "SR", "Recoil", "JES", "UP")
        assert out_path.exists()
        import uproot as _uproot
        values, _ = _uproot.open(str(out_path))["ttbar"].to_numpy()
        np.testing.assert_array_equal(values, [11.0, 11.0, 11.0])

    def test_passthrough_region_histogram_copies_systematic_variants(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _write_region_histogram(input_dir / "hist_1b_SR_Recoil_log.root", {"ttbar": [10.0, 10.0, 10.0]})
        _write_region_histogram(
            region_syst_hist_path(str(input_dir), "1b", "SR", "Recoil", "JES", "UP"),
            {"ttbar": [11.0, 11.0, 11.0]},
        )

        passthrough_region_histogram(str(input_dir), str(output_dir), "1b", "SR", "Recoil",
                                      syst_suffixes=["JES"])

        out_path = region_syst_hist_path(str(output_dir), "1b", "SR", "Recoil", "JES", "UP")
        assert out_path.exists()


class TestCombineDatacardWriter:
    """Test CombineDatacardWriter against real fixture histograms."""

    @pytest.fixture
    def region_root_dir(self, tmp_path):
        d = tmp_path / "region_root"
        _write_region_histogram(d / "hist_1b_SR_Recoil_log.root", {
            "TotalBkg": [10.0, 20.0, 15.0],
            "ttbar": [4.0, 8.0, 6.0],
            "DYto2LJets": [1.0, 2.0, 1.5],
            "sig_MH3_600_MH4_150_Mchi_1": [0.5, 1.0, 0.8],
        })
        _write_region_histogram(d / "hist_1b_Zmumu_Recoil_log.root", {
            "TotalBkg": [5.0, 6.0, 7.0],
            "ttbar": [0.0, 0.0, 0.0],
            "DYto2LJets": [5.0, 6.0, 7.0],
            "data_obs": [5.0, 6.0, 8.0],
        })
        return str(d)

    def test_get_number_of_bins(self, minimal_combine_config, region_root_dir):
        writer = CombineDatacardWriter(minimal_combine_config)
        n_bins = writer._get_number_of_bins(region_root_dir, "1b", "SR", "Recoil")
        assert n_bins == 3

    def test_get_observation_values_blind(self, minimal_combine_config, region_root_dir):
        writer = CombineDatacardWriter(minimal_combine_config)
        obs = writer._get_observation_values(region_root_dir, "1b", "SR", "Recoil", blind=True)
        assert obs == [10.0, 20.0, 15.0]

    def test_get_observation_values_unblind(self, minimal_combine_config, region_root_dir):
        writer = CombineDatacardWriter(minimal_combine_config)
        obs = writer._get_observation_values(region_root_dir, "1b", "Zmumu", "Recoil", blind=False)
        assert obs == [5.0, 6.0, 8.0]

    def test_get_observation_values_unblind_missing_data_raises(self, minimal_combine_config, region_root_dir):
        writer = CombineDatacardWriter(minimal_combine_config)
        with pytest.raises(KeyError):
            writer._get_observation_values(region_root_dir, "1b", "SR", "Recoil", blind=False)

    def test_get_process_rate_background(self, minimal_combine_config, region_root_dir):
        writer = CombineDatacardWriter(minimal_combine_config)
        rate = writer._get_process_rate(region_root_dir, "1b", "SR", "Recoil", "ttbar", None)
        assert rate == pytest.approx(18.0)

    def test_get_process_rate_signal(self, minimal_combine_config, region_root_dir):
        writer = CombineDatacardWriter(minimal_combine_config)
        rate = writer._get_process_rate(region_root_dir, "1b", "SR", "Recoil", "signal",
                                         "MH3_600_MH4_150_Mchi_1")
        assert rate == pytest.approx(2.3)

    def test_write_datacard_sr_bin(self, minimal_combine_config, region_root_dir, tmp_path):
        writer = CombineDatacardWriter(minimal_combine_config)
        out_dir = tmp_path / "out"
        datacard_file = writer.write_datacard(
            region_root_dir, str(out_dir), "1b", "SR", "Recoil",
            mass_point="MH3_600_MH4_150_Mchi_1", blind=True, year="2024",
        )
        content = Path(datacard_file).read_text()

        # unbinned mode (default): one Combine channel per region, imax 1 —
        # the shape histogram carries all bins internally, not one imax per bin.
        # observation is always -1 (shape-derived), matching Run2's actual
        # convention exactly — the real-vs-Asimov distinction lives in
        # shapes.root's data_obs content (write_shapes(blind=...)), not here.
        assert "imax 1" in content
        assert "observation          -1.0" in content
        assert "lumi" in content
        assert "btagSF" in content

        # region_root_dir fixture has no *_weight_btagUP/DOWN_log.root files —
        # write_datacard must not claim the shape systematic applies (1.0) when
        # write_shapes() would have nothing to write for it; every process gets
        # "-" instead. Region applicability (gated_by_cut) is a separate, prior
        # gate — this checks the underlying shape file/key actually exists.
        btag_line = next(l for l in content.split("\n") if l.startswith("btagSF"))
        tokens = btag_line.split()[2:]
        assert all(t == "-" for t in tokens)

    def test_sr_rateparam_unmerged_gets_both_per_channel_entries(
            self, minimal_combine_config, region_root_dir, tmp_path):
        """combine_emu: false -> SR gets BOTH per-channel rateParams (each
        independently scales SR's zjets) — user-confirmed: "when we have
        electron channel WCR ... and muon channel Wmunu then both
        rateparameter will contribute to wjet in SR"."""
        minimal_combine_config["combine_emu"] = False
        writer = CombineDatacardWriter(minimal_combine_config)
        out_dir = tmp_path / "out_sr_unmerged"
        datacard_file = writer.write_datacard(
            region_root_dir, str(out_dir), "1b", "SR", "Recoil",
            mass_point="MH3_600_MH4_150_Mchi_1", blind=True, year="2024",
        )
        content = Path(datacard_file).read_text()
        assert "zjets_1b_Zmumu rateParam" in content
        assert "zjets_1b_Zee rateParam" in content
        assert "zjets_1b_Zll rateParam" not in content

    def test_sr_rateparam_merged_gets_one_merged_entry(
            self, minimal_combine_config, region_root_dir, tmp_path):
        """combine_emu: true (default) -> SR gets ONLY the merged rateParam,
        not both per-channel ones — reproduced directly this session:
        listing both unmerged entries for SR double-applied the correction
        even though only one merged CR channel actually exists."""
        writer = CombineDatacardWriter(minimal_combine_config)
        out_dir = tmp_path / "out_sr_merged"
        datacard_file = writer.write_datacard(
            region_root_dir, str(out_dir), "1b", "SR", "Recoil",
            mass_point="MH3_600_MH4_150_Mchi_1", blind=True, year="2024",
        )
        content = Path(datacard_file).read_text()
        assert "zjets_1b_Zll rateParam" in content
        assert "zjets_1b_Zmumu rateParam" not in content
        assert "zjets_1b_Zee rateParam" not in content

    def test_cr_rateparam_merged_substitutes_merged_name(
            self, minimal_combine_config, region_root_dir, tmp_path):
        """The CR file itself (e.g. the merged Zll channel, region_role still
        CR_Zmumu per make_datacard's first-encountered-role dedup) must use
        the MERGED rateParam name too when combine_emu: true, not the stale
        per-channel one — otherwise the merged CR and SR reference two
        DIFFERENT rateParam names for the same physical merged channel,
        breaking the tie (reproduced directly: without this substitution the
        merged CR carried both wjets_1b_Wlnu AND wjets_1b_Wmunu)."""
        writer = CombineDatacardWriter(minimal_combine_config)
        out_dir = tmp_path / "out_cr_merged"
        datacard_file = writer.write_datacard(
            region_root_dir, str(out_dir), "1b", "CR_Zmumu", "Recoil",
            blind=False, year="2024",
        )
        content = Path(datacard_file).read_text()
        assert "zjets_1b_Zll rateParam" in content
        assert "zjets_1b_Zmumu rateParam" not in content

    def test_write_datacard_shape_systematic_present_when_files_exist(
            self, minimal_combine_config, region_root_dir, tmp_path):
        """Shape systematic row must show 1.0 (not '-') for a process/region
        where write_shapes() actually has both Up/Down variant files to copy."""
        d = Path(region_root_dir)
        edges = np.array([0.0, 50.0, 100.0, 150.0], dtype=float)
        for direction in ("UP", "DOWN"):
            _write_region_histogram(
                d / f"hist_1b_SR_Recoil_weight_btag{direction}_log.root",
                {"ttbar": [4.5, 8.5, 6.5], "DYto2LJets": [1.1, 2.1, 1.6]},
                edges=edges,
            )

        writer = CombineDatacardWriter(minimal_combine_config)
        out_dir = tmp_path / "out_with_shapes"
        datacard_file = writer.write_datacard(
            region_root_dir, str(out_dir), "1b", "SR", "Recoil",
            mass_point="MH3_600_MH4_150_Mchi_1", blind=True, year="2024",
        )
        content = Path(datacard_file).read_text()
        btag_line = next(l for l in content.split("\n") if l.startswith("btagSF"))
        tokens = btag_line.split()[2:]
        # process order for SR: signal, ttbar, zjets — ttbar has both files
        # (idx 1), zjets's plot_group_label is DYto2LJets which also has files
        # (idx 2); signal has no shape files written in this fixture (idx 0).
        assert tokens[0] == "-"     # signal: no shape files
        assert tokens[1] == "1.0"  # ttbar: has both Up/Down
        assert tokens[2] == "1.0"  # zjets (DYto2LJets): has both Up/Down

    def test_resolve_systematic_name(self, minimal_combine_config):
        writer = CombineDatacardWriter(minimal_combine_config)
        assert writer._resolve_systematic_name(
            "btagSF", {"name_template": "CMS{year}_eff_b"}, "2024") == "CMS2024_eff_b"
        assert writer._resolve_systematic_name("JES", {}, "2024") == "JES"

    def test_name_template_matches_between_datacard_and_shapes_keys(
            self, minimal_combine_config, region_root_dir, tmp_path):
        """Regression: name_template (e.g. CMS{year}_eff_b, for cross-era
        decorrelation matching Run2's real CMS2016_eff_b/CMS2017_PU/...
        convention) previously only applied to the datacard's systematics ROW
        label — write_shapes still wrote histogram keys using the raw yaml
        key (btagSF), producing a mismatch text2workspace.py can't resolve
        ("$PROCESS_$SYSTEMATIC" -> ttbar_CMS2024_eff_bUp doesn't exist,
        only ttbar_btagSFUp does). Both must derive the Combine-facing name
        from the same _resolve_systematic_name call."""
        d = Path(region_root_dir)
        edges = np.array([0.0, 50.0, 100.0, 150.0], dtype=float)
        for direction in ("UP", "DOWN"):
            _write_region_histogram(
                d / f"hist_1b_SR_Recoil_weight_btag{direction}_log.root",
                {"ttbar": [4.5, 8.5, 6.5]}, edges=edges,
            )

        minimal_combine_config["datacard"]["systematics"]["btagSF"]["name_template"] = "CMS{year}_eff_b"
        writer = CombineDatacardWriter(minimal_combine_config)
        out_dir = tmp_path / "out_templated"

        datacard_file = writer.write_datacard(
            region_root_dir, str(out_dir), "1b", "SR", "Recoil",
            mass_point="MH3_600_MH4_150_Mchi_1", blind=True, year="2024",
        )
        writer.write_shapes(
            region_root_dir, str(out_dir), "1b", "SR", "Recoil",
            mass_point="MH3_600_MH4_150_Mchi_1", blind=True, year="2024",
        )

        content = Path(datacard_file).read_text()
        assert "CMS2024_eff_b" in content
        assert "btagSF" not in content.split("\n")[0]  # not the raw key anywhere as a row label

        with uproot.open(str(out_dir / "shapes_2024_1b_SR_MH3_600_MH4_150_Mchi_1.root")) as f:
            keys = {k.split(";")[0] for k in f.keys()}
        assert "ttbar_CMS2024_eff_bUp" in keys
        assert "ttbar_CMS2024_eff_bDown" in keys
        assert "ttbar_btagSFUp" not in keys

    def test_write_datacard_binned_mode_channel_per_bin(self, minimal_combine_config,
                                                          region_root_dir, tmp_path):
        """binned mode splits each region's histogram into N single-bin
        Combine channels (Run2's actual production convention, verified
        against bbDMlimitmodelrateParam_oneRP/datacards/*/*.txt)."""
        minimal_combine_config["datacard"]["binning_mode"] = "binned"
        writer = CombineDatacardWriter(minimal_combine_config)
        out_dir = tmp_path / "out_binned"
        datacard_file = writer.write_datacard(
            region_root_dir, str(out_dir), "1b", "SR", "Recoil",
            mass_point="MH3_600_MH4_150_Mchi_1", blind=True, year="2024",
        )
        content = Path(datacard_file).read_text()

        assert "imax 3" in content
        assert "SR_bin1" in content
        assert "SR_bin2" in content
        assert "SR_bin3" in content
        assert "SR_bin4" not in content  # region_root_dir fixture has 3 bins only
        # binned mode: observation AND rate are shape-derived (-1) for every
        # channel, unconditionally — matches Run2's exact convention.
        assert "observation          -1.0 -1.0 -1.0" in content
        assert "-1.000000" in content

    def test_write_datacard_binned_mode_rateparam_per_bin_naming(
            self, minimal_combine_config, region_root_dir, tmp_path):
        """Binned mode must float each bin's rateParam INDEPENDENTLY (a
        _binN suffix per bin, e.g. zjets_1b_Zll_bin1, _bin2, _bin3 — 3
        DISTINCT rateParams, not one name repeated with different targets)
        — matches Run2's real binned convention exactly (verified against
        the actual combined-Run2 datacard: ratewjets_1b_2016_bin1..bin4)."""
        minimal_combine_config["datacard"]["binning_mode"] = "binned"
        writer = CombineDatacardWriter(minimal_combine_config)
        out_dir = tmp_path / "out_binned_rateparam"
        datacard_file = writer.write_datacard(
            region_root_dir, str(out_dir), "1b", "CR_Zmumu", "Recoil",
            blind=False, year="2024",
        )
        content = Path(datacard_file).read_text()
        # region_root_dir fixture's Zmumu histogram has 3 bins.
        assert "zjets_1b_Zll_bin1 rateParam Zmumu_bin1" in content
        assert "zjets_1b_Zll_bin2 rateParam Zmumu_bin2" in content
        assert "zjets_1b_Zll_bin3 rateParam Zmumu_bin3" in content

    def test_write_datacard_zcr_excludes_btagsf(self, minimal_combine_config, region_root_dir, tmp_path):
        """Z CR has no b-tag cut — btagSF row must show '-' for every process
        in that bin (gated_by_cut derived from configs/regions.yaml)."""
        writer = CombineDatacardWriter(minimal_combine_config)
        out_dir = tmp_path / "out_zcr"
        datacard_file = writer.write_datacard(
            region_root_dir, str(out_dir), "1b", "CR_Zmumu", "Recoil",
            blind=False, year="2024",
        )
        content = Path(datacard_file).read_text()
        btag_lines = [l for l in content.split("\n") if l.startswith("btagSF")]
        assert len(btag_lines) == 1
        assert "-" in btag_lines[0]
        # No numeric (non "-") systematic value tokens on the btagSF row
        tokens = btag_lines[0].split()[2:]
        assert all(t == "-" for t in tokens)

        # CR bins have no mass_point (background-only) — the filename must
        # drop the {mass_point} token cleanly, not duplicate {region}
        # (regression: datacard_2024_1b_CR_Zmumu_CR_Zmumu.txt).
        assert Path(datacard_file).name == "datacard_2024_1b_CR_Zmumu.txt"
        assert "CR_Zmumu_CR_Zmumu" not in Path(datacard_file).name

        # Regression: kmax must count only lnN/shape systematics (2: lumi,
        # btagSF), NOT the zjets_1b_Zmumu rateParam applicable to this region.
        # Combine's own parser (HiggsAnalysis.CombinedLimit.DatacardParser)
        # tracks rateParam separately from systematics and rejects a datacard
        # whose declared kmax includes it ("Found N systematics, expected N+1")
        # — reproduced against every real CR datacard before this fix.
        kmax_line = next(l for l in content.split("\n") if l.startswith("kmax"))
        assert kmax_line.split()[1] == "2", f"kmax should count only systematics (2), got: {kmax_line}"

    def test_write_datacard_region_dir_override_reads_merged_histogram(
            self, minimal_combine_config, region_root_dir, tmp_path):
        """combine_emu's merge_emu writes merged e/mu CR histograms under a
        different region_dir ("Wlnu") than the per-channel region_role
        ("CR_Wmunu") that combine.yaml's control_regions list still uses.
        write_datacard must read from region_dir_override when given, not
        region_dir_from_role(region_role) — verified by making "Wmunu" not
        exist at all in the fixture dir, so the test would raise
        FileNotFoundError if the override weren't actually being used."""
        _write_region_histogram(Path(region_root_dir) / "hist_1b_Wlnu_Recoil_log.root", {
            "TotalBkg": [3.0, 4.0, 5.0],
            "ttbar": [1.0, 1.0, 1.0],
            "DYto2LJets": [0.0, 0.0, 0.0],
            "data_obs": [3.0, 5.0, 4.0],
        })
        writer = CombineDatacardWriter(minimal_combine_config)
        out_dir = tmp_path / "out_wlnu"
        datacard_file = writer.write_datacard(
            region_root_dir, str(out_dir), "1b", "CR_Wmunu", "Recoil",
            blind=False, year="2024",
            region_dir_override="Wlnu", filename_region="Wlnu",
        )
        writer.write_shapes(
            region_root_dir, str(out_dir), "1b", "CR_Wmunu", "Recoil",
            blind=False, region_dir_override="Wlnu", filename_region="Wlnu",
        )
        content = Path(datacard_file).read_text()

        # Bin name derives from region_dir (override), not region_role.
        assert "Wlnu_1b" in content
        assert "Wmunu_1b" not in content
        # filename_region controls the output filename, independent of
        # region_role (which stays "CR_Wmunu" internally for gated_by_cut /
        # rate_parameter matching against regions.yaml / combine.yaml).
        assert Path(datacard_file).name == "datacard_2024_1b_Wlnu.txt"

    def test_merged_region_dir_for_role(self):
        assert merged_region_dir_for_role("1b", "CR_Wmunu") == "Wlnu"
        assert merged_region_dir_for_role("1b", "CR_Wenu") == "Wlnu"
        assert merged_region_dir_for_role("1b", "CR_Zmumu") == "Zll"
        assert merged_region_dir_for_role("2b", "CR_Topmunu") == "Topl"
        assert merged_region_dir_for_role("1b", "SR") is None

    def test_write_datacard_with_rateparam_parses_via_combine_datacard_parser(
            self, minimal_combine_config, region_root_dir, tmp_path):
        """End-to-end regression for the kmax/rateParam bug: a datacard for a
        region with an applicable rateParam must actually parse via Combine's
        own DatacardParser, not just look right by eye. This is the real
        validation combineCards.py/combine run at fit time — a wrong kmax
        raises "Found N systematics, expected N+1" here exactly as it did
        against the real generated CR datacards."""
        HiggsAnalysis = pytest.importorskip("HiggsAnalysis.CombinedLimit.DatacardParser")

        writer = CombineDatacardWriter(minimal_combine_config)
        out_dir = tmp_path / "out_parse"
        datacard_file = writer.write_datacard(
            region_root_dir, str(out_dir), "1b", "CR_Zmumu", "Recoil",
            blind=False, year="2024",
        )
        # write_datacard's shapes-file reference must resolve — write_shapes
        # writes it into the same out_dir, matching how the CLI always calls both.
        writer.write_shapes(region_root_dir, str(out_dir), "1b", "CR_Zmumu", "Recoil",
                             blind=False)

        class _Options:
            fileName = datacard_file
            stat = False
            bin = True  # "shapes" directives require binary/shape mode
            nuisancesToExclude = []
            noJMax = False
            allowNoSignal = True
            allowNoBackground = True
            evaluateEdits = True

        with open(datacard_file) as f:
            HiggsAnalysis.parseCard(f, _Options())

    def test_write_shapes(self, minimal_combine_config, region_root_dir, tmp_path):
        writer = CombineDatacardWriter(minimal_combine_config)
        out_dir = tmp_path / "shapes_out"
        shapes_file = writer.write_shapes(
            region_root_dir, str(out_dir), "1b", "SR", "Recoil",
            mass_point="MH3_600_MH4_150_Mchi_1", blind=True,
        )
        assert Path(shapes_file).exists()

        with uproot.open(shapes_file) as f:
            keys = {k.split(";")[0] for k in f.keys()}
        assert "ttbar" in keys
        assert "signal" in keys
        # datacard's observation is always -1 (shape-derived) -> shapes.root
        # MUST contain a data_obs key or text2workspace.py/combine will fail.
        assert "data_obs" in keys

        ttbar_values, _ = uproot.open(shapes_file)["ttbar"].to_numpy()
        np.testing.assert_array_equal(ttbar_values, [4.0, 8.0, 6.0])

    def test_write_shapes_floors_exactly_zero_bins(self, minimal_combine_config, tmp_path):
        """Regression: a process with exactly zero yield in a bin crashes
        Combine's binned mode ("Null norm for channel X, process Y" —
        reproduced against real combine 10.6.1: singletop had 0 events in
        every bin of a real Z CR, a physically plausible low-yield
        background). write_shapes must floor exactly-zero bins to
        ZERO_FLOOR rather than writing a true zero."""
        region_root_dir = tmp_path / "region_root"
        _write_region_histogram(
            region_root_dir / "hist_1b_SR_Recoil_log.root",
            {
                "ttbar": [0.0, 5.0, 0.0], "DYto2LJets": [1.0, 2.0, 1.5],
                "TotalBkg": [10.0, 20.0, 15.0],
                "sig_MH3_600_MH4_150_Mchi_1": [0.5, 1.0, 0.8],
            },
        )
        writer = CombineDatacardWriter(minimal_combine_config)
        shapes_file = writer.write_shapes(
            str(region_root_dir), str(tmp_path / "out"), "1b", "SR", "Recoil",
            mass_point="MH3_600_MH4_150_Mchi_1", blind=True,
        )
        ttbar_values, _ = uproot.open(shapes_file)["ttbar"].to_numpy()
        assert ttbar_values[0] > 0
        assert ttbar_values[2] > 0
        assert ttbar_values[0] < 1e-3  # floored, not a real yield
        assert ttbar_values[1] == 5.0  # untouched — already nonzero

    def test_write_shapes_blind_uses_totalbkg_as_data_obs(self, minimal_combine_config,
                                                            region_root_dir, tmp_path):
        writer = CombineDatacardWriter(minimal_combine_config)
        out_dir = tmp_path / "shapes_blind"
        shapes_file = writer.write_shapes(
            region_root_dir, str(out_dir), "1b", "SR", "Recoil",
            mass_point="MH3_600_MH4_150_Mchi_1", blind=True,
        )
        data_obs_values, _ = uproot.open(shapes_file)["data_obs"].to_numpy()
        np.testing.assert_array_equal(data_obs_values, [10.0, 20.0, 15.0])  # region_root_dir's TotalBkg

    def test_write_shapes_unblind_missing_data_raises(self, minimal_combine_config,
                                                        region_root_dir, tmp_path):
        writer = CombineDatacardWriter(minimal_combine_config)
        out_dir = tmp_path / "shapes_unblind_missing"
        with pytest.raises(KeyError):
            writer.write_shapes(
                region_root_dir, str(out_dir), "1b", "SR", "Recoil",
                mass_point="MH3_600_MH4_150_Mchi_1", blind=False,
            )  # region_root_dir's SR fixture has no data_obs key

    def test_create_workspace_missing_binary_raises_clear_error(self, minimal_combine_config, tmp_path):
        writer = CombineDatacardWriter(minimal_combine_config)
        datacard_file = tmp_path / "datacard.txt"
        datacard_file.write_text("imax 1\njmax 2\nkmax 1\n")

        writer.advanced_config["combine_commands"]["text2workspace"] = "definitely-not-a-real-binary"
        with pytest.raises(RuntimeError, match="not found on PATH"):
            writer.create_workspace(str(datacard_file), str(tmp_path))

    def test_create_workspace_mocked_subprocess(self, minimal_combine_config, tmp_path):
        writer = CombineDatacardWriter(minimal_combine_config)
        datacard_file = tmp_path / "datacard.txt"
        datacard_file.write_text("imax 1\njmax 2\nkmax 1\n")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            workspace_file = writer.create_workspace(str(datacard_file), str(tmp_path))

        assert Path(workspace_file).name == "workspace.root"
        called_args = mock_run.call_args[0][0]
        assert "--channel-masks" in called_args
        # text2workspace.py is run with cwd=output_dir (production bug: without
        # this, relative `shapes *` filenames in the datacard can't be found)
        # -> the datacard/workspace args passed to it must be basenames only,
        # not paths that would double up with cwd.
        assert called_args[1] == "datacard.txt"
        assert mock_run.call_args.kwargs.get("cwd") == str(tmp_path)

    def test_merge_categories(self, minimal_combine_config, tmp_path):
        writer = CombineDatacardWriter(minimal_combine_config)
        mass_point = "MH3_600_MH4_150_Mchi_1"
        for cat in ("1b", "2b"):
            d = tmp_path / "input" / cat / mass_point
            d.mkdir(parents=True)
            (d / "datacard.txt").write_text(f"imax 1\njmax 2\nkmax 1\n# {cat}\n")

        fake_merged_output = f"imax 2\njmax 4\nkmax 1\n"
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=fake_merged_output, stderr="")
            merged = writer.merge_categories(str(tmp_path / "input"), str(tmp_path / "output"), mass_point)

        assert Path(merged).exists()
        called_args = mock_run.call_args[0][0]
        assert called_args[0] == "combineCards.py"
        assert any(a.startswith("cat_1b=") for a in called_args)
        assert any(a.startswith("cat_2b=") for a in called_args)

    def test_merge_categories_missing_card_raises(self, minimal_combine_config, tmp_path):
        writer = CombineDatacardWriter(minimal_combine_config)
        with pytest.raises(FileNotFoundError):
            writer.merge_categories(str(tmp_path / "nonexistent"), str(tmp_path / "output"), "some_point")

    def test_read_bin_names_single(self, minimal_combine_config, tmp_path):
        card = tmp_path / "datacard.txt"
        card.write_text("imax 1\njmax 2\nkmax 1\n\nbin          Wlnu_1b\nobservation  -1\n")
        assert CombineDatacardWriter._read_bin_names(card) == ["Wlnu_1b"]

    def test_read_bin_names_multi(self, minimal_combine_config, tmp_path):
        card = tmp_path / "datacard.txt"
        card.write_text(
            "imax 4\njmax 2\nkmax 1\n\n"
            "bin          SR_bin1  SR_bin2  SR_bin3  SR_bin4\n"
            "observation  -1       -1       -1       -1\n"
        )
        assert CombineDatacardWriter._read_bin_names(card) == ["SR_bin1", "SR_bin2", "SR_bin3", "SR_bin4"]

    def test_merge_region(self, minimal_combine_config, tmp_path):
        """SR + CR datacards must merge into one card via combineCards.py,
        each input labeled with its OWN bin name (extracted from its "bin"
        line) rather than combineCards.py's default ch1/ch2/ch3 renaming —
        verified this session: an unlabeled positional-args combineCards.py
        call discards the original bin names entirely, which would break
        GoF/pulls channel-masking (mask_SR_1b) downstream."""
        writer = CombineDatacardWriter(minimal_combine_config)
        mass_point = "MH3_600_MH4_150_Mchi_1"

        sr_dir = tmp_path / "input" / "1b" / mass_point
        sr_dir.mkdir(parents=True)
        (sr_dir / "datacard_sr.txt").write_text(
            "imax 1\njmax 2\nkmax 1\n\nbin          SR_1b\nobservation  -1\n")

        for cr_dir_name, bin_name in (("Wlnu", "Wlnu_1b"), ("Zll", "Zll_1b")):
            cr_dir = tmp_path / "input" / "1b" / cr_dir_name
            cr_dir.mkdir(parents=True)
            (cr_dir / "datacard_cr.txt").write_text(
                f"imax 1\njmax 1\nkmax 1\n\nbin          {bin_name}\nobservation  -1\n")

        fake_merged_output = "imax 3\njmax 4\nkmax 1\n"
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=fake_merged_output, stderr="")
            merged = writer.merge_region(
                str(tmp_path / "input"), str(tmp_path / "output"), "1b", mass_point,
                ["Wlnu", "Zll"],
            )

        assert Path(merged).exists()
        called_args = mock_run.call_args[0][0]
        assert called_args[0] == "combineCards.py"
        # Each card must be labeled with its OWN existing bin name, not a
        # generic cat_1b=/cat_2b=-style prefix (SR/CR bins are already unique
        # per category, so no disambiguating prefix is needed — only
        # preservation of the original name).
        assert any(a == "SR_1b=" + str(sr_dir / "datacard_sr.txt") for a in called_args)
        assert any(a.startswith("Wlnu_1b=") for a in called_args)
        assert any(a.startswith("Zll_1b=") for a in called_args)

    def test_merge_region_missing_sr_raises(self, minimal_combine_config, tmp_path):
        writer = CombineDatacardWriter(minimal_combine_config)
        with pytest.raises(FileNotFoundError):
            writer.merge_region(str(tmp_path / "nonexistent"), str(tmp_path / "output"),
                                 "1b", "some_point", ["Wlnu", "Zll"])

    def test_merge_region_missing_cr_raises(self, minimal_combine_config, tmp_path):
        writer = CombineDatacardWriter(minimal_combine_config)
        mass_point = "MH3_600_MH4_150_Mchi_1"
        sr_dir = tmp_path / "input" / "1b" / mass_point
        sr_dir.mkdir(parents=True)
        (sr_dir / "datacard_sr.txt").write_text(
            "imax 1\njmax 2\nkmax 1\n\nbin          SR_1b\nobservation  -1\n")

        with pytest.raises(FileNotFoundError):
            writer.merge_region(str(tmp_path / "input"), str(tmp_path / "output"),
                                 "1b", mass_point, ["Wlnu", "Zll"])

    def test_merge_eras(self, minimal_combine_config, tmp_path):
        writer = CombineDatacardWriter(minimal_combine_config)
        mass_point = "MH3_600_MH4_150_Mchi_1"
        d = tmp_path / "input" / "2024" / "C" / mass_point
        d.mkdir(parents=True)
        (d / "datacard.txt").write_text("imax 1\njmax 2\nkmax 1\n")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="imax 1\njmax 2\nkmax 1\n", stderr="")
            merged = writer.merge_eras(str(tmp_path / "input"), str(tmp_path / "output"), mass_point, ["2024"])

        assert Path(merged).exists()
        called_args = mock_run.call_args[0][0]
        assert any(a.startswith("era_2024=") for a in called_args)


class TestCombineRunner:
    """Test CombineRunner's template-driven command building."""

    def test_run_asymptotic_limits_blind(self, minimal_combine_config, tmp_path):
        runner = CombineRunner(minimal_combine_config)
        workspace = tmp_path / "workspace.root"
        workspace.write_text("")

        def fake_run(cmd, **kwargs):
            (tmp_path / "higgsCombineTest.AsymptoticLimits.mH120.root").write_text("")
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("subprocess.run", side_effect=fake_run) as mock_run:
            results_file = runner.run_asymptotic_limits(str(workspace), str(tmp_path), blind=True)

        # combine's real output filename (its default -n Test label), not a
        # made-up "asymptotic_limits.root" that combine never actually writes.
        assert Path(results_file).name == "higgsCombineTest.AsymptoticLimits.mH120.root"
        # blind=True -> single call, includes -t -1, no second "observed" call
        assert mock_run.call_count == 1
        called_args = mock_run.call_args[0][0]
        assert "-t" in called_args and "-1" in called_args

    def test_run_asymptotic_limits_unblind_runs_expected_and_observed(self, minimal_combine_config, tmp_path):
        runner = CombineRunner(minimal_combine_config)
        workspace = tmp_path / "workspace.root"
        workspace.write_text("")

        def fake_run(cmd, **kwargs):
            (tmp_path / "higgsCombineTest.AsymptoticLimits.mH120.root").write_text("")
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("subprocess.run", side_effect=fake_run) as mock_run:
            runner.run_asymptotic_limits(str(workspace), str(tmp_path), blind=False)

        assert mock_run.call_count == 2
        first_args = mock_run.call_args_list[0][0][0]
        second_args = mock_run.call_args_list[1][0][0]
        assert "-t" not in first_args
        assert "--run=observed" in second_args

    def test_run_goodness_of_fit_two_steps_no_toys_conflict(self, minimal_combine_config, tmp_path):
        """Regression: GoF must NOT combine -t -1 (blind_args) with --toys — that
        double-specifies toy generation and combine rejects it ("--toys cannot
        be specified more than once"). GoF runs as 2 separate combine calls
        (observed, then background-only toys), matching Run2's actual script."""
        runner = CombineRunner(minimal_combine_config)
        workspace = tmp_path / "workspace.root"
        workspace.write_text("")
        datacard = tmp_path / "datacard.txt"
        datacard.write_text("imax 1\njmax 2\nkmax 1\n\nbin          SR_1b\nobservation  -1\n")

        def fake_run(cmd, **kwargs):
            (tmp_path / "higgsCombineObserved.GoodnessOfFit.mH120.root").write_text("")
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("subprocess.run", side_effect=fake_run) as mock_run:
            runner.run_goodness_of_fit(str(workspace), str(tmp_path), str(datacard), blind=True)

        assert mock_run.call_count == 2
        observed_args, toys_args = (c[0][0] for c in mock_run.call_args_list)
        assert "-t" not in observed_args
        assert "-n" in observed_args and "Observed" in observed_args
        assert "-t" in toys_args and "--toysFrequentist" in toys_args
        assert "-n" in toys_args and "Toys" in toys_args
        # No call anywhere uses a separate --toys= flag on top of -t
        # (that combination is exactly what combine used to reject)
        assert not any(a.startswith("--toys=") for a in observed_args)
        assert not any(a.startswith("--toys=") for a in toys_args)
        # Channel-masking: nuisances fit with SR masked, eval unmasked, r
        # frozen at 0 — NOT a plain unmasked fit (Run2's real GoF pattern).
        assert "--setParametersForFit" in observed_args
        assert "mask_SR_1b=1" in observed_args
        assert "--setParametersForEval" in observed_args
        assert "mask_SR_1b=0" in observed_args
        assert "r=0,mask_SR_1b=1" in toys_args

    def test_parse_goodness_of_fit_computes_pvalue(self, tmp_path):
        observed_file = tmp_path / "higgsCombineObserved.GoodnessOfFit.mH120.root"
        toys_file = tmp_path / "higgsCombineToys.GoodnessOfFit.mH120.root"
        with uproot.recreate(str(observed_file)) as f:
            f["limit"] = {"limit": np.array([15.0])}
        with uproot.recreate(str(toys_file)) as f:
            f["limit"] = {"limit": np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0])}

        runner = CombineRunner({"fit": {}, "output": {}, "advanced": {}})
        results = runner._parse_goodness_of_fit(str(observed_file))

        assert results["observed"] == pytest.approx(15.0)
        assert results["toys"] is not None
        # 3 of 6 toys (14,16,18,20... wait: values >= 15.0 -> 16,18,20 = 3/6) = 0.5
        assert results["p_value"] == pytest.approx(0.5)

    def test_parse_goodness_of_fit_no_toys_file_gives_none_pvalue(self, tmp_path):
        observed_file = tmp_path / "higgsCombineObserved.GoodnessOfFit.mH120.root"
        with uproot.recreate(str(observed_file)) as f:
            f["limit"] = {"limit": np.array([15.0])}

        runner = CombineRunner({"fit": {}, "output": {}, "advanced": {}})
        results = runner._parse_goodness_of_fit(str(observed_file))

        assert results["observed"] == pytest.approx(15.0)
        assert results["toys"] is None
        assert results["p_value"] is None

    def test_parse_goodness_of_fit_seed_suffixed_filename(self, tmp_path):
        """Real combine appends the --seed value to its output filename
        (higgsCombineToys.GoodnessOfFit.mH120.12345.root, not a fixed
        mH120.root) whenever --seed is passed — must be found via glob,
        not an exact-name lookup (regression from a real production run)."""
        observed_file = tmp_path / "higgsCombineObserved.GoodnessOfFit.mH120.root"
        toys_file = tmp_path / "higgsCombineToys.GoodnessOfFit.mH120.12345.root"
        with uproot.recreate(str(observed_file)) as f:
            f["limit"] = {"limit": np.array([15.0])}
        with uproot.recreate(str(toys_file)) as f:
            f["limit"] = {"limit": np.array([10.0, 20.0])}

        runner = CombineRunner({"fit": {}, "output": {}, "advanced": {}})
        results = runner._parse_goodness_of_fit(str(observed_file))

        assert results["toys"] is not None
        assert results["p_value"] == pytest.approx(0.5)

    def test_run_goodness_of_fit_finds_seed_suffixed_observed_file(self, minimal_combine_config, tmp_path):
        """run_goodness_of_fit's return value must glob for the observed
        output too, since real combine may seed-suffix that file as well."""
        runner = CombineRunner(minimal_combine_config)
        workspace = tmp_path / "workspace.root"
        workspace.write_text("")
        datacard = tmp_path / "datacard.txt"
        datacard.write_text("imax 1\njmax 2\nkmax 1\n\nbin          SR_1b\nobservation  -1\n")

        def fake_run(cmd, **kwargs):
            # simulate combine writing a seed-suffixed observed file
            (tmp_path / "higgsCombineObserved.GoodnessOfFit.mH120.99.root").write_text("")
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("subprocess.run", side_effect=fake_run):
            results_file = runner.run_goodness_of_fit(str(workspace), str(tmp_path), str(datacard), blind=True)

        assert Path(results_file).name == "higgsCombineObserved.GoodnessOfFit.mH120.99.root"

    def test_sr_bin_names_single_card(self, tmp_path):
        datacard = tmp_path / "datacard.txt"
        datacard.write_text("imax 1\njmax 2\nkmax 1\n\nbin          SR_1b\nobservation  -1\n")
        assert CombineRunner._sr_bin_names(str(datacard)) == ["SR_1b"]

    def test_sr_bin_names_region_merged_card(self, tmp_path):
        """SR + CR bins in one card (merge_region output) — only the SR bin
        should match, not Wlnu_1b/Zll_1b."""
        datacard = tmp_path / "datacard.txt"
        datacard.write_text(
            "imax 3\njmax 7\nkmax 1\n\n"
            "bin          SR_1b    Wlnu_1b  Zll_1b\nobservation  -1       -1       -1\n"
        )
        assert CombineRunner._sr_bin_names(str(datacard)) == ["SR_1b"]

    def test_sr_bin_names_category_merged_card(self, tmp_path):
        """After merge_categories, bins are cat_1b_SR_1b/cat_2b_SR_2b etc —
        both SR bins (one per category) must match, no CR bins."""
        datacard = tmp_path / "datacard.txt"
        datacard.write_text(
            "imax 6\njmax 7\nkmax 1\n\n"
            "bin          cat_1b_SR_1b    cat_1b_Wlnu_1b  cat_1b_Zll_1b   "
            "cat_2b_SR_2b    cat_2b_Topl_2b  cat_2b_Zll_2b\n"
            "observation  -1              -1              -1              -1              -1              -1\n"
        )
        assert CombineRunner._sr_bin_names(str(datacard)) == ["cat_1b_SR_1b", "cat_2b_SR_2b"]

    def test_run_collect_goodness_of_fit(self, minimal_combine_config, tmp_path):
        runner = CombineRunner(minimal_combine_config)
        observed = tmp_path / "higgsCombineObserved.GoodnessOfFit.mH120.root"
        observed.write_text("")
        toys = tmp_path / "higgsCombineToys.GoodnessOfFit.mH120.12345.root"
        toys.write_text("")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            output_json = runner.run_collect_goodness_of_fit(str(observed), str(tmp_path))

        assert Path(output_json).name == "gof.json"
        called_args = mock_run.call_args[0][0]
        assert called_args[0] == "combineTool.py"
        assert "CollectGoodnessOfFit" in called_args
        assert "higgsCombineObserved.GoodnessOfFit.mH120.root" in called_args
        assert "higgsCombineToys.GoodnessOfFit.mH120.12345.root" in called_args

    def test_run_collect_goodness_of_fit_missing_toys_raises(self, minimal_combine_config, tmp_path):
        runner = CombineRunner(minimal_combine_config)
        observed = tmp_path / "higgsCombineObserved.GoodnessOfFit.mH120.root"
        observed.write_text("")

        with pytest.raises(FileNotFoundError):
            runner.run_collect_goodness_of_fit(str(observed), str(tmp_path))

    def test_run_plot_gof(self, minimal_combine_config, tmp_path):
        runner = CombineRunner(minimal_combine_config)
        gof_json = tmp_path / "gof.json"
        gof_json.write_text(json.dumps({
            "120.0": {"obs": [95.0], "p": 0.5, "toy": [80.0 + i * 0.5 for i in range(40)]},
        }))

        plot_file = runner.run_plot_gof(str(gof_json), str(tmp_path), algo="saturated")

        assert Path(plot_file).name == "gof.pdf"
        assert Path(plot_file).exists()
        assert (tmp_path / "gof.png").exists()

    def test_run_plot_gof_embeds_mass_point_in_filename(self, minimal_combine_config, tmp_path):
        runner = CombineRunner(minimal_combine_config)
        gof_json = tmp_path / "gof.json"
        gof_json.write_text(json.dumps({
            "120.0": {"obs": [95.0], "p": 0.5, "toy": [80.0 + i * 0.5 for i in range(40)]},
        }))

        plot_file = runner.run_plot_gof(str(gof_json), str(tmp_path), algo="saturated",
                                         mass_point="MH3_600_MH4_150_Mchi_1")

        assert Path(plot_file).name == "gof_MH3_600_MH4_150_Mchi_1.pdf"
        assert Path(plot_file).exists()
        assert (tmp_path / "gof_MH3_600_MH4_150_Mchi_1.png").exists()

    def test_run_plot_gof_handles_observed_out_of_range(self, minimal_combine_config, tmp_path):
        runner = CombineRunner(minimal_combine_config)
        gof_json = tmp_path / "gof.json"
        gof_json.write_text(json.dumps({
            "120.0": {"obs": [1e6], "p": 0.0, "toy": [80.0 + i * 0.5 for i in range(40)]},
        }))

        plot_file = runner.run_plot_gof(str(gof_json), str(tmp_path), algo="saturated")
        assert Path(plot_file).exists()

    def test_resolve_lumi_text(self, minimal_combine_config, tmp_path):
        year_config = tmp_path / "2024.yaml"
        year_config.write_text("lumi: 109.82\n")
        minimal_combine_config["eras"][0]["year_config"] = str(year_config)
        runner = CombineRunner(minimal_combine_config)
        assert runner._resolve_lumi_text("2024") == "109.82 fb^{-1} (2024)"

    def test_resolve_lumi_text_missing_year_returns_empty(self, minimal_combine_config):
        runner = CombineRunner(minimal_combine_config)
        assert runner._resolve_lumi_text("1999") == ""

    def test_run_pulls_cronly_masks_sr_and_skips_diffnuisances(self, minimal_combine_config, tmp_path):
        """CRonly mode must mask SR via --setParameters mask_<bin>=1, derived
        from the datacard's own bin line — same mechanism as GoodnessOfFit.
        It must NOT call diffNuisances.py (which requires a fit_s
        RooFitResult that doesn't exist when SR is masked — reproduced
        against real combine 10.6.1, "does not contain the output of the
        signal fit 'fit_s'") — it reads fit_b directly via
        plotPostNuisance_combine.C instead (only 2 subprocess calls: combine,
        then root — no diffNuisances.py in between)."""
        runner = CombineRunner(minimal_combine_config)
        workspace = tmp_path / "workspace.root"
        workspace.write_text("")
        datacard = tmp_path / "datacard.txt"
        datacard.write_text("imax 1\njmax 2\nkmax 1\n\nbin          SR_1b\nobservation  -1\n")

        def fake_run(cmd, **kwargs):
            if cmd[0] == "combine":
                (tmp_path / "fitDiagnostics_CRonly.root").write_text("")
            elif cmd[0] == "root":
                (tmp_path / "pulls_C_2024_CRonly_1.pdf").write_text("")
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("subprocess.run", side_effect=fake_run) as mock_run:
            plot = runner.run_pulls(str(workspace), str(tmp_path), str(datacard),
                                     mode="CRonly", year="2024")

        assert mock_run.call_count == 2
        combine_call = mock_run.call_args_list[0][0][0]
        assert "--setParameters" in combine_call
        assert "mask_SR_1b=1" in combine_call
        assert not any(c[0][0][0] == "diffNuisances.py" for c in mock_run.call_args_list)

        # Persisted per-mode copy under fitDiagnosticsDir/, matching Run2's
        # --out fitDiagnosticsDir convention — not overwritten by other modes.
        assert (tmp_path / "fitDiagnosticsDir" / "fitDiagnostics_C_2024_CRonly.root").exists()

        root_call = mock_run.call_args_list[1][0][0]
        assert root_call[0] == "root"
        assert "plotPostNuisance_combine.C" in root_call[-1]

        assert Path(plot).name == "pulls_C_2024_CRonly_1.pdf"

    def test_run_pulls_sb_t0_uses_diffnuisances(self, minimal_combine_config, tmp_path):
        """Unlike CRonly, the other 3 pulls modes (asimov_t0/sb_t0/sb_t1) run
        unmasked and DO go through diffNuisances.py + PlotPulls.C — verified
        against Run2's real pulls_oneRP.sh, which only special-cases CRonly."""
        runner = CombineRunner(minimal_combine_config)
        workspace = tmp_path / "workspace.root"
        workspace.write_text("")
        datacard = tmp_path / "datacard.txt"
        datacard.write_text("imax 1\njmax 2\nkmax 1\n\nbin          SR_1b\nobservation  -1\n")

        def fake_run(cmd, **kwargs):
            if cmd[0] == "combine":
                (tmp_path / "fitDiagnostics_SbT0.root").write_text("")
            elif cmd[0] == "diffNuisances.py":
                Path(cmd[-1]).write_text("")
            elif cmd[0] == "root":
                (tmp_path / "pulls_C_2024_sb_t0_1_.pdf").write_text("")
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("subprocess.run", side_effect=fake_run) as mock_run:
            plot = runner.run_pulls(str(workspace), str(tmp_path), str(datacard),
                                     mode="sb_t0", year="2024")

        assert mock_run.call_count == 3
        diff_nuisances_call = mock_run.call_args_list[1][0][0]
        assert diff_nuisances_call[0] == "diffNuisances.py"
        assert "--abs" in diff_nuisances_call and "--all" in diff_nuisances_call

        root_call = mock_run.call_args_list[2][0][0]
        assert root_call[0] == "root"
        assert "PlotPulls.C" in root_call[-1]

        assert Path(plot).name == "pulls_C_2024_sb_t0_1_.pdf"

    def test_run_pulls_mass_point_added_not_replacing_catg_year_mode(self, minimal_combine_config, tmp_path):
        """mass_point is added alongside catg/year/mode, not replacing them —
        matches Run2's real pulls_${catg}_${year}_${mode}_${dirname}_{page}_.pdf
        naming (pulls_oneRP.sh) exactly, with mass_point filling ${dirname}."""
        runner = CombineRunner(minimal_combine_config)
        workspace = tmp_path / "workspace.root"
        workspace.write_text("")
        datacard = tmp_path / "datacard.txt"
        datacard.write_text("imax 1\njmax 2\nkmax 1\n\nbin          SR_1b\nobservation  -1\n")

        def fake_run(cmd, **kwargs):
            if cmd[0] == "combine":
                (tmp_path / "fitDiagnostics_SbT1.root").write_text("")
            elif cmd[0] == "diffNuisances.py":
                Path(cmd[-1]).write_text("")
            elif cmd[0] == "root":
                (tmp_path / "pulls_C_2024_sb_t1_MH3_600_MH4_150_Mchi_1_1_.pdf").write_text("")
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("subprocess.run", side_effect=fake_run):
            plot = runner.run_pulls(str(workspace), str(tmp_path), str(datacard),
                                     mode="sb_t1", year="2024",
                                     mass_point="MH3_600_MH4_150_Mchi_1", category="C")

        assert Path(plot).name == "pulls_C_2024_sb_t1_MH3_600_MH4_150_Mchi_1_1_.pdf"
        assert (tmp_path / "fitDiagnosticsDir" /
                "fitDiagnostics_C_2024_sb_t1_MH3_600_MH4_150_Mchi_1.root").exists()

    def test_run_pulls_unknown_mode_raises(self, minimal_combine_config, tmp_path):
        runner = CombineRunner(minimal_combine_config)
        with pytest.raises(ValueError, match="Unknown pulls mode"):
            runner.run_pulls("ws.root", str(tmp_path), "datacard.txt", mode="nonexistent")

    def test_run_impacts_three_steps(self, minimal_combine_config, tmp_path):
        runner = CombineRunner(minimal_combine_config)
        workspace = tmp_path / "workspace.root"
        workspace.write_text("")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            runner.run_impacts(str(workspace), str(tmp_path), blind=True)

        assert mock_run.call_count == 3
        step_flags = [call[0][0] for call in mock_run.call_args_list]
        assert any("--doInitialFit" in args for args in step_flags)
        assert any("--doFits" in args for args in step_flags)
        assert any("-o" in args for args in step_flags)

        # general_args (e.g. --cminDefaultMinimizerStrategy=0) must not also be
        # hardcoded in a step's own args — combineTool.py's --parallel dispatch
        # mis-resolves cwd for its post-doFits file lookup when a flag is
        # duplicated, causing a spurious FileNotFoundError on the initial-fit
        # results file (reproduced against real combineTool.py 10.6.1).
        for args in step_flags:
            assert len(args) == len(set(args)), f"duplicate flags in {args}"

        # -o's value must be a bare basename (cwd=output_dir already), not a
        # path re-including output_dir — that double-nests the write target.
        for args in step_flags:
            if "-o" in args:
                o_value = args[args.index("-o") + 1]
                assert "/" not in o_value, f"-o value should be a basename, got {o_value}"

    def test_run_impacts_mass_point_added_not_replacing_plain_name(self, minimal_combine_config, tmp_path):
        """mass_point is appended to impacts.json (impacts_{mass_point}.json),
        added alongside the plain name, not replacing it — same convention as
        run_plot_impacts/run_pulls/run_collect_goodness_of_fit. The returned
        path and the actual -o value passed to combineTool.py must agree."""
        runner = CombineRunner(minimal_combine_config)
        workspace = tmp_path / "workspace.root"
        workspace.write_text("")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            results_file = runner.run_impacts(str(workspace), str(tmp_path), blind=True,
                                               mass_point="MH3_600_MH4_150_Mchi_1")

        assert Path(results_file).name == "impacts_MH3_600_MH4_150_Mchi_1.json"
        step_flags = [call[0][0] for call in mock_run.call_args_list]
        o_step = next(args for args in step_flags if "-o" in args)
        o_value = o_step[o_step.index("-o") + 1]
        assert o_value == "impacts_MH3_600_MH4_150_Mchi_1.json"

    def test_run_plot_impacts(self, minimal_combine_config, tmp_path):
        runner = CombineRunner(minimal_combine_config)
        impacts_json = tmp_path / "impacts.json"
        impacts_json.write_text("{}")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            plot_file = runner.run_plot_impacts(str(impacts_json), str(tmp_path))

        assert Path(plot_file).name == "impacts_C_asimov_t0.pdf"
        called_args = mock_run.call_args[0][0]
        assert called_args[0] == "plotImpacts.py"
        assert "-i" in called_args and str(impacts_json) in called_args

    def test_run_plot_impacts_mass_point_and_unblind(self, minimal_combine_config, tmp_path):
        """mass_point is added alongside catg/mode, not replacing them —
        matches Run2's real impacts_${catg}_${mode}_${dirname}.pdf naming
        (impacts.sh) exactly, with mass_point filling the ${dirname} slot."""
        runner = CombineRunner(minimal_combine_config)
        impacts_json = tmp_path / "impacts.json"
        impacts_json.write_text("{}")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            plot_file = runner.run_plot_impacts(
                str(impacts_json), str(tmp_path),
                mass_point="MH3_600_MH4_150_Mchi_1", category="C", blind=False)

        assert Path(plot_file).name == "impacts_C_data_t0_MH3_600_MH4_150_Mchi_1.pdf"

    def test_real_combine_yaml_impacts_steps_have_no_duplicate_flags(self):
        """Regression test against the actual configs/combine.yaml (not the
        test fixture): the Impacts steps previously hardcoded
        --cminDefaultMinimizerStrategy=0 in addition to advanced.commands.
        general_args already adding it, and step 3 used "{output_dir}/impacts.json"
        for -o even though every step already runs with cwd=output_dir — both
        caused combineTool.py --parallel to fail with a spurious
        FileNotFoundError against real combine binaries."""
        real_config = yaml.safe_load((REPO_ROOT / "configs" / "combine.yaml").read_text())
        commands = real_config["advanced"]["commands"]
        general_args = set(commands.get("general_args", []))

        for step in commands["Impacts"]["steps"]:
            args = step["args"]
            assert len(args) == len(set(args)), f"duplicate token in Impacts step args: {args}"
            assert not (general_args & set(args)), (
                f"Impacts step args {args} redundantly hardcode a flag already "
                f"in general_args {general_args}"
            )
            if "-o" in args:
                o_value = args[args.index("-o") + 1]
                assert "/" not in o_value, f"-o value should be a basename, got {o_value}"

    def test_real_combine_yaml_asymptotic_limits_rmax_wide_enough(self):
        """Regression test: rMax=10 was too tight for AsymptoticLimits — combine
        silently drops any quantile row whose fitted r exceeds rMax (raises an
        internal exception it catches rather than erroring out), so low-
        sensitivity mass points in the real 29-point signal grid lost their
        +2sigma expected row entirely, breaking collect_limits' "incomplete
        limit quantiles" check for every single mass point (reproduced against
        real combine 10.6.1: MH3_600_MH4_50 needs r~11.8 for +2sigma)."""
        real_config = yaml.safe_load((REPO_ROOT / "configs" / "combine.yaml").read_text())
        args = real_config["advanced"]["commands"]["AsymptoticLimits"]["steps"][0]["args"]
        rmax_token = next(a for a in args if a.startswith("--rMax="))
        rmax = float(rmax_token.split("=")[1])
        assert rmax >= 1000, f"AsymptoticLimits --rMax={rmax} too tight for low-sensitivity mass points"

    def test_run_missing_binary_raises_clear_error(self, minimal_combine_config, tmp_path):
        runner = CombineRunner(minimal_combine_config)
        runner.advanced_config["combine_commands"]["combine"] = "definitely-not-a-real-binary"
        workspace = tmp_path / "workspace.root"
        workspace.write_text("")
        datacard = tmp_path / "datacard.txt"
        datacard.write_text("imax 1\njmax 2\nkmax 1\n\nbin          SR_1b\nobservation  -1\n")

        with pytest.raises(RuntimeError, match="not found on PATH"):
            runner.run_goodness_of_fit(str(workspace), str(tmp_path), str(datacard), blind=True)

    def test_parse_asymptotic_limits(self, tmp_path):
        results_file = tmp_path / "asymptotic_limits.root"
        with uproot.recreate(str(results_file)) as f:
            f["limit"] = {
                "limit": np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0]),
                "quantileExpected": np.array([0.025, 0.16, 0.5, 0.84, 0.975, -1.0]),
            }

        runner = CombineRunner({"fit": {}, "output": {}, "advanced": {}})
        results = runner._parse_asymptotic_limits(str(results_file))

        assert results["observed"] == pytest.approx(3.0)
        assert results["expected"] == pytest.approx(1.5)
        assert results["expected_minus_2sigma"] == pytest.approx(0.5)
        assert results["expected_plus_2sigma"] == pytest.approx(2.5)

    def test_parse_impacts(self, tmp_path):
        results_file = tmp_path / "impacts.json"
        results_file.write_text(json.dumps({
            "params": [
                {"name": "lumi_2024", "impact_r": 0.1},
                {"name": "btagSF", "impact_r": 0.05},
            ]
        }))

        runner = CombineRunner({"fit": {}, "output": {}, "advanced": {}})
        results = runner._parse_impacts(str(results_file))

        assert results["impacts"]["lumi_2024"] == 0.1
        assert results["impacts"]["btagSF"] == 0.05

    def test_parse_mass_point_label(self):
        runner = CombineRunner({"fit": {}, "output": {}, "advanced": {}})
        parsed = runner._parse_mass_point_label("MH3_600_MH4_150_Mchi_1")
        assert parsed == {"MH3": 600.0, "MH4": 150.0, "Mchi": 1.0}

    def test_parse_mass_point_label_non_conforming_returns_none(self):
        runner = CombineRunner({"fit": {}, "output": {}, "advanced": {}})
        assert runner._parse_mass_point_label("not_a_mass_point") is None

    def test_collect_limits_writes_summary_table_and_root(self, tmp_path):
        """Regression: Run2's limits pipeline always produced an aggregated
        summary (limits_bbDM_C_2018.txt/.root), one row per mass point — not
        just the raw per-mass-point higgsCombine*.root files."""
        card_dir = tmp_path / "cards"
        mass_points = ["MH3_600_MH4_150_Mchi_1", "MH3_600_MH4_50_Mchi_1"]
        quantiles = {
            "MH3_600_MH4_150_Mchi_1": (0.16, 0.24, 0.35, 0.50, 0.71, 0.30),
            "MH3_600_MH4_50_Mchi_1": (0.12, 0.19, 0.27, 0.40, 0.58, 0.25),
        }
        for mp in mass_points:
            mp_dir = card_dir / mp
            mp_dir.mkdir(parents=True)
            m2, m1, exp, p1, p2, obs = quantiles[mp]
            with uproot.recreate(str(mp_dir / "higgsCombineTest.AsymptoticLimits.mH120.root")) as f:
                f["limit"] = {
                    "limit": np.array([m2, m1, exp, p1, p2, obs]),
                    "quantileExpected": np.array([0.025, 0.16, 0.5, 0.84, 0.975, -1.0]),
                }

        runner = CombineRunner({"fit": {}, "output": {}, "advanced": {}})
        output_dir = tmp_path / "limits"
        txt_file = runner.collect_limits(str(card_dir), mass_points, str(output_dir), "limits_bbDM_2024")

        assert Path(txt_file).name == "limits_bbDM_2024.txt"
        lines = Path(txt_file).read_text().strip().split("\n")
        assert len(lines) == 2
        # sorted by (MH3, MH4) -> MH4=50 before MH4=150
        assert lines[0].startswith("600 50 ")
        assert lines[1].startswith("600 150 ")
        tokens = lines[0].split()
        assert len(tokens) == 8
        assert float(tokens[2]) == pytest.approx(0.12)  # exp-2sigma
        assert float(tokens[7]) == pytest.approx(0.25)  # observed

        root_file = output_dir / "limits_bbDM_2024.root"
        assert root_file.exists()
        with uproot.open(str(root_file)) as f:
            keys = {k.split(";")[0] for k in f.keys()}
        assert keys == {"exp2", "exp1", "expmed", "obs"}

    def test_collect_limits_skips_missing_mass_points(self, tmp_path):
        card_dir = tmp_path / "cards"
        card_dir.mkdir()
        # Only one of two requested mass points has output on disk.
        mp_dir = card_dir / "MH3_600_MH4_150_Mchi_1"
        mp_dir.mkdir()
        with uproot.recreate(str(mp_dir / "higgsCombineTest.AsymptoticLimits.mH120.root")) as f:
            f["limit"] = {
                "limit": np.array([0.16, 0.24, 0.35, 0.50, 0.71, 0.30]),
                "quantileExpected": np.array([0.025, 0.16, 0.5, 0.84, 0.975, -1.0]),
            }

        runner = CombineRunner({"fit": {}, "output": {}, "advanced": {}})
        txt_file = runner.collect_limits(
            str(card_dir), ["MH3_600_MH4_150_Mchi_1", "MH3_600_MH4_999_Mchi_1"],
            str(tmp_path / "limits"), "limits_test",
        )
        lines = Path(txt_file).read_text().strip().split("\n")
        assert len(lines) == 1

    def test_plot_limits_one_pdf_per_mh3_slice(self, tmp_path):
        """collect_limits' .txt columns are r (signal-strength) limits, not
        absolute cross sections — plot_limits must multiply by the real
        per-mass-point theory xsec (pb -> fb, x1000) from xsection_signal.json,
        not hardcode it per point the way Run2's reference notebook did.
        Also verifies one PDF is written per distinct MH3 value (not one
        combined plot mixing mA slices on a single x-axis)."""
        limits_txt = tmp_path / "limits.txt"
        limits_txt.write_text(
            "600 50 1.0 1.5 2.0 2.5 3.0 1.8\n"
            "600 100 0.8 1.2 1.6 2.0 2.4 1.5\n"
            "1500 10 0.5 0.7 0.9 1.1 1.3 0.0\n"  # only 1 point at MH3=1500 -> skipped
        )
        xsection_json = tmp_path / "xsec.json"
        xsection_json.write_text(json.dumps({
            "2HDMa": {
                "MH3_600_MH4_50_Mchi_1": 0.001,   # pb -> 1.0 fb
                "MH3_600_MH4_100_Mchi_1": 0.002,  # pb -> 2.0 fb
                "MH3_1500_MH4_10_Mchi_1": 0.003,
            },
        }))

        runner = CombineRunner({"fit": {}, "output": {}, "advanced": {}})
        plot_files = runner.plot_limits(
            str(limits_txt), str(xsection_json), "2HDMa",
            str(tmp_path / "plots"), "limits_test",
        )

        # Only MH3=600 has >=2 points -> exactly one PDF, MH3=1500 skipped.
        assert len(plot_files) == 1
        assert Path(plot_files[0]).name == "limits_test_MH3_600.pdf"
        assert Path(plot_files[0]).exists()
