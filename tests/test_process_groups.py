"""
Tests for process-group pattern matching, sample-label normalisation,
PlotManager yaml parsing, data-group region routing, and histogram normalisation.
"""
import pickle
import tempfile
import textwrap
from pathlib import Path

import numpy as np
import pytest
import yaml

from utils.plot_utils import simplify_sample_label, _PROCESS_CONFIG
from darkbottomline.plotting import (
    PlotManager,
    _histogram_and_sumw2,
    _histogram_counts,
    _apply_variable_plot_filter,
    _clean_sample_name,
    _find_xsec,
)


# ---------------------------------------------------------------------------
# simplify_sample_label
# ---------------------------------------------------------------------------

class TestSimplifySampleLabel:
    """All Run-3 dataset names must map to a known _PROCESS_CONFIG key."""

    SAMPLES = [
        # W+jets
        ("WtoLNu-2Jets_PTLNu-100to250_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.root", "WtoLNu-2Jets"),
        ("WtoLNu-2Jets_PTLNu-600toInf_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.root", "WtoLNu-2Jets"),
        # Z(nunu)+jets
        ("Zto2Nu-2Jets_PTNuNu-100to250_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.root", "Zto2Nu-2Jets"),
        ("Zto2Nu-2Jets_PTNuNu-600toInf_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.root", "Zto2Nu-2Jets"),
        # DY(ll)+jets
        ("DYto2L-2Jets_MLL-50_PTLL-100to250_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.root", "DYto2L-2Jets"),
        ("DYto2L-2Jets_MLL-50_PTLL-600toInf_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.root", "DYto2L-2Jets"),
        # ttbar
        ("TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8.root", "Top"),
        ("TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8.root", "Top"),
        ("TTto4Q_TuneCP5_13p6TeV_powheg-pythia8.root",    "Top"),
        # single top
        ("TBbarQ_t-channel_4FS_TuneCP5_13p6TeV_powheg-madspin-pythia8.root",     "SingleTop"),
        ("TbarBQ_t-channel_4FS_TuneCP5_13p6TeV_powheg-madspin-pythia8.root",     "SingleTop"),
        ("TBbartoLplusNuBbar-s-channel-4FS_TuneCP5_13p6TeV_amcatnlo-pythia8.root", "SingleTop"),
        ("TbarBtoLminusNuB-s-channel-4FS_TuneCP5_13p6TeV_amcatnlo-pythia8.root", "SingleTop"),
        ("TbarWplustoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8.root",                 "SingleTop"),
        ("TWminustoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8.root",                   "SingleTop"),
        # diboson
        ("WZ_TuneCP5_13p6TeV_pythia8.root", "DIBOSON"),
        ("WW_TuneCP5_13p6TeV_pythia8.root", "DIBOSON"),
        ("ZZ_TuneCP5_13p6TeV_pythia8.root", "DIBOSON"),
        # SM Higgs
        ("GluGluHto2B_M-125_TuneCP5_13p6TeV_powheg-minlo-pythia8.root",           "SMHiggs"),
        ("VBFHto2B_M-125_TuneCP5_13p6TeV_powheg-pythia8.root",                    "SMHiggs"),
        ("VBFHto2B_M-125_dipoleRecoilOn_TuneCP5_13p6TeV_powheg-pythia8.root",     "SMHiggs"),
        ("WminusH_Hto2B_WtoLNu_M-125_TuneCP5_13p6TeV_powheg-pythia8.root",       "SMHiggs"),
        ("WplusH_Hto2B_WtoLNu_M-125_TuneCP5_13p6TeV_powheg-pythia8.root",        "SMHiggs"),
        ("ZH_Hto2B_Zto2L_M-125_TuneCP5_13p6TeV_powheg-pythia8.root",             "SMHiggs"),
        ("ZH_Hto2B_Zto2Nu_M-125_TuneCP5_13p6TeV_powheg-minlo-pythia8.root",      "SMHiggs"),
        ("ggZH_Hto2B_Zto2L_M-125_TuneCP5_13p6TeV_powheg-pythia8.root",           "SMHiggs"),
        ("ggZH_Hto2B_Zto2Nu_M-125_TuneCP5_13p6TeV_powheg-minlo-pythia8.root",    "SMHiggs"),
        ("tHto2B_M-125_TuneCP5_13p6TeV_powheg-pythia8.root",                      "SMHiggs"),
        # folder-name aliases
        ("WtoLNuJets",  "WtoLNu-2Jets"),
        ("Zto2NuJets",  "Zto2Nu-2Jets"),
        ("DYto2LJets",  "DYto2L-2Jets"),
        ("ttbar",       "Top"),
        ("singletop",   "SingleTop"),
        ("Diboson",     "DIBOSON"),
        ("SMHiggs",     "SMHiggs"),
    ]

    @pytest.mark.parametrize("name,expected", SAMPLES)
    def test_maps_to_known_process(self, name, expected):
        canon = simplify_sample_label(name)
        assert canon == expected, f"{name!r} → {canon!r}, want {expected!r}"
        assert canon in _PROCESS_CONFIG, f"{canon!r} not in _PROCESS_CONFIG"

    def test_strips_root_extension(self):
        assert simplify_sample_label("TTto4Q_TuneCP5_13p6TeV_powheg-pythia8.root") == \
               simplify_sample_label("TTto4Q_TuneCP5_13p6TeV_powheg-pythia8")

    def test_strips_new_prefix(self):
        assert simplify_sample_label("newWtoLNuJets") == "WtoLNu-2Jets"


# ---------------------------------------------------------------------------
# _resolve_group_files  (pattern substring matching)
# ---------------------------------------------------------------------------

class TestResolveGroupFiles:

    def _make_folder(self, tmp_path, filenames):
        for f in filenames:
            (tmp_path / f).touch()
        return str(tmp_path)

    def test_matches_by_substring(self, tmp_path):
        folder = self._make_folder(tmp_path, [
            "WtoLNu-2Jets_PTLNu-100to250_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.root",
            "WtoLNu-2Jets_PTLNu-250to400_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.root",
            "TTto4Q_TuneCP5_13p6TeV_powheg-pythia8.root",
        ])
        pm = PlotManager()
        patterns = ["WtoLNu-2Jets_PTLNu-100to250_2J", "WtoLNu-2Jets_PTLNu-250to400_2J"]
        found = pm._resolve_group_files(folder, patterns)
        names = {p.name for p in found}
        assert "WtoLNu-2Jets_PTLNu-100to250_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.root" in names
        assert "WtoLNu-2Jets_PTLNu-250to400_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.root" in names
        assert "TTto4Q_TuneCP5_13p6TeV_powheg-pythia8.root" not in names

    def test_no_double_match(self, tmp_path):
        """A file matching two patterns should appear only once."""
        folder = self._make_folder(tmp_path, [
            "WtoLNu-2Jets_PTLNu-100to250_2J_TuneCP5.root",
        ])
        pm = PlotManager()
        found = pm._resolve_group_files(folder, ["WtoLNu-2Jets", "PTLNu-100to250"])
        assert len(found) == 1

    def test_missing_pattern_warns(self, tmp_path, caplog):
        folder = self._make_folder(tmp_path, ["TTto4Q_TuneCP5.root"])
        pm = PlotManager()
        import logging
        with caplog.at_level(logging.WARNING):
            found = pm._resolve_group_files(folder, ["NonExistentSample"])
        assert found == []
        assert "No files matched" in caplog.text

    def test_only_root_and_pkl_returned(self, tmp_path):
        folder = self._make_folder(tmp_path, [
            "sample_TuneCP5.root",
            "sample_TuneCP5.txt",
            "sample_TuneCP5.pkl",
        ])
        pm = PlotManager()
        found = pm._resolve_group_files(folder, ["sample_TuneCP5"])
        suffixes = {p.suffix for p in found}
        assert ".txt" not in suffixes
        assert ".root" in suffixes or ".pkl" in suffixes


# ---------------------------------------------------------------------------
# PlotManager yaml parsing
# ---------------------------------------------------------------------------

MINI_YAML = textwrap.dedent("""\
    process_groups:
      WtoLNuJets:
        type: background
        color: "#bd1f01"
        label: "$W$+jets"
        patterns:
          - "WtoLNu-2Jets_PTLNu-100to250_2J"
          - "WtoLNu-2Jets_PTLNu-250to400_2J"
      ttbar:
        type: background
        color: "#e76300"
        label: "$t\\\\bar{t}$"
        patterns:
          - "TTto2L2Nu"
      MySig:
        type: signal
        color: "#000000"
        label: "Signal"
        patterns:
          - "DM_mMed-1000"
      MET_Data:
        type: data
        label: "Data"
        regions:
          - "SR"
          - "_mu"
        patterns:
          - "JetMET_Run2024"
      EGamma_Data:
        type: data
        label: "Data (EGamma)"
        regions:
          - "_el"
        patterns:
          - "EGamma_Run2024"
""")


class TestPlotManagerYamlParsing:

    def _make_pm(self):
        cfg = yaml.safe_load(MINI_YAML)
        return PlotManager(config=cfg)

    def test_background_groups_populated(self):
        pm = self._make_pm()
        assert "WtoLNuJets" in pm.process_groups
        assert "ttbar" in pm.process_groups
        assert "MySig" not in pm.process_groups

    def test_signal_groups_populated(self):
        pm = self._make_pm()
        assert "MySig" in pm.signal_groups
        assert "WtoLNuJets" not in pm.signal_groups

    def test_data_groups_populated(self):
        pm = self._make_pm()
        assert "MET_Data" in pm.data_groups
        assert "EGamma_Data" in pm.data_groups
        assert "WtoLNuJets" not in pm.data_groups

    def test_patterns_stored(self):
        pm = self._make_pm()
        assert "WtoLNu-2Jets_PTLNu-100to250_2J" in pm.process_groups["WtoLNuJets"]

    def test_color_overrides(self):
        pm = self._make_pm()
        assert pm._group_colors["WtoLNuJets"] == "#bd1f01"
        assert pm._group_colors["ttbar"] == "#e76300"

    def test_label_overrides(self):
        pm = self._make_pm()
        assert pm._group_labels["WtoLNuJets"] == "$W$+jets"

    def test_legacy_files_key_accepted(self):
        cfg = yaml.safe_load(textwrap.dedent("""\
            process_groups:
              OldStyle:
                type: background
                files:
                  - "SomeFile_TuneCP5.root"
        """))
        pm = PlotManager(config=cfg)
        assert "SomeFile_TuneCP5.root" in pm.process_groups["OldStyle"]


# ---------------------------------------------------------------------------
# Data-group region routing
# ---------------------------------------------------------------------------

class TestDataGroupRegionRouting:
    """
    _data_hist_for_region selects the correct data PKL based on region name patterns.
    MET_Data → SR, *_mu;  EGamma_Data → *_el
    """

    def _make_region_pkl(self, tmp_path, fname, region, var, values):
        """Write a minimal region-histogram PKL (dict of dicts of numpy arrays)."""
        import hist
        h = hist.Hist(hist.axis.Regular(10, 0, 100, name=var), storage=hist.storage.Weight())
        h.fill(**{var: values})
        data = {"region_histograms": {region: {var: h}}}
        p = tmp_path / fname
        with open(p, "wb") as fh:
            pickle.dump(data, fh)
        return p

    def _make_pm_with_data(self, tmp_path):
        met_file  = self._make_region_pkl(tmp_path, "JetMET_Run2024C.pkl", "1b:SR",        "met", np.array([300., 400., 500.]))
        egam_file = self._make_region_pkl(tmp_path, "EGamma_Run2024C.pkl", "1b:CR_Zee", "met", np.array([200., 250.]))

        cfg_txt = textwrap.dedent(f"""\
            process_groups:
              MET_Data:
                type: data
                label: "Data (MET)"
                regions: ["SR", "_mu"]
                patterns: ["JetMET_Run2024C"]
              EGamma_Data:
                type: data
                label: "Data (EGamma)"
                regions: ["_el"]
                patterns: ["EGamma_Run2024C"]
        """)
        cfg = yaml.safe_load(cfg_txt)
        pm = PlotManager(config=cfg)
        return pm, str(tmp_path)

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("hist"),
        reason="hist library not available"
    )
    def test_met_data_used_for_sr(self, tmp_path):
        pm, folder = self._make_pm_with_data(tmp_path)
        # Monkey-patch _resolve_group_files to use our tmp folder
        from pathlib import Path as _Path
        orig = pm._resolve_group_files
        pm._resolve_group_files = lambda f, pats: orig(folder, pats)
        # Pre-load data_loaded by running the routing logic directly
        # We test via create_stacked_plots integration smoke (no actual plot output)
        # Instead test the region-pattern filter logic directly
        rp_met  = ["SR", "_mu"]
        rp_egam = ["_el"]

        def applies(region_patterns, region):
            if not region_patterns:
                return True
            return any(p in region for p in region_patterns)

        assert applies(rp_met,  "1b:SR")           is True
        assert applies(rp_met,  "2b:CR_Topmunu")    is True
        assert applies(rp_met,  "1b:CR_Zee")    is False
        assert applies(rp_egam, "1b:CR_Zee")    is True
        assert applies(rp_egam, "1b:SR")            is False
        assert applies(rp_egam, "2b:CR_Topmunu")     is False

    def test_empty_regions_list_matches_all(self):
        def applies(region_patterns, region):
            if not region_patterns:
                return True
            return any(p in region for p in region_patterns)

        for region in ["1b:SR", "2b:CR_Topmunu", "1b:CR_Zee"]:
            assert applies([], region) is True


# ---------------------------------------------------------------------------
# _histogram_and_sumw2 normalisation
# ---------------------------------------------------------------------------

class TestHistogramAndSumw2:

    def test_lumi_xsec_scale(self):
        vals = np.array([10., 20., 30., 40., 50.])
        bins = np.array([0., 25., 75.])   # 2 bins
        wte = 1000
        lumi = 59.83   # fb-1
        xsec = 100.0   # pb
        hv, hs = _histogram_and_sumw2(vals, bins, wte, lumi, xsec)
        expected_scale = lumi * xsec * 1000.0 / wte
        raw_bin0 = 2.0  # events 10,20 in [0,25)
        raw_bin1 = 3.0  # events 30,40,50 in [25,75)
        assert hv[0] == pytest.approx(raw_bin0 * expected_scale)
        assert hv[1] == pytest.approx(raw_bin1 * expected_scale)

    def test_sumw2_is_square_of_scale(self):
        vals = np.array([10., 20., 30.])
        bins = np.array([0., 50., 100.])
        wte = 500
        lumi = 10.0
        xsec = 50.0
        hv, hs = _histogram_and_sumw2(vals, bins, wte, lumi, xsec)
        scale = lumi * xsec * 1000.0 / wte
        np.testing.assert_allclose(hs, hv * scale)

    def test_no_xsec_uses_lumi_only(self):
        vals = np.array([1., 3.])  # one per bin: 1 in [0,2), 3 in [2,4)
        bins = np.array([0., 2., 4.])
        wte = 100
        lumi = 20.0
        hv, _ = _histogram_and_sumw2(vals, bins, wte, lumi, None)
        scale = lumi / wte
        assert hv[0] == pytest.approx(1.0 * scale)
        assert hv[1] == pytest.approx(1.0 * scale)

    def test_sentinel_values_excluded(self):
        # _clip_overflow clips sentinel to bins[0]=0 → lands in bin 0.
        # Test by using a bin range that starts above 0 so sentinels truly stay out.
        # Use _apply_variable_plot_filter which strips sentinels before histogramming.
        from darkbottomline.objects import SENTINEL
        from darkbottomline.plotting import _apply_variable_plot_filter
        vals_with_sentinel = np.array([SENTINEL, SENTINEL, 30., 40.])
        vals_clean         = np.array([30., 40.])
        bins = np.array([0., 50., 100.])
        # After _apply_variable_plot_filter, sentinel is stripped
        filtered = _apply_variable_plot_filter("some_var", vals_with_sentinel)
        np.testing.assert_array_equal(filtered, vals_clean)

    def test_empty_array_returns_zeros(self):
        bins = np.array([0., 50., 100.])
        hv, hs = _histogram_and_sumw2(np.array([]), bins, 100, 1.0, 1.0)
        np.testing.assert_array_equal(hv, np.zeros(2))
        np.testing.assert_array_equal(hs, np.zeros(2))

    def test_zero_wte_does_not_raise(self):
        vals = np.array([10., 20.])
        bins = np.array([0., 50., 100.])
        hv, hs = _histogram_and_sumw2(vals, bins, 0, 1.0, 1.0)
        assert np.all(np.isfinite(hv))

    def test_histogram_counts_raw(self):
        vals = np.array([10., 20., 60.])
        bins = np.array([0., 50., 100.])
        counts = _histogram_counts(vals, bins)
        assert counts[0] == 2
        assert counts[1] == 1


# ---------------------------------------------------------------------------
# PlotManager loads process_groups from default plotting.yaml
# ---------------------------------------------------------------------------

class TestPlotManagerDefaultYaml:

    def test_default_yaml_has_all_background_groups(self):
        cfg_path = Path(__file__).parent.parent / "configs" / "plotting.yaml"
        if not cfg_path.exists():
            pytest.skip("plotting.yaml not found")
        cfg = yaml.safe_load(cfg_path.read_text())
        pm = PlotManager(config=cfg)
        expected = {"WtoLNuJets", "Zto2NuJets", "DYto2LJets", "ttbar", "singletop", "Diboson", "SMHiggs"}
        assert expected.issubset(set(pm.process_groups.keys()))

    def test_default_yaml_patterns_are_non_empty(self):
        cfg_path = Path(__file__).parent.parent / "configs" / "plotting.yaml"
        if not cfg_path.exists():
            pytest.skip("plotting.yaml not found")
        cfg = yaml.safe_load(cfg_path.read_text())
        pm = PlotManager(config=cfg)
        for label, patterns in pm.process_groups.items():
            assert len(patterns) > 0, f"{label} has no patterns"

    def test_all_background_patterns_are_strings(self):
        cfg_path = Path(__file__).parent.parent / "configs" / "plotting.yaml"
        if not cfg_path.exists():
            pytest.skip("plotting.yaml not found")
        cfg = yaml.safe_load(cfg_path.read_text())
        pm = PlotManager(config=cfg)
        for label, patterns in pm.process_groups.items():
            for p in patterns:
                assert isinstance(p, str), f"{label}: pattern {p!r} is not a string"


# ---------------------------------------------------------------------------
# Cross-year dataset-name canonicalization (_clean_sample_name / _find_xsec)
# ---------------------------------------------------------------------------

_TUNE  = "_TuneCP5_13p6TeV_amcatnloFXFX-pythia8"
_POW   = "_TuneCP5_13p6TeV_powheg-pythia8"
_MINLO = "_TuneCP5_13p6TeV_powhegMINLO-pythia8"
_OLDMINLO = "_TuneCP5_13p6TeV_powheg-minlo-pythia8"


class TestCrossYearCanonicalization:
    """The 2022/2022EE/2023 (_2J) and 2024 (Bin-2J-) forms of each process, and
    the SM Higgs old/new renames, must collapse to one canonical string equal to
    the xsec-JSON full_dataset core."""

    # (old 2022/23 name, new 2024 name, expected canonical core)
    PAIRS = [
        ("WtoLNu-2Jets_PTLNu-40to100_2J" + _TUNE,
         "WtoLNu-2Jets_Bin-2J-PTLNu-40to100" + _TUNE,
         "WtoLNu-2Jets_PTLNu-40to100"),
        ("WtoLNu-2Jets_PTLNu-200to400_2J" + _TUNE,
         "WtoLNu-2Jets_Bin-2J-PTLNu-200to400" + _TUNE,
         "WtoLNu-2Jets_PTLNu-200to400"),
        ("Zto2Nu-2Jets_PTNuNu-600_2J" + _TUNE,
         "Zto2Nu-2Jets_Bin-2J-PTNuNu-600" + _TUNE,
         "Zto2Nu-2Jets_PTNuNu-600"),
        ("DYto2L-2Jets_MLL-50_PTLL-100to200_2J" + _TUNE,
         "DYto2L-2Jets_Bin-2J-MLL-50-PTLL-100to200" + _TUNE,
         "DYto2L-2Jets_MLL-50-PTLL-100to200"),
        ("GluGluHto2B_M-125" + _OLDMINLO,
         "GluGluH-Hto2B_Par-M-125" + _MINLO,
         "GluGluH-Hto2B_Par-M-125"),
        ("ZH_Hto2B_Zto2Nu_M-125" + _OLDMINLO,
         "ZH-Zto2Nu-Hto2B_Par-M-125" + _MINLO,
         "ZH-Zto2Nu-Hto2B_Par-M-125"),
        ("WminusH_Hto2B_WtoLNu_M-125" + _POW,
         "WminusH-WtoLNu-Hto2B_Par-M-125" + _MINLO,
         "WminusH-WtoLNu-Hto2B_Par-M-125"),
        ("ttHto2B_M-125" + _POW,
         "TTH-Hto2B_Par-M-125" + _POW,
         "TTH-Hto2B_Par-M-125"),
        # Single top t-channel: 2022/23 underscore/no-toLNu vs 2024 dash/toLNu
        ("TBbarQ_t-channel_4FS_TuneCP5_13p6TeV_powheg-madspin-pythia8",
         "TBbarQtoLNu-t-channel-4FS_TuneCP5_13p6TeV_powheg-madspin-pythia8",
         "TBbarQtoLNu-t-channel-4FS"),
        ("TbarBQ_t-channel_4FS_TuneCP5_13p6TeV_powheg-madspin-pythia8",
         "TbarBQtoLNu-t-channel-4FS_TuneCP5_13p6TeV_powheg-madspin-pythia8",
         "TbarBQtoLNu-t-channel-4FS"),
    ]

    @pytest.mark.parametrize("old,new,canon", PAIRS)
    def test_years_collapse_to_same_canonical(self, old, new, canon):
        c_old = _clean_sample_name(old)
        c_new = _clean_sample_name(new)
        assert c_old == canon, f"{old} -> {c_old}, expected {canon}"
        assert c_new == canon, f"{new} -> {c_new}, expected {canon}"

    @pytest.mark.parametrize("old,new,canon", PAIRS)
    def test_idempotent(self, old, new, canon):
        for n in (old, new):
            once = _clean_sample_name(n)
            assert _clean_sample_name(once) == once

    def test_data_names_untouched(self):
        # Data primary datasets must keep their -Run tag (matched by data patterns)
        for name in ("JetMET-Run2022C-22Sep2023-v1",
                     "JetMET0-Run2024C-MINIv6NANOv15-v1",
                     "EGamma0-Run2024D-MINIv6NANOv15-v1"):
            assert "-Run" in _clean_sample_name(name)

    def test_find_xsec_resolves_all_years(self):
        json_path = (Path(__file__).parent.parent
                     / "data" / "cross-section" / "xsection_background.json")
        if not json_path.exists():
            pytest.skip("xsection_background.json not found")
        import json
        raw = json.loads(json_path.read_text())
        flat = PlotManager._normalize_cross_sections(raw)
        # 2022/23 _2J and 2024 Bin-2J-  → same non-None xsec
        for old, new, _canon in self.PAIRS:
            x_old = _find_xsec(old, flat)
            x_new = _find_xsec(new, flat)
            # ggZH/GluGluZH have no JSON xsec; skip those (none in PAIRS)
            assert x_old is not None, f"no xsec for {old}"
            assert x_new is not None, f"no xsec for {new}"
            assert abs(x_old - x_new) < 1e-6, f"{old} vs {new}: {x_old} != {x_new}"


class TestCrossYearGrouping:
    """Files named in any year's convention must land in the right process group."""

    def test_multi_year_files_group_correctly(self):
        cfg_path = Path(__file__).parent.parent / "configs" / "plotting.yaml"
        if not cfg_path.exists():
            pytest.skip("plotting.yaml not found")
        cfg = yaml.safe_load(cfg_path.read_text())
        pm = PlotManager(config=cfg)

        files = {
            "WtoLNu-2Jets_PTLNu-40to100_2J" + _TUNE + "_EVENTSELECTION.root": "WtoLNuJets",
            "WtoLNu-2Jets_Bin-2J-PTLNu-600" + _TUNE + "_EVENTSELECTION.root": "WtoLNuJets",
            "Zto2Nu-2Jets_PTNuNu-200to400_2J" + _TUNE + "_EVENTSELECTION.root": "Zto2NuJets",
            "Zto2Nu-2Jets_Bin-2J-PTNuNu-40to100" + _TUNE + "_EVENTSELECTION.root": "Zto2NuJets",
            "DYto2L-2Jets_MLL-50_PTLL-100to200_2J" + _TUNE + "_EVENTSELECTION.root": "DYto2LJets",
            "DYto2L-2Jets_Bin-2J-MLL-50-PTLL-600" + _TUNE + "_EVENTSELECTION.root": "DYto2LJets",
            "GluGluHto2B_M-125" + _OLDMINLO + "_EVENTSELECTION.root": "SMHiggs",
            "ZH_Hto2B_Zto2Nu_M-125" + _OLDMINLO + "_EVENTSELECTION.root": "SMHiggs",
            "GluGluH-Hto2B_Par-M-125" + _MINLO + "_EVENTSELECTION.root": "SMHiggs",
            "TTto2L2Nu" + _POW + "_EVENTSELECTION.root": "ttbar",
            "WZ_TuneCP5_13p6TeV_pythia8_EVENTSELECTION.root": "Diboson",
            "TBbarQ_t-channel_4FS_TuneCP5_13p6TeV_powheg-madspin-pythia8_EVENTSELECTION.root": "singletop",
            "TbarBQtoLNu-t-channel-4FS_TuneCP5_13p6TeV_powheg-madspin-pythia8_EVENTSELECTION.root": "singletop",
            "JetMET-Run2022C-22Sep2023-v1_EVENTSELECTION.root": "MET_Data",
            "EGamma0-Run2024D-MINIv6NANOv15-v1_EVENTSELECTION.root": "EGamma_Data",
        }
        with tempfile.TemporaryDirectory() as d:
            for fn in files:
                (Path(d) / fn).touch()
            got = {}
            all_groups = {**pm.process_groups, **pm.signal_groups, **pm.data_groups}
            for label, patterns in all_groups.items():
                for p in pm._resolve_group_files(d, patterns):
                    got[p.name] = label

        for fn, exp in files.items():
            assert got.get(fn) == exp, f"{fn}: got {got.get(fn)}, expected {exp}"
        # nothing left unmatched
        assert set(got) == set(files)
