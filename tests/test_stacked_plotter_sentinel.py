"""Tests for sentinel (-9.0) handling in event_stacked_plotter."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Script is not a package — import directly.
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from event_stacked_plotter import _SENTINEL, _apply_variable_plot_filter, _make_bins


class TestApplyVariablePlotFilter:
    def test_sentinel_stripped(self):
        values = np.array([-9.0, 1.0, 2.0, -9.0, 3.0])
        result = _apply_variable_plot_filter("costheta_star", values)
        assert -9.0 not in result
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_all_sentinel_returns_empty(self):
        values = np.array([-9.0, -9.0, -9.0])
        result = _apply_variable_plot_filter("DetaJet12", values)
        assert result.size == 0

    def test_empty_input_passthrough(self):
        values = np.array([], dtype=float)
        result = _apply_variable_plot_filter("met_pt", values)
        assert result.size == 0

    def test_no_sentinel_unchanged(self):
        values = np.array([10.0, 20.0, 30.0])
        result = _apply_variable_plot_filter("jet_pt", values)
        np.testing.assert_array_equal(result, values)

    def test_met_pt_strips_sentinel_then_applies_100gev_cut(self):
        values = np.array([-9.0, 80.0, 120.0, 200.0, -9.0])
        result = _apply_variable_plot_filter("met_pt", values)
        assert -9.0 not in result
        assert all(v >= 100.0 for v in result)
        np.testing.assert_array_equal(result, [120.0, 200.0])

    def test_met_pt_case_insensitive(self):
        values = np.array([-9.0, 150.0])
        result = _apply_variable_plot_filter("MET_pt", values)
        np.testing.assert_array_equal(result, [150.0])

    def test_sentinel_constant_value(self):
        assert _SENTINEL == -9.0


class TestMakeBins:
    def test_sentinel_excluded_from_range(self):
        # Use non-integer floats so linspace path is taken (not integer-bin path).
        # Without fix bins[0] would be -9.0; with fix bins[0] >= 1.5.
        values = np.array([-9.0, -9.0, 1.5, 2.5, 3.5])
        bins = _make_bins([values], n_bins=10)
        assert bins is not None
        assert bins[0] >= 1.5, f"bins[0]={bins[0]} — sentinel leaked into range"
        assert bins[-1] <= 3.5 + 1e-9

    def test_all_sentinel_returns_none(self):
        values = np.array([-9.0, -9.0])
        result = _make_bins([values], n_bins=10)
        assert result is None

    def test_nan_and_sentinel_both_excluded(self):
        # Non-integer floats → linspace path.
        values = np.array([-9.0, np.nan, np.inf, 5.5, 10.5])
        bins = _make_bins([values], n_bins=10)
        assert bins is not None
        assert bins[0] >= 5.5 - 1e-9

    def test_mixed_arrays_sentinel_excluded(self):
        # Non-integer floats → linspace path.
        a = np.array([-9.0, 1.5, 2.5])
        b = np.array([-9.0, 3.5, 4.5])
        bins = _make_bins([a, b], n_bins=10)
        assert bins is not None
        assert bins[0] >= 1.5
        assert bins[-1] <= 4.5 + 1e-9

    def test_normal_values_unaffected(self):
        # Non-integer floats → linspace path; bins span [data_min, data_max].
        values = np.array([0.5, 1.5, 2.5, 3.5])
        bins = _make_bins([values], n_bins=4)
        assert bins is not None
        assert np.isclose(bins[0], 0.5)
        assert np.isclose(bins[-1], 3.5)
