"""Focused tests for DNN-only signed-weight cancellation."""

import numpy as np
import pandas as pd
import pytest

from darkbottomline.dnn_trainer import build_local_cancellation_weights
from darkbottomline.dnn_trainer import (
    apply_dnn_weight_models,
    apply_local_cancellation_mapping,
    fit_dnn_weight_models,
    fit_local_cancellation_mapping,
)


def _config(levels, bins=None):
    return {
        "mode": "local_cancellation",
        "min_events": 1,
        "min_effective_events": 0.0,
        "levels": levels,
        "bins": bins or {},
    }


def test_single_cell_is_nonnegative_and_preserves_signed_yield():
    features = pd.DataFrame({"unused": [0.0, 0.0, 0.0]})
    signed = np.array([2.0, 1.5, -1.0])

    train, stats = build_local_cancellation_weights(
        features,
        signed,
        _config([[]]),
    )

    expected = np.abs(signed) * (signed.sum() / np.abs(signed).sum())
    np.testing.assert_allclose(train, expected)
    assert np.all(train >= 0.0)
    assert train.sum() == pytest.approx(signed.sum())
    assert stats["closure_error"] == pytest.approx(0.0)


def test_positive_local_cells_close_independently():
    features = pd.DataFrame({"njets": [1.0, 1.0, 2.0, 2.0]})
    signed = np.array([3.0, -1.0, 4.0, -1.0])

    train, _ = build_local_cancellation_weights(
        features,
        signed,
        _config([["njets"], []], bins={"njets": [1.5]}),
    )

    np.testing.assert_allclose(train[:2], [1.5, 0.5])
    np.testing.assert_allclose(train[2:], [2.4, 0.6])
    assert train[:2].sum() == pytest.approx(signed[:2].sum())
    assert train[2:].sum() == pytest.approx(signed[2:].sum())


def test_negative_fine_cell_falls_back_without_stranding_negative_yield():
    features = pd.DataFrame({"njets": [1.0, 2.0]})
    signed = np.array([-1.0, 3.0])

    train, stats = build_local_cancellation_weights(
        features,
        signed,
        _config([["njets"], []], bins={"njets": [1.5]}),
    )

    np.testing.assert_allclose(train, [0.5, 1.5])
    assert stats["levels"][0]["events_assigned"] == 0
    assert stats["levels"][1]["events_assigned"] == 2
    assert train.sum() == pytest.approx(signed.sum())


def test_missing_jet2_is_a_separate_local_cell():
    features = pd.DataFrame({"Jet2Pt": [-9999.0, -9999.0, 100.0, 120.0]})
    signed = np.array([3.0, -1.0, 4.0, -1.0])

    train, _ = build_local_cancellation_weights(
        features,
        signed,
        _config([["has_jet2"], []]),
    )

    np.testing.assert_allclose(train[:2], [1.5, 0.5])
    np.testing.assert_allclose(train[2:], [2.4, 0.6])


def test_nonpositive_process_yield_is_rejected():
    features = pd.DataFrame({"unused": [0.0, 0.0]})
    with pytest.raises(ValueError, match="non-positive signed yield"):
        build_local_cancellation_weights(features, np.array([1.0, -2.0]), _config([[]]))


def test_validation_uses_training_alpha_without_refitting():
    train_features = pd.DataFrame({"njets": [1.0, 1.0, 2.0, 2.0]})
    train_signed = np.array([3.0, -1.0, 4.0, -1.0])
    mapping, _ = fit_local_cancellation_mapping(
        train_features,
        train_signed,
        _config([["njets"], []], bins={"njets": [1.5]}),
    )

    validation_features = pd.DataFrame({"njets": [1.0, 1.0, 2.0, 2.0]})
    validation_signed = np.array([10.0, -9.0, 2.0, -1.0])
    validation_local = apply_local_cancellation_mapping(
        validation_features,
        validation_signed,
        mapping,
    )

    # Training alphas are 0.5 and 0.6. Validation's own alphas would both be
    # different, so this also guards against accidental split-wise refitting.
    np.testing.assert_allclose(validation_local, [5.0, 4.5, 1.2, 0.6])
    assert validation_local.sum() != pytest.approx(validation_signed.sum())


def test_unseen_evaluation_cell_uses_training_global_fallback():
    mapping, _ = fit_local_cancellation_mapping(
        pd.DataFrame({"njets": [1.0, 1.0]}),
        np.array([3.0, -1.0]),
        _config([["njets"], []], bins={"njets": [1.5]}),
    )

    evaluation_local = apply_local_cancellation_mapping(
        pd.DataFrame({"njets": [2.0, 2.0]}),
        np.array([4.0, -2.0]),
        mapping,
    )

    np.testing.assert_allclose(evaluation_local, [2.0, 1.0])


def test_samples_are_fitted_independently_and_reused_on_other_splits():
    train_features = pd.DataFrame({"unused": np.zeros(6)})
    train_signed = np.array([4.0, -1.0, 1.0, 5.0, -2.0, 1.0])
    train_samples = np.array(["slice_a"] * 3 + ["slice_b"] * 3)
    cfg = _config([[]])

    models, train_local, stats = fit_dnn_weight_models(
        train_features,
        train_signed,
        train_samples,
        cfg,
    )

    assert set(models) == {"slice_a", "slice_b"}
    assert train_local[:3].sum() == pytest.approx(train_signed[:3].sum())
    assert train_local[3:].sum() == pytest.approx(train_signed[3:].sum())
    assert stats["slice_a"]["closure_error"] == pytest.approx(0.0)
    assert stats["slice_b"]["closure_error"] == pytest.approx(0.0)

    eval_features = pd.DataFrame({"unused": np.zeros(4)})
    eval_signed = np.array([2.0, -2.0, 3.0, -1.0])
    eval_samples = np.array(["slice_a", "slice_a", "slice_b", "slice_b"])
    eval_local = apply_dnn_weight_models(
        eval_features,
        eval_signed,
        eval_samples,
        models,
        cfg,
    )

    np.testing.assert_allclose(eval_local[:2], np.abs(eval_signed[:2]) * (4.0 / 6.0))
    np.testing.assert_allclose(eval_local[2:], np.abs(eval_signed[2:]) * (4.0 / 8.0))


def test_evaluation_sample_missing_from_training_is_rejected():
    features = pd.DataFrame({"unused": [0.0, 0.0]})
    models, _, _ = fit_dnn_weight_models(
        features,
        np.array([2.0, -1.0]),
        np.array(["known", "known"]),
        _config([[]]),
    )

    with pytest.raises(ValueError, match="no training events"):
        apply_dnn_weight_models(
            pd.DataFrame({"unused": [0.0]}),
            np.array([1.0]),
            np.array(["unseen"]),
            models,
            _config([[]]),
        )
