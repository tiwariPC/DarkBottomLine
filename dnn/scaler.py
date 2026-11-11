"""Feature scaling utilities for tabular DNN training.

Purpose:
- Fit a per-feature standardization (mean/std) on the training sample.
- Transform feature matrices consistently for both training and inference.
- Handle missing/sentinel values (default: -9999) without polluting statistics.
- MassScaler: the same idea for the (MH3, MH4) inputs of a *parametric*
  model, which are physical parameters rather than engineered features.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


def _apply_log(M: np.ndarray, use_log: np.ndarray) -> np.ndarray:
    """log10 the columns flagged in *use_log*, leaving non-positive entries raw."""
    T = np.array(M, dtype="f8", copy=True)
    for j in np.flatnonzero(np.asarray(use_log, dtype=bool)):
        col = M[:, j]
        ok = np.isfinite(col) & (col > 0)
        T[ok, j] = np.log10(col[ok])
    return T


@dataclass(frozen=True)
class MassScaler:
    """Standardize the ``(MH3, MH4)`` input columns of a parametric model.

    Every engineered feature goes through :class:`StandardScaler`; the mass
    hypothesis is the one input that does not, because it is a physical
    parameter.  Fed raw, ``MH4`` (10-1450 GeV) and ``MH3`` (600/1500) sit two
    orders of magnitude away from the standardized features and the MLP trains
    poorly.  Both columns are therefore standardized on the training split
    before they are concatenated to the features, and ``MH4`` is
    log10-transformed first because the mass dependence is multiplicative
    rather than linear.

    This is only ever constructed when ``model.parametric_input`` is true;
    non-parametric models never see a mass input.
    """

    mean: np.ndarray
    std: np.ndarray
    use_log: np.ndarray

    @staticmethod
    def fit(mass: np.ndarray) -> "MassScaler":
        M = np.asarray(mass, dtype="f8")
        if M.ndim != 2:
            raise ValueError(f"Expected 2D mass array, got shape={M.shape}")
        # Column 0 = MH3, column 1 = MH4 (see MASS_DIM in dnn_trainer).
        use_log = np.zeros(M.shape[1], dtype=bool)
        if M.shape[1] > 1:
            use_log[1] = True
        T = _apply_log(M, use_log)
        mean = np.zeros(M.shape[1], dtype="f8")
        std = np.ones(M.shape[1], dtype="f8")
        for j in range(T.shape[1]):
            col = T[np.isfinite(T[:, j]), j]
            if col.size == 0:
                continue
            mean[j] = float(np.mean(col))
            s = float(np.std(col))
            std[j] = s if s > 0 else 1.0
        return MassScaler(mean=mean, std=std, use_log=use_log)

    def transform(self, mass: np.ndarray) -> np.ndarray:
        M = np.asarray(mass, dtype="f8")
        if M.ndim != 2:
            raise ValueError(f"Expected 2D mass array, got shape={M.shape}")
        if M.shape[1] != self.mean.shape[0]:
            raise ValueError(
                f"mass array has {M.shape[1]} columns but the scaler expects "
                f"{self.mean.shape[0]}"
            )
        T = _apply_log(M, self.use_log)
        T = (T - self.mean[None, :]) / self.std[None, :]
        bad = ~np.isfinite(T)
        if np.any(bad):
            T = T.copy()
            T[bad] = 0.0
        return T

    def to_jsonable(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "use_log": self.use_log.tolist(),
        }

    @staticmethod
    def from_jsonable(d: dict) -> "MassScaler":
        return MassScaler(
            mean=np.asarray(d["mean"], dtype="f8"),
            std=np.asarray(d["std"], dtype="f8"),
            use_log=np.asarray(d.get("use_log", []), dtype=bool),
        )


@dataclass(frozen=True)
class StandardScaler:
    """A simple, serializable standard scaler for numpy arrays."""

    mean: np.ndarray
    std: np.ndarray
    missing_sentinel: float = -9999.0

    @staticmethod
    def fit(X: np.ndarray, *, missing_sentinel: float = -9999.0) -> "StandardScaler":
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape={X.shape}")

        mask = np.isfinite(X) & (X != float(missing_sentinel))
        mean = np.zeros(X.shape[1], dtype="f8")
        std = np.ones(X.shape[1], dtype="f8")

        for j in range(X.shape[1]):
            col = X[:, j]
            m = mask[:, j]
            if not np.any(m):
                mean[j] = 0.0
                std[j] = 1.0
                continue
            v = col[m]
            mean[j] = float(np.mean(v))
            s = float(np.std(v))
            std[j] = s if s > 0 else 1.0

        return StandardScaler(mean=mean, std=std, missing_sentinel=float(missing_sentinel))

    def transform(self, X: np.ndarray, *, copy: bool = True) -> np.ndarray:
        """Standardize *X*.

        With ``copy=False`` the transformation is applied to *X* in place.
        The caller then owns the returned array and must not reuse *X*.  This
        matters for training on large samples: the copy-free path avoids a
        second full feature matrix in memory.
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape={X.shape}")

        mean = self.mean.astype(X.dtype, copy=False)
        std = self.std.astype(X.dtype, copy=False)
        # Detect the sentinel *before* scaling: after subtracting the mean a
        # -9999 sentinel is no longer recognizable.
        missing = (~np.isfinite(X)) | (X == float(self.missing_sentinel))
        if copy:
            Xn = (X - mean[None, :]) / std[None, :]
        else:
            Xn = X
            Xn -= mean[None, :]
            Xn /= std[None, :]
        if np.any(missing):
            Xn[missing] = 0.0
        return Xn

    def to_jsonable(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "missing_sentinel": float(self.missing_sentinel),
        }

    @staticmethod
    def from_jsonable(d: dict) -> "StandardScaler":
        mean = np.asarray(d["mean"], dtype="f8")
        std = np.asarray(d["std"], dtype="f8")
        missing_sentinel = float(d.get("missing_sentinel", -9999.0))
        return StandardScaler(mean=mean, std=std, missing_sentinel=missing_sentinel)
