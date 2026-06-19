"""Feature scaling utilities for tabular DNN training.

Purpose:
- Fit a per-feature standardization (mean/std) on the training sample.
- Transform feature matrices consistently for both training and inference.
- Handle missing/sentinel values (default: -9999) without polluting statistics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class StandardScaler:
    """A simple, serializable standard scaler for numpy arrays."""

    mean: np.ndarray
    std: np.ndarray
    missing_sentinel: float = -9999.0

    @staticmethod
    def fit(X: np.ndarray, *, missing_sentinel: float = -9999.0) -> "StandardScaler":
        X = np.asarray(X, dtype="f8")
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

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype="f8")
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape={X.shape}")

        Xn = (X - self.mean[None, :]) / self.std[None, :]
        missing = (~np.isfinite(X)) | (X == float(self.missing_sentinel))
        if np.any(missing):
            Xn = Xn.copy()
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
