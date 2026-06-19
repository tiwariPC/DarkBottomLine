"""Shared helpers for training/applying classifiers.

Purpose:
- Define signal-identification rules (`DEFAULT_SIGNAL_PATTERNS`, `is_signal`).
- Hold small data structures (`DatasetSpec`).
- Provide feature sanitation (`sanitize_feature_frame`) to make training robust
    against non-finite values and boolean columns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


DEFAULT_SIGNAL_PATTERNS = (
    r"chichi",
    r"bbchichi",
    r"\bdm\b",
    r"dark",
    r"invisible",
)


def is_signal(sample_name: str, patterns: Iterable[str] = DEFAULT_SIGNAL_PATTERNS) -> bool:
    s = str(sample_name)
    return any(re.search(p, s, flags=re.IGNORECASE) for p in patterns)


@dataclass(frozen=True)
class DatasetSpec:
    root_path: str
    region: str
    features: list[str]
    weight_branch: str


def sanitize_feature_frame(df):
    # Local import to keep this module lightweight
    import numpy as np

    df = df.copy()
    for c in df.columns:
        a = df[c].to_numpy()
        if a.dtype == bool:
            df[c] = a.astype("int8")
        # Replace non-finite with sentinel
        df[c] = df[c].replace([np.inf, -np.inf], np.nan).fillna(-9999.0)
    return df
