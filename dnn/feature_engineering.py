"""Feature engineering helpers for ppbbchichi DNN workflows.

This module provides:
- A default feature set requested for training/analysis, used as a fallback
  when a caller doesn't have an explicit list (e.g. from configs/dnn.yaml's
  features: key) — callers should prefer their own config-provided list.
- build_feature_frame_from_tree: reads features from a tree by exact branch
  name. A feature name must match a real branch in the ROOT file; there is
  no aliasing or derivation — fix the name in configs/dnn.yaml instead.
"""

from __future__ import annotations

import numpy as np

# Fallback feature list — length is whatever's listed here, not a fixed count.
# Prefer passing an explicit list (e.g. configs/dnn.yaml's features:) to
# build_feature_frame_from_tree instead of relying on this default.
REQUESTED_FEATURES = [
    "costheta_star",
    "JetHT",
    "Jet1BTagScore",
    "dRJet12",
    "dPhiJet12",
    "Jet1Pt",
    "Jet2BTagScore",
    "Jet2Eta",
    "Jet2Pt",
    "eta_Jet1Jet2",
    "M_Jet1Jet2",
    "dPhi_jetMET",
    "dEtaJet12",
    "MET_significance",
    "pT_Jet1Jet2",
    "ratioJet1PtMET",
    "ratioPtJet21",
    "del_plus",
    "Jet1Eta",
    "del_minus",
    "MET_phi",
    "Jet1Phi",
    "MET_pt",
    "Jet2Phi",
    "phi_Jet1Jet2",
]


def _trim_arrays(arrays: dict, max_events: int | None) -> dict[str, np.ndarray]:
    if not arrays:
        return {}
    if max_events is None:
        return {k: np.asarray(v, dtype="f8") for k, v in arrays.items()}
    n = int(max_events)
    return {k: np.asarray(v, dtype="f8")[:n] for k, v in arrays.items()}


def build_feature_frame_from_tree(
    tree,
    features: list[str],
    max_events: int | None = None,
):
    """Build a feature DataFrame from a tree by exact branch name.

    Returns: (dataframe, source_map, used_branches)
    - source_map: feature -> source descriptor ("branch:<name>" or "missing:not_a_branch")
    - used_branches: branch names actually read from tree

    Any *features* entry that isn't a real branch on *tree* comes back as an
    all-NaN column (later filled to SENTINEL by sanitize_feature_frame) — fix
    the name in configs/dnn.yaml rather than adding an alias here.
    """
    import pandas as pd

    available = {str(k) for k in tree.keys()}
    to_read = sorted(b for b in features if b in available)

    arrays_raw = tree.arrays(to_read, library="np") if to_read else {}
    arrays = _trim_arrays(arrays_raw, max_events=max_events)

    if arrays:
        n = len(next(iter(arrays.values())))
    else:
        n_entries = int(tree.num_entries)
        n = min(n_entries, int(max_events)) if max_events is not None else n_entries

    nan_vec = np.full(n, np.nan, dtype="f8")

    source_map: dict[str, str] = {}
    out: dict[str, np.ndarray] = {}
    for feat in features:
        if feat in arrays:
            out[feat] = np.asarray(arrays[feat], dtype="f8")
            source_map[feat] = f"branch:{feat}"
        else:
            out[feat] = nan_vec.copy()
            source_map[feat] = "missing:not_a_branch"

    df = pd.DataFrame(out)
    return df, source_map, to_read


def get_default_feature_csv() -> str:
    return ",".join(REQUESTED_FEATURES)
