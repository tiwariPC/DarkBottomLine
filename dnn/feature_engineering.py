"""Feature engineering helpers for ppbbchichi DNN workflows.

This module provides:
- A canonical 25-feature set requested for training/analysis.
- Branch alias resolution for different ROOT schemas.
- Derived-feature construction when canonical branches are missing.
"""

from __future__ import annotations

from typing import Any

import numpy as np

REQUESTED_FEATURES_25 = [
    "costheta_star",
    "JetHT",
    "Jet1deepCSV",
    "dRJet12",
    "dPhiJet12",
    "Jet1Pt",
    "Jet2deepCSV",
    "Jet2Eta",
    "Jet2Pt",
    "eta_Jet1Jet2",
    "M_Jet1Jet2",
    "dPhi_jetMET",
    "dEtaJet12",
    "pfMetCorrSig",
    "pT_Jet1Jet2",
    "rJet1PtMET",
    "ratioPtJet21",
    "del_plus",
    "Jet1Eta",
    "del_minus",
    "METPhi",
    "Jet1Phi",
    "MET",
    "Jet2Phi",
    "phi_Jet1Jet2",
]

FEATURE_ALIASES = {
    "JetHT": ["JetHT", "jetHT", "HT", "ht"],
    "Jet1deepCSV": ["Jet1deepCSV", "lead_bjet_deepCSV", "lead_bjet_PNetB", "lead_bjet_btagDeepFlavB"],
    "dRJet12": ["dRJet12", "DeltaR_j1j2", "DeltaR_jj"],
    "dPhiJet12": ["dPhiJet12", "DeltaPhi_j1j2", "DeltaPhi_jj"],
    "Jet1Pt": ["Jet1Pt", "jet1_pt", "lead_bjet_pt"],
    "Jet2deepCSV": ["Jet2deepCSV", "sublead_bjet_deepCSV", "sublead_bjet_PNetB", "sublead_bjet_btagDeepFlavB"],
    "Jet2Eta": ["Jet2Eta", "jet2_eta", "sublead_bjet_eta"],
    "Jet2Pt": ["Jet2Pt", "jet2_pt", "sublead_bjet_pt"],
    "eta_Jet1Jet2": ["eta_Jet1Jet2", "dijet_eta"],
    "M_Jet1Jet2": ["M_Jet1Jet2", "dijet_mass"],
    "dPhi_jetMET": ["dPhi_jetMET", "DeltaPhi_j1MET", "DeltaPhi_jetMET"],
    "dEtaJet12": ["dEtaJet12", "DeltaEta_j1j2", "DeltaEta_jj"],
    "pfMetCorrSig": ["pfMetCorrSig", "MET_significance", "met_significance", "puppiMET_significance"],
    "pT_Jet1Jet2": ["pT_Jet1Jet2", "dijet_pt"],
    "Jet1Eta": ["Jet1Eta", "jet1_eta", "lead_bjet_eta"],
    "METPhi": ["METPhi", "pfMetCorrPhi", "puppiMET_phi", "met_phi"],
    "Jet1Phi": ["Jet1Phi", "jet1_phi", "lead_bjet_phi"],
    "MET": ["MET", "pfMetCorrPt", "puppiMET_pt", "met_pt"],
    "Jet2Phi": ["Jet2Phi", "jet2_phi", "sublead_bjet_phi"],
    "phi_Jet1Jet2": ["phi_Jet1Jet2", "dijet_phi"],
}

RAW_ALIASES = {
    "jet1_pt": ["Jet1Pt", "jet1_pt", "lead_bjet_pt"],
    "jet2_pt": ["Jet2Pt", "jet2_pt", "sublead_bjet_pt"],
    "jet1_eta": ["Jet1Eta", "jet1_eta", "lead_bjet_eta"],
    "jet2_eta": ["Jet2Eta", "jet2_eta", "sublead_bjet_eta"],
    "jet1_phi": ["Jet1Phi", "jet1_phi", "lead_bjet_phi"],
    "jet2_phi": ["Jet2Phi", "jet2_phi", "sublead_bjet_phi"],
    "met_pt": ["MET", "pfMetCorrPt", "puppiMET_pt", "met_pt"],
    "met_phi": ["METPhi", "pfMetCorrPhi", "puppiMET_phi", "met_phi"],
    "jet1_btag": ["Jet1deepCSV", "lead_bjet_deepCSV", "lead_bjet_PNetB", "lead_bjet_btagDeepFlavB"],
    "jet2_btag": ["Jet2deepCSV", "sublead_bjet_deepCSV", "sublead_bjet_PNetB", "sublead_bjet_btagDeepFlavB"],
    "dijet_mass": ["M_Jet1Jet2", "dijet_mass"],
    "dijet_pt": ["pT_Jet1Jet2", "dijet_pt"],
    "dijet_eta": ["eta_Jet1Jet2", "dijet_eta"],
    "dijet_phi": ["phi_Jet1Jet2", "dijet_phi"],
    "jet_ht": ["JetHT", "jetHT", "HT", "ht"],
    "met_sig": ["pfMetCorrSig", "MET_significance", "met_significance", "puppiMET_significance"],
    "deta_j12": ["dEtaJet12", "DeltaEta_j1j2", "DeltaEta_jj"],
    "dphi_j12": ["dPhiJet12", "DeltaPhi_j1j2", "DeltaPhi_jj"],
    "dr_j12": ["dRJet12", "DeltaR_j1j2", "DeltaR_jj"],
    "dphi_j1_met": ["dPhi_jetMET", "DeltaPhi_j1MET", "DeltaPhi_jetMET"],
}

KNOWN_BRANCHES = set()
for vals in FEATURE_ALIASES.values():
    KNOWN_BRANCHES.update(vals)
for vals in RAW_ALIASES.values():
    KNOWN_BRANCHES.update(vals)


def _wrap_phi(delta_phi: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(delta_phi), np.cos(delta_phi))


def _safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.zeros_like(num, dtype="f8")
    ok = np.isfinite(num) & np.isfinite(den) & (np.abs(den) > 1e-12)
    out[ok] = num[ok] / den[ok]
    return out


def _find_first_available(arrays: dict[str, Any], aliases: list[str]) -> tuple[np.ndarray | None, str | None]:
    for name in aliases:
        if name in arrays:
            return np.asarray(arrays[name], dtype="f8"), name
    return None, None


def _trim_arrays(arrays: dict[str, Any], max_events: int | None) -> dict[str, np.ndarray]:
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
    """Build a feature DataFrame from a tree.

    Returns: (dataframe, source_map, used_branches)
    - source_map: feature -> source descriptor
    - used_branches: branch names actually read from tree
    """
    import pandas as pd

    available = {str(k) for k in tree.keys()}
    to_read = sorted([b for b in available if b in KNOWN_BRANCHES])

    arrays_raw = tree.arrays(to_read, library="np") if to_read else {}
    arrays = _trim_arrays(arrays_raw, max_events=max_events)

    if arrays:
        n = len(next(iter(arrays.values())))
    else:
        n_entries = int(tree.num_entries)
        n = min(n_entries, int(max_events)) if max_events is not None else n_entries

    nan_vec = np.full(n, np.nan, dtype="f8")

    raw_cache: dict[str, tuple[np.ndarray | None, str | None]] = {}
    feat_cache: dict[str, np.ndarray] = {}
    source_map: dict[str, str] = {}

    def raw(name: str) -> tuple[np.ndarray | None, str | None]:
        if name in raw_cache:
            return raw_cache[name]
        aliases = RAW_ALIASES.get(name, [name])
        arr, alias = _find_first_available(arrays, aliases)
        raw_cache[name] = (arr, alias)
        return raw_cache[name]

    def save_feat(name: str, value: np.ndarray, source: str) -> np.ndarray:
        feat_cache[name] = np.asarray(value, dtype="f8")
        source_map[name] = source
        return feat_cache[name]

    def get_feat(name: str) -> np.ndarray:
        if name in feat_cache:
            return feat_cache[name]

        aliases = FEATURE_ALIASES.get(name, [name])
        direct, direct_name = _find_first_available(arrays, aliases)
        if direct is not None:
            return save_feat(name, direct, f"branch:{direct_name}")

        if name == "Jet1Pt":
            v, src = raw("jet1_pt")
            return save_feat(name, v if v is not None else nan_vec.copy(), f"derived:{src or 'missing'}")

        if name == "Jet2Pt":
            v, src = raw("jet2_pt")
            return save_feat(name, v if v is not None else nan_vec.copy(), f"derived:{src or 'missing'}")

        if name == "Jet1Eta":
            v, src = raw("jet1_eta")
            return save_feat(name, v if v is not None else nan_vec.copy(), f"derived:{src or 'missing'}")

        if name == "Jet2Eta":
            v, src = raw("jet2_eta")
            return save_feat(name, v if v is not None else nan_vec.copy(), f"derived:{src or 'missing'}")

        if name == "Jet1Phi":
            v, src = raw("jet1_phi")
            return save_feat(name, v if v is not None else nan_vec.copy(), f"derived:{src or 'missing'}")

        if name == "Jet2Phi":
            v, src = raw("jet2_phi")
            return save_feat(name, v if v is not None else nan_vec.copy(), f"derived:{src or 'missing'}")

        if name == "MET":
            v, src = raw("met_pt")
            return save_feat(name, v if v is not None else nan_vec.copy(), f"derived:{src or 'missing'}")

        if name == "METPhi":
            v, src = raw("met_phi")
            return save_feat(name, v if v is not None else nan_vec.copy(), f"derived:{src or 'missing'}")

        if name == "Jet1deepCSV":
            v, src = raw("jet1_btag")
            return save_feat(name, v if v is not None else nan_vec.copy(), f"derived:{src or 'missing'}")

        if name == "Jet2deepCSV":
            v, src = raw("jet2_btag")
            return save_feat(name, v if v is not None else nan_vec.copy(), f"derived:{src or 'missing'}")

        if name == "dEtaJet12":
            a = get_feat("Jet1Eta")
            b = get_feat("Jet2Eta")
            return save_feat(name, a - b, "derived:Jet1Eta-Jet2Eta")

        if name == "dPhiJet12":
            a = get_feat("Jet1Phi")
            b = get_feat("Jet2Phi")
            return save_feat(name, _wrap_phi(a - b), "derived:wrap(Jet1Phi-Jet2Phi)")

        if name == "dRJet12":
            deta = get_feat("dEtaJet12")
            dphi = get_feat("dPhiJet12")
            return save_feat(name, np.sqrt(np.maximum(deta * deta + dphi * dphi, 0.0)), "derived:sqrt(dEta^2+dPhi^2)")

        if name == "costheta_star":
            deta = get_feat("dEtaJet12")
            return save_feat(name, np.abs(np.tanh(0.5 * deta)), "derived:abs(tanh(dEtaJet12/2))")

        if name == "JetHT":
            v, src = raw("jet_ht")
            if v is not None:
                return save_feat(name, v, f"derived:{src}")
            j1 = get_feat("Jet1Pt")
            j2 = get_feat("Jet2Pt")
            return save_feat(name, j1 + j2, "derived_approx:Jet1Pt+Jet2Pt")

        if name == "dPhi_jetMET":
            v, src = raw("dphi_j1_met")
            if v is not None:
                return save_feat(name, v, f"derived:{src}")
            jphi = get_feat("Jet1Phi")
            mphi = get_feat("METPhi")
            return save_feat(name, _wrap_phi(jphi - mphi), "derived:wrap(Jet1Phi-METPhi)")

        if name == "pT_Jet1Jet2":
            v, src = raw("dijet_pt")
            if v is not None:
                return save_feat(name, v, f"derived:{src}")
            pt1 = get_feat("Jet1Pt")
            pt2 = get_feat("Jet2Pt")
            phi1 = get_feat("Jet1Phi")
            phi2 = get_feat("Jet2Phi")
            px = pt1 * np.cos(phi1) + pt2 * np.cos(phi2)
            py = pt1 * np.sin(phi1) + pt2 * np.sin(phi2)
            return save_feat(name, np.sqrt(np.maximum(px * px + py * py, 0.0)), "derived:sqrt((px1+px2)^2+(py1+py2)^2)")

        if name == "M_Jet1Jet2":
            v, src = raw("dijet_mass")
            if v is not None:
                return save_feat(name, v, f"derived:{src}")
            pt1 = get_feat("Jet1Pt")
            pt2 = get_feat("Jet2Pt")
            eta1 = get_feat("Jet1Eta")
            eta2 = get_feat("Jet2Eta")
            phi1 = get_feat("Jet1Phi")
            phi2 = get_feat("Jet2Phi")
            px1 = pt1 * np.cos(phi1)
            py1 = pt1 * np.sin(phi1)
            pz1 = pt1 * np.sinh(eta1)
            px2 = pt2 * np.cos(phi2)
            py2 = pt2 * np.sin(phi2)
            pz2 = pt2 * np.sinh(eta2)
            e1 = np.sqrt(np.maximum(px1 * px1 + py1 * py1 + pz1 * pz1, 0.0))
            e2 = np.sqrt(np.maximum(px2 * px2 + py2 * py2 + pz2 * pz2, 0.0))
            e = e1 + e2
            px = px1 + px2
            py = py1 + py2
            pz = pz1 + pz2
            m2 = np.maximum(e * e - (px * px + py * py + pz * pz), 0.0)
            return save_feat(name, np.sqrt(m2), "derived:massless_dijet_invariant_mass")

        if name == "eta_Jet1Jet2":
            v, src = raw("dijet_eta")
            if v is not None:
                return save_feat(name, v, f"derived:{src}")
            pt1 = get_feat("Jet1Pt")
            pt2 = get_feat("Jet2Pt")
            eta1 = get_feat("Jet1Eta")
            eta2 = get_feat("Jet2Eta")
            phi1 = get_feat("Jet1Phi")
            phi2 = get_feat("Jet2Phi")
            px = pt1 * np.cos(phi1) + pt2 * np.cos(phi2)
            py = pt1 * np.sin(phi1) + pt2 * np.sin(phi2)
            pz = pt1 * np.sinh(eta1) + pt2 * np.sinh(eta2)
            p = np.sqrt(np.maximum(px * px + py * py + pz * pz, 0.0))
            num = np.maximum(p + pz, 1e-12)
            den = np.maximum(p - pz, 1e-12)
            return save_feat(name, 0.5 * np.log(num / den), "derived:dijet_eta_from_vector_sum")

        if name == "phi_Jet1Jet2":
            v, src = raw("dijet_phi")
            if v is not None:
                return save_feat(name, v, f"derived:{src}")
            pt1 = get_feat("Jet1Pt")
            pt2 = get_feat("Jet2Pt")
            phi1 = get_feat("Jet1Phi")
            phi2 = get_feat("Jet2Phi")
            px = pt1 * np.cos(phi1) + pt2 * np.cos(phi2)
            py = pt1 * np.sin(phi1) + pt2 * np.sin(phi2)
            return save_feat(name, np.arctan2(py, px), "derived:atan2(py1+py2,px1+px2)")

        if name == "rJet1PtMET":
            pt1 = get_feat("Jet1Pt")
            met = get_feat("MET")
            return save_feat(name, _safe_divide(pt1, met), "derived:Jet1Pt/MET")

        if name == "ratioPtJet21":
            pt2 = get_feat("Jet2Pt")
            pt1 = get_feat("Jet1Pt")
            return save_feat(name, _safe_divide(pt2, pt1), "derived:Jet2Pt/Jet1Pt")

        if name == "del_plus":
            d1 = get_feat("dPhi_jetMET")
            d2 = get_feat("dPhiJet12")
            return save_feat(name, np.abs(d1 + d2 - np.pi), "derived:abs(dPhi_jetMET+dPhiJet12-pi)")

        if name == "del_minus":
            d1 = get_feat("dPhi_jetMET")
            d2 = get_feat("dPhiJet12")
            return save_feat(name, d1 - d2, "derived:dPhi_jetMET-dPhiJet12")

        if name == "pfMetCorrSig":
            v, src = raw("met_sig")
            if v is not None:
                return save_feat(name, v, f"derived:{src}")
            return save_feat(name, nan_vec.copy(), "missing:no_formula")

        return save_feat(name, nan_vec.copy(), "missing:unknown_feature")

    out = {}
    for feat in features:
        out[feat] = get_feat(feat)

    df = pd.DataFrame(out)
    return df, source_map, to_read


def get_default_feature_csv() -> str:
    return ",".join(REQUESTED_FEATURES_25)
