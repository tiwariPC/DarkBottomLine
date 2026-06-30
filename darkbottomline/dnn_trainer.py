"""
Parametric DNN training for DarkBottomLine framework.

Uses dnn.model / dnn.scaler / dnn.feature_engineering primitives.
All training logic mirrors dnn/train_classifier.py:
  - 60/20/20 stratified train/val/test split
  - weighted BCE loss + optional class balancing
  - topology decorrelation penalty
  - Asimov significance-based feature ranking / top-K selection
  - per-feature 1D DNN scans
  - ROC, loss/AUC, score distribution plots

Public API (DNNTrainer, ParametricDNN) preserved for cli.py and tests.
"""

from __future__ import annotations

import csv
import json
import logging
import yaml
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dnn.model import ModelSpec, build_mlp, parse_hidden_layers, save_checkpoint, load_checkpoint
from dnn.scaler import StandardScaler as _StandardScaler


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def _resolve_device(cfg: dict) -> torch.device:
    d = cfg.get("hardware", {}).get("device", "auto")
    if d == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if d == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda requested but CUDA not available")
    return torch.device(d)


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

def _drop_constant_features(df, *, missing_sentinel: float = -9999.0, eps: float = 1e-12):
    kept, dropped = [], []
    for c in list(df.columns):
        a = np.asarray(df[c].to_numpy(), dtype="f8")
        m = np.isfinite(a) & (a != float(missing_sentinel))
        s = float(np.std(a[m])) if np.any(m) else 0.0
        (kept if s > float(eps) else dropped).append(str(c))
    if not kept:
        raise ValueError("All features dropped as near-constant; cannot train.")
    return df[kept], kept, dropped


def _weighted_percentile(values: np.ndarray, weights: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.full_like(quantiles, np.nan, dtype="f8")
    sorter = np.argsort(values)
    v = values[sorter]
    w = weights[sorter]
    cdf = np.cumsum(w)
    if cdf.size == 0 or cdf[-1] <= 0:
        return np.percentile(v, quantiles * 100.0)
    cdf = cdf / cdf[-1]
    return np.interp(quantiles, cdf, v)


def _asimov_significance_from_hist(sig: np.ndarray, bkg: np.ndarray, eps: float = 1e-12) -> float:
    s = np.maximum(np.asarray(sig, dtype="f8"), 0.0)
    b = np.maximum(np.asarray(bkg, dtype="f8"), 0.0)
    term = np.where(
        b > eps,
        (s + b) * np.log1p(np.divide(s, b, out=np.zeros_like(s), where=b > eps)) - s,
        0.0,
    )
    return float(np.sqrt(max(2.0 * float(np.sum(np.maximum(term, 0.0))), 0.0)))


def _asimov_significance_from_hist_syst(
    sig: np.ndarray,
    bkg: np.ndarray,
    sigma_rel: float,
    eps: float = 1e-12,
) -> float:
    s = np.maximum(np.asarray(sig, dtype="f8"), 0.0)
    b = np.maximum(np.asarray(bkg, dtype="f8"), 0.0)
    if sigma_rel <= 0.0:
        return _asimov_significance_from_hist(s, b, eps=eps)
    sigma_b2 = (sigma_rel * b) ** 2
    mask_syst = (b > eps) & (sigma_b2 > eps)
    term = np.zeros_like(s, dtype="f8")
    if np.any(mask_syst):
        s_m = s[mask_syst]; b_m = b[mask_syst]; sb2_m = sigma_b2[mask_syst]
        num = (s_m + b_m) * (b_m + sb2_m)
        den = b_m * b_m + (s_m + b_m) * sb2_m
        ratio1 = np.where(den > eps, num / den, 1.0)
        term1 = np.where(ratio1 > 1.0 + eps, (s_m + b_m) * np.log(ratio1), 0.0)
        ratio2 = sb2_m * s_m / np.maximum(b_m * (b_m + sb2_m), eps)
        term2 = (b_m * b_m / np.maximum(sb2_m, eps)) * np.log1p(ratio2)
        term[mask_syst] = term1 - term2
    mask_pure = ~mask_syst & (b > eps)
    if np.any(mask_pure):
        s_p = s[mask_pure]; b_p = b[mask_pure]
        term[mask_pure] = (s_p + b_p) * np.log1p(s_p / b_p) - s_p
    z2 = 2.0 * np.sum(np.maximum(term, 0.0))
    return float(np.sqrt(max(z2, 0.0)))


def _compute_feature_significance(
    X_df,
    y: np.ndarray,
    w: np.ndarray,
    features: list[str],
    outdir: Path,
    source_map: dict | None = None,
    n_bins: int = 40,
    sig_syst: float = 0.0,
) -> list[dict]:
    from sklearn.metrics import roc_auc_score

    outdir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    y_i = np.asarray(y, dtype="i4")
    w_f = np.maximum(np.asarray(w, dtype="f8"), 0.0)

    # Counting-experiment baseline (pure S/√B)
    S_total = float(np.sum(w_f[y_i == 1]))
    B_total = float(np.sum(w_f[y_i == 0]))
    sig_syst_val = float(sig_syst)
    z_counting_stat = _asimov_significance_from_hist(
        np.array([max(S_total, 0.0)]), np.array([max(B_total, 0.0)]),
    )
    if sig_syst_val > 0.0:
        z_counting_syst = _asimov_significance_from_hist_syst(
            np.array([max(S_total, 0.0)]), np.array([max(B_total, 0.0)]), sig_syst_val,
        )
    else:
        z_counting_syst = z_counting_stat

    for feat in features:
        x = np.asarray(X_df[feat].to_numpy(), dtype="f8")
        m = np.isfinite(x)
        x, yy, ww = x[m], y_i[m], w_f[m]

        if x.size == 0 or np.sum(ww[yy == 1]) <= 0 or np.sum(ww[yy == 0]) <= 0:
            rows.append({"feature": feat, "source": (source_map or {}).get(feat, "unknown"),
                         "auc": float("nan"), "asimov_z": 0.0, "asimov_z_syst": 0.0,
                         "delta_z": 0.0, "delta_z_syst": 0.0})
            continue

        qlo, qhi = _weighted_percentile(x, ww, np.array([0.01, 0.99], dtype="f8"))
        lo = float(np.nanmin(x) if not np.isfinite(qlo) else qlo)
        hi = float(np.nanmax(x) if not np.isfinite(qhi) else qhi)
        if hi <= lo:
            hi = lo + 1.0

        edges = np.linspace(lo, hi, int(n_bins) + 1, dtype="f8")
        hs, _ = np.histogram(x[yy == 1], bins=edges, weights=ww[yy == 1])
        hb, _ = np.histogram(x[yy == 0], bins=edges, weights=ww[yy == 0])
        z = _asimov_significance_from_hist(hs, hb)
        z_syst = _asimov_significance_from_hist_syst(hs, hb, float(sig_syst))
        auc = float(roc_auc_score(yy, x, sample_weight=ww))
        rows.append({"feature": feat, "source": (source_map or {}).get(feat, "unknown"),
                     "auc": auc, "asimov_z": z, "asimov_z_syst": z_syst,
                     "delta_z": z - z_counting_stat, "delta_z_syst": z_syst - z_counting_syst})

    sort_key = "asimov_z_syst" if float(sig_syst) > 0.0 else "asimov_z"
    rows.sort(key=lambda r: (r[sort_key], abs((r["auc"] if np.isfinite(r["auc"]) else 0.5) - 0.5)), reverse=True)

    import pandas as pd
    pd.DataFrame(rows).to_csv(outdir / "feature_significance.csv", index=False)
    (outdir / "feature_significance.json").write_text(json.dumps(rows, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [r["feature"] for r in rows]
    vals_stat = [float(r["asimov_z"]) for r in rows]
    vals_syst = [float(r["asimov_z_syst"]) for r in rows]
    has_syst = float(sig_syst) > 0.0

    def _draw_bars(ax, vals, ylabel, title, color="#3f90da"):
        ax.bar(labels, vals, color=color, edgecolor="black", linewidth=0.8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=40, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    if has_syst:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(16, 1.1 * len(labels)), 5.5))
        _draw_bars(ax1, vals_stat, "Asimov Z (pure stat)", "Per-feature significance (pure statistical)")
        syst_label = f"syst-aware (\u03c3_rel={float(sig_syst)*100:.0f}%)"
        _draw_bars(ax2, vals_syst, f"Asimov Z ({syst_label})", f"Per-feature significance ({syst_label})", "#d62728")
        fig.tight_layout()
        fig.savefig(outdir / "feature_significance.png", dpi=160)
        plt.close(fig)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(max(12, 0.7 * len(labels)), 5.5))
        _draw_bars(ax, vals_stat, "Asimov Z (pure stat)", "Per-feature significance (pure statistical)")
        fig.tight_layout()
        fig.savefig(outdir / "feature_significance.png", dpi=160)
        plt.close(fig)

    # Delta-Z plot
    vals_delta_stat = [float(r["delta_z"]) for r in rows]
    vals_delta_syst = [float(r["delta_z_syst"]) for r in rows]
    if has_syst:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(16, 1.1 * len(labels)), 5.5))
        _draw_bars(ax1, vals_delta_stat, "\u0394Z (pure stat)",
                   f"Effective significance (pure stat)\nbaseline = S/\u221aB = {z_counting_stat:.2f}\u03c3")
        ax1.axhline(y=0, color="gray", linewidth=0.8, linestyle="--")
        syst_label = f"syst-aware (\u03c3_rel={float(sig_syst)*100:.0f}%)"
        _draw_bars(ax2, vals_delta_syst, f"\u0394Z ({syst_label})",
                   f"Effective significance ({syst_label})\nbaseline = S/\u221aB (syst) = {z_counting_syst:.2f}\u03c3", "#d62728")
        ax2.axhline(y=0, color="gray", linewidth=0.8, linestyle="--")
        fig.tight_layout()
        fig.savefig(outdir / "feature_significance_delta.png", dpi=160)
        plt.close(fig)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(max(12, 0.7 * len(labels)), 5.5))
        _draw_bars(ax, vals_delta_stat, "\u0394Z (pure stat)",
                   f"Effective significance\nbaseline = S/\u221aB = {z_counting_stat:.2f}\u03c3")
        ax.axhline(y=0, color="gray", linewidth=0.8, linestyle="--")
        fig.tight_layout()
        fig.savefig(outdir / "feature_significance_delta.png", dpi=160)
        plt.close(fig)

    return rows


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _plot_score_distribution(
    y_true: np.ndarray,
    y_score: np.ndarray,
    weights: np.ndarray,
    out_path: Path,
    title: str,
    n_bins: int = 50,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = np.asarray(y_true, dtype="i4")
    s = np.clip(np.asarray(y_score, dtype="f8"), 0.0, 1.0)
    w = np.maximum(np.asarray(weights, dtype="f8"), 0.0)
    m = np.isfinite(s) & np.isfinite(w)
    y, s, w = y[m], s[m], w[m]

    bins = np.linspace(0.0, 1.0, int(n_bins) + 1, dtype="f8")
    hs, _ = np.histogram(s[y == 1], bins=bins, weights=w[y == 1])
    hb, _ = np.histogram(s[y == 0], bins=bins, weights=w[y == 0])
    hs = hs / max(float(np.sum(hs)), 1e-12)
    hb = hb / max(float(np.sum(hb)), 1e-12)
    centers = 0.5 * (bins[:-1] + bins[1:])

    plt.figure(figsize=(7.0, 5.5))
    plt.step(centers, hs, where="mid", linewidth=2.0, color="#bd1f01", label="Signal")
    plt.step(centers, hb, where="mid", linewidth=2.0, color="#3f90da", label="Background")
    plt.xlim(0.0, 1.0)
    plt.xlabel("DNN score")
    plt.ylabel("Normalized events")
    plt.title(title)
    plt.grid(alpha=0.22)
    plt.legend(loc="best")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def _write_score_table(
    y_true: np.ndarray,
    y_score: np.ndarray,
    weights: np.ndarray,
    out_path: Path,
    n_bins: int = 50,
) -> None:
    import pandas as pd

    y = np.asarray(y_true, dtype="i4")
    s = np.clip(np.asarray(y_score, dtype="f8"), 0.0, 1.0)
    w = np.maximum(np.asarray(weights, dtype="f8"), 0.0)
    m = np.isfinite(s) & np.isfinite(w)
    y, s, w = y[m], s[m], w[m]

    bins = np.linspace(0.0, 1.0, int(n_bins) + 1, dtype="f8")
    hs, _ = np.histogram(s[y == 1], bins=bins, weights=w[y == 1])
    hb, _ = np.histogram(s[y == 0], bins=bins, weights=w[y == 0])
    hs = hs / max(float(np.sum(hs)), 1e-12)
    hb = hb / max(float(np.sum(hb)), 1e-12)

    pd.DataFrame({
        "bin_low": bins[:-1], "bin_high": bins[1:],
        "bin_center": 0.5 * (bins[:-1] + bins[1:]),
        "signal_norm": hs, "background_norm": hb,
    }).to_csv(out_path, index=False)


# ---------------------------------------------------------------------------
# Topology decorrelation penalty
# ---------------------------------------------------------------------------

def _topology_decorrelation_penalty(
    logits,
    inputs,
    weights,
    signal_mask,
    nuisance_indices: list[int],
    *,
    eps: float = 1e-12,
):
    if not nuisance_indices or signal_mask.sum().item() < 4:
        return logits.new_tensor(0.0)

    sig_logits = logits[signal_mask]
    sig_inputs = inputs[signal_mask][:, nuisance_indices]
    sig_weights = weights[signal_mask]

    wsum = sig_weights.sum()
    if not torch.isfinite(wsum) or float(wsum.detach().cpu().item()) <= eps:
        return logits.new_tensor(0.0)

    w = sig_weights / (wsum + eps)
    log_c = sig_logits - torch.sum(w * sig_logits)
    feat_c = sig_inputs - torch.sum(w[:, None] * sig_inputs, dim=0, keepdim=True)

    log_var = torch.sum(w * log_c * log_c)
    feat_var = torch.sum(w[:, None] * feat_c * feat_c, dim=0)
    cov = torch.sum(w[:, None] * log_c[:, None] * feat_c, dim=0)
    corr = cov / (torch.sqrt(log_var + eps) * torch.sqrt(feat_var + eps) + eps)
    corr = torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.mean(corr * corr)


# ---------------------------------------------------------------------------
# Per-feature 1D DNN scan
# ---------------------------------------------------------------------------

def _train_single_feature_dnn(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    w_train: np.ndarray,
    w_test: np.ndarray,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    dropout: float,
    device,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    from sklearn.metrics import roc_auc_score

    xtr = np.asarray(x_train, dtype="f8").reshape(-1, 1)
    xte = np.asarray(x_test, dtype="f8").reshape(-1, 1)
    scaler = _StandardScaler.fit(xtr, missing_sentinel=-9999.0)
    xtr_n = scaler.transform(xtr).astype("float32")
    xte_n = scaler.transform(xte).astype("float32")

    torch.manual_seed(int(seed))
    spec = ModelSpec(n_inputs=1, hidden_layers=(16, 16), dropout=float(min(max(dropout, 0.0), 0.2)))
    model = build_mlp(spec).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=float(lr))
    bce = nn.BCEWithLogitsLoss(reduction="none")

    Xtr = torch.from_numpy(xtr_n)
    ytr = torch.from_numpy(np.asarray(y_train, dtype="float32"))
    wtr = torch.from_numpy(np.asarray(w_train, dtype="float32"))
    Xte = torch.from_numpy(xte_n)

    loader = DataLoader(TensorDataset(Xtr, ytr, wtr), batch_size=max(256, int(batch_size)), shuffle=True, drop_last=False)
    best_auc, best_state, bad = -np.inf, None, 0

    for _ in range(max(1, epochs)):
        model.train()
        for xb, yb, wb in loader:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            optim.zero_grad(set_to_none=True)
            loss = (bce(model(xb).squeeze(1), yb) * (wb / (wb.mean() + 1e-12))).mean()
            loss.backward()
            optim.step()

        model.eval()
        with torch.no_grad():
            score_te = torch.sigmoid(model(Xte.to(device)).squeeze(1)).cpu().numpy()
        auc_te = float(roc_auc_score(np.asarray(y_test, dtype="i4"), score_te, sample_weight=np.asarray(w_test, dtype="f8")))

        if auc_te > best_auc + 1e-6:
            best_auc = auc_te
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if patience > 0 and bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        score_tr = torch.sigmoid(model(Xtr.to(device)).squeeze(1)).cpu().numpy()
        score_te = torch.sigmoid(model(Xte.to(device)).squeeze(1)).cpu().numpy()

    auc_tr = float(roc_auc_score(np.asarray(y_train, dtype="i4"), score_tr, sample_weight=np.asarray(w_train, dtype="f8")))
    auc_te = float(roc_auc_score(np.asarray(y_test, dtype="i4"), score_te, sample_weight=np.asarray(w_test, dtype="f8")))
    return score_tr, score_te, auc_tr, auc_te


# ---------------------------------------------------------------------------
# ParametricDNN — thin wrapper so existing code still works
# ---------------------------------------------------------------------------

class ParametricDNN(nn.Module):
    """Wraps dnn.model.build_mlp. mass is concatenated when parametric_input=True."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.parametric_input = bool(config.get("parametric_input", True))

        n_inputs = int(config.get("input_features", 25))
        if self.parametric_input:
            n_inputs += 1

        hidden = config.get("hidden_layers", [128, 128])
        if isinstance(hidden, str):
            hidden = list(parse_hidden_layers(hidden))

        spec = ModelSpec(
            n_inputs=n_inputs,
            hidden_layers=tuple(int(h) for h in hidden),
            dropout=float(config.get("dropout", 0.1)),
        )
        self._net = build_mlp(spec)
        self._spec = spec

    def forward(self, x: torch.Tensor, mass: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.parametric_input and mass is not None:
            x = torch.cat([x, mass.unsqueeze(-1)], dim=-1)
        return torch.sigmoid(self._net(x))


# ---------------------------------------------------------------------------
# DNNTrainer
# ---------------------------------------------------------------------------

class DNNTrainer:
    """
    DNN trainer for signal-background classification.

    Wraps dnn.model / dnn.scaler / dnn.data / dnn.feature_engineering.
    Training logic mirrors dnn/train_classifier.py (weighted BCE, topology
    decorrelation, Asimov feature ranking, per-feature scans).
    """

    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.model_config = self.config.get("model", {})
        self.training_config = self.config.get("training", {})
        self.preprocessing_config = self.config.get("preprocessing", {})
        self.features_config = self.config.get("features", {})
        self.data_config = self.config.get("data", {})
        self.output_config = self.config.get("output", {})
        self.evaluation_config = self.config.get("evaluation", {})
        self.feature_selection_config = self.config.get("feature_selection", {})
        self.topo_config = self.config.get("topology_decorrelation", {})

        self.model = ParametricDNN(self.model_config)
        self._dnn_scaler: Optional[_StandardScaler] = None
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "train_auc": [], "val_auc": [],
        }

        self.device = _resolve_device(self.config)
        self.model.to(self.device)

        logging.info("Initialized DNNTrainer with %d parameters", sum(p.numel() for p in self.model.parameters()))

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data(
        self,
        signal_files: List[str],
        background_files: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load training data from ROOT files using dnn.data helpers."""
        import uproot
        from dnn.data import list_sample_region_trees, read_tree_as_arrays
        from dnn.feature_engineering import build_feature_frame_from_tree, REQUESTED_FEATURES_25
        from dnn.common import sanitize_feature_frame

        region = self.data_config.get("region", "preselection")
        features_req = list(REQUESTED_FEATURES_25)
        weight_branch = f"weight_{region}"
        max_ev = int(self.data_config.get("max_events_per_sample", 200000))

        X_parts, y_parts, w_parts = [], [], []

        def _load_root(path: str, label: int):
            with uproot.open(path) as f:
                for _sample, tpath in list_sample_region_trees(f, region):
                    tree = f[tpath]
                    df, _, _ = build_feature_frame_from_tree(tree, features_req, max_events=max_ev)
                    df = sanitize_feature_frame(df)
                    arrs = read_tree_as_arrays(f, tpath, branches=[weight_branch], max_events=max_ev)
                    w = np.asarray(arrs[weight_branch], dtype="f8")
                    w = np.where(np.isfinite(w), np.maximum(w, 0.0), 0.0)
                    n = min(len(df), len(w))
                    X_parts.append(df.iloc[:n].to_numpy(dtype="f8"))
                    y_parts.append(np.full(n, label, dtype="i4"))
                    w_parts.append(w[:n])

        for fp in signal_files:
            _load_root(fp, 1)
        for fp in background_files:
            _load_root(fp, 0)

        if not X_parts:
            raise ValueError("No events loaded — check file paths and region name.")

        X = np.vstack(X_parts)
        y = np.concatenate(y_parts).astype("f8")
        w = np.concatenate(w_parts)
        masses = np.zeros(len(X))
        logging.info("Loaded %d events (%d signal, %d background)", len(X), int((y == 1).sum()), int((y == 0).sum()))
        return X, y, masses

    # ------------------------------------------------------------------
    # Preprocessing  (legacy path: used when caller manages splits)
    # ------------------------------------------------------------------

    def preprocess(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        masses: np.ndarray,
    ) -> Tuple[torch.Tensor, ...]:
        """Scale features and do a simple train/val split (legacy two-way split)."""
        from sklearn.model_selection import train_test_split

        seed = int(self.preprocessing_config.get("random_seed", 42))
        val_frac = 1.0 - float(self.preprocessing_config.get("train_validation_split", 0.8))

        X_tr, X_va, y_tr, y_va, m_tr, m_va = train_test_split(
            features, labels, masses,
            test_size=val_frac, random_state=seed, stratify=labels.astype("i4"),
        )

        self._dnn_scaler = _StandardScaler.fit(X_tr.astype("f8"), missing_sentinel=-9999.0)
        X_tr_n = self._dnn_scaler.transform(X_tr).astype("float32")
        X_va_n = self._dnn_scaler.transform(X_va).astype("float32")

        def _t(a): return torch.from_numpy(a.astype("float32"))

        logging.info("Preprocessed: %d train, %d val", len(X_tr_n), len(X_va_n))
        return _t(X_tr_n), _t(y_tr), _t(m_tr), _t(X_va_n), _t(y_va), _t(m_va)

    # ------------------------------------------------------------------
    # Training from pre-loaded arrays (no intermediate file)
    # Called by CLI train-dnn when input = flat event-selection ROOT files
    # ------------------------------------------------------------------

    def train_from_arrays(
        self,
        X,  # pandas DataFrame, columns = feature names
        y: np.ndarray,
        w: np.ndarray,
        feature_sources: Optional[Dict[str, str]] = None,
        outdir: str = "data/dnn",
        plot_dir: str = "outputs/dnn",
    ) -> Dict[str, Any]:
        """Full training pipeline from pre-loaded (X, y, w) arrays.

        Identical logic to train_from_root() but skips the ROOT/uproot loading phase.
        Useful when inputs are flat event-selection ROOT files (not ppbbchichi-trees).
        """
        features = list(X.columns)
        return self._run_training_pipeline(
            X=X,
            y=y,
            w=w,
            features=features,
            feature_sources=feature_sources or {},
            outdir=outdir,
            plot_dir=plot_dir,
            region="preselection",
            root_path="<in-memory>",
        )

    # ------------------------------------------------------------------
    # Full training pipeline (mirrors train_classifier.py)
    # ------------------------------------------------------------------

    def train_from_root(
        self,
        root_path: str,
        region: str = "preselection",
        outdir: str = "data/dnn",
        plot_dir: str = "outputs/dnn",
        features: Optional[List[str]] = None,
        label_csv: Optional[str] = None,
        signal_patterns: Optional[List[str]] = None,
        signal_prefix: Optional[str] = None,
        exclude_prefixes: str = "run",
        max_events_per_sample: int = 200000,
    ) -> Dict[str, Any]:
        """Full training pipeline from a ppbbchichi ROOT file."""
        import uproot
        import pandas as pd
        from dnn.data import list_sample_region_trees, read_tree_as_arrays
        from dnn.feature_engineering import build_feature_frame_from_tree, REQUESTED_FEATURES_25
        from dnn.common import DEFAULT_SIGNAL_PATTERNS, is_signal, sanitize_feature_frame

        if features is None:
            cfg_feats = self.config.get("features", None)
            if isinstance(cfg_feats, list) and cfg_feats:
                features = [str(f) for f in cfg_feats]
            else:
                features = list(REQUESTED_FEATURES_25)

        tc = self.training_config
        weight_clip = float(tc.get("weight_clip", 100.0))
        excl = tuple(p.strip().lower() for p in str(exclude_prefixes).split(",") if p.strip())
        patterns = tuple(signal_patterns) if signal_patterns else DEFAULT_SIGNAL_PATTERNS
        weight_branch = f"weight_{region}"

        label_map: Optional[Dict[str, int]] = None
        if label_csv:
            label_map = {}
            with open(label_csv, "r", newline="") as fp:
                reader = csv.DictReader(fp)
                for row in reader:
                    s = str(row["sample"]).strip()
                    if s:
                        label_map[s] = int(row["label"])

        X_parts, y_parts, w_parts = [], [], []
        feature_sources: Dict[str, str] = {}
        used_samples: Dict[str, int] = {}

        with uproot.open(root_path) as f:
            trees = list_sample_region_trees(f, region)
            if not trees:
                raise FileNotFoundError(f"No trees for region '{region}' in {root_path}")

            for sample, tpath in trees:
                sample_l = str(sample).lower()
                if excl and any(sample_l.startswith(p) for p in excl):
                    continue

                tree = f[tpath]
                df, source_map, _ = build_feature_frame_from_tree(tree, features, max_events=max_events_per_sample)
                df = sanitize_feature_frame(df)

                arrs = read_tree_as_arrays(f, tpath, branches=[weight_branch], max_events=max_events_per_sample)
                w = np.asarray(arrs[weight_branch], dtype="f8")
                w = np.where(np.isfinite(w), np.maximum(w, 0.0), 0.0)
                w = np.minimum(w, weight_clip)

                if len(w) != len(df):
                    n = min(len(w), len(df))
                    df = df.iloc[:n].reset_index(drop=True)
                    w = w[:n]

                if label_map is not None:
                    if sample not in label_map:
                        raise KeyError(f"Sample '{sample}' not in label CSV.")
                    label = int(label_map[sample])
                else:
                    label = 1 if (signal_prefix and sample_l.startswith(signal_prefix)) else (1 if is_signal(sample, patterns) else 0)

                y = np.full(df.shape[0], label, dtype=np.int8)
                X_parts.append(df)
                y_parts.append(y)
                w_parts.append(w)
                used_samples[sample] = int(df.shape[0])
                for feat in features:
                    if feat not in feature_sources:
                        feature_sources[feat] = source_map.get(feat, "unknown")

        if not X_parts:
            raise ValueError("No training events selected.")

        X = pd.concat(X_parts, axis=0, ignore_index=True)
        y = np.concatenate(y_parts)
        w = np.concatenate(w_parts)

        if np.unique(y).size < 2:
            raise ValueError("Only one class found; check signal/background rules.")
        if float(np.sum(w)) <= 0.0:
            raise ValueError("All event weights are zero.")

        return self._run_training_pipeline(
            X=X, y=y, w=w, features=list(features),
            feature_sources=feature_sources,
            outdir=outdir, plot_dir=plot_dir,
            region=region, root_path=root_path,
            used_samples=used_samples,
        )

    # ------------------------------------------------------------------
    # Shared pipeline: feature ranking → split → scale → train → plots
    # Called by both train_from_root() and train_from_arrays()
    # ------------------------------------------------------------------

    def _run_training_pipeline(
        self,
        X,  # pandas DataFrame
        y: np.ndarray,
        w: np.ndarray,
        features: List[str],
        feature_sources: Dict[str, str],
        outdir: str,
        plot_dir: str,
        region: str,
        root_path: str,
        used_samples: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score, roc_curve

        outdir_p = Path(outdir)
        outdir_p.mkdir(parents=True, exist_ok=True)
        plot_dir_p = Path(plot_dir)
        plot_dir_p.mkdir(parents=True, exist_ok=True)

        requested_features = list(features)

        tc = self.training_config
        seed = int(tc.get("seed", 7))
        val_size = float(tc.get("val_size", 0.2))
        test_size = float(tc.get("test_size", 0.3))
        epochs = int(tc.get("epochs", 50))
        batch_size = int(tc.get("batch_size", 8192))
        lr = float(tc.get("learning_rate", 1e-3))
        patience = int(tc.get("early_stopping", {}).get("patience", 10))
        balance_classes = bool(tc.get("balance_classes", True))
        balance_strength = float(tc.get("balance_strength", 1.0))

        fsc = self.feature_selection_config
        top_k = int(fsc.get("top_k_significance", 5))
        single_feat_epochs = int(fsc.get("single_feature_epochs", 20))
        drop_constant = bool(fsc.get("drop_constant_features", False))

        topo_w = float(self.topo_config.get("weight", 0.0))
        topo_feats = [s.strip() for s in str(self.topo_config.get("features", "M_Jet1Jet2,dRJet12")).split(",") if s.strip()]
        topo_min_sig = int(self.topo_config.get("min_signal_events", 16))

        # Feature significance ranking (syst-aware when configured)
        sig_syst_val = float(self.training_config.get("sig_syst", 0.0))
        signif_rows = _compute_feature_significance(X, y, w, features, plot_dir_p, source_map=feature_sources,
                                                     sig_syst=sig_syst_val)
        logging.info("Feature significance written to %s", plot_dir_p / "feature_significance.csv")

        if top_k > 0:
            ranked = [str(r["feature"]) for r in signif_rows if str(r.get("feature", "")) in X.columns]
            keep = ranked[: min(top_k, len(ranked))]
            X = X[keep].copy()
            features = list(keep)
            logging.info("Selected top-%d features: %s", len(features), features)

        if drop_constant:
            X, kept, dropped = _drop_constant_features(X, missing_sentinel=-9999.0)
            if dropped:
                logging.info("Dropped near-constant features: %s", dropped)
            features = kept

        topo_indices = [int(features.index(f)) for f in topo_feats if f in features]

        # 60/20/20 stratified split
        X_train, X_temp, y_train, y_temp, w_train, w_temp = train_test_split(
            X, y, w, test_size=val_size + test_size, random_state=seed, stratify=y,
        )
        test_frac_of_temp = test_size / (val_size + test_size)
        X_val, X_test, y_val, y_test, w_val, w_test = train_test_split(
            X_temp, y_temp, w_temp, test_size=test_frac_of_temp, random_state=seed, stratify=y_temp,
        )

        y_train_i = np.asarray(y_train, dtype="i4")
        y_val_i = np.asarray(y_val, dtype="i4")
        y_test_i = np.asarray(y_test, dtype="i4")
        w_train_eff = np.asarray(w_train, dtype="f8").copy()
        w_val_eff = np.asarray(w_val, dtype="f8").copy()
        w_test_eff = np.asarray(w_test, dtype="f8").copy()

        class_balance_factors = {"background": 1.0, "signal": 1.0}
        if balance_classes:
            eps = 1e-12
            sum_b = float(np.sum(w_train_eff[y_train_i == 0]))
            sum_s = float(np.sum(w_train_eff[y_train_i == 1]))
            if sum_b <= eps or sum_s <= eps:
                raise ValueError("Cannot balance classes: one class has zero weight sum.")
            full_f_b = 0.5 / sum_b
            full_f_s = 0.5 / sum_s
            f_b = full_f_b ** balance_strength
            f_s = full_f_s ** balance_strength
            scale = (sum_b + sum_s) / float(sum_b * f_b + sum_s * f_s)
            f_b *= scale
            f_s *= scale
            class_balance_factors = {"background": float(f_b), "signal": float(f_s)}
            w_train_eff = w_train_eff * np.where(y_train_i == 1, f_s, f_b)
            w_val_eff = w_val_eff * np.where(y_val_i == 1, f_s, f_b)
            w_test_eff = w_test_eff * np.where(y_test_i == 1, f_s, f_b)

        # Scale
        X_train_np = X_train.to_numpy(dtype="f8")
        X_val_np = X_val.to_numpy(dtype="f8")
        X_test_np = X_test.to_numpy(dtype="f8")
        self._dnn_scaler = _StandardScaler.fit(X_train_np, missing_sentinel=-9999.0)
        X_train_np = self._dnn_scaler.transform(X_train_np).astype("float32")
        X_val_np = self._dnn_scaler.transform(X_val_np).astype("float32")
        X_test_np = self._dnn_scaler.transform(X_test_np).astype("float32")

        # Build model matching actual feature count
        spec = ModelSpec(
            n_inputs=int(X_train_np.shape[1]),
            hidden_layers=parse_hidden_layers(str(self.model_config.get("hidden_layers", "128,128"))),
            dropout=float(self.model_config.get("dropout", 0.1)),
        )
        torch.manual_seed(seed)
        net = build_mlp(spec).to(self.device)
        self.model._net = net
        self.model._spec = spec

        optim = torch.optim.AdamW(net.parameters(), lr=lr)
        bce_loss = nn.BCEWithLogitsLoss(reduction="none")

        Xtr = torch.from_numpy(X_train_np)
        ytr = torch.from_numpy(y_train_i.astype("float32"))
        wtr = torch.from_numpy(w_train_eff.astype("float32"))
        Xva = torch.from_numpy(X_val_np)
        yva = torch.from_numpy(y_val_i.astype("float32"))
        wva_t = torch.from_numpy(w_val_eff.astype("float32"))
        Xte = torch.from_numpy(X_test_np)

        loader = DataLoader(TensorDataset(Xtr, ytr, wtr), batch_size=batch_size, shuffle=True, drop_last=False)

        best_auc, best_state, bad = -np.inf, None, 0
        train_losses: List[float] = []
        val_losses: List[float] = []
        val_aucs: List[float] = []
        epoch_ids: List[int] = []

        for epoch in range(1, epochs + 1):
            net.train()
            running, n_batches = 0.0, 0

            for xb, yb, wb in loader:
                xb, yb, wb = xb.to(self.device), yb.to(self.device), wb.to(self.device)
                optim.zero_grad(set_to_none=True)
                logits = net(xb).squeeze(1)
                wnorm = wb / (wb.mean() + 1e-12)
                loss = (bce_loss(logits, yb) * wnorm).mean()

                if topo_w > 0.0 and topo_indices:
                    sig_mask = yb > 0.5
                    if int(sig_mask.sum().item()) >= topo_min_sig:
                        loss = loss + topo_w * _topology_decorrelation_penalty(logits, xb, wb, sig_mask, topo_indices)

                loss.backward()
                optim.step()
                running += float(loss.detach().cpu().item())
                n_batches += 1

            net.eval()
            with torch.no_grad():
                logits_va = net(Xva.to(self.device)).squeeze(1)
                y_score_va = torch.sigmoid(logits_va).cpu().numpy()
                wva_d = wva_t.to(self.device)
                loss_val = float(
                    (bce_loss(logits_va, yva.to(self.device)) * (wva_d / (wva_d.mean() + 1e-12))).mean().detach().cpu()
                )

            auc = float(roc_auc_score(y_val_i, y_score_va, sample_weight=w_val_eff))
            avg_loss = running / max(1, n_batches)
            train_losses.append(avg_loss)
            val_losses.append(loss_val)
            val_aucs.append(auc)
            epoch_ids.append(epoch)

            self.history["train_loss"].append(avg_loss)
            self.history["val_loss"].append(loss_val)
            self.history["val_auc"].append(auc)

            logging.info("[Epoch %03d] train_loss=%.6f  val_loss=%.6f  auc=%.6f", epoch, avg_loss, loss_val, auc)

            if auc > best_auc + 1e-6:
                best_auc = auc
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if patience > 0 and bad >= patience:
                    logging.info("EarlyStop at epoch %d", epoch)
                    break

        if best_state is not None:
            net.load_state_dict(best_state)

        # Final evaluation
        net.eval()
        with torch.no_grad():
            y_score_test = torch.sigmoid(net(Xte.to(self.device)).squeeze(1)).cpu().numpy()
            y_score_train = torch.sigmoid(net(Xtr.to(self.device)).squeeze(1)).cpu().numpy()
            y_score_val = torch.sigmoid(net(Xva.to(self.device)).squeeze(1)).cpu().numpy()

        auc_test = float(roc_auc_score(y_test_i, y_score_test, sample_weight=w_test_eff))
        auc_train = float(roc_auc_score(y_train_i, y_score_train, sample_weight=w_train_eff))
        auc_val = float(roc_auc_score(y_val_i, y_score_val, sample_weight=w_val_eff))

        fpr_test, tpr_test, _ = roc_curve(y_test_i, y_score_test, sample_weight=w_test_eff)
        fpr_train, tpr_train, _ = roc_curve(y_train_i, y_score_train, sample_weight=w_train_eff)

        # Save model
        model_path = outdir_p / "dnn_model.pt"
        save_checkpoint(str(model_path), model=net, spec=spec)
        if self._dnn_scaler is not None:
            (outdir_p / "scaler.json").write_text(json.dumps(self._dnn_scaler.to_jsonable(), indent=2) + "\n")

        # Metrics JSON
        metrics = {
            "root": root_path, "region": region,
            "requested_features": requested_features, "features": features,
            "feature_sources": feature_sources,
            "top_k_significance": top_k,
            "topology_decorrelation_weight": topo_w,
            "topology_decorrelation_features": topo_feats,
            "balance_classes": balance_classes,
            "class_balance_factors": class_balance_factors,
            "used_samples": used_samples or {},
            "model_spec": {"n_inputs": int(spec.n_inputs), "hidden_layers": list(spec.hidden_layers), "dropout": float(spec.dropout)},
            "auc_train": auc_train, "auc_val": auc_val, "auc_test": auc_test,
            "n_train": int(len(X_train)), "n_val": int(len(X_val)), "n_test": int(len(X_test)),
            "seed": seed,
        }

        (outdir_p / "train_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        (outdir_p / "features.json").write_text(json.dumps(features, indent=2) + "\n")
        (outdir_p / "feature_significance.json").write_text(json.dumps(signif_rows, indent=2) + "\n")

        # Plots
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6.5, 6))
        plt.plot(fpr_train, tpr_train, label=f"Train AUC={auc_train:.4f}", color="#3f90da")
        plt.plot(fpr_test, tpr_test, label=f"Test AUC={auc_test:.4f}", color="#bd1f01")
        plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Train vs Test ({region})")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(plot_dir_p / "roc_train_vs_test.png", dpi=160)
        plt.close()

        for split, yt, ys, wt in [
            ("test", y_test_i, y_score_test, w_test_eff),
            ("train", y_train_i, y_score_train, w_train_eff),
        ]:
            _plot_score_distribution(yt, ys, wt, plot_dir_p / f"score_distribution_{split}.png", f"DNN score ({split}, {region})")
            _write_score_table(yt, ys, wt, plot_dir_p / f"score_distribution_{split}.csv")

        plt.figure(figsize=(7, 5))
        plt.plot(epoch_ids, train_losses, marker="o", linewidth=1.5, label="Train loss")
        plt.plot(epoch_ids, val_losses, marker="s", linewidth=1.5, label="Val loss")
        plt.xlabel("Epoch")
        plt.ylabel("Weighted BCE loss")
        plt.title(f"Loss vs Epoch ({region})")
        plt.grid(alpha=0.25)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(plot_dir_p / "loss_curve.png", dpi=160)
        plt.close()

        plt.figure(figsize=(7, 5))
        plt.plot(epoch_ids, val_aucs, marker="o", linewidth=1.5, color="#bd1f01")
        plt.xlabel("Epoch")
        plt.ylabel("Validation AUC")
        plt.title(f"Validation AUC vs Epoch ({region})")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(plot_dir_p / "auc_curve.png", dpi=160)
        plt.close()

        # Per-feature 1D DNN scans
        top_feature_scan_rows = []
        for feat in list(features):
            xtr_f = np.asarray(X_train[feat].to_numpy(), dtype="f8")
            xte_f = np.asarray(X_test[feat].to_numpy(), dtype="f8")
            _, score_te_f, auc_tr_f, auc_te_f = _train_single_feature_dnn(
                xtr_f, xte_f,
                y_train_i, y_test_i,
                w_train_eff, w_test_eff,
                seed=seed, epochs=single_feat_epochs, batch_size=batch_size,
                lr=lr, patience=patience, dropout=float(self.model_config.get("dropout", 0.1)),
                device=self.device,
            )
            safe_feat = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in str(feat))
            _plot_score_distribution(
                y_test_i, score_te_f, w_test_eff,
                plot_dir_p / f"score_distribution_feature_{safe_feat}.png",
                f"1D DNN score ({feat}, test)",
            )
            _write_score_table(y_test_i, score_te_f, w_test_eff, plot_dir_p / f"score_distribution_feature_{safe_feat}.csv")
            feat_signif = next((r for r in signif_rows if str(r.get("feature")) == str(feat)), None)
            top_feature_scan_rows.append({
                "feature": feat,
                "source": feature_sources.get(feat, "unknown"),
                "feature_asimov_z": None if feat_signif is None else float(feat_signif.get("asimov_z", 0.0)),
                "dnn_auc_train": float(auc_tr_f),
                "dnn_auc_test": float(auc_te_f),
            })

        if top_feature_scan_rows:
            pd.DataFrame(top_feature_scan_rows).to_csv(plot_dir_p / "top_feature_dnn_scores.csv", index=False)
            metrics["top_feature_dnn_scores"] = top_feature_scan_rows
            (outdir_p / "train_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

        logging.info("[OK] AUC(val)=%.4f  AUC(test)=%.4f  AUC(train)=%.4f", auc_val, auc_test, auc_train)
        return metrics

    # ------------------------------------------------------------------
    # Low-level train() — used when caller has already preprocessed tensors
    # ------------------------------------------------------------------

    def train(
        self,
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        train_masses: torch.Tensor,
        val_features: torch.Tensor,
        val_labels: torch.Tensor,
        val_masses: torch.Tensor,
    ) -> Dict[str, Any]:
        from sklearn.metrics import roc_auc_score

        tc = self.training_config
        batch_size = int(tc.get("batch_size", 8192))
        lr = float(tc.get("learning_rate", 1e-3))
        epochs = int(tc.get("epochs", 50))
        patience = int(tc.get("early_stopping", {}).get("patience", 10))
        min_delta = float(tc.get("early_stopping", {}).get("min_delta", 1e-6))

        net = self.model._net
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
        bce = nn.BCEWithLogitsLoss(reduction="none")

        loader = DataLoader(
            TensorDataset(train_features, train_labels, train_masses),
            batch_size=batch_size, shuffle=True, drop_last=False,
        )

        Xva = val_features.to(self.device)
        yva = val_labels.to(self.device)
        best_auc, best_state, bad = -np.inf, None, 0
        epoch = 0

        for epoch in range(1, epochs + 1):
            net.train()
            running, n_batches = 0.0, 0

            for xb, yb, _mb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = net(xb).squeeze(1)
                loss = (bce(logits, yb) / (bce(logits, yb).mean() + 1e-12)).mean()
                loss.backward()
                optimizer.step()
                running += float(loss.detach().cpu())
                n_batches += 1

            net.eval()
            with torch.no_grad():
                score_va = torch.sigmoid(net(Xva).squeeze(1)).cpu().numpy()

            y_va_np = yva.cpu().numpy().astype("i4")
            auc = float(roc_auc_score(y_va_np, score_va)) if len(np.unique(y_va_np)) > 1 else 0.0
            avg_loss = running / max(1, n_batches)
            self.history["train_loss"].append(avg_loss)
            self.history["val_loss"].append(avg_loss)
            self.history["val_auc"].append(auc)

            if epoch % 10 == 0:
                logging.info("Epoch %03d  loss=%.6f  val_auc=%.6f", epoch, avg_loss, auc)

            if auc > best_auc + float(min_delta):
                best_auc = auc
                best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if patience > 0 and bad >= patience:
                    logging.info("EarlyStop at epoch %d", epoch)
                    break

        if best_state is not None:
            net.load_state_dict(best_state)

        logging.info("Training complete. Best val AUC=%.4f", best_auc)
        return {"best_val_auc": best_auc, "final_epoch": epoch, "history": self.history}

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        masses: torch.Tensor,
    ) -> Dict[str, float]:
        from sklearn.metrics import roc_auc_score

        self.model.eval()
        with torch.no_grad():
            scores = self.model(features.to(self.device), masses.to(self.device)).squeeze().cpu()

        preds = (scores > 0.5).float()
        y = labels.cpu()
        accuracy = (preds == y).float().mean().item()

        y_np = y.numpy().astype("i4")
        s_np = scores.numpy()
        auc = float(roc_auc_score(y_np, s_np)) if len(np.unique(y_np)) > 1 else 0.0

        tp = float(((preds == 1) & (y == 1)).sum())
        fp = float(((preds == 1) & (y == 0)).sum())
        fn = float(((preds == 0) & (y == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {"accuracy": accuracy, "auc": auc, "precision": precision, "recall": recall, "f1_score": f1}

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save_model(self, model_path: str):
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(model_path, model=self.model._net, spec=self.model._spec)
        if self._dnn_scaler is not None:
            scaler_path = str(model_path).replace(".pt", "_scaler.json")
            Path(scaler_path).write_text(json.dumps(self._dnn_scaler.to_jsonable(), indent=2) + "\n")
        logging.info("Model saved to %s", model_path)

    def load_model(self, model_path: str):
        net, spec = load_checkpoint(model_path, map_location="cpu")
        self.model._net = net.to(self.device)
        self.model._spec = spec
        scaler_path = str(model_path).replace(".pt", "_scaler.json")
        if Path(scaler_path).exists():
            self._dnn_scaler = _StandardScaler.from_jsonable(json.loads(Path(scaler_path).read_text()))
        logging.info("Model loaded from %s", model_path)

    def predict(self, features: np.ndarray, masses: np.ndarray) -> np.ndarray:
        self.model.eval()
        X = features.astype("f8")
        if self._dnn_scaler is not None:
            X = self._dnn_scaler.transform(X)
        Xt = torch.from_numpy(X.astype("float32"))
        Mt = torch.from_numpy(masses.astype("float32"))
        with torch.no_grad():
            out = self.model(Xt.to(self.device), Mt.to(self.device)).cpu().numpy()
        return out

    # ------------------------------------------------------------------
    # Plotting (legacy two-panel history plot)
    # ------------------------------------------------------------------

    def plot_training_history(self, save_path: Optional[str] = None):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].plot(self.history["train_loss"], label="Train loss")
        axes[0].plot(self.history["val_loss"], label="Val loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Loss curve")
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(self.history["val_auc"], label="Val AUC", color="#bd1f01")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("AUC")
        axes[1].set_title("Validation AUC")
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=160, bbox_inches="tight")
            logging.info("Training history saved to %s", save_path)
        plt.close()
