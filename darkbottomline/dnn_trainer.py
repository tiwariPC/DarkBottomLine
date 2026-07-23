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
import os
import yaml
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dnn.model import ModelSpec, build_mlp, parse_hidden_layers, save_checkpoint, load_checkpoint
from dnn.scaler import StandardScaler as _StandardScaler


def _variable_label(name: str, variable_labels: Optional[Dict[str, str]] = None) -> str:
    """x-axis label for *name*, from dnn.yaml's variable_labels — falls back to
    the raw name if absent."""
    return (variable_labels or {}).get(name, name)


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
    plot_dir: Path | None = None,
    variable_labels: dict | None = None,
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

    feature_plot_dir = Path(plot_dir) / "features" if plot_dir is not None else None
    if feature_plot_dir is not None:
        feature_plot_dir.mkdir(parents=True, exist_ok=True)

    from darkbottomline.objects import SENTINEL

    for feat in features:
        x = np.asarray(X_df[feat].to_numpy(), dtype="f8")
        m = np.isfinite(x) & (x != SENTINEL)
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

        if feature_plot_dir is not None:
            _plot_feature_distribution(edges, hs, hb, feat, feature_plot_dir / f"feature_{feat}.png",
                                        variable_labels=variable_labels)

    sort_key = "asimov_z_syst" if float(sig_syst) > 0.0 else "asimov_z"
    rows.sort(key=lambda r: (r[sort_key], abs((r["auc"] if np.isfinite(r["auc"]) else 0.5) - 0.5)), reverse=True)

    import pandas as pd
    pd.DataFrame(rows).to_csv(outdir / "feature_significance.csv", index=False)
    (outdir / "feature_significance.json").write_text(json.dumps(rows, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from utils.plot_utils import CMSPlotStyle, _PALETTE
    import mplhep as hep

    CMSPlotStyle().set_style()

    feat_names = [r["feature"] for r in rows]
    labels = [_variable_label(f, variable_labels) for f in feat_names]
    vals_stat = [float(r["asimov_z"]) for r in rows]
    vals_syst = [float(r["asimov_z_syst"]) for r in rows]
    has_syst = float(sig_syst) > 0.0

    def _draw_bars(ax, vals, ylabel, title, color=_PALETTE[0]):
        ax.bar(labels, vals, color=color, edgecolor="black", linewidth=0.8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel(ylabel)
        hep.cms.label(llabel="Work in Progress", data=False, com=13.6, ax=ax, loc=0)
        ax.text(0.02, 0.88, title, transform=ax.transAxes, fontsize=11,
                va="top", ha="left")

    fig_width = max(14.0, 0.55 * len(labels))

    if has_syst:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_width, 13.0))
        _draw_bars(ax1, vals_stat, "Asimov Z (pure stat)", "Per-feature significance (pure statistical)")
        syst_label = f"syst-aware (σ_rel={float(sig_syst)*100:.0f}%)"
        _draw_bars(ax2, vals_syst, f"Asimov Z ({syst_label})", f"Per-feature significance ({syst_label})", _PALETTE[2])
        fig.tight_layout()
        fig.savefig(outdir / "feature_significance.png", dpi=300)
        plt.close(fig)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(fig_width, 6.5))
        _draw_bars(ax, vals_stat, "Asimov Z (pure stat)", "Per-feature significance (pure statistical)")
        fig.tight_layout()
        fig.savefig(outdir / "feature_significance.png", dpi=300)
        plt.close(fig)

    # Delta-Z plot
    vals_delta_stat = [float(r["delta_z"]) for r in rows]
    vals_delta_syst = [float(r["delta_z_syst"]) for r in rows]
    if has_syst:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_width, 13.0))
        _draw_bars(ax1, vals_delta_stat, "ΔZ (pure stat)",
                   f"Effective significance (pure stat), baseline S/√B = {z_counting_stat:.2f}σ")
        ax1.axhline(y=0, color="gray", linewidth=0.8, linestyle="--")
        syst_label = f"syst-aware (σ_rel={float(sig_syst)*100:.0f}%)"
        _draw_bars(ax2, vals_delta_syst, f"ΔZ ({syst_label})",
                   f"Effective significance ({syst_label}), baseline = {z_counting_syst:.2f}σ", _PALETTE[2])
        ax2.axhline(y=0, color="gray", linewidth=0.8, linestyle="--")
        fig.tight_layout()
        fig.savefig(outdir / "feature_significance_delta.png", dpi=300)
        plt.close(fig)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(fig_width, 6.5))
        _draw_bars(ax, vals_delta_stat, "ΔZ (pure stat)",
                   f"Effective significance, baseline S/√B = {z_counting_stat:.2f}σ")
        ax.axhline(y=0, color="gray", linewidth=0.8, linestyle="--")
        fig.tight_layout()
        fig.savefig(outdir / "feature_significance_delta.png", dpi=300)
        plt.close(fig)

    return rows


def _plot_feature_distribution(
    edges: np.ndarray,
    hs: np.ndarray,
    hb: np.ndarray,
    feature_name: str,
    out_path: "Path",
    variable_labels: dict | None = None,
) -> None:
    """CMS-style filled signal-vs-background distribution for one feature."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from utils.plot_utils import CMSPlotStyle, _PALETTE
    import mplhep as hep

    CMSPlotStyle().set_style()
    color_bkg, color_sig = _PALETTE[0], "#e76300"  # #3f90da, #e76300

    hs_norm = hs / max(float(np.sum(hs)), 1e-12)
    hb_norm = hb / max(float(np.sum(hb)), 1e-12)

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    hep.histplot(hb_norm, bins=edges, ax=ax, histtype="fill",
                 color=color_bkg, alpha=0.65, edgecolor=color_bkg, label="Background")
    hep.histplot(hs_norm, bins=edges, ax=ax, histtype="fill",
                 color=color_sig, alpha=0.65, edgecolor=color_sig, label="Signal")
    ax.set_xlabel(_variable_label(feature_name, variable_labels))
    ax.set_ylabel("Normalized events")
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="best")
    hep.cms.label(llabel="Work in Progress", data=False, com=13.6, ax=ax, loc=0)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _weighted_corrcoef(sx: np.ndarray, sy: np.ndarray, sw: np.ndarray) -> float:
    """Weighted Pearson correlation coefficient.

    r_w = cov_w(x,y) / sqrt(cov_w(x,x) * cov_w(y,y)), with weighted mean/covariance
    mu_w(x) = sum(w*x)/sum(w), cov_w(x,y) = sum(w*(x-mu_x)*(y-mu_y))/sum(w).
    """
    wsum = float(np.sum(sw))
    if wsum <= 0.0:
        return 0.0
    mx = float(np.sum(sw * sx) / wsum)
    my = float(np.sum(sw * sy) / wsum)
    cov_xy = float(np.sum(sw * (sx - mx) * (sy - my)) / wsum)
    cov_xx = float(np.sum(sw * (sx - mx) ** 2) / wsum)
    cov_yy = float(np.sum(sw * (sy - my) ** 2) / wsum)
    if cov_xx <= 0.0 or cov_yy <= 0.0:
        return 0.0
    return cov_xy / np.sqrt(cov_xx * cov_yy)


def _plot_feature_correlation(
    X_df,
    features: list[str],
    out_path: Path,
    variable_labels: dict | None = None,
    weights: np.ndarray | None = None,
) -> None:
    """CMS-style Pearson correlation heatmap between input features.

    Sentinel-filled entries (darkbottomline.objects.SENTINEL) are masked out
    per-column-pair before computing correlation, same as significance/plots.

    When *weights* is given, uses weighted Pearson correlation (event weights,
    e.g. lumi*xsec/wte) instead of unweighted — reflects correlations in the
    physically-normalized sample rather than raw event counts.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from utils.plot_utils import CMSPlotStyle
    import mplhep as hep
    from darkbottomline.objects import SENTINEL

    CMSPlotStyle().set_style()

    n = len(features)
    X = np.asarray(X_df[features].to_numpy(), dtype="f8")
    X = np.where((X == SENTINEL) | ~np.isfinite(X), np.nan, X)
    w_all = np.maximum(np.asarray(weights, dtype="f8"), 0.0) if weights is not None else None

    corr = np.full((n, n), np.nan, dtype="f8")
    for i in range(n):
        for j in range(i, n):
            xi, xj = X[:, i], X[:, j]
            m = np.isfinite(xi) & np.isfinite(xj)
            if w_all is not None:
                m = m & np.isfinite(w_all) & (w_all > 0.0)
            if m.sum() < 2:
                c = 0.0
            else:
                sx, sy = xi[m], xj[m]
                if np.std(sx) == 0.0 or np.std(sy) == 0.0:
                    c = 1.0 if i == j else 0.0
                elif w_all is not None:
                    c = _weighted_corrcoef(sx, sy, w_all[m])
                else:
                    c = float(np.corrcoef(sx, sy)[0, 1])
            corr[i, j] = c
            corr[j, i] = c

    labels = [_variable_label(f, variable_labels) for f in features]

    fig_size = max(9.0, 0.42 * n)
    fig, ax = plt.subplots(figsize=(fig_size + 1.5, fig_size))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="equal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fontsize_cell = 6 if n > 20 else 7
    for i in range(n):
        for j in range(n):
            v = corr[i, j]
            color = "white" if abs(v) > 0.6 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=fontsize_cell, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Weighted correlation coefficient" if w_all is not None else "Correlation coefficient")

    subtitle = "Event-weighted" if w_all is not None else "Unweighted"
    hep.cms.label(llabel="Work in Progress", rlabel=f"{subtitle}, 13.6 TeV", ax=ax, loc=0)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


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
    from utils.plot_utils import CMSPlotStyle, _PALETTE
    import mplhep as hep

    CMSPlotStyle().set_style()
    color_bkg, color_sig = _PALETTE[0], "#e76300"  # #3f90da, #e76300

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

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    hep.histplot(hb, bins=bins, ax=ax, histtype="fill",
                 color=color_bkg, alpha=0.65, edgecolor=color_bkg, label="Background")
    hep.histplot(hs, bins=bins, ax=ax, histtype="fill",
                 color=color_sig, alpha=0.65, edgecolor=color_sig, label="Signal")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("DNN score")
    ax.set_ylabel("Normalized events")
    ax.legend(loc="best")
    hep.cms.label(llabel="Work in Progress", com=13.6, ax=ax, loc=0)
    ax.text(0.02, 0.92, title, transform=ax.transAxes, fontsize=11,
            va="top", ha="left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


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


def _feature_scan_worker(task: tuple) -> dict:
    """Train + plot the 1D DNN scan for one feature. Module-level (picklable)
    so it can run as a multiprocessing worker — each feature's mini-DNN is
    trained fully independently of every other feature's, so the serial
    per-feature loop (previously the largest single stage of train-dnn wall
    time: ~50 features x their own epochs, one at a time on one core) is
    dispatched across CPU cores instead."""
    # Env vars alone don't reach torch's thread pool here: under the "spawn"
    # start method, torch is already imported (and its intra-op thread pool
    # initialized) by the time this module's top-level `import torch` runs in
    # the child, before this function body executes — so OMP_NUM_THREADS set
    # here is too late. torch.set_num_threads() works at any point instead.
    torch.set_num_threads(1)

    (feat, xtr_f, xte_f, y_train_i, y_test_i, w_train_eff, w_test_eff,
     seed, single_feat_epochs, batch_size, lr, patience, dropout,
     variable_labels, feature_sources, signif_row,
     plot_dir_str, safe_feat) = task

    # Small (16x16) per-feature models gain nothing from GPU and CUDA
    # contexts aren't safely shared across spawned processes — always CPU here.
    device = torch.device("cpu")

    _, score_te_f, auc_tr_f, auc_te_f = _train_single_feature_dnn(
        xtr_f, xte_f,
        y_train_i, y_test_i,
        w_train_eff, w_test_eff,
        seed=seed, epochs=single_feat_epochs, batch_size=batch_size,
        lr=lr, patience=patience, dropout=dropout,
        device=device,
    )

    plot_dir_p = Path(plot_dir_str)
    _plot_score_distribution(
        y_test_i, score_te_f, w_test_eff,
        plot_dir_p / f"score_distribution_feature_{safe_feat}.png",
        f"1D DNN score ({_variable_label(feat, variable_labels)}, test)",
    )
    _write_score_table(y_test_i, score_te_f, w_test_eff, plot_dir_p / f"score_distribution_feature_{safe_feat}.csv")

    return {
        "feature": feat,
        "source": feature_sources.get(feat, "unknown"),
        "feature_asimov_z": None if signif_row is None else float(signif_row.get("asimov_z", 0.0)),
        "dnn_auc_train": float(auc_tr_f),
        "dnn_auc_test": float(auc_te_f),
    }


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

MASS_DIM = 2  # (MH3, MH4) — Mchi is fixed at 1 GeV across the signal grid


class ParametricDNN(nn.Module):
    """Wraps dnn.model.build_mlp. mass (MH3, MH4) is concatenated when parametric_input=True."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.parametric_input = bool(config.get("parametric_input", True))

        n_inputs = int(config.get("input_features", 25))
        if self.parametric_input:
            n_inputs += MASS_DIM

        hidden = config.get("hidden_layers", [128, 128])
        if isinstance(hidden, str):
            hidden = list(parse_hidden_layers(hidden))

        spec = ModelSpec(
            n_inputs=n_inputs,
            hidden_layers=tuple(int(h) for h in hidden),
            dropout=float(config.get("dropout", 0.1)),
            parametric=self.parametric_input,
        )
        self._net = build_mlp(spec)
        self._spec = spec

    def forward(self, x: torch.Tensor, mass: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.parametric_input and mass is not None:
            x = torch.cat([x, mass], dim=-1)
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
        from dnn.feature_engineering import build_feature_frame_from_tree
        from dnn.common import sanitize_feature_frame

        region = self.data_config.get("region", "preselection")
        features_req = self.config.get("features")
        if not features_req:
            raise ValueError("dnn.yaml has no features: list — required for DNN training.")
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
        mass: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Full training pipeline from pre-loaded (X, y, w) arrays.

        Identical logic to train_from_root() but skips the ROOT/uproot loading phase.
        Useful when inputs are flat event-selection ROOT files (not ppbbchichi-trees).

        *mass*, when given, is an (N, MASS_DIM) array of (MH3, MH4) values aligned
        row-for-row with X — only used when model.parametric_input is true.
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
            mass=mass,
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
        from dnn.feature_engineering import build_feature_frame_from_tree
        from dnn.common import DEFAULT_SIGNAL_PATTERNS, is_signal, sanitize_feature_frame

        if features is None:
            cfg_feats = self.config.get("features", None)
            if isinstance(cfg_feats, list) and cfg_feats:
                features = [str(f) for f in cfg_feats]
            else:
                raise ValueError("dnn.yaml has no features: list — required for DNN training.")

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
        mass: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score, roc_curve

        parametric = bool(self.model_config.get("parametric_input", False)) and mass is not None
        if parametric and (mass.shape[0] != len(X) or mass.shape[1] != MASS_DIM):
            raise ValueError(
                f"mass array shape {mass.shape} does not match X rows={len(X)} / MASS_DIM={MASS_DIM}"
            )

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
                                                     sig_syst=sig_syst_val, plot_dir=plot_dir_p,
                                                     variable_labels=self.config.get("variable_labels"))
        logging.info("Feature significance written to %s", plot_dir_p / "feature_significance.csv")

        _plot_feature_correlation(X, features, plot_dir_p / "feature_correlation.png",
                                   variable_labels=self.config.get("variable_labels"))
        logging.info("Feature correlation matrix written to %s", plot_dir_p / "feature_correlation.png")

        _plot_feature_correlation(X, features, plot_dir_p / "feature_correlation_weighted.png",
                                   variable_labels=self.config.get("variable_labels"), weights=w)
        logging.info("Weighted feature correlation matrix written to %s", plot_dir_p / "feature_correlation_weighted.png")

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
        if parametric:
            X_train, X_temp, y_train, y_temp, w_train, w_temp, mass_train, mass_temp = train_test_split(
                X, y, w, mass, test_size=val_size + test_size, random_state=seed, stratify=y,
            )
            test_frac_of_temp = test_size / (val_size + test_size)
            X_val, X_test, y_val, y_test, w_val, w_test, mass_val, mass_test = train_test_split(
                X_temp, y_temp, w_temp, mass_temp, test_size=test_frac_of_temp, random_state=seed, stratify=y_temp,
            )
        else:
            X_train, X_temp, y_train, y_temp, w_train, w_temp = train_test_split(
                X, y, w, test_size=val_size + test_size, random_state=seed, stratify=y,
            )
            test_frac_of_temp = test_size / (val_size + test_size)
            X_val, X_test, y_val, y_test, w_val, w_test = train_test_split(
                X_temp, y_temp, w_temp, test_size=test_frac_of_temp, random_state=seed, stratify=y_temp,
            )
            mass_train = mass_val = mass_test = None

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

        # Build model matching actual feature count (+ MASS_DIM when parametric)
        n_inputs = int(X_train_np.shape[1]) + (MASS_DIM if parametric else 0)
        spec = ModelSpec(
            n_inputs=n_inputs,
            hidden_layers=parse_hidden_layers(str(self.model_config.get("hidden_layers", "128,128"))),
            dropout=float(self.model_config.get("dropout", 0.1)),
            parametric=parametric,
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

        if parametric:
            Mtr = torch.from_numpy(np.asarray(mass_train, dtype="float32"))
            Mva = torch.from_numpy(np.asarray(mass_val, dtype="float32"))
            Mte = torch.from_numpy(np.asarray(mass_test, dtype="float32"))
            loader = DataLoader(TensorDataset(Xtr, ytr, wtr, Mtr), batch_size=batch_size, shuffle=True, drop_last=False)
        else:
            Mva = Mte = None
            loader = DataLoader(TensorDataset(Xtr, ytr, wtr), batch_size=batch_size, shuffle=True, drop_last=False)

        best_auc, best_state, bad = -np.inf, None, 0
        train_losses: List[float] = []
        val_losses: List[float] = []
        val_aucs: List[float] = []
        epoch_ids: List[int] = []

        for epoch in range(1, epochs + 1):
            net.train()
            running, n_batches = 0.0, 0

            for batch in loader:
                if parametric:
                    xb, yb, wb, mb = batch
                    mb = mb.to(self.device)
                else:
                    xb, yb, wb = batch
                    mb = None
                xb, yb, wb = xb.to(self.device), yb.to(self.device), wb.to(self.device)
                optim.zero_grad(set_to_none=True)
                net_in = torch.cat([xb, mb], dim=-1) if parametric else xb
                logits = net(net_in).squeeze(1)
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
                Xva_in = torch.cat([Xva.to(self.device), Mva.to(self.device)], dim=-1) if parametric else Xva.to(self.device)
                logits_va = net(Xva_in).squeeze(1)
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
            if parametric:
                Xte_in = torch.cat([Xte.to(self.device), Mte.to(self.device)], dim=-1)
                Xtr_in = torch.cat([Xtr.to(self.device), Mtr.to(self.device)], dim=-1)
                Xva_in = torch.cat([Xva.to(self.device), Mva.to(self.device)], dim=-1)
            else:
                Xte_in = Xte.to(self.device)
                Xtr_in = Xtr.to(self.device)
                Xva_in = Xva.to(self.device)
            y_score_test = torch.sigmoid(net(Xte_in).squeeze(1)).cpu().numpy()
            y_score_train = torch.sigmoid(net(Xtr_in).squeeze(1)).cpu().numpy()
            y_score_val = torch.sigmoid(net(Xva_in).squeeze(1)).cpu().numpy()

        auc_test = float(roc_auc_score(y_test_i, y_score_test, sample_weight=w_test_eff))
        auc_train = float(roc_auc_score(y_train_i, y_score_train, sample_weight=w_train_eff))
        auc_val = float(roc_auc_score(y_val_i, y_score_val, sample_weight=w_val_eff))

        fpr_test, tpr_test, _ = roc_curve(y_test_i, y_score_test, sample_weight=w_test_eff)
        fpr_train, tpr_train, _ = roc_curve(y_train_i, y_score_train, sample_weight=w_train_eff)

        # Save model
        if parametric:
            mass_grid = sorted({tuple(row) for row in np.asarray(mass, dtype="f8").tolist()})
            spec = ModelSpec(
                n_inputs=spec.n_inputs, hidden_layers=spec.hidden_layers, dropout=spec.dropout,
                parametric=True, mass_grid=[list(m) for m in mass_grid],
            )
        model_path = outdir_p / "dnn_model.pt"
        save_checkpoint(str(model_path), model=net, spec=spec)
        if self._dnn_scaler is not None:
            scaler_json = json.dumps(self._dnn_scaler.to_jsonable(), indent=2) + "\n"
            (outdir_p / "scaler.json").write_text(scaler_json)
            # DNNInference looks for "<model_stem>_scaler.json" next to the checkpoint.
            (outdir_p / f"{model_path.stem}_scaler.json").write_text(scaler_json)

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
            "model_spec": {
                "n_inputs": int(spec.n_inputs), "hidden_layers": list(spec.hidden_layers), "dropout": float(spec.dropout),
                "parametric": bool(spec.parametric), "mass_grid": spec.mass_grid,
            },
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
        from utils.plot_utils import CMSPlotStyle, _PALETTE
        import mplhep as hep

        CMSPlotStyle().set_style()

        fig, ax = plt.subplots(figsize=(8.0, 7.0))
        ax.plot(fpr_train, tpr_train, label=f"Train AUC={auc_train:.4f}", color=_PALETTE[0], linewidth=2.0)
        ax.plot(fpr_test, tpr_test, label=f"Test AUC={auc_test:.4f}", color=_PALETTE[2], linewidth=2.0)
        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
        hep.cms.label(llabel="Work in Progress", com=13.6, ax=ax, loc=0)
        ax.text(0.02, 0.97, f"ROC ({region})", transform=ax.transAxes, fontsize=11,
                va="top", ha="left")
        fig.tight_layout()
        fig.savefig(plot_dir_p / "roc_train_vs_test.png", dpi=300)
        plt.close(fig)

        for split, yt, ys, wt in [
            ("test", y_test_i, y_score_test, w_test_eff),
            ("train", y_train_i, y_score_train, w_train_eff),
        ]:
            _plot_score_distribution(yt, ys, wt, plot_dir_p / f"score_distribution_{split}.png", f"DNN score ({split}, {region})")
            _write_score_table(yt, ys, wt, plot_dir_p / f"score_distribution_{split}.csv")

        fig, ax = plt.subplots(figsize=(8.5, 6.5))
        ax.plot(epoch_ids, train_losses, marker="o", linewidth=1.5, color=_PALETTE[0], label="Train loss")
        ax.plot(epoch_ids, val_losses, marker="s", linewidth=1.5, color=_PALETTE[2], label="Val loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Weighted BCE loss")
        ax.legend(loc="best")
        hep.cms.label(llabel="Work in Progress", com=13.6, ax=ax, loc=0)
        ax.text(0.02, 0.97, f"Loss vs Epoch ({region})", transform=ax.transAxes, fontsize=11,
                va="top", ha="left")
        fig.tight_layout()
        fig.savefig(plot_dir_p / "loss_curve.png", dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.5, 6.5))
        ax.plot(epoch_ids, val_aucs, marker="o", linewidth=1.5, color=_PALETTE[2])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation AUC")
        hep.cms.label(llabel="Work in Progress", com=13.6, ax=ax, loc=0)
        ax.text(0.02, 0.03, f"Validation AUC vs Epoch ({region})", transform=ax.transAxes, fontsize=11,
                va="bottom", ha="left")
        fig.tight_layout()
        fig.savefig(plot_dir_p / "auc_curve.png", dpi=300)
        plt.close(fig)

        # Per-feature 1D DNN scans — each feature trains its own small,
        # independent model, so dispatch across CPU cores instead of one at a
        # time (this loop was previously the single largest wall-time stage
        # of train-dnn: N features x their own epoch loop, serial, one core).
        dropout_val = float(self.model_config.get("dropout", 0.1))
        variable_labels = self.config.get("variable_labels")
        scan_tasks = []
        for feat in list(features):
            xtr_f = np.asarray(X_train[feat].to_numpy(), dtype="f8")
            xte_f = np.asarray(X_test[feat].to_numpy(), dtype="f8")
            safe_feat = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in str(feat))
            feat_signif = next((r for r in signif_rows if str(r.get("feature")) == str(feat)), None)
            scan_tasks.append((
                feat, xtr_f, xte_f, y_train_i, y_test_i, w_train_eff, w_test_eff,
                seed, single_feat_epochs, batch_size, lr, patience, dropout_val,
                variable_labels, feature_sources, feat_signif,
                str(plot_dir_p), safe_feat,
            ))

        num_workers = int(os.environ.get("DNN_SCAN_WORKERS", max(1, (os.cpu_count() or 1))))
        num_workers = max(1, min(num_workers, len(scan_tasks)))

        if num_workers > 1 and len(scan_tasks) > 1:
            import multiprocessing as mp

            logging.info(f"Running per-feature DNN scan for {len(scan_tasks)} feature(s) with {num_workers} worker processes")
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=num_workers) as pool:
                top_feature_scan_rows = pool.map(_feature_scan_worker, scan_tasks)
        else:
            top_feature_scan_rows = [_feature_scan_worker(t) for t in scan_tasks]

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

    def predict(self, features: np.ndarray, masses: Optional[np.ndarray]) -> np.ndarray:
        self.model.eval()
        X = features.astype("f8")
        if self._dnn_scaler is not None:
            X = self._dnn_scaler.transform(X)
        Xt = torch.from_numpy(X.astype("float32")).to(self.device)
        Mt = None if masses is None else torch.from_numpy(masses.astype("float32")).to(self.device)
        with torch.no_grad():
            out = self.model(Xt, Mt).cpu().numpy()
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
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logging.info("Training history saved to %s", save_path)
        plt.close()
