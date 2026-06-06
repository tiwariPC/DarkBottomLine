"""Train a classifier using ppbbchichi output trees.

Purpose:
- Read trees from `ppbbchichi-trees.root` (per sample / region).
- Build a feature matrix from selected branches and train a binary classifier.
- Write the trained model + training metadata (AUC, features) to an output directory.

Notes:
- This script trains a small tabular DNN (MLP) using PyTorch.
- Output artifacts are written to an output directory (checkpoint, scaler, metrics).
"""
from __future__ import annotations

import os

# --- Limit CPU threads (set BEFORE numpy/pandas/torch are imported) ---
_NCPUS = int(os.environ.get("TRAIN_NUM_THREADS", "4"))
os.environ.setdefault("OMP_NUM_THREADS", str(_NCPUS))
os.environ.setdefault("MKL_NUM_THREADS", str(_NCPUS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_NCPUS))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(_NCPUS))

import argparse
import json
from pathlib import Path
import csv

import numpy as np
import uproot

try:
    # When installed as part of darkbottomline package
    from darkbottomline.dnn.common import DEFAULT_SIGNAL_PATTERNS, is_signal, sanitize_feature_frame
    from darkbottomline.dnn.data import list_sample_region_trees, read_tree_as_arrays
    from darkbottomline.dnn.feature_engineering import REQUESTED_FEATURES_25, build_feature_frame_from_tree, get_default_feature_csv
    from darkbottomline.dnn.model import ModelSpec, build_mlp, parse_hidden_layers, save_checkpoint
    from darkbottomline.dnn.scaler import StandardScaler
except ImportError:
    try:
        # When executed as a module: `python -m darkbottomline.dnn.train_classifier ...`
        from dnn.common import DEFAULT_SIGNAL_PATTERNS, is_signal, sanitize_feature_frame
        from dnn.data import list_sample_region_trees, read_tree_as_arrays
        from dnn.feature_engineering import REQUESTED_FEATURES_25, build_feature_frame_from_tree, get_default_feature_csv
        from dnn.model import ModelSpec, build_mlp, parse_hidden_layers, save_checkpoint
        from dnn.scaler import StandardScaler
    except ImportError:
        # When executed as a script: `python darkbottomline/dnn/train_classifier.py ...`
        from common import DEFAULT_SIGNAL_PATTERNS, is_signal, sanitize_feature_frame
        from data import list_sample_region_trees, read_tree_as_arrays
        from feature_engineering import REQUESTED_FEATURES_25, build_feature_frame_from_tree, get_default_feature_csv
        from model import ModelSpec, build_mlp, parse_hidden_layers, save_checkpoint
        from scaler import StandardScaler


DEFAULT_FEATURES = list(REQUESTED_FEATURES_25)


def _load_label_csv(path: str) -> dict[str, int]:
    """Load sample->label mapping from a CSV.

    Expected columns: sample,label
    - sample: top-level directory name inside ppbbchichi-trees.root
    - label: 0 (background) or 1 (signal)
    """
    mapping: dict[str, int] = {}
    with open(path, "r", newline="") as fp:
        reader = csv.DictReader(fp)
        if not reader.fieldnames or "sample" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError("label CSV must have columns: sample,label")
        for row in reader:
            s = str(row["sample"]).strip()
            if not s:
                continue
            v = int(row["label"])
            if v not in (0, 1):
                raise ValueError(f"Invalid label for sample '{s}': {v} (expected 0 or 1)")
            mapping[s] = v
    if not mapping:
        raise ValueError(f"No labels loaded from: {path}")
    return mapping


def _parse_prefixes(prefix_text: str | None, default: str | None = None) -> tuple[str, ...]:
    if prefix_text is None:
        prefix_text = default
    if prefix_text is None:
        return tuple()
    vals = [p.strip().lower() for p in str(prefix_text).split(",") if p.strip()]
    return tuple(vals)


def _resolve_signal_prefix(
    signal_prefix: str | None,
    signal_category: str | None,
) -> str | None:
    if signal_prefix:
        return str(signal_prefix).strip().lower()
    if signal_category:
        cat = str(signal_category).strip().lower()
        if not cat:
            return None
        return cat if cat.startswith("new") else f"new{cat}"
    return None


def _drop_constant_features(df, *, missing_sentinel: float = -9999.0, eps: float = 1e-12):
    """Drop near-constant features (common for LHEF-derived btag placeholders).

    Uses std computed on finite, non-sentinel values.
    Returns: (df_kept, kept_features, dropped_features)
    """
    import numpy as _np

    kept: list[str] = []
    dropped: list[str] = []
    for c in list(df.columns):
        a = _np.asarray(df[c].to_numpy(), dtype="f8")
        m = _np.isfinite(a) & (a != float(missing_sentinel))
        s = float(_np.std(a[m])) if _np.any(m) else 0.0
        if s <= float(eps):
            dropped.append(str(c))
        else:
            kept.append(str(c))

    if not kept:
        raise ValueError("All features were dropped as (near-)constant; cannot train.")

    return df[kept], kept, dropped


def _resolve_device(arg: str):
    import torch

    if arg == "cpu":
        return torch.device("cpu")
    if arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available")
        return torch.device("cuda")
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError(f"Unknown device: {arg}")


def _topology_decorrelation_penalty(
    logits,
    inputs,
    weights,
    signal_mask,
    nuisance_indices: list[int],
    *,
    eps: float = 1e-12,
):
    """Penalize correlation between score logits and topology nuisance variables.

    The penalty is computed on signal events only and uses weighted Pearson
    correlation so that the classifier is discouraged from splitting one signal
    class into multiple topology-driven score modes.
    """
    import torch

    if not nuisance_indices:
        return logits.new_tensor(0.0)

    if signal_mask.sum().item() < 4:
        return logits.new_tensor(0.0)

    sig_logits = logits[signal_mask]
    sig_inputs = inputs[signal_mask][:, nuisance_indices]
    sig_weights = weights[signal_mask]

    wsum = sig_weights.sum()
    if not torch.isfinite(wsum) or float(wsum.detach().cpu().item()) <= eps:
        return logits.new_tensor(0.0)

    w = sig_weights / (wsum + eps)
    log_centered = sig_logits - torch.sum(w * sig_logits)
    feat_centered = sig_inputs - torch.sum(w[:, None] * sig_inputs, dim=0, keepdim=True)

    log_var = torch.sum(w * log_centered * log_centered)
    feat_var = torch.sum(w[:, None] * feat_centered * feat_centered, dim=0)
    cov = torch.sum(w[:, None] * log_centered[:, None] * feat_centered, dim=0)

    corr = cov / (torch.sqrt(log_var + eps) * torch.sqrt(feat_var + eps) + eps)
    corr = torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.mean(corr * corr)


def _weighted_percentile(values: np.ndarray, weights: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """Compute weighted percentiles for 1D arrays."""
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
    """Compute binned Asimov significance from per-bin signal/background yields."""
    s = np.maximum(np.asarray(sig, dtype="f8"), 0.0)
    b = np.maximum(np.asarray(bkg, dtype="f8"), 0.0)
    term = np.where(
        b > eps,
        (s + b) * np.log1p(np.divide(s, b, out=np.zeros_like(s), where=b > eps)) - s,
        0.0,
    )
    z2 = 2.0 * np.sum(np.maximum(term, 0.0))
    return float(np.sqrt(max(z2, 0.0)))


def _asimov_significance_from_hist_syst(
    sig: np.ndarray,
    bkg: np.ndarray,
    sigma_rel: float,
    eps: float = 1e-12,
) -> float:
    """Compute binned Asimov significance with background systematic uncertainty.

    Uses the Cowan, Cranmer, Gross, Vitells (2011) asymptotic formula with a
    Gaussian nuisance parameter on the per-bin background yield:

        Z² = 2 Σ [(sᵢ+bᵢ) ln(((sᵢ+bᵢ)(bᵢ+σbᵢ²))/(bᵢ²+(sᵢ+bᵢ)σbᵢ²))
                 - (bᵢ²/σbᵢ²) ln(1 + σbᵢ²·sᵢ/(bᵢ(bᵢ+σbᵢ²)))]

    where σbᵢ = sigma_rel * bᵢ.  When sigma_rel → 0 this reduces to the pure
    statistical Asimov Z.
    """
    s = np.maximum(np.asarray(sig, dtype="f8"), 0.0)
    b = np.maximum(np.asarray(bkg, dtype="f8"), 0.0)

    if sigma_rel <= 0.0:
        return _asimov_significance_from_hist(s, b, eps=eps)

    sigma_b2 = (sigma_rel * b) ** 2

    # Bins where the systematic is non-negligible
    mask_syst = (b > eps) & (sigma_b2 > eps)
    term = np.zeros_like(s, dtype="f8")

    if np.any(mask_syst):
        s_m = s[mask_syst]
        b_m = b[mask_syst]
        sb2_m = sigma_b2[mask_syst]

        # First log term
        num = (s_m + b_m) * (b_m + sb2_m)
        den = b_m * b_m + (s_m + b_m) * sb2_m
        ratio1 = np.where(den > eps, num / den, 1.0)
        term1 = np.where(ratio1 > 1.0 + eps, (s_m + b_m) * np.log(ratio1), 0.0)

        # Second log term
        ratio2 = sb2_m * s_m / np.maximum(b_m * (b_m + sb2_m), eps)
        term2 = (b_m * b_m / np.maximum(sb2_m, eps)) * np.log1p(ratio2)

        term[mask_syst] = term1 - term2

    # Bins where the systematic is negligible → pure stat
    mask_pure = ~mask_syst & (b > eps)
    if np.any(mask_pure):
        s_p = s[mask_pure]
        b_p = b[mask_pure]
        term[mask_pure] = (s_p + b_p) * np.log1p(s_p / b_p) - s_p

    z2 = 2.0 * np.sum(np.maximum(term, 0.0))
    return float(np.sqrt(max(z2, 0.0)))


def _compute_feature_significance(
    X_df,
    y: np.ndarray,
    w: np.ndarray,
    features: list[str],
    outdir: Path,
    source_map: dict[str, str] | None = None,
    n_bins: int = 40,
    sig_syst: float = 0.0,
) -> list[dict]:
    """Compute per-feature separation metrics and save ranking outputs.

    Parameters
    ----------
    sig_syst : float
        Fractional systematic uncertainty on background (0 = pure stat, 0.20 = 20%).
        When non-zero, ``asimov_z_syst`` is computed via the Cowan et al. 2011
        asymptotic formula and used for feature ranking.
    """
    from sklearn.metrics import roc_auc_score

    outdir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    y_i = np.asarray(y, dtype="i4")
    w_f = np.maximum(np.asarray(w, dtype="f8"), 0.0)

    # ---- Counting-experiment baseline (pure S/√B, no shape information) ----
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
        m_all = np.isfinite(x)
        x = x[m_all]
        yy = y_i[m_all]
        ww = w_f[m_all]

        if x.size == 0:
            rows.append(
                {
                    "feature": feat,
                    "source": (source_map or {}).get(feat, "unknown"),
                    "auc": float("nan"),
                    "asimov_z": 0.0,
                    "asimov_z_syst": 0.0,
                    "delta_z": 0.0,
                    "delta_z_syst": 0.0,
                }
            )
            continue

        x_sig = x[yy == 1]
        x_bkg = x[yy == 0]
        w_sig = ww[yy == 1]
        w_bkg = ww[yy == 0]

        if x_sig.size == 0 or x_bkg.size == 0 or np.sum(w_sig) <= 0.0 or np.sum(w_bkg) <= 0.0:
            rows.append(
                {
                    "feature": feat,
                    "source": (source_map or {}).get(feat, "unknown"),
                    "auc": float("nan"),
                    "asimov_z": 0.0,
                    "asimov_z_syst": 0.0,
                    "delta_z": 0.0,
                    "delta_z_syst": 0.0,
                }
            )
            continue

        qlo, qhi = _weighted_percentile(x, ww, np.array([0.01, 0.99], dtype="f8"))
        lo = float(np.nanmin(x) if not np.isfinite(qlo) else qlo)
        hi = float(np.nanmax(x) if not np.isfinite(qhi) else qhi)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = float(np.nanmin(x))
            hi = float(np.nanmax(x))
        if hi <= lo:
            hi = lo + 1.0

        edges = np.linspace(lo, hi, int(n_bins) + 1, dtype="f8")
        hs, _ = np.histogram(x_sig, bins=edges, weights=w_sig)
        hb, _ = np.histogram(x_bkg, bins=edges, weights=w_bkg)
        z = _asimov_significance_from_hist(hs, hb)
        z_syst = _asimov_significance_from_hist_syst(hs, hb, float(sig_syst))

        auc = float(roc_auc_score(yy, x, sample_weight=ww))
        rows.append(
            {
                "feature": feat,
                "source": (source_map or {}).get(feat, "unknown"),
                "auc": auc,
                "asimov_z": z,
                "asimov_z_syst": z_syst,
                "delta_z": z - z_counting_stat,
                "delta_z_syst": z_syst - z_counting_syst,
            }
        )

    # Sort by syst-aware Z when available, else pure stat Z
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
    vals_delta_stat = [float(r["delta_z"]) for r in rows]
    vals_delta_syst = [float(r["delta_z_syst"]) for r in rows]
    has_syst = float(sig_syst) > 0.0

    def _draw_feature_bars(ax, vals, ylabel, title, color="#3f90da"):
        ax.bar(labels, vals, color=color, edgecolor="black", linewidth=0.8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=40, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    # ---- Raw Z plot(s) ----
    if has_syst:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(16, 1.1 * len(labels)), 5.5))
        _draw_feature_bars(ax1, vals_stat, "Asimov Z (pure stat)",
                           "Per-feature significance (pure statistical)")
        syst_label = f"syst-aware (\u03c3_rel={float(sig_syst)*100:.0f}%)"
        _draw_feature_bars(ax2, vals_syst, f"Asimov Z ({syst_label})",
                           f"Per-feature significance ({syst_label})", "#d62728")
        fig.tight_layout()
        fig.savefig(outdir / "feature_significance.png", dpi=160)
        plt.close(fig)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(max(12, 0.7 * len(labels)), 5.5))
        _draw_feature_bars(ax, vals_stat, "Asimov Z (pure stat)",
                           "Per-feature significance (pure statistical)")
        fig.tight_layout()
        fig.savefig(outdir / "feature_significance.png", dpi=160)
        plt.close(fig)

    # ---- Delta-Z plot(s) (counting baseline S/√B subtracted) ----
    if has_syst:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(16, 1.1 * len(labels)), 5.5))
        _draw_feature_bars(ax1, vals_delta_stat, "\u0394Z (pure stat)",
                           f"Effective significance (pure stat)\nbaseline = counting S/\u221aB = {z_counting_stat:.2f}\u03c3")
        ax1.axhline(y=0, color="gray", linewidth=0.8, linestyle="--")
        syst_label = f"syst-aware (\u03c3_rel={float(sig_syst)*100:.0f}%)"
        _draw_feature_bars(ax2, vals_delta_syst, f"\u0394Z ({syst_label})",
                           f"Effective significance ({syst_label})\nbaseline = counting S/\u221aB (syst) = {z_counting_syst:.2f}\u03c3",
                           "#d62728")
        ax2.axhline(y=0, color="gray", linewidth=0.8, linestyle="--")
        fig.tight_layout()
        fig.savefig(outdir / "feature_significance_delta.png", dpi=160)
        plt.close(fig)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(max(12, 0.7 * len(labels)), 5.5))
        _draw_feature_bars(ax, vals_delta_stat, "\u0394Z (pure stat)",
                           f"Effective significance\nbaseline = counting S/\u221aB = {z_counting_stat:.2f}\u03c3")
        ax.axhline(y=0, color="gray", linewidth=0.8, linestyle="--")
        fig.tight_layout()
        fig.savefig(outdir / "feature_significance_delta.png", dpi=160)
        plt.close(fig)

    return rows


def _plot_score_distribution(
    y_true: np.ndarray,
    y_score: np.ndarray,
    weights: np.ndarray,
    out_path: Path,
    title: str,
    n_bins: int = 50,
) -> None:
    """Plot normalized score distributions for signal/background on [0, 1]."""
    y = np.asarray(y_true, dtype="i4")
    s = np.asarray(y_score, dtype="f8")
    w = np.maximum(np.asarray(weights, dtype="f8"), 0.0)

    m = np.isfinite(s) & np.isfinite(w)
    y = y[m]
    s = np.clip(s[m], 0.0, 1.0)
    w = w[m]

    bins = np.linspace(0.0, 1.0, int(n_bins) + 1, dtype="f8")
    hs, _ = np.histogram(s[y == 1], bins=bins, weights=w[y == 1])
    hb, _ = np.histogram(s[y == 0], bins=bins, weights=w[y == 0])

    hs = hs / max(np.sum(hs), 1e-12)
    hb = hb / max(np.sum(hb), 1e-12)

    centers = 0.5 * (bins[:-1] + bins[1:])

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
    """Write normalized score distributions for signal/background to CSV."""
    import pandas as pd

    y = np.asarray(y_true, dtype="i4")
    s = np.asarray(y_score, dtype="f8")
    w = np.maximum(np.asarray(weights, dtype="f8"), 0.0)

    m = np.isfinite(s) & np.isfinite(w)
    y = y[m]
    s = np.clip(s[m], 0.0, 1.0)
    w = w[m]

    bins = np.linspace(0.0, 1.0, int(n_bins) + 1, dtype="f8")
    hs, _ = np.histogram(s[y == 1], bins=bins, weights=w[y == 1])
    hb, _ = np.histogram(s[y == 0], bins=bins, weights=w[y == 0])

    hs = hs / max(np.sum(hs), 1e-12)
    hb = hb / max(np.sum(hb), 1e-12)

    df = pd.DataFrame(
        {
            "bin_low": bins[:-1],
            "bin_high": bins[1:],
            "bin_center": 0.5 * (bins[:-1] + bins[1:]),
            "signal_norm": hs,
            "background_norm": hb,
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


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
    """Train a tiny 1D DNN and return train/test scores + AUCs."""
    import torch
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader, TensorDataset

    xtr = np.asarray(x_train, dtype="f8").reshape(-1, 1)
    xte = np.asarray(x_test, dtype="f8").reshape(-1, 1)

    scaler = StandardScaler.fit(xtr, missing_sentinel=-9999.0)
    xtr_n = scaler.transform(xtr).astype("float32")
    xte_n = scaler.transform(xte).astype("float32")

    torch.manual_seed(int(seed))
    spec = ModelSpec(n_inputs=1, hidden_layers=(16, 16), dropout=float(min(max(dropout, 0.0), 0.2)))
    model = build_mlp(spec).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=float(lr))
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")

    Xtr = torch.from_numpy(xtr_n)
    ytr = torch.from_numpy(np.asarray(y_train, dtype="float32"))
    wtr = torch.from_numpy(np.asarray(w_train, dtype="float32"))

    Xte = torch.from_numpy(xte_n)
    yte = torch.from_numpy(np.asarray(y_test, dtype="float32"))
    wte = torch.from_numpy(np.asarray(w_test, dtype="float32"))

    loader = DataLoader(TensorDataset(Xtr, ytr, wtr), batch_size=max(256, int(batch_size)), shuffle=True, drop_last=False)

    best_auc = -np.inf
    best_state = None
    bad_epochs = 0

    for _epoch in range(1, int(max(1, epochs)) + 1):
        model.train()
        for xb, yb, wb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)

            optim.zero_grad(set_to_none=True)
            logits = model(xb).squeeze(1)
            loss_vec = bce(logits, yb)
            loss = (loss_vec * (wb / (wb.mean() + 1e-12))).mean()
            loss.backward()
            optim.step()

        model.eval()
        with torch.no_grad():
            score_te = torch.sigmoid(model(Xte.to(device)).squeeze(1)).cpu().numpy()
        auc_te = float(
            roc_auc_score(
                np.asarray(y_test, dtype="i4"),
                score_te,
                sample_weight=np.asarray(w_test, dtype="f8"),
            )
        )

        if auc_te > best_auc + 1e-6:
            best_auc = auc_te
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if int(patience) > 0 and bad_epochs >= int(patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        score_tr = torch.sigmoid(model(Xtr.to(device)).squeeze(1)).cpu().numpy()
        score_te = torch.sigmoid(model(Xte.to(device)).squeeze(1)).cpu().numpy()

    auc_tr = float(
        roc_auc_score(
            np.asarray(y_train, dtype="i4"),
            score_tr,
            sample_weight=np.asarray(w_train, dtype="f8"),
        )
    )
    auc_te = float(
        roc_auc_score(
            np.asarray(y_test, dtype="i4"),
            score_te,
            sample_weight=np.asarray(w_test, dtype="f8"),
        )
    )
    return score_tr, score_te, auc_tr, auc_te


def main():
    ap = argparse.ArgumentParser(description="Train a tabular DNN classifier for pp→bbχχ using ppbbchichi output trees.")
    ap.add_argument("--root", required=True, help="Path to ppbbchichi-trees.root")
    ap.add_argument("--region", default="preselection", help="Region tree to use for training")
    ap.add_argument(
        "--features",
        default=get_default_feature_csv(),
        help="Comma-separated feature branch names",
    )
    ap.add_argument(
        "--signal-pattern",
        action="append",
        default=None,
        help="Regex to identify signal sample names (repeatable). Default matches: " + ", ".join(DEFAULT_SIGNAL_PATTERNS),
    )
    ap.add_argument(
        "--signal-prefix",
        default=None,
        help=(
            "Optional signal prefix rule, e.g. 'newdiboson'. If set, samples starting "
            "with this prefix are signal and all other non-excluded samples are background."
        ),
    )
    ap.add_argument(
        "--signal-category",
        default=None,
        help=(
            "Optional shorthand for signal prefix. Example: --signal-category diboson "
            "means signal prefix 'newdiboson'."
        ),
    )
    ap.add_argument(
        "--exclude-prefixes",
        default="",
        help="Comma-separated sample prefixes to exclude from ML entirely. Default: none",
    )
    ap.add_argument(
        "--sig-syst",
        type=float,
        default=0.20,
        help=(
            "Fractional systematic uncertainty on background for syst-aware Asimov Z "
            "(Cowan et al. 2011). 0 = pure stat, 0.20 = 20%%. "
            "Feature ranking uses syst-aware Z when > 0."
        ),
    )
    ap.add_argument(
        "--label-csv",
        default=None,
        help="Optional CSV mapping sample->label with columns 'sample,label'. If provided, overrides --signal-pattern.",
    )
    ap.add_argument(
        "--background-prefix",
        default=None,
        help=(
            "Optional naming rule: label samples starting with this prefix as background (0), all others as signal (1). "
            "Useful for EOS EVENTSELECTION inputs where backgrounds are named like 'newDIBOSON*'. "
            "Ignored if --label-csv is provided."
        ),
    )
    ap.add_argument("--max-events-per-sample", type=int, default=200000, help="Cap events per sample for training")
    ap.add_argument("--test-size", type=float, default=0.3, help="Fraction for test split")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--outdir", default="outputs_dnn", help="Output directory")
    ap.add_argument(
        "--plot-dir",
        default="plot",
        help="Directory for training plots and feature significance outputs",
    )
    ap.add_argument("--hidden", default="128,128", help="Hidden layer sizes, e.g. '128,128'")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-clip", type=float, default=100.0, help="Clip event weights to this maximum")
    ap.add_argument(
        "--balance-classes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Balance class contributions in BCE loss by reweighting signal/background "
            "weights using train-split class sums (default: enabled)."
        ),
    )
    ap.add_argument(
        "--balance-strength",
        type=float,
        default=1.0,
        help=(
            "Strength of class balancing in [0,1]. 0 disables balancing, 1 applies the full "
            "class-sum equalization, intermediate values interpolate on a log scale."
        ),
    )
    ap.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help="Fraction of the full dataset reserved for validation (default: 0.2, i.e. 60/20/20 split).",
    )
    ap.add_argument("--patience", type=int, default=10, help="Early-stopping patience (epochs)")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument(
        "--top-k-significance",
        type=int,
        default=5,
        help="Train final model with top-K features ranked by Asimov significance (0 keeps all).",
    )
    ap.add_argument(
        "--single-feature-epochs",
        type=int,
        default=20,
        help="Epochs for per-feature 1D DNN score scans on selected top-K features.",
    )
    ap.add_argument(
        "--topology-decorrelation-weight",
        type=float,
        default=0.0,
        help=(
            "Extra loss weight that suppresses score correlation with topology proxies "
            "(set >0 to reduce topology-driven multimodality)."
        ),
    )
    ap.add_argument(
        "--topology-decorrelation-features",
        default="M_Jet1Jet2,dRJet12",
        help="Comma-separated feature names used as topology nuisances for decorrelation.",
    )
    ap.add_argument(
        "--topology-decorrelation-min-signal",
        type=int,
        default=16,
        help="Minimum number of signal events in a batch to activate the topology penalty.",
    )
    ap.add_argument(
        "--drop-constant-features",
        action="store_true",
        help="Drop near-constant features automatically (useful for LHEF inputs with placeholder btag).",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    features = [s.strip() for s in args.features.split(",") if s.strip()]
    requested_features = list(features)
    if not features:
        raise ValueError("No features specified")

    region = args.region
    region_safe = region.replace(":", "_")
    weight_branch = f"weight_{region_safe}"
    patterns = tuple(args.signal_pattern) if args.signal_pattern else DEFAULT_SIGNAL_PATTERNS
    label_map = _load_label_csv(args.label_csv) if args.label_csv else None
    bg_prefix = str(args.background_prefix).strip().lower() if args.background_prefix else None
    exclude_prefixes = _parse_prefixes(args.exclude_prefixes, default="run")
    signal_prefix = _resolve_signal_prefix(args.signal_prefix, args.signal_category)

    if signal_prefix and label_map is not None:
        print("[WARN] --label-csv is provided; --signal-prefix/--signal-category are ignored for labels.")

    X_parts = []
    y_parts = []
    w_parts = []
    sample_parts = []
    used_samples: dict[str, int] = {}
    excluded_samples: dict[str, str] = {}
    sample_labels: dict[str, int] = {}
    label_counts = {0: 0, 1: 0}
    feature_sources: dict[str, str] = {}

    with uproot.open(args.root) as f:
        trees = list_sample_region_trees(f, region)
        if not trees:
            raise FileNotFoundError(f"No trees found for region '{region}' in {args.root}")

        for sample, tpath in trees:
            sample_l = str(sample).lower()
            if exclude_prefixes and any(sample_l.startswith(p) for p in exclude_prefixes):
                excluded_samples[sample] = "excluded_prefix"
                continue

            tree = f[tpath]
            df, source_map, _used_branch_names = build_feature_frame_from_tree(
                tree,
                features,
                max_events=args.max_events_per_sample,
            )
            df = sanitize_feature_frame(df)

            arrs = read_tree_as_arrays(f, tpath, branches=[weight_branch], max_events=args.max_events_per_sample)

            w = np.asarray(arrs[weight_branch], dtype="f8")
            w = np.where(np.isfinite(w), w, 0.0)
            # Training with negative weights is ill-defined for BCE; clamp to 0.
            w = np.maximum(w, 0.0)
            if args.weight_clip is not None:
                w = np.minimum(w, float(args.weight_clip))

            if len(w) != len(df):
                n = min(len(w), len(df))
                df = df.iloc[:n].reset_index(drop=True)
                w = w[:n]

            if label_map is not None:
                if sample not in label_map:
                    raise KeyError(
                        f"Sample '{sample}' not found in --label-csv mapping. "
                        "Add it to the CSV or remove --label-csv to use --signal-pattern."
                    )
                label = int(label_map[sample])
            else:
                if signal_prefix:
                    label = 1 if sample_l.startswith(signal_prefix) else 0
                elif bg_prefix:
                    label = 0 if str(sample).lower().startswith(bg_prefix) else 1
                else:
                    label = 1 if is_signal(sample, patterns) else 0
            y = np.full(df.shape[0], label, dtype=np.int8)

            X_parts.append(df)
            y_parts.append(y)
            w_parts.append(w)
            sample_parts.append(np.array([sample] * df.shape[0], dtype=object))
            used_samples[sample] = int(df.shape[0])
            sample_labels[sample] = int(label)
            label_counts[int(label)] += 1

            for feat in features:
                if feat not in feature_sources:
                    feature_sources[feat] = source_map.get(feat, "unknown")

    if not X_parts:
        raise ValueError(
            "No training events selected after sample filtering. "
            "Check --exclude-prefixes, --signal-prefix/--signal-category, and input ROOT content."
        )

    import pandas as pd

    X = pd.concat(X_parts, axis=0, ignore_index=True)
    y = np.concatenate(y_parts)
    w = np.concatenate(w_parts)
    _sample_names = np.concatenate(sample_parts)

    if np.unique(y).size < 2:
        raise ValueError(
            "Training labels contain only one class. "
            "Check signal/background rules and the sample names inside the ROOT file."
        )
    if not np.isfinite(w).all():
        w = np.where(np.isfinite(w), w, 0.0)
    if float(np.sum(w)) <= 0.0:
        raise ValueError("All event weights are zero after sanitation/clipping; cannot train.")

    # Compute per-feature significance before training.
    signif_rows = _compute_feature_significance(
        X, y, w, features, plot_dir, source_map=feature_sources, sig_syst=args.sig_syst,
    )
    print(f"[OK] Wrote feature significance to: {plot_dir / 'feature_significance.csv'}")

    if int(args.top_k_significance) > 0:
        k = int(args.top_k_significance)
        ranked = [str(r["feature"]) for r in signif_rows if str(r.get("feature", "")) in X.columns]
        if not ranked:
            raise ValueError("No features available after significance ranking; cannot continue.")
        keep = ranked[: min(k, len(ranked))]
        X = X[keep].copy()
        features = list(keep)
        print(f"[INFO] Selected top-{len(features)} features for final training: {features}")

    # ---- Auto-plot top-K feature signal/background overlays ----
    _top_features_script = Path(__file__).resolve().parent / "plot_top_features.py"
    if _top_features_script.exists():
        import subprocess, sys
        _cmd = [
            sys.executable, str(_top_features_script),
            "--significance-json", str(plot_dir / "feature_significance.json"),
            "--root", str(args.root),
            "--region", str(args.region),
            "--outdir", str(plot_dir),
            "--top-k", str(max(5, int(args.top_k_significance)) if int(args.top_k_significance) > 0 else "5"),
        ]
        if signal_prefix:
            _cmd += ["--signal-prefix", signal_prefix]
        if args.signal_category:
            _cmd += ["--signal-category", str(args.signal_category)]
        if args.background_prefix:
            _cmd += ["--background-prefix", str(args.background_prefix)]
        if args.exclude_prefixes:
            _cmd += ["--exclude-prefixes", str(args.exclude_prefixes)]
        _cmd += ["--max-events-per-sample", str(args.max_events_per_sample)]
        print(f"[INFO] Plotting top-feature distributions: {' '.join(_cmd)}")
        try:
            subprocess.run(_cmd, check=True, timeout=600)
        except subprocess.TimeoutExpired:
            print("[WARN] Top-feature plotting timed out (10 min), continuing.")
        except subprocess.CalledProcessError as e:
            print(f"[WARN] Top-feature plotting failed (exit {e.returncode}), continuing.")
    else:
        print(f"[WARN] Top-feature plot script not found at: {_top_features_script}")

    if args.drop_constant_features:
        X, kept, dropped = _drop_constant_features(X, missing_sentinel=-9999.0)
        if dropped:
            print(f"[INFO] Dropped near-constant features: {dropped}")
        features = kept

    topology_nuisance_features = [s.strip() for s in str(args.topology_decorrelation_features).split(",") if s.strip()]
    topology_nuisance_indices = [int(features.index(f)) for f in topology_nuisance_features if f in features]
    if float(args.topology_decorrelation_weight) > 0.0 and not topology_nuisance_indices:
        raise ValueError(
            "--topology-decorrelation-weight > 0 requires at least one nuisance feature present in the training inputs."
        )
    print(
        "[INFO] Sample selection summary: "
        f"used={len(used_samples)}, excluded={len(excluded_samples)}, "
        f"signal_samples={label_counts[1]}, background_samples={label_counts[0]}"
    )

    from sklearn.model_selection import train_test_split

    val_size = float(args.val_size)
    test_size = float(args.test_size)
    if val_size <= 0.0 or test_size <= 0.0:
        raise ValueError("--val-size and --test-size must both be > 0 for a 60/20/20-style split.")
    if val_size + test_size >= 1.0:
        raise ValueError("--val-size + --test-size must be < 1.0.")

    X_train, X_temp, y_train, y_temp, w_train, w_temp = train_test_split(
        X, y, w,
        test_size=val_size + test_size,
        random_state=int(args.seed),
        stratify=y,
    )

    test_frac_of_temp = test_size / (val_size + test_size)
    X_val, X_test, y_val, y_test, w_val, w_test = train_test_split(
        X_temp, y_temp, w_temp,
        test_size=test_frac_of_temp,
        random_state=int(args.seed),
        stratify=y_temp,
    )

    y_train_i = np.asarray(y_train, dtype="i4")
    y_val_i = np.asarray(y_val, dtype="i4")
    y_test_i = np.asarray(y_test, dtype="i4")
    w_train_eff = np.asarray(w_train, dtype="f8").copy()
    w_val_eff = np.asarray(w_val, dtype="f8").copy()
    w_test_eff = np.asarray(w_test, dtype="f8").copy()
    class_balance_factors = {"background": 1.0, "signal": 1.0}
    balance_strength = float(args.balance_strength)
    if balance_strength < 0.0 or balance_strength > 1.0:
        raise ValueError("--balance-strength must be in the range [0, 1].")

    if bool(args.balance_classes):
        eps = 1e-12
        sum_b = float(np.sum(w_train_eff[y_train_i == 0]))
        sum_s = float(np.sum(w_train_eff[y_train_i == 1]))
        if sum_b <= eps or sum_s <= eps:
            raise ValueError("Cannot balance classes: one class has zero train weight sum.")

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

        print(
            "[INFO] Enabled class-balanced loss weighting: "
            f"strength={balance_strength:.3f}, f_background={f_b:.6e}, f_signal={f_s:.6e}"
        )

    X_train_np = X_train.to_numpy(dtype="f8")
    X_val_np = X_val.to_numpy(dtype="f8")
    X_test_np = X_test.to_numpy(dtype="f8")
    scaler = StandardScaler.fit(X_train_np, missing_sentinel=-9999.0)
    X_train_np = scaler.transform(X_train_np).astype("float32")
    X_val_np = scaler.transform(X_val_np).astype("float32")
    X_test_np = scaler.transform(X_test_np).astype("float32")

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(int(args.seed))
    device = _resolve_device(args.device)

    spec = ModelSpec(
        n_inputs=int(X_train_np.shape[1]),
        hidden_layers=parse_hidden_layers(args.hidden),
        dropout=float(args.dropout),
    )
    model = build_mlp(spec).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=float(args.lr))
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    topo_weight = float(args.topology_decorrelation_weight)
    topo_min_sig = int(args.topology_decorrelation_min_signal)

    Xtr = torch.from_numpy(X_train_np)
    ytr = torch.from_numpy(np.asarray(y_train, dtype="float32"))
    wtr = torch.from_numpy(np.asarray(w_train_eff, dtype="float32"))

    Xva = torch.from_numpy(X_val_np)
    yva = torch.from_numpy(np.asarray(y_val, dtype="float32"))
    wva = torch.from_numpy(np.asarray(w_val_eff, dtype="float32"))

    Xte = torch.from_numpy(X_test_np)
    yte = torch.from_numpy(np.asarray(y_test, dtype="float32"))
    wte = torch.from_numpy(np.asarray(w_test_eff, dtype="float32"))

    train_loader = DataLoader(
        TensorDataset(Xtr, ytr, wtr),
        batch_size=int(args.batch_size),
        shuffle=True,
        drop_last=False,
    )

    from sklearn.metrics import roc_auc_score, roc_curve

    best_auc = -np.inf
    best_state = None
    bad_epochs = 0
    train_losses: list[float] = []
    val_losses: list[float] = []
    val_aucs: list[float] = []
    epoch_ids: list[int] = []

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        running = 0.0
        n_batches = 0

        for xb, yb, wb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)

            optim.zero_grad(set_to_none=True)
            logits = model(xb).squeeze(1)
            loss_vec = bce(logits, yb)
            wnorm = wb / (wb.mean() + 1e-12)
            loss = (loss_vec * wnorm).mean()

            if topo_weight > 0.0 and len(topology_nuisance_indices) > 0:
                signal_mask = yb > 0.5
                if int(signal_mask.sum().item()) >= topo_min_sig:
                    topo_penalty = _topology_decorrelation_penalty(
                        logits, xb, wb, signal_mask, topology_nuisance_indices,
                    )
                    loss = loss + topo_weight * topo_penalty
                else:
                    topo_penalty = logits.new_tensor(0.0)
            else:
                topo_penalty = logits.new_tensor(0.0)

            loss.backward()
            optim.step()

            running += float(loss.detach().cpu().item())
            n_batches += 1

        model.eval()
        with torch.no_grad():
            logits_val = model(Xva.to(device)).squeeze(1)
            y_score = torch.sigmoid(logits_val).cpu().numpy()
            loss_val_vec = bce(logits_val, yva.to(device))
            wva_n = wva.to(device) / (wva.to(device).mean() + 1e-12)
            loss_val = float((loss_val_vec * wva_n).mean().detach().cpu().item())

        auc = float(roc_auc_score(np.asarray(y_val, dtype="int8"), y_score, sample_weight=np.asarray(w_val, dtype="f8")))
        avg_loss = running / max(1, n_batches)
        train_losses.append(float(avg_loss))
        val_losses.append(float(loss_val))
        val_aucs.append(float(auc))
        epoch_ids.append(int(epoch))

        print(f"[Epoch {epoch:03d}] train_loss={avg_loss:.6f}  val_loss={loss_val:.6f}  auc={auc:.6f}  device={device}")

        if auc > best_auc + 1e-6:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if int(args.patience) > 0 and bad_epochs >= int(args.patience):
                print(f"[EarlyStop] No AUC improvement for {bad_epochs} epochs")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits_te = model(Xte.to(device)).squeeze(1)
        y_score_test = torch.sigmoid(logits_te).cpu().numpy()
        logits_tr = model(Xtr.to(device)).squeeze(1)
        y_score_train = torch.sigmoid(logits_tr).cpu().numpy()
        logits_va = model(Xva.to(device)).squeeze(1)
        y_score_val = torch.sigmoid(logits_va).cpu().numpy()

    auc_test = float(roc_auc_score(np.asarray(y_test, dtype="int8"), y_score_test, sample_weight=np.asarray(w_test, dtype="f8")))
    auc_train = float(roc_auc_score(np.asarray(y_train, dtype="int8"), y_score_train, sample_weight=np.asarray(w_train, dtype="f8")))
    auc_val = float(roc_auc_score(np.asarray(y_val, dtype="int8"), y_score_val, sample_weight=np.asarray(w_val, dtype="f8")))

    fpr_test, tpr_test, _ = roc_curve(np.asarray(y_test, dtype="int8"), y_score_test, sample_weight=np.asarray(w_test, dtype="f8"))
    fpr_val, tpr_val, _ = roc_curve(np.asarray(y_val, dtype="int8"), y_score_val, sample_weight=np.asarray(w_val, dtype="f8"))
    fpr_train, tpr_train, _ = roc_curve(np.asarray(y_train, dtype="int8"), y_score_train, sample_weight=np.asarray(w_train, dtype="f8"))

    model_path = outdir / "dnn_model.pt"
    save_checkpoint(str(model_path), model=model, spec=spec)
    (outdir / "scaler.json").write_text(json.dumps(scaler.to_jsonable(), indent=2) + "\n")

    metrics = {
        "root": str(args.root),
        "region": region,
        "requested_features": requested_features,
        "features": features,
        "weight_branch": weight_branch,
        "signal_patterns": list(patterns),
        "signal_prefix": signal_prefix,
        "signal_category": args.signal_category,
        "exclude_prefixes": list(exclude_prefixes),
        "feature_sources": feature_sources,
        "top_k_significance": int(args.top_k_significance),
        "topology_decorrelation_weight": float(args.topology_decorrelation_weight),
        "topology_decorrelation_features": list(topology_nuisance_features),
        "topology_decorrelation_min_signal": int(args.topology_decorrelation_min_signal),
        "balance_classes": bool(args.balance_classes),
        "class_balance_factors": class_balance_factors,
        "used_samples": used_samples,
        "excluded_samples": excluded_samples,
        "sample_labels": sample_labels,
        "label_sample_counts": {
            "background": int(label_counts[0]),
            "signal": int(label_counts[1]),
        },
        "model_type": "pytorch_mlp",
        "model_path": str(model_path),
        "scaler_path": str(outdir / "scaler.json"),
        "model_spec": {
            "n_inputs": int(spec.n_inputs),
            "hidden_layers": list(spec.hidden_layers),
            "dropout": float(spec.dropout),
        },
        "auc": auc_test,
        "auc_train": auc_train,
        "auc_val": auc_val,
        "auc_test": auc_test,
        "n_total": int(len(X)),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "seed": int(args.seed),
        "val_size": float(val_size),
        "test_size": float(args.test_size),
    }

    (outdir / "train_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (outdir / "features.json").write_text(json.dumps(features, indent=2) + "\n")
    (outdir / "feature_significance.json").write_text(json.dumps(signif_rows, indent=2) + "\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 6))
    plt.plot(fpr_test, tpr_test, label=f"AUC={auc_test:.4f}")
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Test ({region})")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(outdir / "roc.png", dpi=150)
    plt.savefig(outdir / "roc_test.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.plot(fpr_train, tpr_train, label=f"AUC={auc_train:.4f}", color="#3f90da")
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Train ({region})")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(outdir / "roc_train.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6.5, 6))
    plt.plot(fpr_train, tpr_train, label=f"Train AUC={auc_train:.4f}", color="#3f90da")
    plt.plot(fpr_test, tpr_test, label=f"Test AUC={auc_test:.4f}", color="#bd1f01")
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Train vs Test ({region})")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(plot_dir / "roc_train_vs_test.png", dpi=160)
    plt.close()

    _plot_score_distribution(
        np.asarray(y_test, dtype="i4"), np.asarray(y_score_test, dtype="f8"), np.asarray(w_test, dtype="f8"),
        plot_dir / "score_distribution_test.png", f"DNN score distribution (test, {region})",
    )
    _write_score_table(
        np.asarray(y_test, dtype="i4"), np.asarray(y_score_test, dtype="f8"), np.asarray(w_test, dtype="f8"),
        plot_dir / "score_distribution_test.csv",
    )
    _plot_score_distribution(
        np.asarray(y_train, dtype="i4"), np.asarray(y_score_train, dtype="f8"), np.asarray(w_train, dtype="f8"),
        plot_dir / "score_distribution_train.png", f"DNN score distribution (train, {region})",
    )
    _write_score_table(
        np.asarray(y_train, dtype="i4"), np.asarray(y_score_train, dtype="f8"), np.asarray(w_train, dtype="f8"),
        plot_dir / "score_distribution_train.csv",
    )

    y_all = np.concatenate([np.asarray(y_train, dtype="i4"), np.asarray(y_test, dtype="i4")])
    s_all = np.concatenate([np.asarray(y_score_train, dtype="f8"), np.asarray(y_score_test, dtype="f8")])
    w_all = np.concatenate([np.asarray(w_train, dtype="f8"), np.asarray(w_test, dtype="f8")])
    _plot_score_distribution(y_all, s_all, w_all, plot_dir / "score_distribution_all.png", f"DNN score distribution (all, {region})")
    _write_score_table(y_all, s_all, w_all, plot_dir / "score_distribution_all.csv")

    top_feature_scan_rows = []
    for feat in list(features):
        xtr_feat = np.asarray(X_train[feat].to_numpy(), dtype="f8")
        xte_feat = np.asarray(X_test[feat].to_numpy(), dtype="f8")

        score_tr_f, score_te_f, auc_tr_f, auc_te_f = _train_single_feature_dnn(
            xtr_feat, xte_feat,
            np.asarray(y_train, dtype="i4"), np.asarray(y_test, dtype="i4"),
            np.asarray(w_train_eff, dtype="f8"), np.asarray(w_test_eff, dtype="f8"),
            seed=int(args.seed),
            epochs=int(args.single_feature_epochs),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            patience=int(args.patience),
            dropout=float(args.dropout),
            device=device,
        )

        safe_feat = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in str(feat))

        _plot_score_distribution(
            np.asarray(y_test, dtype="i4"), np.asarray(score_te_f, dtype="f8"), np.asarray(w_test, dtype="f8"),
            plot_dir / f"score_distribution_feature_{safe_feat}.png",
            f"1D DNN score distribution ({feat}, test)",
        )
        _write_score_table(
            np.asarray(y_test, dtype="i4"), np.asarray(score_te_f, dtype="f8"), np.asarray(w_test, dtype="f8"),
            plot_dir / f"score_distribution_feature_{safe_feat}.csv",
        )

        feat_signif = next((r for r in signif_rows if str(r.get("feature")) == str(feat)), None)
        top_feature_scan_rows.append(
            {
                "feature": feat,
                "source": feature_sources.get(feat, "unknown"),
                "feature_auc": None if feat_signif is None else float(feat_signif.get("auc", float("nan"))),
                "feature_asimov_z": None if feat_signif is None else float(feat_signif.get("asimov_z", 0.0)),
                "feature_asimov_z_syst": None if feat_signif is None else float(feat_signif.get("asimov_z_syst", 0.0)),
                "dnn_auc_train": float(auc_tr_f),
                "dnn_auc_test": float(auc_te_f),
                "score_plot": str(plot_dir / f"score_distribution_feature_{safe_feat}.png"),
                "score_table": str(plot_dir / f"score_distribution_feature_{safe_feat}.csv"),
            }
        )

    if top_feature_scan_rows:
        import pandas as pd

        pd.DataFrame(top_feature_scan_rows).to_csv(plot_dir / "top5_feature_dnn_scores.csv", index=False)
        (outdir / "top5_feature_dnn_scores.json").write_text(json.dumps(top_feature_scan_rows, indent=2) + "\n")
        metrics["top5_feature_dnn_scores"] = top_feature_scan_rows

    (outdir / "train_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    plt.figure(figsize=(7, 5))
    plt.plot(epoch_ids, train_losses, marker="o", linewidth=1.5, label="Train loss")
    plt.plot(epoch_ids, val_losses, marker="s", linewidth=1.5, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Weighted BCE loss")
    plt.title(f"Loss vs Epoch ({region})")
    plt.grid(alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(plot_dir / "loss_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(epoch_ids, val_aucs, marker="o", linewidth=1.5, color="#bd1f01")
    plt.xlabel("Epoch")
    plt.ylabel("Validation AUC")
    plt.title(f"Validation AUC vs Epoch ({region})")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(plot_dir / "auc_curve.png", dpi=160)
    plt.close()

    print(f"[OK] Trained DNN model. AUC(val)={auc_val:.4f}, AUC(test)={auc_test:.4f}, AUC(train)={auc_train:.4f}")
    print(f"[OK] Wrote: {model_path}")
    print(f"[OK] Wrote: {outdir / 'train_metrics.json'}")
    print(f"[OK] Wrote: {plot_dir / 'loss_curve.png'}")
    print(f"[OK] Wrote: {plot_dir / 'score_distribution_test.png'}")


if __name__ == "__main__":
    main()
