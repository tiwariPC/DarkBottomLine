#!/usr/bin/env python3
"""CMS-style normalized distribution comparison for the three DNN weight schemes.

For every feature (kinematic parameter) we plot the *normalized* (unit-area)
distribution under three event-weighting policies, overlaid on one figure:

  * signed : the raw signed event weight (weight_branch, xsec-scaled exactly as
             in train-dnn) — negative where MC@NLO produces negative weights
  * abs    : absolute value of the signed weight
  * local  : local_cancellation weights — non-negative, fitted cell-by-cell per
             sample on the *training* split so the signed yield is preserved
             (exactly the policy the DNN trainer used)

Inputs are the flat EVENTSELECTION ROOT files, loaded through the *same*
pipeline as `darkbottomline train-dnn` (configs/dnn.yaml features + xsec /
per-masspoint weighting), so the comparison reflects the real training data.

Usage (from the DarkBottomLine repo root, conda env `darkbottomline`):

    python dnn/plot_weight_comparison.py \
        --input /home/zzq/eventsel-merged \
        --dnn-config configs/dnn.yaml \
        --outdir outputs/dnn_25feature/weight_comparison
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("plot_weight_comparison")

SENTINEL = -9999.0


# ---------------------------------------------------------------------------
# Weighted binning helpers
# ---------------------------------------------------------------------------

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


def _is_discrete(x: np.ndarray, max_unique: int = 15) -> bool:
    xf = x[np.isfinite(x)]
    if xf.size == 0:
        return False
    u = np.unique(xf)
    return u.size <= max_unique and np.allclose(u, np.round(u))


def _get_edges(x: np.ndarray, w_nonneg: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile-based (1-99%) edges for a continuous feature."""
    qlo, qhi = _weighted_percentile(x, w_nonneg, np.array([0.01, 0.99], dtype="f8"))
    lo = float(np.nanmin(x) if not np.isfinite(qlo) else qlo)
    hi = float(np.nanmax(x) if not np.isfinite(qhi) else qhi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    return np.linspace(lo, hi, int(n_bins) + 1, dtype="f8")


def _get_integer_edges(x: np.ndarray) -> np.ndarray:
    xf = x[np.isfinite(x)]
    lo, hi = int(np.floor(np.nanmin(xf))), int(np.ceil(np.nanmax(xf)))
    if hi <= lo:
        hi = lo + 1
    return np.arange(lo - 0.5, hi + 1.5, 1.0, dtype="f8")


# ---------------------------------------------------------------------------
# Per-feature plotting
# ---------------------------------------------------------------------------

def _plot_feature_overlay(
    feat: str,
    x: np.ndarray,
    w_signed: np.ndarray,
    w_local: np.ndarray,
    w_abs: np.ndarray,
    edges: np.ndarray,
    outdir: Path,
    xlabel: str,
    n_bins: int,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _norm_hist(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, _ = np.histogram(x, bins=edges, weights=w)
        denom = float(np.sum(np.abs(h)))
        if denom > 0:
            h = h / denom
        return h, 0.5 * (edges[:-1] + edges[1:])

    h_signed, centers = _norm_hist(w_signed)
    h_local, _ = _norm_hist(w_local)
    h_abs, _ = _norm_hist(w_abs)

    fig, ax = plt.subplots(figsize=(8.0, 6.0))

    # Draw order + zorder chosen so that all three curves stay visible even
    # where they overlap (signed == local by construction on many features):
    #   abs    -> red dashed line            (bottom)
    #   local  -> blue line + translucent fill (fill keeps it visible under signed)
    #   signed -> black solid line on top    (top)
    ax.step(centers, h_abs, where="mid", linewidth=1.8, color="#d62728",
            linestyle="--", label="abs", alpha=0.9, zorder=3)
    ax.fill_between(centers, h_local, 0.0, step="mid", color="#1f6fb2",
                    alpha=0.18, zorder=1, label=None)
    ax.step(centers, h_local, where="mid", linewidth=2.0, color="#1f6fb2",
            linestyle="-", label="local", alpha=0.9, zorder=4)
    ax.step(centers, h_signed, where="mid", linewidth=1.6, color="#111111",
            linestyle="-", label="signed", alpha=0.95, zorder=5)

    ax.axhline(0.0, color="k", linewidth=0.6, alpha=0.4)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel("Normalized events", fontsize=13)
    sums = dict(
        signed=float(np.sum(w_signed)), local=float(np.sum(w_local)), abs=float(np.sum(w_abs)),
    )
    ax.set_title(
        f"{feat}\n$\\sum w_{{\\rm signed}}$={sums['signed']:.3g}   "
        f"$\\sum w_{{\\rm local}}$={sums['local']:.3g}   "
        f"$\\sum |w|$={sums['abs']:.3g}",
        fontsize=12,
    )
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=11)

    try:
        import mplhep as hep
        hep.cms.label("Work in progress", loc=0, com=13.6, ax=ax)
    except Exception:
        pass

    fig.tight_layout()
    fig.savefig(outdir / f"{feat}_weight_comparison.png", dpi=170)
    fig.savefig(outdir / f"{feat}_weight_comparison.pdf")
    plt.close(fig)
    log.info("Wrote %s_weight_comparison.png", feat)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Weight-scheme normalized distribution comparison")
    ap.add_argument("--input", required=True, help="Folder of EVENTSELECTION.root files (as in train-dnn)")
    ap.add_argument("--dnn-config", default="configs/dnn.yaml", help="DNN config YAML (features + weight handling)")
    ap.add_argument("--xsection-signal-json", default="data/cross-section/xsection_signal.json")
    ap.add_argument("--xsection-json", default="data/cross-section/xsection_background_run3.json")
    ap.add_argument("--weight-branch", default="full_event_weight")
    ap.add_argument("--max-events-per-sample", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=7, help="Must match the training seed for a faithful local-cancellation fit")
    ap.add_argument("--outdir", default="outputs/dnn_25feature/weight_comparison")
    ap.add_argument("--n-bins", type=int, default=40)
    args = ap.parse_args()

    import yaml
    with open(args.dnn_config) as f:
        cfg = yaml.safe_load(f)

    features = [str(s) for s in cfg.get("features", []) if str(s).strip()]
    if not features:
        raise ValueError(f"'{args.dnn_config}' has no features: list — required.")
    variable_labels = cfg.get("variable_labels", {}) or {}
    nwh_cfg = cfg.get("training", {}).get("negative_weight_handling", {}) or {}
    seed = args.seed

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 0) Expand the input folder / file list exactly as train-dnn does.
    from darkbottomline.cli import _get_input_files
    raw_inputs = _get_input_files([args.input])
    input_files: list[str] = []
    for p in raw_inputs:
        if os.path.isdir(p):
            input_files.extend(sorted(str(f) for f in Path(p).iterdir() if f.suffix == ".root"))
        else:
            input_files.append(p)
    if not input_files:
        raise ValueError(f"No .root input files found from: {args.input}")
    log.info("Using %d input file(s) from %s", len(input_files), args.input)

    # 1) Load X, y, signed weights and per-event sample ids with the *exact*
    #    train-dnn loader (xsec / per-masspoint weighting included).
    from darkbottomline.cli import _load_training_data_from_eventsel
    X, y, w_signed, feature_sources, _mass, sample_ids = _load_training_data_from_eventsel(
        input_files=input_files,
        region="preselection",
        signal_patterns=None,
        signal_prefix=None,
        label_csv=None,
        weight_branch=args.weight_branch,
        max_events_per_file=args.max_events_per_sample,
        signal_cross_sections=_load_xsec(args.xsection_signal_json) if args.xsection_signal_json else None,
        background_cross_sections=_load_xsec(args.xsection_json) if args.xsection_json else None,
        lumi=1.0,
        features=features,
        parametric_input=False,
        mass_grid=None,
        seed=seed,
    )
    w_signed = np.asarray(w_signed, dtype="f8")
    w_signed = np.where(np.isfinite(w_signed), w_signed, 0.0)
    y = np.asarray(y, dtype="i4")
    sample_ids = np.asarray(sample_ids).astype(str)
    n_events = len(X)
    log.info("Loaded %d events, %d features, sum(signed)=%.6g", n_events, len(features), float(np.sum(w_signed)))

    # 2) Replicate the trainer's split + local-cancellation fit
    #    (fit on train rows per sample, apply to val/test).
    from sklearn.model_selection import train_test_split
    from darkbottomline.dnn_trainer import fit_dnn_weight_models, apply_dnn_weight_models

    indices = np.arange(n_events, dtype="i8")
    val_size = float(cfg.get("training", {}).get("val_size", 0.2))
    test_size = float(cfg.get("training", {}).get("test_size", 0.3))
    train_idx, temp_idx = train_test_split(indices, test_size=val_size + test_size,
                                           random_state=seed, stratify=y)
    test_frac_of_temp = test_size / (val_size + test_size)
    val_idx, test_idx = train_test_split(temp_idx, test_size=test_frac_of_temp,
                                         random_state=seed, stratify=y[temp_idx])

    weight_models, w_train_local, _stats = fit_dnn_weight_models(
        X.iloc[train_idx], w_signed[train_idx], sample_ids[train_idx], nwh_cfg,
    )
    w_val_local = apply_dnn_weight_models(X.iloc[val_idx], w_signed[val_idx], sample_ids[val_idx], weight_models, nwh_cfg)
    w_test_local = apply_dnn_weight_models(X.iloc[test_idx], w_signed[test_idx], sample_ids[test_idx], weight_models, nwh_cfg)

    w_local = np.empty_like(w_signed)
    w_local[train_idx] = w_train_local
    w_local[val_idx] = w_val_local
    w_local[test_idx] = w_test_local
    w_abs = np.abs(w_signed)

    # 3) Summary table for the user
    _print_weight_summary(w_signed, w_local, w_abs, y)

    # 4) One normalized overlay plot per feature
    for feat in features:
        x = X[feat].to_numpy(dtype="f8")
        finite = np.isfinite(x) & (x != SENTINEL)
        xv = x[finite]
        ws = w_signed[finite]
        wl = w_local[finite]
        wa = w_abs[finite]
        if xv.size == 0 or np.sum(wa) <= 0.0:
            log.warning("Skipping %s: no usable finite events", feat)
            continue

        if _is_discrete(xv):
            edges = _get_integer_edges(xv)
        else:
            edges = _get_edges(xv, wa, args.n_bins)
        _plot_feature_overlay(
            feat, xv, ws, wl, wa, edges, outdir,
            xlabel=variable_labels.get(feat, feat),
            n_bins=args.n_bins,
        )

    log.info("Done — plots written to %s", outdir)


def _load_xsec(path: str):
    import json
    with open(path) as f:
        raw = json.load(f)
    # Same normalization as PlotManager._normalize_cross_sections: flatten nested dicts.
    flat: dict[str, float] = {}
    for _k, v in raw.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                if not str(k2).startswith("_") and isinstance(v2, (int, float)):
                    flat[str(k2)] = float(v2)
        elif isinstance(v, (int, float)):
            flat[str(_k)] = float(v)
    return flat


def _print_weight_summary(w_signed, w_local, w_abs, y) -> None:
    sig = y == 1
    bkg = ~sig
    rows = [
        ("all", w_signed, w_local, w_abs),
        ("signal", w_signed[sig], w_local[sig], w_abs[sig]),
        ("background", w_signed[bkg], w_local[bkg], w_abs[bkg]),
    ]
    print("\n===== Weight scheme summary (sums) =====")
    print(f"{'class':<12} {'N':>10} {'sum_signed':>14} {'sum_abs':>14} {'sum_local':>14} {'neg_frac':>10}")
    for name, ws, wl, wa in rows:
        neg_frac = float(np.count_nonzero(ws < 0.0)) / max(ws.size, 1)
        print(f"{name:<12} {ws.size:>10d} {np.sum(ws):>14.6g} {np.sum(wa):>14.6g} "
              f"{np.sum(wl):>14.6g} {neg_frac:>10.3%}")
    print("=====\n")


if __name__ == "__main__":
    main()
