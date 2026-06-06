#!/usr/bin/env python3
"""Compute DNN significance at a given signal-efficiency working point.

Loads a trained DNN model, scores the test set, and reports the syst-aware
Asimov Z (Cowan et al. 2011) at a user-specified signal efficiency cut.

Usage:
    python dnn/compute_dnn_significance.py \
        --model-dir outputs_dnn_<tag> \
        --root outputfiles/merged/<tag>/ppbbchichi-trees.root \
        --sig-syst 0.25 --min-signal-eff 0.20

Outputs:
    - dnn_significance.json  : Z vs. signal efficiency scan
    - dnn_significance.png   : scan plot + score distribution
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
#  Z formulas (same as elsewhere)
# ---------------------------------------------------------------------------

def _asimov_z_syst(s: float, b: float, sigma_rel: float, eps: float = 1e-12) -> float:
    """Syst-aware Asimov Z (Cowan et al. 2011)."""
    if sigma_rel <= 0.0:
        if b <= eps or s <= eps:
            return 0.0
        term = (s + b) * np.log1p(s / b) - s
        return float(np.sqrt(2.0 * max(term, 0.0)))

    sb2 = (sigma_rel * b) ** 2
    num = (s + b) * (b + sb2)
    den = b * b + (s + b) * sb2
    ratio1 = num / max(den, eps)
    term1 = 0.0 if ratio1 <= 1.0 else (s + b) * np.log(ratio1)
    ratio2 = sb2 * s / max(b * (b + sb2), eps)
    term2 = (b * b / max(sb2, eps)) * np.log1p(ratio2) if sb2 > eps else 0.0
    z2 = 2.0 * max(term1 - term2, 0.0)
    return float(np.sqrt(max(z2, 0.0)))


def _asimov_significance_from_hist_syst(
    sig: np.ndarray, bkg: np.ndarray, sigma_rel: float, eps: float = 1e-12,
) -> float:
    """Binned syst-aware Asimov Z."""
    s = np.maximum(np.asarray(sig, dtype="f8"), 0.0)
    b = np.maximum(np.asarray(bkg, dtype="f8"), 0.0)
    if sigma_rel <= 0.0:
        term = np.where(b > eps, (s + b) * np.log1p(np.divide(s, b, out=np.zeros_like(s), where=b > eps)) - s, 0.0)
        return float(np.sqrt(2.0 * np.sum(np.maximum(term, 0.0))))

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
    return float(np.sqrt(2.0 * np.sum(np.maximum(term, 0.0))))


# ---------------------------------------------------------------------------
#  main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="DNN significance at signal efficiency working points.")
    ap.add_argument("--model-dir", required=True, help="Path to DNN output dir (contains dnn_model.pt, scaler.json, features.json)")
    ap.add_argument("--root", required=True, help="Path to ppbbchichi-trees.root")
    ap.add_argument("--region", default="preselection")
    ap.add_argument("--signal-prefix", default="run3", help="Signal sample prefix")
    ap.add_argument("--max-events-per-sample", type=int, default=200000)
    ap.add_argument("--test-size", type=float, default=0.3)
    ap.add_argument("--val-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--sig-syst", type=float, default=0.25,
                    help="Fractional systematic on background (default: 0.25 = 25%% from your table).")
    ap.add_argument("--min-signal-eff", type=float, default=0.20,
                    help="Target signal efficiency for the summary (default: 0.20).")
    ap.add_argument("--outdir", default=None,
                    help="Output directory for plots/JSON (default: model-dir/significance_scan/).")
    ap.add_argument("--n-bins", type=int, default=50, help="Bins for binned Z.")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    outdir = Path(args.outdir) if args.outdir else model_dir / "significance_scan"
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- Load model ----
    import torch
    try:
        from darkbottomline.dnn.model import ModelSpec, build_mlp
        from darkbottomline.dnn.scaler import StandardScaler
    except ImportError:
        try:
            from dnn.model import ModelSpec, build_mlp
            from dnn.scaler import StandardScaler
        except ImportError:
            from model import ModelSpec, build_mlp
            from scaler import StandardScaler

    ckpt = torch.load(model_dir / "dnn_model.pt", map_location="cpu", weights_only=False)
    spec = ModelSpec(
        n_inputs=ckpt["spec"]["n_inputs"],
        hidden_layers=tuple(ckpt["spec"]["hidden_layers"]),
        dropout=ckpt["spec"]["dropout"],
    )
    model = build_mlp(spec)
    model.load_state_dict(ckpt["state_dict"])
    model.to(args.device)
    model.eval()

    sd = json.loads((model_dir / "scaler.json").read_text())
    scaler = StandardScaler(
        mean=np.array(sd["mean"]), std=np.array(sd["std"]),
        missing_sentinel=sd["missing_sentinel"],
    )
    features = json.loads((model_dir / "features.json").read_text())
    print(f"[OK] Loaded DNN: {len(features)} features, spec={spec}")

    # ---- Load data ----
    import uproot
    try:
        from darkbottomline.dnn.common import sanitize_feature_frame
        from darkbottomline.dnn.data import list_sample_region_trees, read_tree_as_arrays
        from darkbottomline.dnn.feature_engineering import build_feature_frame_from_tree
    except ImportError:
        try:
            from dnn.common import sanitize_feature_frame
            from dnn.data import list_sample_region_trees, read_tree_as_arrays
            from dnn.feature_engineering import build_feature_frame_from_tree
        except ImportError:
            from common import sanitize_feature_frame
            from data import list_sample_region_trees, read_tree_as_arrays
            from feature_engineering import build_feature_frame_from_tree

    X_parts, y_parts, w_parts = [], [], []
    region = args.region
    weight_branch = f"weight_{region}"

    with uproot.open(args.root) as f:
        trees = list_sample_region_trees(f, region)
        for sample, tpath in trees:
            sl = str(sample).lower()
            # Exclude run* but keep the signal (run3*)
            if sl.startswith("run") and not sl.startswith(args.signal_prefix):
                continue

            df, _, _ = build_feature_frame_from_tree(f[tpath], features, max_events=args.max_events_per_sample)
            df = sanitize_feature_frame(df)
            arrs = read_tree_as_arrays(f, tpath, branches=[weight_branch], max_events=args.max_events_per_sample)
            w = np.asarray(arrs[weight_branch], dtype="f8")
            w = np.where(np.isfinite(w), w, 0.0)
            w = np.clip(w, -100.0, 100.0)

            n = min(len(w), len(df))
            df = df.iloc[:n].reset_index(drop=True); w = w[:n]

            label = 1 if sl.startswith(args.signal_prefix) else 0
            y_arr = np.full(len(df), label, dtype=np.int8)

            X_parts.append(df)
            y_parts.append(y_arr)
            w_parts.append(w)

    import pandas as pd
    X = pd.concat(X_parts, axis=0, ignore_index=True)
    y = np.concatenate(y_parts)
    w = np.concatenate(w_parts)

    # ---- Train/val/test split (same as training) ----
    from sklearn.model_selection import train_test_split
    val_sz = float(args.val_size); test_sz = float(args.test_size)
    _, X_temp, _, y_temp, _, w_temp = train_test_split(
        X, y, w, test_size=val_sz + test_sz, random_state=args.seed, stratify=y,
    )
    t_frac = test_sz / (val_sz + test_sz)
    X_test, _, y_test, _, w_test, _ = train_test_split(
        X_temp, y_temp, w_temp, test_size=t_frac, random_state=args.seed, stratify=y_temp,
    )
    print(f"[OK] Test set: N={len(X_test):,}  S={(y_test==1).sum():,}  B={(y_test==0).sum():,}")

    # ---- DNN inference ----
    X_np = scaler.transform(X_test.to_numpy(dtype="f8")).astype("float32")
    with torch.no_grad():
        scores = torch.sigmoid(model(torch.from_numpy(X_np).to(args.device)).squeeze(1)).cpu().numpy()

    y_i = y_test.astype("i4")
    w_i = w_test.copy()  # preserve negative weights (NLO cancellation)
    s_total = float(w_i[y_i == 1].sum())
    b_total = float(w_i[y_i == 0].sum())
    s_raw = int((y_i == 1).sum())
    b_raw = int((y_i == 0).sum())

    print(f"[OK] Test set: N={len(X_test):,}  S_raw={s_raw:,}  B_raw={b_raw:,}")
    print(f"[OK] Test yields: S_w={s_total:.1f}  B_w={b_total:.1f}  S/B={s_total/max(b_total,1e-12):.4f}")

    # ---- Full binned Z (all scores) ----
    n_bins = int(args.n_bins)
    hs_all, _ = np.histogram(scores[y_i == 1], bins=n_bins, range=(0, 1), weights=w_i[y_i == 1])
    hb_all, _ = np.histogram(scores[y_i == 0], bins=n_bins, range=(0, 1), weights=w_i[y_i == 0])
    z_full_binned = _asimov_significance_from_hist_syst(hs_all, hb_all, float(args.sig_syst))
    print(f"[OK] Full DNN binned Z_syst ({n_bins}b, σ_rel={args.sig_syst*100:.0f}%) = {z_full_binned:.4f}")

    # ---- Scan: significance vs signal efficiency ----
    thresholds = np.linspace(0.01, 0.999, 200)
    scan_rows = []
    for thr in thresholds:
        mask = scores > thr
        mask_s = mask & (y_i == 1)
        mask_b = mask & (y_i == 0)
        s_cut = float(w_i[mask_s].sum())
        b_cut = float(w_i[mask_b].sum())
        s_raw_cut = int(mask_s.sum())
        b_raw_cut = int(mask_b.sum())
        eff_sig = s_cut / max(s_total, 1e-12)
        eff_bkg = b_cut / max(b_total, 1e-12)

        if s_cut <= 0 or b_cut <= 0:
            z_cut = 0.0
        else:
            z_cut = _asimov_z_syst(s_cut, b_cut, float(args.sig_syst))

        # Also binned Z above threshold
        if s_raw_cut > 0:
            hs, _ = np.histogram(scores[mask_s], bins=n_bins, range=(thr, 1), weights=w_i[mask_s])
            hb, _ = np.histogram(scores[mask_b], bins=n_bins, range=(thr, 1), weights=w_i[mask_b])
            z_binned = _asimov_significance_from_hist_syst(hs, hb, float(args.sig_syst))
        else:
            z_binned = 0.0

        scan_rows.append({
            "threshold": float(thr),
            "signal_eff": float(eff_sig),
            "background_eff": float(eff_bkg),
            "s_cut": float(s_cut),
            "b_cut": float(b_cut),
            "s_raw_cut": int(s_raw_cut),
            "b_raw_cut": int(b_raw_cut),
            "s_over_b": float(s_cut / max(b_cut, 1e-12)),
            "z_syst_counting": float(z_cut),
            "z_syst_binned": float(z_binned),
        })

    # ---- Summary at target signal efficiency ----
    target_eff = float(args.min_signal_eff)
    best_at_target = max(
        (r for r in scan_rows if r["signal_eff"] >= target_eff),
        key=lambda r: r["z_syst_counting"],
        default=None,
    )

    if best_at_target:
        print(f"\n{'='*60}")
        print(f"=== DNN at ε_sig ≥ {target_eff*100:.0f}% ===")
        print(f"  Score threshold : {best_at_target['threshold']:.4f}")
        print(f"  S_w (weighted)  : {best_at_target['s_cut']:.1f}")
        print(f"  B_w (weighted)  : {best_at_target['b_cut']:.1f}")
        print(f"  S_raw (events)  : {best_at_target.get('s_raw_cut', 'N/A')}")
        print(f"  B_raw (events)  : {best_at_target.get('b_raw_cut', 'N/A')}")
        print(f"  S/B             : {best_at_target['s_over_b']:.4f}")
        print(f"  ε_sig           : {best_at_target['signal_eff']*100:.1f}%")
        print(f"  ε_bkg           : {best_at_target['background_eff']*100:.2f}%")
        print(f"  Z_syst (counting): {best_at_target['z_syst_counting']:.4f}")
        print(f"  Z_syst (binned)  : {best_at_target['z_syst_binned']:.4f}")
        print(f"{'='*60}")

    # ---- Save JSON ----
    result = {
        "model_dir": str(model_dir),
        "sig_syst": float(args.sig_syst),
        "n_bins": int(n_bins),
        "test_s_total": float(s_total),
        "test_b_total": float(b_total),
        "test_s_raw": int(s_raw),
        "test_b_raw": int(b_raw),
        "z_full_binned": float(z_full_binned),
        "target_signal_eff": float(target_eff),
        "best_at_target": best_at_target,
        "scan": scan_rows,
    }
    (outdir / "dnn_significance.json").write_text(json.dumps(result, indent=2) + "\n")

    # ---- Plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # (0,0): Z_syst vs signal efficiency
    ax = axes[0, 0]
    effs = [r["signal_eff"] for r in scan_rows]
    z_vals = [r["z_syst_counting"] for r in scan_rows]
    ax.plot(effs, z_vals, color="#d62728", linewidth=2)
    ax.axvline(target_eff, color="gray", linestyle="--", alpha=0.5, label=f"ε_sig={target_eff*100:.0f}%")
    ax.set_xlabel("Signal efficiency"); ax.set_ylabel(f"Z_syst (σ_rel={args.sig_syst*100:.0f}%)")
    ax.set_title("Counting-experiment Z_syst vs signal efficiency")
    ax.grid(alpha=0.3); ax.legend()

    # (0,1): Z_syst (binned) vs signal efficiency
    ax = axes[0, 1]
    z_bv = [r["z_syst_binned"] for r in scan_rows]
    ax.plot(effs, z_bv, color="#2ca02c", linewidth=2)
    ax.axvline(target_eff, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Signal efficiency"); ax.set_ylabel(f"Z_syst binned ({n_bins}b)")
    ax.set_title("Binned Z_syst vs signal efficiency")
    ax.grid(alpha=0.3)

    # (1,0): Score distributions
    ax = axes[1, 0]
    bins = np.linspace(0, 1, 51)
    ax.hist(scores[y_i == 1], bins=bins, weights=w_i[y_i == 1], histtype="step",
            linewidth=2, color="#bd1f01", label="Signal", density=True)
    ax.hist(scores[y_i == 0], bins=bins, weights=w_i[y_i == 0], histtype="step",
            linewidth=2, color="#3f90da", label="Background", density=True)
    if best_at_target:
        ax.axvline(best_at_target["threshold"], color="gray", linestyle="--", alpha=0.5,
                    label=f"threshold={best_at_target['threshold']:.2f}")
    ax.set_xlabel("DNN score"); ax.set_ylabel("Normalised events")
    ax.set_title(f"Score distributions (Z_full_binned={z_full_binned:.2f})")
    ax.legend(fontsize=9); ax.grid(alpha=0.2)

    # (1,1): S/B vs signal efficiency
    ax = axes[1, 1]
    sb_vals = [r["s_over_b"] for r in scan_rows]
    ax.plot(effs, sb_vals, color="#1f77b4", linewidth=2)
    ax.axvline(target_eff, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Signal efficiency"); ax.set_ylabel("S/B")
    ax.set_title("S/B vs signal efficiency")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(outdir / "dnn_significance.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Wrote: {outdir / 'dnn_significance.json'}")
    print(f"[OK] Wrote: {outdir / 'dnn_significance.png'}")


if __name__ == "__main__":
    main()
