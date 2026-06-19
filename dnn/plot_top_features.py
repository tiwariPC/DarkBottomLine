#!/usr/bin/env python3
"""Plot top-K feature distributions (signal vs background overlays).

Reads the feature significance ranking from training output
(``feature_significance.json``) and plots normalised signal/background
overlays for the top-K most significant features.

Data source: ``ppbbchichi-trees.root`` (per-sample per-region trees).

Usage (standalone):
    python dnn/plot_top_features.py \
        --significance-json outputs_dnn_<tag>/feature_significance.json \
        --root outputfiles/merged/<tag>/ppbbchichi-trees.root \
        --region preselection --signal-prefix newdiboson \
        --outdir plot/<tag>/training --top-k 5

Called automatically by ``train_classifier.py`` after feature significance
is computed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import uproot

try:
    from darkbottomline.dnn.common import DEFAULT_SIGNAL_PATTERNS, is_signal, sanitize_feature_frame
    from darkbottomline.dnn.data import list_sample_region_trees, read_tree_as_arrays
    from darkbottomline.dnn.feature_engineering import build_feature_frame_from_tree
except ImportError:
    try:
        from dnn.common import DEFAULT_SIGNAL_PATTERNS, is_signal, sanitize_feature_frame
        from dnn.data import list_sample_region_trees, read_tree_as_arrays
        from dnn.feature_engineering import build_feature_frame_from_tree
    except ImportError:
        from common import DEFAULT_SIGNAL_PATTERNS, is_signal, sanitize_feature_frame
        from data import list_sample_region_trees, read_tree_as_arrays
        from feature_engineering import build_feature_frame_from_tree


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


def _parse_prefixes(prefix_text: str | None, default: str | None = None) -> tuple[str, ...]:
    if prefix_text is None:
        prefix_text = default
    if prefix_text is None:
        return tuple()
    vals = [p.strip().lower() for p in str(prefix_text).split(",") if p.strip()]
    return tuple(vals)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot top-K feature distributions (signal vs background)."
    )
    ap.add_argument(
        "--significance-json",
        required=True,
        help="Path to feature_significance.json (from training output).",
    )
    ap.add_argument(
        "--root",
        required=True,
        help="Path to ppbbchichi-trees.root.",
    )
    ap.add_argument(
        "--region",
        default="preselection",
        help="Region tree to read (default: preselection).",
    )
    ap.add_argument(
        "--outdir",
        required=True,
        help="Output directory for plots.",
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top features to plot (default: 5). Ignored when --all-features is set.",
    )
    ap.add_argument(
        "--all-features",
        action="store_true",
        help="Plot all features in the significance ranking (overrides --top-k).",
    )
    ap.add_argument(
        "--signal-prefix",
        default=None,
        help="Signal sample prefix, e.g. 'newdiboson'.",
    )
    ap.add_argument(
        "--signal-category",
        default=None,
        help="Signal category shorthand (e.g. 'diboson' -> newdiboson).",
    )
    ap.add_argument(
        "--background-prefix",
        default=None,
        help="Optional: label samples starting with this prefix as background.",
    )
    ap.add_argument(
        "--exclude-prefixes",
        default="run",
        help="Comma-separated sample prefixes to exclude (default: run).",
    )
    ap.add_argument(
        "--max-events-per-sample",
        type=int,
        default=200000,
        help="Cap events per sample (default: 200000).",
    )
    ap.add_argument(
        "--n-bins",
        type=int,
        default=40,
        help="Number of histogram bins (default: 40).",
    )
    ap.add_argument(
        "--bjet-filter",
        choices=["1b", "2bplus", "all"],
        default="all",
        help=(
            "Select which b-jet multiplicity events to use: "
            "'1b' = only background events with 1 b-jet; "
            "'2bplus' = only events with >=2 b-jets; "
            "'all' = all events (default)."
        ),
    )
    args = ap.parse_args()

    sig_path = Path(args.significance_json)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Read top-K features from significance JSON ----
    with sig_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    # Determine sort key (prefer syst-aware if available)
    has_syst = any("asimov_z_syst" in r for r in rows)
    sort_key = "asimov_z_syst" if has_syst else "asimov_z"
    rows_sorted = sorted(rows, key=lambda r: float(r.get(sort_key, 0.0)), reverse=True)

    top_features = []
    for r in rows_sorted:
        feat = str(r.get("feature", ""))
        if feat and feat not in top_features:
            top_features.append(feat)
        if not args.all_features and len(top_features) >= int(args.top_k):
            break

    if not top_features:
        raise ValueError(f"No features found in: {sig_path}")

    if args.all_features:
        print(f"[INFO] Plotting all {len(top_features)} features from significance ranking:")
    else:
        print(f"[INFO] Top-{len(top_features)} features from significance ranking:")
    for i, f in enumerate(top_features, 1):
        z = float(rows_sorted[i - 1].get(sort_key, 0.0))
        print(f"  {i}. {f}  (Z={z:.3f})")

    # ---- 2. Load data from ROOT ----
    region = args.region
    weight_branch = f"weight_{region}"
    signal_prefix = _resolve_signal_prefix(args.signal_prefix, args.signal_category)
    exclude_prefixes = _parse_prefixes(args.exclude_prefixes, default="run")
    bg_prefix = str(args.background_prefix).strip().lower() if args.background_prefix else None

    import pandas as pd

    # Branch names used for 1-bjet detection.
    BJET_DETECT_BRANCHES = ["sublead_bjet_pt"]

    X_parts, y_parts, w_parts = [], [], []
    used_samples: dict[str, int] = {}
    excluded_samples: dict[str, str] = {}

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
            df, _source_map, _used = build_feature_frame_from_tree(
                tree, top_features, max_events=args.max_events_per_sample,
            )
            df = sanitize_feature_frame(df)

            arrs = read_tree_as_arrays(
                f, tpath, branches=[weight_branch], max_events=args.max_events_per_sample,
            )
            w = np.asarray(arrs[weight_branch], dtype="f8")
            w = np.where(np.isfinite(w), w, 0.0)
            w = np.maximum(w, 0.0)

            n = min(len(w), len(df))
            df = df.iloc[:n].reset_index(drop=True)
            w = w[:n]

            # ---- Tag 1-bjet events and set costheta_star = -1 ----
            has_one_bjet = np.zeros(n, dtype=bool)
            try:
                bjet_arrs = read_tree_as_arrays(
                    f, tpath, branches=BJET_DETECT_BRANCHES, max_events=args.max_events_per_sample,
                )
                sublead_pt = np.asarray(bjet_arrs.get("sublead_bjet_pt", []), dtype="f8")[:n]
                has_one_bjet = np.abs(sublead_pt - (-9999.0)) < 1e-6
            except Exception:
                pass

            if signal_prefix:
                label = 1 if sample_l.startswith(signal_prefix) else 0
            elif bg_prefix:
                label = 0 if sample_l.startswith(bg_prefix) else 1
            else:
                label = 1 if is_signal(sample, DEFAULT_SIGNAL_PATTERNS) else 0

            y_arr = np.full(len(df), label, dtype=np.int8)

            # Set costheta_star = -1 for BACKGROUND 1-bjet events only
            is_bkg = (y_arr == 0)
            if "costheta_star" in df.columns and np.any(has_one_bjet & is_bkg):
                df.loc[has_one_bjet & is_bkg, "costheta_star"] = -1.0

            # ---- Apply b-jet filter (per-sample level) ----
            bjet_filter = args.bjet_filter
            if bjet_filter == "1b":
                keep_mask = (has_one_bjet & (y_arr == 0)) | (y_arr == 1)
            elif bjet_filter == "2bplus":
                keep_mask = ~has_one_bjet
            else:  # "all"
                keep_mask = np.ones(n, dtype=bool)

            if np.any(keep_mask):
                X_parts.append(df.iloc[keep_mask].reset_index(drop=True))
                y_parts.append(y_arr[keep_mask])
                w_parts.append(w[keep_mask])
                used_samples[str(sample)] = int(np.sum(keep_mask))
            else:
                excluded_samples[str(sample)] = "bjet_filter_empty"

    if not X_parts:
        raise ValueError("No events loaded. Check --exclude-prefixes, --bjet-filter, and signal/background rules.")

    X = pd.concat(X_parts, axis=0, ignore_index=True)
    y = np.concatenate(y_parts)
    w = np.concatenate(w_parts)

    y_i = np.asarray(y, dtype="i4")
    w_f = np.maximum(np.asarray(w, dtype="f8"), 0.0)
    sig_mask = y_i == 1
    bkg_mask = y_i == 0

    if not np.any(sig_mask) or not np.any(bkg_mask):
        raise ValueError("Only one class found after labeling. Check --signal-prefix or --bjet-filter.")

    # Count 1-bjet tagged events (costheta_star == -1)
    if "costheta_star" in X.columns:
        one_bjet_mask = np.abs(np.asarray(X["costheta_star"].to_numpy(), dtype="f8") - (-1.0)) < 1e-6
    else:
        one_bjet_mask = np.zeros(len(X), dtype=bool)

    print(f"[INFO] Loaded events: signal={int(sig_mask.sum()):,}  background={int(bkg_mask.sum()):,}")
    print(f"[INFO] 1-bjet tagged events (costheta_star=-1): {int(one_bjet_mask.sum()):,}")
    print(f"[INFO] >=2-bjet events: {int((~one_bjet_mask).sum()):,}")
    print(f"[INFO] Samples used: {len(used_samples)}, excluded: {len(excluded_samples)}")
    print(f"[INFO] B-jet filter: {args.bjet_filter}")

    # ---- 3. Plot per-feature overlays ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        import mplhep as hep
        hep.style.use("CMS")
        use_hep = True
    except Exception:
        use_hep = False

    for feat in top_features:
        x = np.asarray(X[feat].to_numpy(), dtype="f8")
        ok = np.isfinite(x)
        x = x[ok]
        yy = y_i[ok]
        ww = w_f[ok]

        xs = x[yy == 1]
        xb = x[yy == 0]
        ws = ww[yy == 1]
        wb = ww[yy == 0]

        if xs.size == 0 or xb.size == 0:
            print(f"[SKIP] {feat}: no signal or background events")
            continue

        # Determine bin edges from 1%-99% weighted percentile range
        x_all = np.concatenate([xs, xb])
        w_all = np.concatenate([ws, wb])
        sorter = np.argsort(x_all)
        cdf = np.cumsum(w_all[sorter])
        cdf = cdf / max(cdf[-1], 1e-12)
        qlo = float(np.interp(0.01, cdf, x_all[sorter]))
        qhi = float(np.interp(0.99, cdf, x_all[sorter]))
        lo = float(np.nanmin(x_all) if not np.isfinite(qlo) else qlo)
        hi = float(np.nanmax(x_all) if not np.isfinite(qhi) else qhi)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = float(np.nanmin(x_all))
            hi = float(np.nanmax(x_all))
        if hi <= lo:
            hi = lo + 1.0

        edges = np.linspace(lo, hi, int(args.n_bins) + 1, dtype="f8")
        hs, _ = np.histogram(xs, bins=edges, weights=ws)
        hb, _ = np.histogram(xb, bins=edges, weights=wb)

        hs_norm = hs / max(np.sum(hs), 1e-12)
        hb_norm = hb / max(np.sum(hb), 1e-12)
        centers = 0.5 * (edges[:-1] + edges[1:])
        widths = np.diff(edges)

        # Get the Z value for this feature from the significance ranking
        feat_info = next((r for r in rows if str(r.get("feature")) == feat), {})
        z_val = float(feat_info.get(sort_key, feat_info.get("asimov_z", 0.0)))
        auc_val = float(feat_info.get("auc", float("nan")))

        fig, ax = plt.subplots(figsize=(7.5, 6.0))

        ax.step(centers, hs_norm, where="mid", linewidth=2.0, color="#bd1f01", label="Signal")
        ax.step(centers, hb_norm, where="mid", linewidth=2.0, color="#3f90da", label="Background")
        ax.bar(centers, hb_norm, width=widths, alpha=0.18, color="#3f90da", align="center")

        ax.set_xlabel(feat, fontsize=12)
        ax.set_ylabel("Normalised events", fontsize=12)
        auc_str = f"AUC={auc_val:.3f}" if np.isfinite(auc_val) else "AUC=N/A"
        ax.set_title(f"Top feature: {feat}  |  {auc_str}, Z={z_val:.2f}", fontsize=13)
        ax.grid(alpha=0.2)
        ax.legend(loc="best", fontsize=11)

        if use_hep:
            hep.cms.label("Work in progress", loc=0, com=13.6, ax=ax)

        fig.tight_layout()

        # Sanitise feature name for filename
        safe_feat = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in str(feat))
        file_prefix = "all_features" if args.all_features else "top5_feature"
        out_png = outdir / f"{file_prefix}_{safe_feat}.png"
        fig.savefig(out_png, dpi=170, bbox_inches="tight")
        out_pdf = outdir / f"{file_prefix}_{safe_feat}.pdf"
        fig.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] {feat} -> {out_png}")

    # ---- 4. Summary JSON ----
    summary = {
        "significance_json": str(sig_path),
        "root": str(args.root),
        "region": region,
        "signal_prefix": signal_prefix,
        "top_k": int(args.top_k),
        "all_features": bool(args.all_features),
        "bjet_filter": args.bjet_filter,
        "plotted_features": top_features,
        "used_samples": used_samples,
        "excluded_samples": excluded_samples,
    }
    (outdir / "top_features_plot_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(f"[OK] Summary written to: {outdir / 'top_features_plot_summary.json'}")


if __name__ == "__main__":
    main()
