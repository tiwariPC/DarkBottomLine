#!/usr/bin/env python3
"""Make CMS-style feature comparison plots for signal vs background.

Outputs:
- One normalized comparison plot per feature (signal vs background overlays).
- A per-feature significance ranking (binned Asimov Z) as csv/json/png.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import uproot

try:
    from dnn.common import DEFAULT_SIGNAL_PATTERNS, is_signal, sanitize_feature_frame
    from dnn.data import list_sample_region_trees, read_tree_as_arrays
    from dnn.feature_engineering import REQUESTED_FEATURES_25, build_feature_frame_from_tree, get_default_feature_csv
except ModuleNotFoundError:
    from common import DEFAULT_SIGNAL_PATTERNS, is_signal, sanitize_feature_frame
    from data import list_sample_region_trees, read_tree_as_arrays
    from feature_engineering import REQUESTED_FEATURES_25, build_feature_frame_from_tree, get_default_feature_csv


DEFAULT_FEATURES = list(REQUESTED_FEATURES_25)


def _load_label_csv(path: str) -> dict[str, int]:
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


def _resolve_signal_prefix(signal_prefix: str | None, signal_category: str | None) -> str | None:
    if signal_prefix:
        return str(signal_prefix).strip().lower()
    if signal_category:
        cat = str(signal_category).strip().lower()
        if not cat:
            return None
        return cat if cat.startswith("new") else f"new{cat}"
    return None


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
    z2 = 2.0 * np.sum(np.maximum(term, 0.0))
    return float(np.sqrt(max(z2, 0.0)))


def _get_edges(x_all: np.ndarray, w_all: np.ndarray, n_bins: int) -> np.ndarray:
    qlo, qhi = _weighted_percentile(x_all, w_all, np.array([0.01, 0.99], dtype="f8"))
    lo = float(np.nanmin(x_all) if not np.isfinite(qlo) else qlo)
    hi = float(np.nanmax(x_all) if not np.isfinite(qhi) else qhi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(x_all))
        hi = float(np.nanmax(x_all))
    if hi <= lo:
        hi = lo + 1.0
    return np.linspace(lo, hi, int(n_bins) + 1, dtype="f8")


def main():
    ap = argparse.ArgumentParser(description="CMS-style feature comparison plots from ppbbchichi trees")
    ap.add_argument("--root", required=True, help="Path to ppbbchichi-trees.root")
    ap.add_argument("--region", default="preselection", help="Region tree to read")
    ap.add_argument("--features", default=get_default_feature_csv(), help="Comma-separated features")
    ap.add_argument("--max-events-per-sample", type=int, default=200000)
    ap.add_argument("--signal-pattern", action="append", default=None)
    ap.add_argument("--signal-prefix", default=None)
    ap.add_argument("--signal-category", default=None)
    ap.add_argument("--exclude-prefixes", default="run")
    ap.add_argument("--label-csv", default=None)
    ap.add_argument("--background-prefix", default=None)
    ap.add_argument("--outdir", default="plot", help="Output plot directory")
    ap.add_argument("--n-bins", type=int, default=40)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    features = [s.strip() for s in args.features.split(",") if s.strip()]
    if not features:
        raise ValueError("No features specified")

    patterns = tuple(args.signal_pattern) if args.signal_pattern else DEFAULT_SIGNAL_PATTERNS
    label_map = _load_label_csv(args.label_csv) if args.label_csv else None
    bg_prefix = str(args.background_prefix).strip().lower() if args.background_prefix else None
    signal_prefix = _resolve_signal_prefix(args.signal_prefix, args.signal_category)
    exclude_prefixes = _parse_prefixes(args.exclude_prefixes, default="run")

    if signal_prefix and label_map is not None:
        print("[WARN] --label-csv is provided; --signal-prefix/--signal-category are ignored for labels.")

    import pandas as pd

    X_parts = []
    y_parts = []
    w_parts = []
    used_samples: dict[str, int] = {}
    excluded_samples: dict[str, str] = {}
    sample_labels: dict[str, int] = {}
    label_counts = {0: 0, 1: 0}
    feature_sources: dict[str, str] = {}

    weight_branch = f"weight_{args.region}"

    with uproot.open(args.root) as f:
        trees = list_sample_region_trees(f, args.region)
        if not trees:
            raise FileNotFoundError(f"No trees found for region '{args.region}' in {args.root}")

        for sample, tpath in trees:
            sample_l = str(sample).lower()
            if exclude_prefixes and any(sample_l.startswith(p) for p in exclude_prefixes):
                excluded_samples[sample] = "excluded_prefix"
                continue

            tree = f[tpath]
            df, source_map, _used_branches = build_feature_frame_from_tree(
                tree,
                features,
                max_events=args.max_events_per_sample,
            )
            df = sanitize_feature_frame(df)

            arrs = read_tree_as_arrays(f, tpath, branches=[weight_branch], max_events=args.max_events_per_sample)

            w = np.asarray(arrs[weight_branch], dtype="f8")
            w = np.where(np.isfinite(w), w, 0.0)
            w = np.maximum(w, 0.0)

            if len(w) != len(df):
                n = min(len(w), len(df))
                df = df.iloc[:n].reset_index(drop=True)
                w = w[:n]

            if label_map is not None:
                if sample not in label_map:
                    raise KeyError(f"Sample '{sample}' not found in --label-csv")
                label = int(label_map[sample])
            else:
                if signal_prefix:
                    label = 1 if sample_l.startswith(signal_prefix) else 0
                elif bg_prefix:
                    label = 0 if sample_l.startswith(bg_prefix) else 1
                else:
                    label = 1 if is_signal(sample, patterns) else 0

            X_parts.append(df)
            y_parts.append(np.full(df.shape[0], label, dtype=np.int8))
            w_parts.append(w)
            used_samples[sample] = int(df.shape[0])
            sample_labels[sample] = int(label)
            label_counts[int(label)] += 1

            for feat in features:
                if feat not in feature_sources:
                    feature_sources[feat] = source_map.get(feat, "unknown")

    if not X_parts:
        raise ValueError(
            "No events selected after filtering. "
            "Check --exclude-prefixes and signal/background options."
        )

    X = pd.concat(X_parts, axis=0, ignore_index=True)
    y = np.concatenate(y_parts)
    w = np.concatenate(w_parts)

    if np.unique(y).size < 2:
        raise ValueError("Only one class found after labeling.")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        import mplhep as hep

        hep.style.use("CMS")
        use_hep = True
    except Exception:
        use_hep = False

    from sklearn.metrics import roc_auc_score

    signif_rows = []

    for feat in features:
        x = np.asarray(X[feat].to_numpy(), dtype="f8")
        m_all = np.isfinite(x)
        x = x[m_all]
        yy = np.asarray(y[m_all], dtype="i4")
        ww = np.maximum(np.asarray(w[m_all], dtype="f8"), 0.0)

        xs = x[yy == 1]
        xb = x[yy == 0]
        ws = ww[yy == 1]
        wb = ww[yy == 0]

        if xs.size == 0 or xb.size == 0 or np.sum(ws) <= 0.0 or np.sum(wb) <= 0.0:
            continue

        edges = _get_edges(x, ww, n_bins=int(args.n_bins))
        hs, _ = np.histogram(xs, bins=edges, weights=ws)
        hb, _ = np.histogram(xb, bins=edges, weights=wb)

        auc = float(roc_auc_score(yy, x, sample_weight=ww))
        z = _asimov_significance_from_hist(hs, hb)
        signif_rows.append({"feature": feat, "auc": auc, "asimov_z": z})

        hs_norm = hs / max(np.sum(hs), 1e-12)
        hb_norm = hb / max(np.sum(hb), 1e-12)
        centers = 0.5 * (edges[:-1] + edges[1:])
        widths = np.diff(edges)

        fig, ax = plt.subplots(figsize=(7.5, 6.0))
        ax.step(centers, hs_norm, where="mid", linewidth=2.0, color="#bd1f01", label="Signal")
        ax.step(centers, hb_norm, where="mid", linewidth=2.0, color="#3f90da", label="Background")
        ax.bar(centers, hb_norm, width=widths, alpha=0.18, color="#3f90da", align="center")
        ax.set_xlabel(feat)
        ax.set_ylabel("Normalized events")
        ax.set_title(f"{feat}  |  AUC={auc:.3f}, Asimov Z={z:.2f}")
        ax.grid(alpha=0.2)
        ax.legend(loc="best")

        if use_hep:
            hep.cms.label("Work in progress", loc=0, com=13.6, ax=ax)

        fig.tight_layout()
        fig.savefig(outdir / f"{feat}_comparison.png", dpi=170)
        fig.savefig(outdir / f"{feat}_comparison.pdf")
        plt.close(fig)

    signif_rows.sort(key=lambda r: (r["asimov_z"], abs(r["auc"] - 0.5)), reverse=True)
    (outdir / "feature_significance.json").write_text(json.dumps(signif_rows, indent=2) + "\n")
    (outdir / "sample_selection.json").write_text(
        json.dumps(
            {
                "root": str(args.root),
                "region": str(args.region),
                "signal_prefix": signal_prefix,
                "signal_category": args.signal_category,
                "exclude_prefixes": list(exclude_prefixes),
                "used_samples": used_samples,
                "excluded_samples": excluded_samples,
                "sample_labels": sample_labels,
                "label_sample_counts": {
                    "background": int(label_counts[0]),
                    "signal": int(label_counts[1]),
                },
                "feature_sources": feature_sources,
            },
            indent=2,
        )
        + "\n"
    )

    import pandas as pd

    pd.DataFrame(signif_rows).to_csv(outdir / "feature_significance.csv", index=False)

    if signif_rows:
        labels = [r["feature"] for r in signif_rows]
        vals = [float(r["asimov_z"]) for r in signif_rows]
        plt.figure(figsize=(max(8, 0.55 * len(labels)), 5.5))
        plt.bar(labels, vals, color="#3f90da", edgecolor="black", linewidth=0.8)
        plt.xticks(rotation=40, ha="right")
        plt.ylabel("Feature significance (binned Asimov Z)")
        plt.title("Per-feature significance ranking")
        if use_hep:
            import mplhep as hep

            hep.cms.label("Work in progress", loc=0, com=13.6)
        plt.tight_layout()
        plt.savefig(outdir / "feature_significance.png", dpi=170)
        plt.close()

    print(
        "[OK] Wrote plots and significance into: "
        f"{outdir} (used={len(used_samples)}, excluded={len(excluded_samples)})"
    )


if __name__ == "__main__":
    main()
