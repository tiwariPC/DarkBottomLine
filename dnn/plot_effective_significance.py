#!/usr/bin/env python3
"""Plot effective feature significance (Delta-Z over baseline).

Uses syst-aware Asimov Z (Cowan et al. 2011) when available in the input JSON
(``asimov_z_syst`` field); falls back to pure statistical Asimov Z otherwise.

Input:
- training/feature_significance.json

Output:
- training/effective_significance.png
- training/effective_significance_syst.png  (if syst-aware data available)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot effective significance (Delta-Z)")
    parser.add_argument(
        "--input",
        default="plot/outpus_dnn_eos_4_2_2022_newdiboson_25feature_balanced_auto/training/feature_significance.json",
        help="Path to feature_significance.json",
    )
    parser.add_argument(
        "--output",
        default="plot/outpus_dnn_eos_4_2_2022_newdiboson_25feature_balanced_auto/training/effective_significance.png",
        help="Output PNG path (suffix _syst.png appended for syst-aware plot)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=25,
        help="Show top-K features by Delta-Z (default: 25)",
    )
    parser.add_argument(
        "--sig-syst",
        type=float,
        default=None,
        help=(
            "Override the systematic uncertainty label in plot titles. "
            "If not set, read from input JSON when present."
        ),
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    with input_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    # --- Detect which Z metric to use ---
    has_syst = any("asimov_z_syst" in r for r in rows)
    z_key = "asimov_z_syst" if has_syst else "asimov_z"
    z_label = "asimov_z_syst" if has_syst else "asimov_z"

    # Try to read sig_syst from first row's metadata or use CLI argument
    sig_syst = args.sig_syst
    if sig_syst is None and has_syst:
        # Infer from data: look for a "sig_syst" field or use a default
        sig_syst = 0.20  # sensible default

    # --- Baseline ---
    # Prefer pfMetCorrSig as baseline anchor; fallback to min Z.
    baseline = None
    for r in rows:
        if str(r.get("feature", "")) == "pfMetCorrSig":
            baseline = float(r.get(z_key, r.get("asimov_z", 0.0)))
            break
    if baseline is None:
        baseline = min(float(r.get(z_key, r.get("asimov_z", 0.0))) for r in rows)

    enriched = []
    for r in rows:
        z = float(r.get(z_key, r.get("asimov_z", 0.0)))
        auc = float(r.get("auc", 0.5))
        delta_z = z - baseline
        rel = (delta_z / baseline * 100.0) if baseline > 0.0 else 0.0
        eff_auc = max(auc, 1.0 - auc)
        # Also carry the pure stat Z for reference
        z_stat = float(r.get("asimov_z", float("nan")))
        enriched.append(
            {
                "feature": str(r["feature"]),
                "auc": auc,
                "effective_auc": eff_auc,
                "asimov_z": z_stat,
                "asimov_z_syst": z,
                "delta_z": delta_z,
                "relative_improvement": rel,
            }
        )

    enriched.sort(key=lambda x: x["delta_z"], reverse=True)
    if args.top_k > 0:
        enriched = enriched[: args.top_k]

    def _make_plot(enr, z_label, sig_syst_label, out_path):
        labels = [e["feature"] for e in enr]
        delta_vals = [e["delta_z"] for e in enr]

        import matplotlib.pyplot as plt

        plt.style.use("default")
        fig, ax = plt.subplots(figsize=(18, 8))

        colors = []
        for dz in delta_vals:
            if dz >= 3.0:
                colors.append("#2ca02c")
            elif dz >= 1.0:
                colors.append("#ffbf4d")
            else:
                colors.append("#d62728")

        bars = ax.bar(range(len(labels)), delta_vals, color=colors, alpha=0.9,
                       edgecolor="black", linewidth=0.6)
        ax.set_ylabel("Delta-Z (sigma)", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Feature Effective Significance ({sig_syst_label}, Delta-Z over baseline)",
            fontsize=14, fontweight="bold",
        )
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_ylim(bottom=0)

        for i, (bar, val) in enumerate(zip(bars, delta_vals)):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0, val + 0.02,
                f"{val:.2f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
            )

        legend_text = (
            f"Baseline Z = {baseline:.3f}σ | "
            f"Green: ΔZ≥3 | Amber: 1≤ΔZ<3 | Red: ΔZ<1"
        )
        ax.text(
            0.5, -0.25, legend_text, transform=ax.transAxes,
            fontsize=10, ha="center",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    # --- Plot using the primary (syst-aware) metric ---
    syst_str = f"syst-aware σ_rel={sig_syst*100:.0f}%" if sig_syst else z_label
    _make_plot(enriched, z_label, syst_str, output_path)

    # --- Also produce a pure-stat Delta-Z plot for comparison ---
    if has_syst:
        enriched_stat = []
        for r in rows:
            z_s = float(r.get("asimov_z", 0.0))
            auc = float(r.get("auc", 0.5))
            b_stat = None
            for rr in rows:
                if str(rr.get("feature", "")) == "pfMetCorrSig":
                    b_stat = float(rr.get("asimov_z", 0.0))
                    break
            if b_stat is None:
                b_stat = min(float(rr.get("asimov_z", 0.0)) for rr in rows)
            delta_z_s = z_s - b_stat
            rel_s = (delta_z_s / b_stat * 100.0) if b_stat > 0.0 else 0.0
            enriched_stat.append(
                {
                    "feature": str(r["feature"]),
                    "auc": auc,
                    "effective_auc": max(auc, 1.0 - auc),
                    "asimov_z": z_s,
                    "asimov_z_syst": float(r.get("asimov_z_syst", float("nan"))),
                    "delta_z": delta_z_s,
                    "relative_improvement": rel_s,
                }
            )
        enriched_stat.sort(key=lambda x: x["delta_z"], reverse=True)
        if args.top_k > 0:
            enriched_stat = enriched_stat[: args.top_k]

        # Temporarily override baseline for pure stat plot
        b_stat_val = None
        for rr in rows:
            if str(rr.get("feature", "")) == "pfMetCorrSig":
                b_stat_val = float(rr.get("asimov_z", 0.0))
                break
        if b_stat_val is None:
            b_stat_val = min(float(rr.get("asimov_z", 0.0)) for rr in rows)

        old_baseline = baseline
        baseline = b_stat_val
        stat_out = Path(str(output_path).replace(".png", "_stat.png"))
        _make_plot(enriched_stat, "asimov_z", "pure statistical", stat_out)
        baseline = old_baseline

        print(f"[OK] syst-aware output: {output_path}")
        print(f"[OK] pure stat output : {stat_out}")
    else:
        print(f"[OK] output: {output_path}")

    print(f"[OK] input : {input_path}")
    print(f"[OK] baseline Z ({z_label}) = {baseline:.6f}")
    print("[OK] top features by Delta-Z:")
    for i, e in enumerate(enriched[:5], start=1):
        print(
            f"  {i}. {e['feature']}: Delta-Z={e['delta_z']:.3f}, "
            f"rel={e['relative_improvement']:.1f}%, effAUC={e['effective_auc']:.3f}"
        )


if __name__ == "__main__":
    main()
