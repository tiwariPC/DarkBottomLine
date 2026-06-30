#!/usr/bin/env python3
"""
Plot EGamma HLT scale factor map (pT vs eta) from correctionlib JSON.gz.
Uses mplhep CMS style.

Usage:
    python scripts/plot_egamma_hlt_sf.py \
        --json data/corrections/2022/Run3-22CDSep23-Summer22-NanoAODv12_electronHlt.json.gz \
        --path HLT_SF_Ele30_TightID \
        --lumi 8.1 --com 13.6 \
        [--year 2022Re-recoBCD] [--valtype sf] \
        [--outdir plots/hlt_sf/] [--pt-min 32] [--pt-max 300]
"""

import argparse
import gzip
import json
import math
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

mpl.use("Agg")
plt.style.use(hep.style.CMS)


# ---------------------------------------------------------------------------
# Correctionlib JSON walker
# ---------------------------------------------------------------------------

def _load_json(path: str) -> dict:
    if path.endswith(".gz"):
        with gzip.open(path) as f:
            return json.load(f)
    with open(path) as f:
        return json.load(f)


def _find_correction(data: dict, name: str) -> dict:
    for c in data["corrections"]:
        if c["name"] == name:
            return c
    raise KeyError(
        f"Correction '{name}' not found. "
        f"Available: {[c['name'] for c in data['corrections']]}"
    )


def _walk_category(node: dict, key: str) -> dict:
    for entry in node["content"]:
        if entry["key"] == key:
            return entry["value"]
    raise KeyError(
        f"Key '{key}' not found. "
        f"Available: {[e['key'] for e in node['content']]}"
    )


def _parse_edges(raw: list) -> np.ndarray:
    out = []
    for v in raw:
        if isinstance(v, str):
            v = v.strip().lower()
            if v in ("-inf", "-infinity"):
                out.append(-math.inf)
            elif v in ("inf", "infinity", "+inf"):
                out.append(math.inf)
            else:
                out.append(float(v))
        else:
            out.append(float(v))
    return np.array(out, dtype=float)


def extract_2d_sf(
    json_path: str,
    corr_name: str,
    year: str,
    valtype: str,
    path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (eta_edges, pt_edges, sf_grid) where sf_grid[i_eta, i_pt]."""
    data = _load_json(json_path)
    corr = _find_correction(data, corr_name)
    node = corr["data"]
    node = _walk_category(node, year)
    node = _walk_category(node, valtype)
    node = _walk_category(node, path)

    assert node["nodetype"] == "multibinning", \
        f"Expected multibinning, got {node['nodetype']}"
    assert node["inputs"] == ["eta", "pt"], \
        f"Unexpected axis order: {node['inputs']}"

    eta_edges = _parse_edges(node["edges"][0])
    pt_edges  = _parse_edges(node["edges"][1])
    values    = np.array(node["content"], dtype=float)

    n_eta = len(eta_edges) - 1
    n_pt  = len(pt_edges) - 1
    assert len(values) == n_eta * n_pt, \
        f"Value count mismatch: {len(values)} vs {n_eta}×{n_pt}"

    # correctionlib multibinning: last axis (pt) innermost → (n_eta, n_pt)
    return eta_edges, pt_edges, values.reshape(n_eta, n_pt)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _finite(edges: np.ndarray, lo: float, hi: float) -> np.ndarray:
    out = edges.copy()
    if not math.isfinite(out[0]):
        out[0] = lo
    if not math.isfinite(out[-1]):
        out[-1] = hi
    return out


def plot_sf_map(
    eta_edges: np.ndarray,
    pt_edges: np.ndarray,
    sf_grid: np.ndarray,
    lumi: float | str | None = None,
    com: float | str | None = 13.6,
    cms_label: str = "Internal",
    pt_min: float = 30.0,
    pt_max: float = 500.0,
    outpath: str = "egamma_hlt_sf.png",
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
):
    # Crop pT range
    pt_mask = (pt_edges[1:] > pt_min) & (pt_edges[:-1] < pt_max)
    pt_idx = np.where(pt_mask)[0]
    if len(pt_idx) == 0:
        raise ValueError(f"No pT bins in [{pt_min}, {pt_max}] GeV")
    pt_edges_c = pt_edges[pt_idx[0]: pt_idx[-1] + 2]
    sf = sf_grid[:, pt_idx]

    n_eta, n_pt = sf.shape

    # Axis display edges (replace ±inf)
    eta_ax = _finite(eta_edges, -2.6, 2.6)
    # pT axis: extend slightly beyond outermost finite edges
    pt_lo = pt_edges_c[1]  if not math.isfinite(pt_edges_c[0])  else pt_edges_c[0]
    pt_hi = pt_edges_c[-2] if not math.isfinite(pt_edges_c[-1]) else pt_edges_c[-1]
    pt_ax = _finite(pt_edges_c, pt_lo * 0.65, pt_hi * 1.5)

    # Color scale: use on-plateau bins (SF > 0.5) to avoid sub-threshold bias
    plateau = sf[sf > 0.5]
    if vmin is None:
        vmin = max(0.85, float(np.nanmin(plateau)))
    if vmax is None:
        vmax = min(1.15, float(np.nanmax(plateau)))

    fig, ax = plt.subplots(figsize=(9, 9))

    mesh = ax.pcolormesh(
        eta_ax, pt_ax, sf.T,
        cmap=cmap, vmin=vmin, vmax=vmax,
        linewidth=0, edgecolors="none",
    )

    # Colorbar: same height as axes, ticks on right
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.046, aspect=25)
    cbar.ax.tick_params(labelsize=14, which="both")
    # cbar.set_label("Scale Factor", fontsize=16, labelpad=10)

    # Annotate each bin — black text throughout (matches reference)
    eta_ax_c = _finite(eta_edges, -2.5, 2.5)
    pt_ax_c  = _finite(pt_edges_c, pt_lo * 0.85, pt_hi * 1.2)
    # Pick text color by normalized bin value: dark bins (low in viridis) get white
    cmap_obj = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
    for i in range(n_eta):
        eta_ctr = (eta_ax_c[i] + eta_ax_c[i + 1]) / 2
        for j in range(n_pt):
            val = sf[i, j]
            if not math.isfinite(val) or val < 0.1:
                continue
            pt_ctr = math.sqrt(pt_ax_c[j] * pt_ax_c[j + 1])
            # Luminance of the bin colour: dark → white text, light → black text
            norm_val = (val - vmin) / max(vmax - vmin, 1e-9)
            r, g, b, _ = cmap_obj(float(np.clip(norm_val, 0, 1)))
            lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
            color = "white" if lum < 0.45 else "black"
            ax.text(
                eta_ctr, pt_ctr, f"{val:.3f}",
                ha="center", va="center",
                fontsize=10, color=color,
                fontweight="normal",
            )

    ax.set_yscale("log")
    ax.set_xlabel(r"Electron $|\eta|$", fontsize=16)
    ax.set_ylabel(r"Electron $p_\mathrm{T}$ [GeV]", fontsize=16)
    ax.set_xlim(eta_ax[0], eta_ax[-1])
    ax.set_ylim(pt_ax[0], pt_ax[-1])

    # x ticks: clean half-integer labels independent of bin edges
    x_ticks = [-2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{v:g}" for v in x_ticks], fontsize=15)

    # y: explicit ticks at every bin edge, scalar labels
    finite_pt = _finite(pt_edges_c, pt_lo * 0.65, pt_hi * 1.5)
    ax.set_yticks(finite_pt)
    ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.tick_params(axis="y", which="major", labelsize=13)

    # mplhep CMS label (data=True suppresses "Simulation" prefix)
    hep.cms.label(
        text=cms_label,
        data=True,
        lumi=lumi,
        com=com,
        ax=ax,
        fontsize=18,
    )

    fig.tight_layout()
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    pdf_path = os.path.splitext(outpath)[0] + ".pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")
    print(f"Saved: {pdf_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot EGamma HLT SF map (pT vs eta) from correctionlib JSON"
    )
    parser.add_argument("--json",      required=True,
                        help="Path to .json or .json.gz correctionlib file")
    parser.add_argument("--corr",      default="Electron-HLT-SF",
                        help="Correction name in the JSON")
    parser.add_argument("--year",      default=None,
                        help="Year key (auto-detected if only one present)")
    parser.add_argument("--valtype",   default="sf",
                        help="sf | sfup | sfdown")
    parser.add_argument("--path",      default=None,
                        help="HLT path key. Lists available if omitted.")
    parser.add_argument("--all-paths", action="store_true",
                        help="Plot all HLT paths in the file")
    parser.add_argument("--outdir",    default="plots/hlt_sf")
    parser.add_argument("--pt-min",    type=float, default=30.0,
                        help="Min pT to display [GeV]")
    parser.add_argument("--pt-max",    type=float, default=500.0,
                        help="Max pT to display [GeV]")
    parser.add_argument("--lumi",      type=float, default=None,
                        help="Luminosity value in fb⁻¹ (e.g. 8.1)")
    parser.add_argument("--com",       type=float, default=13.6,
                        help="Centre-of-mass energy in TeV (default 13.6)")
    parser.add_argument("--label",     default="Internal",
                        help="CMS label text: Internal | Preliminary | Work in Progress")
    parser.add_argument("--vmin",      type=float, default=None)
    parser.add_argument("--vmax",      type=float, default=None)
    args = parser.parse_args()

    data = _load_json(args.json)
    corr_node = _find_correction(data, args.corr)

    # Auto-detect year
    year_node = corr_node["data"]
    available_years = [e["key"] for e in year_node["content"]]
    if args.year is None:
        if len(available_years) == 1:
            args.year = available_years[0]
            print(f"Auto-selected year: {args.year}")
        else:
            print(f"Available years: {available_years}")
            parser.error("--year required (multiple options)")

    vt_node   = _walk_category(year_node, args.year)
    p_node    = _walk_category(vt_node, args.valtype)
    all_paths = [e["key"] for e in p_node["content"]]

    if args.path is None and not args.all_paths:
        print("Available HLT paths:")
        for p in all_paths:
            print(f"  {p}")
        parser.error("Provide --path <name> or --all-paths")

    paths_to_plot = all_paths if args.all_paths else [args.path]

    for hlt_path in paths_to_plot:
        eta_edges, pt_edges, sf_grid = extract_2d_sf(
            args.json, args.corr, args.year, args.valtype, hlt_path
        )
        fname = f"{hlt_path}_{args.valtype}.png".replace("/", "_")
        plot_sf_map(
            eta_edges, pt_edges, sf_grid,
            lumi=args.lumi,
            com=args.com,
            cms_label=args.label,
            pt_min=args.pt_min,
            pt_max=args.pt_max,
            outpath=os.path.join(args.outdir, fname),
            vmin=args.vmin,
            vmax=args.vmax,
        )


if __name__ == "__main__":
    main()
