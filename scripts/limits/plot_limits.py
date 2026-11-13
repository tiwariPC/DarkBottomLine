#!/usr/bin/env python
"""Expected-limit plots from the Combine limit tables (limits_bbDM_<tag>.txt: mA ma q2.5 q16 q50 q84 q97.5 observed).

  years   per-year curves + the combination with its Brazil bands and a ratio-to-combined panel, in r and in sigma (fb)
  single  one year (or the combination) with its Brazil bands and the Run 2 curves (observed, theory; expected in r only)

Examples (the layout of the production output is <base>/combine/limits/limits_bbDM_<tag>.txt):
  python scripts/limits/plot_limits.py years  --limits-dir B/combine/limits --out B/plots --years 2024 2025 2026 --combined run3
  python scripts/limits/plot_limits.py years  --limits-dir B/combine/limits --out B/plots --years 2024 2025 2026 --combined run3 --run2 --run2-file run2_limits_mA600.txt
  python scripts/limits/plot_limits.py years  --limits-dir B/combine/limits --combined-dir B2/combine/limits --years 2024 2025 \\
        --combined run3 --combined-label 2024+25 --out B/plots            # combination of a subset of the years
  python scripts/limits/plot_limits.py single --limits-dir B/combine/limits --out B/plots --tag 2026 --run2-file run2_limits_mA600.txt
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import limit_plot_lib as mp
import matplotlib.pyplot as plt, mplhep as hep

LUMI = {"2024": 109.82, "2025": 110.628, "2026": 25.794}
COL = {"2024": "tab:blue", "2025": "tab:red", "2026": "tab:green"}; MK = {"2024": "o", "2025": "s", "2026": "^"}
RUN2_BLUE = "#29B6F6"


def read(limits_dir, tag, to_sigma):
    d = {}
    for ln in open(f"{limits_dir}/limits_bbDM_{tag}.txt"):
        p = ln.split()
        if len(p) < 7:
            continue
        mA, ma = int(p[0]), int(p[1]); k = 1000.0 * mp.XS[f"MH3_{mA}_MH4_{ma}_Mchi_1"] if to_sigma else 1.0
        d[(mA, ma)] = {q: float(v) * k for q, v in zip(mp.Q, p[2:7])}
    return d


def add_run2(ax, mA, to_sigma, expected):
    """Run 2 curves (mA = 600 only): observed, theory (sigma plots), expected (only if `expected`)."""
    run2 = mp.read_run2(to_sigma); x2 = sorted(k[1] for k in run2 if k[0] == mA)
    ax.plot(x2, [run2[(mA, a)][1] for a in x2], "-", color=RUN2_BLUE, lw=2.0, label="Run 2 observed (138 fb$^{-1}$)")
    if expected:
        ax.plot(x2, [run2[(mA, a)][0][2] for a in x2], "--", color=RUN2_BLUE, lw=2.0, label="Run 2 expected (138 fb$^{-1}$)")
    if to_sigma:
        x3 = sorted(mp.RUN2_XS)
        ax.plot(x3, [1000.0 * mp.RUN2_XS[a] for a in x3], color=RUN2_BLUE, ls="-.", lw=2.0, alpha=0.8, label=r"$\sigma_{theory}$ Run 2 (13 TeV)")


def plot_years(a, mA, to_sigma, name):
    comb = read(a.combined_dir or a.limits_dir, a.combined, to_sigma)
    yrs = {y: read(a.limits_dir, y, to_sigma) for y in a.years}
    lumi_comb = sum(LUMI[y] for y in a.years)
    xs = sorted(x for (m, x) in comb if m == mA)
    fig, (ax, rx) = plt.subplots(2, 1, figsize=(8, 10.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.1], "hspace": 0.05})
    lab = a.combined_label
    mp.bands(ax, comb, mA, xs, lw=None, suffix=f" ({lab})", med=f"{lab} combined expected")
    for y, d in yrs.items():
        x = [v for v in xs if (mA, v) in d]
        ax.plot(x, [d[(mA, v)][0.5] for v in x], MK[y] + "--", color=COL[y], ms=5, lw=1.5, label=f"{y} expected ({LUMI[y]:.1f} fb$^{{-1}}$)")
        rx.plot(x, [d[(mA, v)][0.5] / comb[(mA, v)][0.5] for v in x], MK[y] + "--", color=COL[y], ms=5, lw=1.5)
    mp.theory(ax, xs, mA, to_sigma)
    if a.run2 and mA == 600 and to_sigma:
        add_run2(ax, mA, True, expected=False)
    ax.set_xlim(50 if mA == 600 else 10, max(xs))
    if mA == 600:
        ax.set_xticks([50, 100, 200, 300, 400, 500])
    ax.set_yscale("log"); mp.ylabel(ax, to_sigma); mp.headroom(ax, 600.0 if to_sigma else 60.0)
    if a.run2 and mA == 600 and to_sigma:
        ax.set_ylim(ax.get_ylim()[0], a.ytop)       # tight top so the theory line starts just below the benchmark box
    mp.model_box(ax, mA)
    hep.cms.label("Work in Progress", data=True, lumi=round(lumi_comb, 1), com=13.6, ax=ax, fontsize=20)
    ax.text(0.97, 0.58, "1b + 2b", transform=ax.transAxes, ha="right", fontsize=16)
    ax.tick_params(axis="y", labelsize=24); rx.tick_params(axis="x", labelsize=20); rx.tick_params(axis="y", labelsize=20)
    ax.legend(loc="upper left", bbox_to_anchor=(0.36, 0.985), frameon=False, labelspacing=0.35, fontsize=13, ncol=1, handlelength=1.6)
    rx.axhline(1, color="gray", lw=1.2, ls="--"); rx.set_ylabel("ratio to combined", fontsize=20)
    rx.set_xlabel(r"$\mathit{m}_{\mathsf{a}}$ (GeV)", usetex=True)
    # log ratio panel starting at 0.9; the top follows the largest ratio (2026 and the highest m_a reach >4 at mA = 1500)
    rmax = max((d[(mA, v)][0.5] / comb[(mA, v)][0.5] for d in yrs.values() for v in xs if (mA, v) in d), default=3.0)
    rtop = 4.0 if rmax <= 3.6 else rmax * 1.2
    ticks = [1, 2, 3] if rtop <= 4.0 else [1, 2, 5, 10, 20]
    rx.set_yscale("log"); rx.set_ylim(0.9, rtop); rx.set_yticks([t for t in ticks if t < rtop]); rx.set_yticklabels([str(t) for t in ticks if t < rtop])
    rx.yaxis.set_minor_formatter(plt.NullFormatter())
    save(fig, a.out, name, mA)


def plot_single(a, mA, to_sigma, name):
    d0 = read(a.limits_dir, a.tag, to_sigma)
    lumi = a.lumi if a.lumi else LUMI[a.tag]
    fig, ax = plt.subplots(figsize=(8, 8)); xs = sorted(k[1] for k in d0 if k[0] == mA)
    mp.bands(ax, d0, mA, xs, suffix="", lw=None, med="Median expected"); mp.theory(ax, xs, mA, to_sigma)
    if mA == 600:
        add_run2(ax, mA, to_sigma, expected=not to_sigma)
    ax.set_xlim(50 if mA == 600 else 10, max(xs))
    if mA == 600:
        ax.set_xticks([50, 100, 200, 300, 400, 500])
    ax.set_yscale("log"); mp.ylabel(ax, to_sigma); ax.set_xlabel(r"$\mathit{m}_{\mathsf{a}}$ (GeV)", usetex=True)
    ax.tick_params(axis="x", labelsize=20); ax.tick_params(axis="y", labelsize=24)
    mp.model_box(ax, mA)
    hep.cms.label("Work in Progress", data=True, lumi=round(lumi, 1), com=13.6, ax=ax, fontsize=20)
    mp.headroom(ax, 150.0 if to_sigma else 40.0, **({"below": 4.0} if to_sigma else {}))
    ax.legend(loc="upper right", frameon=False, labelspacing=0.5, fontsize=18)
    save(fig, a.out, name, mA)


def save(fig, out, name, mA):
    Path(out).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out}/{name}_mA{mA}.png", dpi=130, bbox_inches="tight"); fig.savefig(f"{out}/{name}_mA{mA}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    for m in ("years", "single"):
        p = sub.add_parser(m); p.add_argument("--limits-dir", required=True, help="directory with limits_bbDM_<tag>.txt")
        p.add_argument("--out", required=True)
        p.add_argument("--mA", type=int, nargs="+", default=[600, 1500])
        p.add_argument("--run2-file", default=None, help="Run 2 limit table (mA ma exp2.5 exp16 exp50 exp84 exp97.5 obs, pb); needed for --run2 / single mode at mA=600")
    y = sub.choices["years"]
    y.add_argument("--years", nargs="+", default=["2024", "2025", "2026"], choices=list(LUMI))
    y.add_argument("--combined", default="run3", help="tag of the combination table (limits_bbDM_<combined>.txt)")
    y.add_argument("--combined-dir", default=None, help="directory of the combination table (default: --limits-dir)")
    y.add_argument("--combined-label", default="2024+25+26")
    y.add_argument("--run2", action="store_true", help="add Run 2 observed and Run 2 theory to the sigma plot (mA=600)")
    y.add_argument("--ytop", type=float, default=1.0e4, help="top of the y axis of the sigma + Run 2 plot (mA=600)")
    s = sub.choices["single"]; s.add_argument("--tag", required=True, help="2024, 2025, 2026 or the combination tag")
    s.add_argument("--lumi", type=float, default=None, help="luminosity label (fb-1); default from --tag if a year")
    a = ap.parse_args()
    mp.RUN2_FILE = a.run2_file
    for mA in a.mA:
        for to_sigma, base in ((False, "r"), (True, "xsec")):
            if a.mode == "years":
                plot_years(a, mA, to_sigma, f"limits_years_{base}" + ("_run2" if (a.run2 and mA == 600 and to_sigma) else ""))
            else:
                plot_single(a, mA, to_sigma, f"limits_{a.tag}_{base}_with_run2")


if __name__ == "__main__":
    main()
