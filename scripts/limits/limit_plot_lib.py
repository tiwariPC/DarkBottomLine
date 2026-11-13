"""Shared helpers for the limit plots (Brazil bands, theory line, Run 2 curves, benchmark box). Used by plot_limits.py.
No path is hardcoded: the repository root is found from this file, the Run 2 limit table is passed in by the caller (RUN2_FILE)."""
import json, os, warnings
from pathlib import Path
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[1]
_tex = REPO / ".local/.TinyTeX/bin/x86_64-linux"
if _tex.is_dir():                    # TinyTeX: latex for usetex (axis labels and the benchmark text box)
    os.environ["PATH"] = f"{_tex}:" + os.environ["PATH"]
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
warnings.filterwarnings("ignore")

RUN2_FILE = None                     # set by the caller: text file "mA ma exp(2.5 16 50 84 97.5 %) observed" in pb (Run 2 limits at mA = 600)
XS = json.load(open(REPO / "data/cross-section/xsection_signal.json"))["2HDMa"]
Q = (0.025, 0.16, 0.5, 0.84, 0.975)

# Run 2 theory cross sections (pb) at mA = 600 GeV (13 TeV), as supplied. The Run 3 sigma_theory (13.6 TeV, xsection_signal.json)
# is 9-25% higher, so the Run 2 limit in r is not comparable with the Run 3 one unless it is expressed with the same sigma_theory.
RUN2_XS = {10: 0.9124, 50: 0.6275, 100: 0.421, 150: 0.2869, 200: 0.1886, 250: 0.1266, 300: 0.08627,
           350: 0.06118, 400: 0.04518, 450: 0.03419, 500: 0.02596}

def read_run2(to_sigma):
    """Run 2 limits at mA = 600: cross section (pb -> fb), or the same limit in units of the RUN 3 sigma_theory.
    mA, ma, exp(2.5, 16, 50, 84, 97.5 %), observed -> {(mA, ma): (expected list, observed)}"""
    out = {}
    if RUN2_FILE is None:
        raise SystemExit("Run 2 curves requested but no Run 2 limit table was given (--run2-file)")
    for line in open(RUN2_FILE):
        if line.startswith("#") or not line.strip():
            continue
        v = line.split(); mA, ma = int(v[0]), int(v[1])
        k = 1000.0 if to_sigma else 1.0 / XS[f"MH3_{mA}_MH4_{ma}_Mchi_1"]       # pb -> fb, or / sigma_theory (pb) of Run 3
        out[(mA, ma)] = ([k * float(x) for x in v[2:7]], k * float(v[7]))
    return out

hep.style.use("CMS")
CYC = hep.style.CMS["axes.prop_cycle"].by_key()["color"]
THEORY = "red"   # as before (bright)
BRAZIL_GREEN, BRAZIL_YELLOW = "#607641", "#F5BB54"   # CombineRunner.plot_limits in the repo
COL = {"srjets": CYC[0], "nobtagWT": CYC[3], "nobtagW": CYC[2]}
plt.rcParams.update({"legend.fontsize": 13, "xtick.labelsize": 15, "ytick.labelsize": 15})   # axis label size: the CMS style default

def ylabel(ax, to_sigma):
    ax.set_ylabel(r"$\mathsf{95\%\ CL\hspace{0.3cm}\sigma(pp\rightarrow b\bar{b}\chi\bar{\chi})}$(fb)" if to_sigma else r"$\mathsf{95\%\ CL\ limit\ on\ \sigma/\sigma_{th}}$", usetex=True, labelpad=0.2)

def theory(ax, xs, mA, to_sigma):
    if to_sigma:
        ax.plot(xs, [1000.0 * XS[f"MH3_{mA}_MH4_{a}_Mchi_1"] for a in xs], color=THEORY, ls="-.", lw=2.0, alpha=0.7, label=r"$\sigma_{theory}$")
    else:
        ax.axhline(1, color=THEORY, ls="-.", lw=2.0, alpha=0.7, label=r"$\sigma/\sigma_{th}=1$")

def bands(ax, d0, mA, xs, suffix=" (incl)", lw=2.0, med="incl, median"):
    ax.fill_between(xs, [d0[(mA, a)][0.025] for a in xs], [d0[(mA, a)][0.975] for a in xs], color=BRAZIL_YELLOW, alpha=0.9, lw=0, label=f"95% expected{suffix}")
    ax.fill_between(xs, [d0[(mA, a)][0.16] for a in xs], [d0[(mA, a)][0.84] for a in xs], color=BRAZIL_GREEN, alpha=0.9, lw=0, label=f"68% expected{suffix}")
    ax.plot(xs, [d0[(mA, a)][0.5] for a in xs], "k--", **({} if lw is None else {"lw": lw}), label=med)

def model_box(ax, mA):
    """Benchmark text box as in the Run 2 paper figure code: heading, then final state, mA, mchi, tan(beta), sin(theta)."""
    ax.text(0.030, 0.92, "2HDM+a", transform=ax.transAxes, fontsize=22, usetex=True)
    body = "\n".join((r"$\mathsf{b\bar{b}+p_T^{miss}}$", rf"$\mathit{{m}}_{{\mathsf{{A}}}}$ = {mA} GeV", r"$\mathit{m}_{\chi}$ = 1 GeV",
                      r"tan$\beta$ = 35", r"sin$\theta$ = 0.7"))
    ax.text(0.045, 0.875, body, transform=ax.transAxes, fontsize=19, va="top", linespacing=1.3, usetex=True)

def headroom(ax, f=6.0, below=1.0):
    lo, hi = ax.get_ylim(); ax.set_ylim(lo / below, hi * f)   # empty band at the top for the legend
