#!/usr/bin/env python3
"""
Background composition pies of the signal region and its two control regions (xkcd style).

One compact figure per b-tag category: the two CR pies on top, the SR pie below, the process
name written next to each slice and its percentage inside.

  1b: Z(ll) CR (Zmumu+Zee),  W(lv) CR (Wmunu+Wenu),     SR
  2b: Z(ll) CR (Zmumu+Zee),  ttbar CR (Topmunu+Topenu), SR

Slices below --min-frac of a pie are merged into a grey "Other" slice (--min-frac 0 shows every
process group). Colours come from the process_groups of configs/plotting.yaml.

The layout is optimised for every figure on its own: the rotation of the SR pie and where each arrow
leaves / enters are searched so that no arrow crosses a label, leader line, title or pie
(--no-optimize uses fixed defaults).

The yields are the integrals of the per-process histograms written by the region analysis
(<base>/<year>/region_incl/root/hist_<cat>_<region>_<var>_log.root), i.e. MC normalised to the
luminosity of each year. A set with several years sums the years.

Usage:
    python scripts/plotting/plot_region_composition_pies.py \
        --base /home/ptiwari/data/e0ba653-11112026/v12112026 --years 2024 2025 2026 \
        --outdir outputs/region_pies
"""
import argparse
import contextlib
import csv
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from matplotlib.patches import Polygon, Wedge
import numpy as np
import uproot

# (process-group key in the ROOT files, label, colour). Colours are overwritten from plotting.yaml.
PROCS = [
    (("Zto2NuJets",), r"$Z(\nu\nu)$+jets", "#a96b59"),
    (("WtoLNuJets",), r"$W(\ell\nu)$+jets", "#bd1f01"),
    (("DYto2LJets",), r"$Z(\ell\ell)$+jets", "#3f90da"),
    (("ttbar",), r"$t\bar{t}$", "#e76300"),
    (("singletop",), "Single-$t$", "#ffa90e"),
    (("Diboson",), "WW/WZ/ZZ", "#832db6"),
    (("QCD",), "QCD", "#92dadd"),
    (("SMHiggs",), "SM Higgs", "#b9ac70"),
]
OTHER = ("Other", "#808080")

# category -> (fit variable, [(panel title, regions, process that defines the CR)])
CATEGORIES = {
    "1b": ("Recoil", [(r"Z$\ell\ell$(1b)", ["Zmumu", "Zee"], "DYto2LJets"),
                      (r"W$\ell\nu$(1b)", ["Wmunu", "Wenu"], "WtoLNuJets")]),
    "2b": ("costheta_star", [(r"Z$\ell\ell$(2b)", ["Zmumu", "Zee"], "DYto2LJets"),
                             (r"$t\bar{t}$(2b)", ["Topmunu", "Topenu"], "ttbar")]),
}


def load_colours(plot_config: str) -> None:
    import yaml

    groups = yaml.safe_load(open(plot_config))["process_groups"]
    for k, (names, label, _) in enumerate(PROCS):
        PROCS[k] = (names, label, groups[names[0]]["color"])


def pidx(key: str) -> int:
    return next(i for i, (names, _, _) in enumerate(PROCS) if names[0] == key)


def read_region(root_dir: Path, cat: str, region: str, var: str) -> np.ndarray:
    """Per-process integral (in PROCS order) of one region."""
    for name in (f"hist_{cat}_{region}_{var}_log.root", f"hist_{cat}_{region}_{var}.root"):
        path = root_dir / name
        if path.exists():
            break
    else:
        raise FileNotFoundError(f"no {var} histogram for {cat}_{region} under {root_dir}")
    out = np.zeros(len(PROCS))
    with uproot.open(path) as f:
        keys = {k.split(";")[0] for k in f.keys()}
        for i, (names, _, _) in enumerate(PROCS):
            for n in names:
                if n in keys:
                    out[i] += float(f[n].values().sum())
    return out


def region_yields(roots, cat, regions, var):
    return sum(read_region(r, cat, reg, var) for r in roots for reg in regions)


def slices(yields: np.ndarray, first_key: str, min_frac: float, last_key: str = None):
    """[(label, colour, fraction)]: `first_key` first, slices below min_frac merged into Other, the rest
    interleaved big/small so the small slices (and their labels) are spread around the pie."""
    frac = yields / yields.sum()
    first = pidx(first_key)
    last = pidx(last_key) if last_key else None
    rest, other = [], 0.0
    for i in range(len(PROCS)):
        if i == first or i == last or frac[i] <= 0:
            continue
        if frac[i] < min_frac:
            other += frac[i]
        else:
            rest.append((PROCS[i][1], PROCS[i][2], float(frac[i])))
    if other > 0:
        rest.append((OTHER[0], OTHER[1], float(other)))
    rest.sort(key=lambda t: -t[2])
    inter = []
    while rest:
        inter.append(rest.pop(0))
        if rest:
            inter.append(rest.pop(-1))
    tail = [(PROCS[last][1], PROCS[last][2], float(frac[last]))] if last is not None and frac[last] >= min_frac else []
    return [(PROCS[first][1], PROCS[first][2], float(frac[first]))] + inter + tail


def spread(ys, gap):
    """Push label heights apart so neighbours are at least `gap` away (keeps the order)."""
    order = sorted(range(len(ys)), key=lambda k: ys[k])
    out = list(ys)
    for a, b in zip(order, order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return out


def text_size_in(text, fs):
    """Rough (width, height) in inches of a label."""
    plain = re.sub(r"\\[A-Za-z]+", "x", text).replace("$", "").replace("{", "").replace("}", "")
    return len(plain) * 0.56 * fs / 72.0, 1.25 * fs / 72.0


def pie_layout(items, radius_in, name_fs, first_center):
    """Slice angles and name-label positions (unit-circle coordinates) of one pie."""
    out, ang = [], first_center - 180.0 * items[0][2]
    for name, colour, f in items:
        t1, t2 = ang, ang + 360.0 * f
        out.append(dict(name=name, colour=colour, f=f, t1=t1, t2=t2, mid=math.radians(0.5 * (t1 + t2))))
        ang = t2
    gap = 1.3 * name_fs / 72.0 / radius_in
    for side in (+1, -1):
        sel = [d for d in out if (math.cos(d["mid"]) >= 0) == (side > 0)]
        ys = spread([1.2 * math.sin(d["mid"]) for d in sel], gap)
        for d, y in zip(sel, ys):
            x = side * math.sqrt(max(1.2 ** 2 - min(y * y, 1.2 ** 2 - 0.09), 0.09))
            d["label"] = (x, y, side)
    return out


def draw_pie(fig, centre_in, radius_in, layout, title, title_fs, name_fs, pct_fs):
    """Draw a pie of radius `radius_in` inches centred at `centre_in` (inches) from a precomputed layout."""
    ext = 1.75
    w_in, h_in = fig.get_size_inches()
    cx, cy = centre_in
    ax = fig.add_axes([(cx - ext * radius_in) / w_in, (cy - ext * radius_in) / h_in,
                       2 * ext * radius_in / w_in, 2 * ext * radius_in / h_in])
    ax.set_xlim(-ext, ext)
    ax.set_ylim(-ext, ext)
    ax.set_aspect("equal")
    ax.axis("off")
    for d in layout:
        ax.add_patch(Wedge((0, 0), 1.0, d["t1"], d["t2"], facecolor=d["colour"], edgecolor="white", linewidth=1.8))
        r_pct = 0.62 if d["f"] >= 0.08 else 0.8
        ax.text(r_pct * math.cos(d["mid"]), r_pct * math.sin(d["mid"]), f"{100 * d['f']:.1f}%", ha="center",
                va="center", fontsize=pct_fs, zorder=5)
        x, y, side = d["label"]
        ax.plot([math.cos(d["mid"]), 0.97 * x], [math.sin(d["mid"]), 0.97 * y], lw=0.8, color="0.3", zorder=4)
        ax.text(x, y, d["name"], ha="left" if side > 0 else "right", va="center", fontsize=name_fs, zorder=5)
    ax.text(0, 1.38, title, ha="center", va="center", fontsize=title_fs)


# ----------------------------------------------------------------------------------------------
# arrows

def rim(centre, radius, deg):
    return np.array([centre[0] + radius * math.cos(math.radians(deg)), centre[1] + radius * math.sin(math.radians(deg))])


def arrow_points(kind, tail_deg, tip_deg, off, cr_pie, sr_pie):
    """Control points (inches) of one arrow: left = down then right, right = down then left."""
    tail, tip = rim(*cr_pie, tail_deg), rim(sr_pie[0], 0.965 * sr_pie[1], tip_deg)
    if kind == "left":
        xv = tail[0] - off
        return [tail, np.array([xv, tail[1] - 0.55 * (tail[1] - tip[1])]), np.array([xv + 0.5 * (tip[0] - xv), tip[1]]), tip]
    xv = tail[0] + off
    return [tail, np.array([xv, tail[1] - 0.5 * (tail[1] - tip[1])]), np.array([tip[0] + 0.5 * (xv - tip[0]), tip[1]]), tip]


def bezier(pts4, n=100):
    p0, p1, p2, p3 = (np.asarray(p, dtype=float) for p in pts4)
    t = np.linspace(0.0, 1.0, n)[:, None]
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def chevron(pts4, head_len=0.34, head_deg=38.0):
    p2, p3 = np.asarray(pts4[2], dtype=float), np.asarray(pts4[3], dtype=float)
    back = -(p3 - p2) / np.hypot(*(p3 - p2))
    segs = []
    for sign in (+1, -1):
        a = math.radians(sign * head_deg)
        rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        segs.append([p3, p3 + head_len * (rot @ back)])
    return segs


def path_arrow(fig, pts4, c0, c1, width_pt=8.0):
    """Arrow along a cubic Bezier with an open chevron head; the colour fades from c0 (tail) to c1 (head)."""
    pts = bezier(pts4, 120)
    segs = [[pts[i], pts[i + 1]] for i in range(len(pts) - 1)]
    a, b = np.array(to_rgb(c0)), np.array(to_rgb(c1))
    cols = [tuple(a + (b - a) * (i / (len(segs) - 1))) for i in range(len(segs))]
    fig.add_artist(LineCollection(segs, colors=cols, linewidths=width_pt, capstyle="round",
                                  transform=fig.dpi_scale_trans, zorder=-1))
    fig.add_artist(LineCollection(chevron(pts4), colors=[c1, c1], linewidths=width_pt, capstyle="round",
                                  transform=fig.dpi_scale_trans, zorder=10))


class Obstacles:
    """Label boxes, leader lines, titles and pie discs that an arrow must not cross (all in inches)."""

    def __init__(self, pies):
        self.rects, self.leaders, self.discs = [], [], []
        for centre, radius, layout, title, title_fs, name_fs in pies:
            cx, cy = centre
            self.discs.append((np.array(centre), radius))
            w, h = text_size_in(title, title_fs)
            self.rects.append((cx - w / 2, cy + 1.38 * radius - h / 2, cx + w / 2, cy + 1.38 * radius + h / 2))
            for d in layout:
                x, y, side = d["label"]
                w, h = text_size_in(d["name"], name_fs)
                px, py = cx + x * radius, cy + y * radius
                x0 = px if side > 0 else px - w
                self.rects.append((x0, py - h / 2, x0 + w, py + h / 2))
                a, b = np.array([cx + radius * math.cos(d["mid"]), cy + radius * math.sin(d["mid"])]), np.array([cx + 0.97 * px - 0.97 * cx + 0.03 * cx, py])
                self.leaders.append(a[None, :] + (np.array([px, py]) - a)[None, :] * np.linspace(0, 1, 8)[:, None])
        self.rects = np.array(self.rects)
        self.leaders = np.vstack(self.leaders)

    def cost(self, pts4, own_discs):
        shaft = bezier(pts4, 100)
        pts = np.vstack([shaft] + [np.linspace(s[0], s[1], 6) for s in chevron(pts4)])
        pad = 0.06
        inside = ((pts[:, None, 0] > self.rects[None, :, 0] - pad) & (pts[:, None, 0] < self.rects[None, :, 2] + pad) &
                  (pts[:, None, 1] > self.rects[None, :, 1] - pad) & (pts[:, None, 1] < self.rects[None, :, 3] + pad))
        c = 10.0 * inside.sum()
        d = np.hypot(pts[:, None, 0] - self.leaders[None, :, 0], pts[:, None, 1] - self.leaders[None, :, 1]).min(axis=1)
        c += 4.0 * (d < 0.09).sum()
        for centre, radius in self.discs:
            dist = np.hypot(pts[:, 0] - centre[0], pts[:, 1] - centre[1])
            c += 10.0 * (dist < radius - 0.1).sum()
        tail, tip = np.asarray(pts4[0]), np.asarray(pts4[3])
        if tip[1] > tail[1] - 0.3:
            c += 500.0
        return c + 0.05 * float(np.hypot(*np.diff(shaft, axis=0).T).sum())


def plan_arrows(sr_items, cr_pie_specs, sr_pie, sr_spec, left_name, right_name, optimize):
    """Pick the SR rotation and the arrow geometry with the lowest overlap cost.
    Returns (first_center, left_params, right_params) with params = (tail_deg, tip_deg, off)."""
    best = None
    centers = [165.0, 180.0, 195.0, 210.0, 225.0, 240.0] if optimize else [195.0]
    for fc in centers:
        sr_layout = pie_layout(sr_items, sr_pie[1], sr_spec["name_fs"], fc)
        slc = {d["name"]: d for d in sr_layout}
        if left_name not in slc or right_name not in slc:
            continue
        obs = Obstacles([(c, r, lay, t, tf, nf) for (c, r), lay, t, tf, nf in cr_pie_specs] +
                        [(sr_pie[0], sr_pie[1], sr_layout, "SR(xb)", sr_spec["title_fs"], sr_spec["name_fs"])])
        L, R = slc[left_name], slc[right_name]
        lefts = []
        for tip in (170.0, 180.0, 190.0) if optimize else (180.0,):
            tip_a = tip + 360.0 * round((0.5 * (L["t1"] + L["t2"]) - tip) / 360.0)
            if not (L["t1"] + 0.05 * (L["t2"] - L["t1"]) <= tip_a <= L["t2"] - 0.05 * (L["t2"] - L["t1"])):
                continue
            for tail in (215.0, 235.0, 255.0) if optimize else (235.0,):
                for off in (0.7, 1.0, 1.3) if optimize else (0.55,):
                    lefts.append(((tail, tip_a, off),
                                  obs.cost(arrow_points("left", tail, tip_a, off, cr_pie_specs[0][0], sr_pie), None)
                                  + 0.3 * abs(tip - 180.0) / 10.0))
        rights = []
        for frac in (0.15, 0.25, 0.35, 0.5, 0.65) if optimize else (0.2,):
            tip_a = R["t1"] + frac * (R["t2"] - R["t1"])
            for tail in (270.0, 285.0, 300.0, 315.0) if optimize else (285.0,):
                for off in (0.6, 0.9, 1.2) if optimize else (0.6,):
                    rights.append(((tail, tip_a, off),
                                   obs.cost(arrow_points("right", tail, tip_a, off, cr_pie_specs[1][0], sr_pie), None)
                                   + 3.0 * abs(frac - 0.35)))
        if not lefts or not rights:
            continue
        bl, br = min(lefts, key=lambda t: t[1]), min(rights, key=lambda t: t[1])
        total = bl[1] + br[1] + 0.01 * abs(fc - 195.0)
        if best is None or total < best[0]:
            best = (total, fc, bl[0], br[0])
    if best is None:
        return 195.0, None, None
    return best[1], best[2], best[3]


def artist_bbox_inches(fig, pad=0.04):
    """Bounding box (inches) of every text, wedge and arrow, so the saved figure has no blank margin."""
    from matplotlib.transforms import Bbox

    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = [a.get_window_extent(r) for ax in fig.axes for a in list(ax.texts) + list(ax.patches)]
    boxes += [a.get_window_extent(r) for a in fig.artists]
    x0, y0 = min(b.x0 for b in boxes), min(b.y0 for b in boxes)
    x1, y1 = max(b.x1 for b in boxes), max(b.y1 for b in boxes)
    return Bbox.from_extents(x0 / fig.dpi - pad, y0 / fig.dpi - pad, x1 / fig.dpi + pad, y1 / fig.dpi + pad)


def make_figure(cat, roots, label, outdir, min_frac, xkcd, optimize=True):
    var, crs = CATEGORIES[cat]
    sr = region_yields(roots, cat, ["SR"], var)
    cr_yields = [region_yields(roots, cat, regs, var) for _, regs, _ in crs]
    cr_pies = [((2.3, 6.15), 1.3), ((6.7, 6.15), 1.25)]
    sr_pie = ((4.3, 2.35), 2.05)
    sr_spec = dict(title_fs=27, name_fs=15, pct_fs=13)
    cr_items = [slices(y, c[2], min_frac) for y, c in zip(cr_yields, crs)]
    sr_items = slices(sr, "Zto2NuJets", min_frac, last_key=crs[1][2])
    cr_layouts = [pie_layout(it, r, 10, 180.0) for it, (_, r) in zip(cr_items, cr_pies)]
    left_name, right_name = PROCS[pidx("Zto2NuJets")][1], PROCS[pidx(crs[1][2])][1]
    fc, lp, rp = plan_arrows(sr_items, [(cp, lay, c[0], 19, 10) for cp, lay, c in zip(cr_pies, cr_layouts, crs)],
                             sr_pie, sr_spec, left_name, right_name, optimize)
    sr_layout = pie_layout(sr_items, sr_pie[1], sr_spec["name_fs"], fc)
    with (plt.xkcd() if xkcd else contextlib.nullcontext()):
        fig = plt.figure(figsize=(9.0, 7.6))
        for (centre, r), lay, c in zip(cr_pies, cr_layouts, crs):
            draw_pie(fig, centre, r, lay, c[0], 19, 10, 9)
        draw_pie(fig, sr_pie[0], sr_pie[1], sr_layout, f"SR({cat})", sr_spec["title_fs"], sr_spec["name_fs"], sr_spec["pct_fs"])
        if lp and rp:
            path_arrow(fig, arrow_points("left", *lp[:2], lp[2], cr_pies[0], sr_pie),
                       PROCS[pidx(crs[0][2])][2], PROCS[pidx("Zto2NuJets")][2])
            path_arrow(fig, arrow_points("right", *rp[:2], rp[2], cr_pies[1], sr_pie),
                       PROCS[pidx(crs[1][2])][2], PROCS[pidx(crs[1][2])][2])
        outdir.mkdir(parents=True, exist_ok=True)
        box = artist_bbox_inches(fig)
        fig.savefig(outdir / f"composition_{cat}_{label}.png", dpi=200, bbox_inches=box)
        fig.savefig(outdir / f"composition_{cat}_{label}.pdf", bbox_inches=box, transparent=True)
        plt.close(fig)
    fr = {}
    for name, y in [("SR", sr)] + [(c[0], yy) for c, yy in zip(crs, cr_yields)]:
        fr[name] = {PROCS[i][1]: (float(y[i] / y.sum()) if y.sum() > 0 else 0.0) for i in range(len(PROCS))}
        fr[name]["_total_yield"] = float(y.sum())
    fr["_layout"] = dict(sr_rotation_deg=fc)
    return fr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="Directory holding <year>/region_incl/root")
    ap.add_argument("--years", nargs="+", default=["2024", "2025", "2026"])
    ap.add_argument("--outdir", default="outputs/region_pies")
    ap.add_argument("--plot-config", default="configs/plotting.yaml",
                    help="plotting.yaml whose process_groups colours are used")
    ap.add_argument("--min-frac", type=float, default=0.015,
                    help="slices below this fraction of a pie are merged into 'Other' (0 = show every group)")
    ap.add_argument("--no-xkcd", action="store_true", help="plain matplotlib style instead of xkcd")
    ap.add_argument("--no-optimize", action="store_true", help="fixed layout instead of the per-plot optimisation")
    ap.add_argument("--categories", nargs="+", default=["1b", "2b"], choices=list(CATEGORIES))
    args = ap.parse_args()

    load_colours(args.plot_config)
    base = Path(args.base)
    roots = {y: base / y / "region_incl" / "root" for y in args.years}
    outdir = Path(args.outdir)
    sets = [(y, [roots[y]]) for y in args.years]
    if len(args.years) > 1:
        label = "run3" if set(args.years) == {"2024", "2025", "2026"} else "+".join(args.years)
        sets.append((label, [roots[y] for y in args.years]))
    summary = {}
    for label, rs in sets:
        for cat in args.categories:
            summary[f"{label}_{cat}"] = make_figure(cat, rs, label, outdir, args.min_frac, not args.no_xkcd, not args.no_optimize)
            print(f"wrote {outdir}/composition_{cat}_{label}.png")
    (outdir / "composition_fractions.json").write_text(json.dumps(summary, indent=1))
    with open(outdir / "composition_fractions.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["set", "region", "process", "fraction"])
        for key, regs in summary.items():
            for reg, d in regs.items():
                if reg.startswith("_"):
                    continue
                for p, v in d.items():
                    if not p.startswith("_"):
                        w.writerow([key, reg, p, f"{v:.5f}"])


if __name__ == "__main__":
    main()
