"""Pre-fit / post-fit summary plots from a FitDiagnostics file.

One plot per category (1b, 2b): every region block (SR, then the CRs) is drawn as its four
one-bin channels side by side, the stacked processes, the post-fit total with its uncertainty,
the pre-fit total, the data, and a ratio pad (data - fit) / fit for the post-fit and the pre-fit
prediction. Ported from the Run 2 crSummary_postfit_*.py scripts; the channel and process
names are read from the file, so the merged e/mu CRs (cat_1b_Wlnu_bin1, ...) and the separate
ones (cat_1b_Wmunu_bin1, cat_1b_Wenu_bin1, ...) both work, and there is no per-year sum
(one era per file).

Needs the post-fit folders of the FitDiagnostics file (--saveShapes --saveWithUncertainties,
with the POI range wide enough for the toy sampling, see combine.yaml PullsCRonly).
"""
import logging
import math
from array import array
import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional

# leg_entry / hist_color of the Run 2 script, keyed by the datacard process names
PROCESS_LABEL = {
    "diboson": "WW/WZ/ZZ", "zjets": "Z(#nu#bar{#nu})+jets", "dyjets": "Z(ll)+jets", "wjets": "W(l#nu)+jets",
    "ttbar": "t#bar{t}", "singletop": "Single t", "smhiggs": "SM Higgs", "qcd": "QCD",
}
REGION_ORDER = ["SR", "Zll", "Zmumu", "Zee", "Wlnu", "Wmunu", "Wenu", "Topl", "Topmunu", "Topenu"]
REGION_LABEL = {
    "SR": "SR", "Zll": "Z(#font[32]{ll}) CR", "Zmumu": "Z(#mu#mu) CR", "Zee": "Z(ee) CR",
    "Wlnu": "W(#font[32]{l}#nu) CR", "Wmunu": "W(#mu#nu) CR", "Wenu": "W(e#nu) CR",
    "Topl": "t#bar{t}(#font[32]{l}#nu) CR", "Topmunu": "t#bar{t}(#mu#nu) CR", "Topenu": "t#bar{t}(e#nu) CR",
}
X_TITLE = {"1b": "p_{T}^{miss} (GeV)", "2b": "|cos#theta*|"}
_CHANNEL = re.compile(r"^(?:.*_)?cat_(?P<cat>[^_]+)_(?P<region>.+)_bin(?P<bin>\d+)$")


def _region_channels(root_file, fit_dir: str, category: str) -> "OrderedDict[str, List[str]]":
    """{region: [channel dirs ordered by bin]} for one category, regions in REGION_ORDER."""
    d = root_file.Get(fit_dir)
    if not d:
        raise KeyError(f"no '{fit_dir}' folder in {root_file.GetName()}")
    found: Dict[str, Dict[int, str]] = {}
    for key in d.GetListOfKeys():
        m = _CHANNEL.match(key.GetName())
        if m and m.group("cat") == category:
            found.setdefault(m.group("region"), {})[int(m.group("bin"))] = key.GetName()
    ordered = sorted(found, key=lambda r: REGION_ORDER.index(r) if r in REGION_ORDER else len(REGION_ORDER))
    return OrderedDict((r, [found[r][b] for b in sorted(found[r])]) for r in ordered)


def _first_bin(root_file, path: str):
    """(content, error) of bin 1 of a one-bin histogram; None when the object is missing or empty."""
    h = root_file.Get(path)
    if not h or not h.InheritsFrom("TH1"):
        return None
    return h.GetBinContent(1), max(h.GetBinErrorUp(1), h.GetBinErrorLow(1))


def make_prefit_postfit_summary(fit_file: str, fit_dir: str, category: str, out_dir: str, tag: str,
                                 lumi_text: str = "", edges: Optional[List[float]] = None,
                                 unblind_sr: bool = False, signal_label: str = "signal",
                                 ratio_range: float = 0.5, colors: Optional[Dict[str, str]] = None,
                                 ticks: Optional[List[float]] = None) -> Optional[Path]:
    """Draw one category's summary; returns the pdf path, or None when the fit file has no
    post-fit shapes for it (a warning says so). colors: {datacard process: '#rrggbb'}, the process
    colours of the data/MC plots (configs/plotting.yaml process_groups)."""
    import ROOT
    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptTitle(0)
    # ROOT line widths are integers; a width-1 line is 1 x ImageScaling pixels thick. A smaller canvas with a larger
    # scaling gives the same image size and makes width 1 equal to a 1.5 line of the 3x-scaled plots.
    ROOT.gStyle.SetFrameLineWidth(1)
    ROOT.gStyle.SetImageScaling(4.5)

    fit_dir = fit_dir if fit_dir.startswith("shapes_") else f"shapes_fit_{fit_dir}"
    f = ROOT.TFile.Open(fit_file)
    if not f or f.IsZombie():
        raise FileNotFoundError(fit_file)
    regions = _region_channels(f, fit_dir, category)
    if not regions or not any(_first_bin(f, f"{fit_dir}/{c}/total_background") for r in regions.values() for c in r):
        logging.warning(f"{fit_file}: no post-fit shapes in {fit_dir} for {category} — skipped (check "
                        f"--saveWithUncertainties and the POI range in the FitDiagnostics command)")
        f.Close()
        return None

    channels = [c for r in regions.values() for c in r]
    nbins = len(channels)
    nblocks = len(regions)
    per_block = len(next(iter(regions.values())))
    if not edges or len(edges) != per_block + 1:      # no usable edges: uniform bins, no tick labels
        edges = list(range(per_block + 1))
    span = edges[-1] - edges[0]
    xedges = [b * span + edges[j] - edges[0] for b in range(nblocks) for j in range(per_block)] + [nblocks * span]
    xmax = float(nblocks * span)
    # axis ticks: a major tick at every labelled value (the labels repeat in every block, so the spacing is the
    # distance between two labels), 5 minor ticks in between; ROOT would otherwise put one tick per bin, at equal spacing
    step = (ticks[1] - ticks[0]) if ticks and len(ticks) > 1 else span / per_block
    xdiv = int(round(xmax / step)) + 100 * 5
    procs = [k.GetName() for k in f.Get(f"shapes_prefit/{channels[0]}").GetListOfKeys()
             if k.GetName() not in ("data", "signal", "total", "total_signal", "total_background", "total_covar")]
    hist = {p: ROOT.TH1F(f"h_{p}_{category}", "", nbins, array("d", xedges))
            for p in procs + ["prefit", "postfit", "data", "signal"]}
    for i, ch in enumerate(channels, start=1):
        is_sr = "_SR_" in ch
        for p in procs:
            v = _first_bin(f, f"{fit_dir}/{ch}/{p}")
            hist[p].SetBinContent(i, v[0] if v else 0.0)
        for name, path in (("prefit", f"shapes_prefit/{ch}/total_background"),
                           ("postfit", f"{fit_dir}/{ch}/total_background")):
            v = _first_bin(f, path)
            if v:
                hist[name].SetBinContent(i, v[0])
                hist[name].SetBinError(i, v[1])
        g = f.Get(f"{fit_dir}/{ch}/data")
        if g and g.GetN() and (unblind_sr or not is_sr):
            hist["data"].SetBinContent(i, g.GetY()[0])
            hist["data"].SetBinError(i, max(g.GetErrorYhigh(0), g.GetErrorYlow(0)))
        if is_sr and fit_dir.endswith("_s"):
            v = _first_bin(f, f"{fit_dir}/{ch}/total_signal")
            if v:
                hist["signal"].SetBinContent(i, v[0])
                hist["signal"].SetBinError(i, v[1])

    colors = {p: ROOT.TColor.GetColor(hexcol) for p, hexcol in (colors or {}).items()}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"{tag}_{category}"
    lx, rx = 0.088, 0.035
    ndc_x = lambda x: lx + x / xmax * (1 - lx - rx)

    c = ROOT.TCanvas(f"c_{tag}_{category}", "", 866, 467)
    up = ROOT.TPad("up", "", 0, 0.30, 1, 1)
    up.SetBottomMargin(0.0)
    up.SetTopMargin(0.115)
    up.SetLeftMargin(lx)
    up.SetRightMargin(rx)
    up.SetLogy(1)
    up.SetTicky(1)
    up.SetTickx(1)
    up.Draw()
    up.cd()
    stack = ROOT.THStack("hs", "")
    order = sorted([p for p in procs if hist[p].Integral() > 0], key=lambda x: hist[x].Integral())
    for p in order:                                   # smallest at the bottom, as in the data/MC plots
        hist[p].SetFillColor(colors.get(p, ROOT.kGray))
        hist[p].SetFillStyle(1001)
        hist[p].SetLineWidth(0)
        stack.Add(hist[p])
    top = max(hist["postfit"].GetMaximum(), hist["prefit"].GetMaximum())
    ymax = 10 ** math.ceil(math.log10(max(top, 10.0)) + 3.0)
    stack.SetMaximum(ymax)
    stack.SetMinimum(3.0)
    stack.Draw("HIST")
    stack.GetYaxis().SetTitle("Events/bin" if category == "1b" else f"Events/{span / per_block:g}")
    stack.GetYaxis().SetTitleSize(0.075)
    stack.GetYaxis().SetTitleOffset(0.47)
    stack.GetYaxis().SetLabelSize(0.06)
    stack.GetYaxis().SetTickLength(0.015)
    stack.GetYaxis().SetNdivisions(509, False)       # few minor ticks per decade (the default is dense)
    stack.GetXaxis().SetNdivisions(xdiv, False)
    stack.GetXaxis().SetLabelSize(0)
    stack.GetXaxis().SetTickLength(0.03)
    outline = hist["postfit"].Clone("outline")        # black line on top of the stack, as in the data/MC plots
    outline.SetLineColor(ROOT.kBlack)
    outline.SetLineWidth(1)
    outline.SetFillStyle(0)
    for i in range(1, nbins + 1):
        outline.SetBinError(i, 0.0)
    outline.Draw("same HIST")
    data = hist["data"]
    data.SetMarkerStyle(20)
    data.SetMarkerSize(0.8)
    data.SetLineColor(ROOT.kBlack)
    data.SetLineWidth(1)
    data.Draw("P e1 same")
    blinded = [i for i in range(1, nbins + 1) if data.GetBinContent(i) <= 0 and hist["postfit"].GetBinContent(i) > 0]
    pseudo = None
    if blinded:     # blinded SR: the post-fit background sum stands in for the data (black points, post-fit uncertainty)
        pseudo = hist["postfit"].Clone("pseudo_data")
        for i in range(1, nbins + 1):
            if i not in blinded:
                pseudo.SetBinContent(i, 0.0)
                pseudo.SetBinError(i, 0.0)
        pseudo.SetMarkerStyle(20)
        pseudo.SetMarkerSize(0.8)
        pseudo.SetLineColor(ROOT.kBlack)
        pseudo.SetLineWidth(1)
        pseudo.Draw("P e1 same")
    sig = hist["signal"]
    has_signal = sig.Integral() > 0
    if has_signal:
        sig.SetLineColor(ROOT.TColor.GetColor("#00e5ff"))
        sig.SetLineWidth(3)
        # ROOT's image output draws dashed styles one pixel thin whatever the width, so the dashed step line
        # is built from solid segments (dash/gap in x, a fixed log step for the vertical joins)
        dash = xmax / 260.0
        segments = []
        prev_y = None
        for i in range(1, nbins + 1):
            y = sig.GetBinContent(i)
            if y <= 0:
                prev_y = None
                continue
            lo, hi = sig.GetBinLowEdge(i), sig.GetBinLowEdge(i) + sig.GetBinWidth(i)
            if prev_y:
                n = max(2, int(abs(math.log10(y / prev_y)) / 0.05))
                for k in range(0, n, 2):
                    y0 = prev_y * (y / prev_y) ** (k / n)
                    y1 = prev_y * (y / prev_y) ** (min(k + 1, n) / n)
                    segments.append((lo, y0, lo, y1))
            x = lo
            while x < hi:
                segments.append((x, y, min(x + dash, hi), y))
                x += 2 * dash
            prev_y = y
        sig_lines = []
        for x0, y0, x1, y1 in segments:
            ln = ROOT.TLine(x0, y0, x1, y1)
            ln.SetLineColor(sig.GetLineColor())
            ln.SetLineWidth(3)
            ln.Draw()
            sig_lines.append(ln)
    legend = ROOT.TLegend(0.40, 0.66, 0.985, 0.875)
    legend.SetNColumns(3)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextSize(0.05)
    for p in reversed(order):
        legend.AddEntry(hist[p], PROCESS_LABEL.get(p, p), "f")
    legend.AddEntry(data, "Data (bkg sum for SR)" if pseudo is not None else "Data", "PEL")
    legend.Draw("same")
    up.Update()

    c.cd()
    low = ROOT.TPad("low", "", 0, 0, 1, 0.30)
    low.SetLeftMargin(lx)
    low.SetRightMargin(rx)
    low.SetTopMargin(0.0)
    low.SetBottomMargin(0.42)
    low.SetTicky(1)
    low.SetTickx(1)
    low.Draw()
    low.cd()
    frame = ROOT.TH1F(f"ratio_frame_{category}", "", nbins, array("d", xedges))
    frame.SetMinimum(0.5)
    frame.SetMaximum(1.5)
    frame.GetYaxis().SetNdivisions(504)
    frame.GetYaxis().SetLabelSize(0)           # own labels below: no 0.5 / 1.5 labels at the pad edges
    frame.GetYaxis().SetTitleSize(0.13)
    frame.GetYaxis().SetTitleOffset(0.36)
    frame.GetYaxis().SetTitle("Data/Pred.")
    frame.GetYaxis().CenterTitle()
    frame.GetXaxis().SetNdivisions(xdiv, False)
    frame.GetXaxis().SetLabelSize(0)
    frame.GetXaxis().SetTickLength(0.06)
    frame.Draw("axis")
    # uncertainty band of the post-fit prediction, around 1
    band = ROOT.TGraphAsymmErrors()
    for i in range(1, nbins + 1):
        n = hist["postfit"].GetBinContent(i)
        if n <= 1e-6:
            continue
        k = band.GetN()
        half = hist["postfit"].GetBinWidth(i) / 2
        e = hist["postfit"].GetBinError(i) / n
        band.SetPoint(k, hist["postfit"].GetBinCenter(i), 1.0)
        band.SetPointError(k, half, half, e, e)
    band.SetFillColor(ROOT.kGray + 1)
    band.SetFillStyle(3144)
    band.Draw("2 same")
    ratios = {}
    for name, ref, style, color, size in (("post", hist["postfit"], 20, ROOT.kBlack, 0.8),
                                           ("pre", hist["prefit"], 29, ROOT.kRed + 2, 1.2)):
        g = ROOT.TGraphAsymmErrors()
        for i in range(1, nbins + 1):
            n = ref.GetBinContent(i)
            if n <= 1e-6:
                continue
            if i in blinded:        # pseudo-data = post-fit background sum: post-fit ratio is 1, pre-fit shows the fit's shift
                num, err = hist["postfit"].GetBinContent(i), hist["postfit"].GetBinError(i)
            elif data.GetBinContent(i) > 0:
                num, err = data.GetBinContent(i), data.GetBinError(i)
            else:
                continue
            k = g.GetN()
            g.SetPoint(k, ref.GetBinCenter(i), num / n)
            w = ref.GetBinWidth(i) / 2 if name == "pre" else 0.0
            g.SetPointError(k, w, w, err / n, err / n)
        g.SetMarkerStyle(style)
        g.SetMarkerColor(color)
        g.SetMarkerSize(size)
        g.SetLineColor(color)
        g.SetLineWidth(1)
        ratios[name] = g
    ratios["pre"].Draw("P same")
    ratios["post"].Draw("P same")
    leg_r = ROOT.TLegend(0.40, 0.76, 0.985, 0.97)
    leg_r.SetNColumns(3)
    leg_r.SetBorderSize(0)
    leg_r.SetFillStyle(0)
    leg_r.SetTextSize(0.14)
    leg_r.AddEntry(band, "unc.(stat+syst)", "f")
    leg_r.AddEntry(ratios["post"], "Postfit", "PEL")       # PEL: marker, vertical error bar and the horizontal line
    leg_r.AddEntry(ratios["pre"], "Prefit", "PEL")
    leg_r.Draw("same")

    # y labels of the ratio pad: only the inner ones, so they do not run into the upper pad or the x labels
    ty = ROOT.TLatex()
    ty.SetNDC()
    ty.SetTextAlign(32)
    ty.SetTextFont(42)
    ty.SetTextSize(0.13)
    for v in (0.75, 1.0, 1.25):
        ty.DrawLatex(lx - 0.008, 0.42 + (v - 0.5) * 0.58, f"{v:g}")

    # tick labels under each block, one axis title per block
    tx = ROOT.TLatex()
    tx.SetNDC()
    tx.SetTextAlign(21)
    tx.SetTextFont(42)
    tx.SetTextSize(0.125)
    for b in range(nblocks):
        for t in ticks if ticks is not None else edges:
            first, last = (t == edges[0]), (t == edges[-1])
            if (first and b > 0) or (last and b < nblocks - 1):
                continue
            tx.DrawLatex(ndc_x(b * span + t - edges[0]), 0.31, f"{t:g}")
    tx.SetTextSize(0.17)
    if category == "1b":
        for b, region in enumerate(regions):
            tx.DrawLatex(ndc_x((b + 0.5) * span), 0.13, "p_{T}^{miss} (GeV)" if region == "SR" else "U (GeV)")
    else:
        tx.SetTextAlign(31)
        tx.DrawLatex(1 - rx, 0.13, "cos#Theta*")

    c.cd()
    # region separators: one red dash-dot line per block edge, from the top of the plot down to the x-axis labels.
    # Built from solid segments, because ROOT's image output draws dashed line styles one pixel thin.
    # top of the lines: just above the tallest bin (data or prediction), in the log scale of the upper pad
    tallest = max(max(hist["postfit"].GetBinContent(i), data.GetBinContent(i)) for i in range(1, nbins + 1))
    frac = (math.log10(tallest * 2.0) - math.log10(3.0)) / (math.log10(ymax) - math.log10(3.0))
    top_y = 0.30 + 0.70 * (1 - 0.115) * min(max(frac, 0.0), 1.0)
    bottom_y = 0.032                                   # down to the bottom of the axis titles
    pattern = [(0.030, True), (0.011, False), (0.006, True), (0.011, False)]      # dash, gap, dot, gap (NDC height)
    sep_lines = []
    for b in range(1, nblocks):
        x = ndc_x(b * span)
        y, k = top_y, 0
        while y > bottom_y:
            length, drawn = pattern[k % 4]
            if drawn:
                ln = ROOT.TLine(x, y, x, max(y - length, bottom_y))
                ln.SetNDC()
                ln.SetLineColor(ROOT.kRed + 1)
                ln.SetLineWidth(2)
                ln.Draw()
                sep_lines.append(ln)
            y -= length
            k += 1
    t = ROOT.TLatex()
    t.SetNDC()
    t.SetTextFont(62)
    t.SetTextSize(0.065)
    t.DrawLatex(0.095, 0.925, "CMS")
    t.SetTextFont(52)
    t.SetTextSize(0.05)
    t.DrawLatex(0.168, 0.927, "Work in progress")
    t.SetTextFont(62)
    t.SetTextSize(0.055)
    t.DrawLatex(0.13, 0.80, f"{category} category")
    if has_signal:      # signal entry drawn by hand (same dash pattern as the curve), as ROOT draws dashed legend lines thin
        dash_ndc = dash / xmax * (1 - lx - rx)
        x0, y = 0.105, 0.752
        x1 = x0 + 9 * dash_ndc
        for k in range(5):
            ln = ROOT.TLine(x0 + 2 * k * dash_ndc, y, x0 + (2 * k + 1) * dash_ndc, y)
            ln.SetNDC()
            ln.SetLineColor(sig.GetLineColor())
            ln.SetLineWidth(3)
            ln.Draw()
            sig_lines.append(ln)
        t.SetTextFont(42)
        t.SetTextSize(0.04)
        t.SetTextAlign(12)
        t.DrawLatex(x1 + 0.012, y, signal_label)
        t.SetTextFont(62)
        t.SetTextAlign(11)
    t.SetTextAlign(21)
    t.SetTextSize(0.047)
    for b, region in enumerate(regions):
        label = "SR" if region == "SR" else REGION_LABEL.get(region, region).replace(" CR", "+jets CR") \
            if region.startswith(("Z", "W")) else REGION_LABEL.get(region, region)
        t.DrawLatex(ndc_x((b + 0.5) * span), 0.685, label)
    t.SetTextAlign(31)
    t.SetTextFont(42)
    t.SetTextSize(0.05)
    t.DrawLatex(0.985, 0.93, lumi_text + " (13.6 TeV)" if lumi_text else "13.6 TeV")

    c.SaveAs(f"{stem}.pdf")
    c.SaveAs(f"{stem}.png")
    fo = ROOT.TFile(f"{stem}.root", "RECREATE")
    for h in list(hist.values()):
        h.Write()
    fo.Close()
    f.Close()
    return Path(f"{stem}.pdf")
