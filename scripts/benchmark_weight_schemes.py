#!/usr/bin/env python3
"""Physics-oriented benchmark of DNN training-weight schemes for NLO MC
samples with negative event weights.

Compares three treatments of the signed MC event weight w_i:

  A. absolute   : w_i -> |w_i|                     (mode = absolute)
  B. local      : local weight aggregation         (mode = local_cancellation)
                  cells C_k in (njets, n_bjets, costheta_star, Recoil, JetHT);
                  w_train_i = |w_i| * alpha_cell,  alpha = sum(w)/sum(|w|)
                  fitted per process/slice on the TRAIN split only, applied
                  fixed to val/test.
  C. clip       : positive-only w_i -> max(w_i, 0) (mode = clip_negative)

Everything else — network architecture, optimizer, hyperparameters, split,
class balancing, feature list — follows the existing DNNTrainer pipeline
(configs/dnn.yaml). Only `training.negative_weight_handling.mode` changes.

Evaluation is physics-first:
  - all physical quantities (yields, sumw2, significance, limits) use the
    SIGNED MC weights;
  - AUC/ROC use each scheme's non-negative local weights;
  - sigma_MC = sqrt(sum w_i^2),  N_eff = (sum w)^2 / sum w^2;
  - sensitivity = binned Asimov Z (stat-only and syst-aware, Cowan et al.
    2011) + Asimov CLs 95% expected upper limit on the signal strength.

Usage:
    python scripts/benchmark_weight_schemes.py \
        --config configs/dnn.yaml \
        --events-dir /home/zzq/eventsel-merged \
        --outdir outputs/weight_scheme_benchmark \
        [--modes absolute local_cancellation clip_negative] \
        [--max-events-signal 200000] [--max-events-bkg 200000] \
        [--xsec-bkg data/cross-section/xsection_background_run3.json] \
        [--xsec-signal data/cross-section/xsection_signal.json] \
        [--lumi 109.82] [--threads 2] [--sig-syst 0.20] \
        [--skip-training]   # reuse trained models, only rerun evaluation
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import logging
import os
import sys
import time
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Thread hygiene: run with a small, fixed thread budget (machine is shared).
# ---------------------------------------------------------------------------
_THREADS = int(os.environ.get("BENCH_THREADS", "2"))
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = os.environ.get(_var, str(_THREADS))
os.environ["DNN_SCAN_WORKERS"] = os.environ.get("DNN_SCAN_WORKERS", "1")
os.environ["DNN_LOAD_WORKERS"] = os.environ.get("DNN_LOAD_WORKERS", "1")

import torch  # noqa: E402

torch.set_num_threads(_THREADS)

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from dnn.common import sanitize_feature_frame  # noqa: E402
from dnn.data import read_branch_as_array, read_tree_branches_as_arrays  # noqa: E402
from dnn.feature_engineering import build_feature_frame_from_tree  # noqa: E402
from dnn.model import load_checkpoint  # noqa: E402
from dnn.scaler import StandardScaler  # noqa: E402
from darkbottomline.dnn_trainer import (  # noqa: E402
    DNNTrainer,
    _asimov_significance_from_hist,
    _asimov_significance_from_hist_syst,
    fit_dnn_weight_models,
    apply_dnn_weight_models,
)

MODES = ("absolute", "local_cancellation", "clip_negative")
SIGNAL_NAME = "BBDM-2HDMa-5f_TuneCP5_13p6TeV_madgraph-pythia8-RunIII2024Summer24NanoAODv15-FSMiniv6_FSNanov15_150X_mcRun3_2024_realistic_v2-v2_EVENTSELECTION"
# Full background composition of the analysis (everything in eventsel-merged
# that is MC and not signal), except two files that uproot cannot read:
#   TTto2L2Nu              — unrecognized compression algorithm b'\x00L'
#   TbarWplustoLNu2Q       — basket-entry structure unsupported by uproot
BKG_GLOBS = ("*_EVENTSELECTION.root",)
BKG_EXCLUDE_PREFIXES = (
    "EGamma", "JetMET",          # data
    "BBDM",                      # signal (incl. the dedicated MH3=500 point)
    "TTto2L2Nu", "TbarWplustoLNu2Q",  # unreadable by uproot 5.6.x
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("benchmark_weight_schemes")


# ---------------------------------------------------------------------------
# Data loading (mirrors DNNTrainer.train_from_root)
# ---------------------------------------------------------------------------

def load_dataset(
    events_dir: str,
    config: dict,
    max_events_signal: int,
    max_events_bkg: int,
    xsec_bkg: Optional[str],
    xsec_signal: Optional[str],
    lumi: float,
):
    """Load signal + background EVENTSELECTION ROOT files into one frame.

    Weights are normalized to physical yields with the SAME convention as the
    framework CLI (darkbottomline.cli._load_one_eventsel_file):
      signal (per masspoint): w_phys = w * (lumi * xsec_masspoint * 1000) / wte
      background (per file) : w_phys = w * (lumi * xsec_file     * 1000) / wte
    with wte = weighted_total_events (full-sample weighted total) read from the
    ROOT file, so the event cap does not bias the physical yields.
    """
    import pandas as pd
    import uproot
    from darkbottomline.plotting import _find_xsec, PlotManager

    features = [str(f) for f in config["features"]]
    weight_clip = float(config.get("training", {}).get("weight_clip", 100.0))
    weight_branch = "full_event_weight"

    # --- background xsec map {dataset: pb} (same flattening as the CLI) -------
    bkg_xsec_map: Dict[str, float] = {}
    if xsec_bkg and Path(xsec_bkg).exists():
        with open(xsec_bkg) as f:
            bkg_xsec_map = PlotManager._normalize_cross_sections(json.load(f))
    # --- signal masspoint xsec map {"MH3_a_MH4_b_Mchi_1": pb} -----------------
    sig_xsec_map: Dict[str, float] = {}
    if xsec_signal and Path(xsec_signal).exists():
        with open(xsec_signal) as f:
            raw = json.load(f)
        for _model, entries in raw.items():
            if isinstance(entries, dict):
                for k, v in entries.items():
                    if not str(k).startswith("_") and isinstance(v, (int, float)):
                        sig_xsec_map[str(k)] = float(v)
    if not bkg_xsec_map and not sig_xsec_map:
        log.warning("No xsec JSONs given — yields/Z will NOT be lumi*xsec normalized!")

    file_specs: List[Tuple[str, int, int]] = []  # (path, label, cap)
    sig_path = Path(events_dir) / f"{SIGNAL_NAME}.root"
    if not sig_path.exists():
        raise FileNotFoundError(f"Signal file not found: {sig_path}")
    file_specs.append((str(sig_path), 1, max_events_signal))
    for pat in BKG_GLOBS:
        for fp in sorted(glob.glob(str(Path(events_dir) / pat))):
            stem = Path(fp).stem
            if any(stem.startswith(p) for p in BKG_EXCLUDE_PREFIXES):
                continue
            file_specs.append((fp, 0, max_events_bkg))
    if not file_specs:
        raise FileNotFoundError("No background files matched.")

    X_parts, y_parts, w_parts, sid_parts = [], [], [], []
    summary = []
    for fp, label, cap in file_specs:
        t0 = time.time()
        sid = Path(fp).stem
        with uproot.open(fp) as f:
            tree = f["Events"]
            df, _, _ = build_feature_frame_from_tree(tree, features, max_events=cap)
            df = sanitize_feature_frame(df)
            w = np.asarray(read_branch_as_array(tree, weight_branch, max_events=cap), dtype="f8")
            w = np.where(np.isfinite(w), w, 0.0)
            w = np.clip(w, -weight_clip, weight_clip)
            n = min(len(df), len(w))
            df = df.iloc[:n].reset_index(drop=True)
            w = w[:n]

            # ---- lumi*xsec normalization (mirrors _load_one_eventsel_file) -----
            wte = 0.0
            for key in ("weighted_total_events", "weighted_total_events;1"):
                if key in f:
                    try:
                        wte = float(f[key].values()[0])
                        break
                    except Exception:
                        pass
            scale_note = "raw (no xsec)"
            if label == 1 and sig_xsec_map:
                gm_cols = sorted(k for k in tree.keys() if str(k).startswith("GenModel_"))
                if gm_cols and wte > 0:
                    gm_arr = read_tree_branches_as_arrays(tree, gm_cols, max_events=n)
                    mp_scale = np.ones(n, dtype="f8")
                    for gmc in gm_cols:
                        mask = gm_arr[gmc][:n].astype(bool)
                        mp_xsec = _find_xsec(gmc[len("GenModel_"):], sig_xsec_map)
                        if mp_xsec is not None:
                            mp_scale[mask] = (lumi * mp_xsec * 1000.0) / wte
                    w = w * mp_scale
                    scale_note = "per-masspoint lumi*xsec*1000/wte"
                else:
                    log.warning("signal %s: no GenModel branches or wte — no normalization", sid)
            elif label == 0 and bkg_xsec_map:
                xsec = _find_xsec(sid, bkg_xsec_map)
                if xsec is not None and wte > 0:
                    w = w * ((lumi * xsec * 1000.0) / wte)
                    scale_note = f"lumi*xsec*1000/wte = {lumi * xsec * 1000.0 / wte:.3e}"
                else:
                    log.warning("no xsec/wte for background %s — raw weights only", sid)

        X_parts.append(df)
        y_parts.append(np.full(n, label, dtype="int8"))
        w_parts.append(w)
        sid_parts.append(np.full(n, sid, dtype=object))
        summary.append({
            "sample": sid, "label": int(label), "events": int(n),
            "negative_events": int(np.count_nonzero(w < 0)),
            "negative_fraction": float(np.mean(w < 0)),
            "sum_signed": float(np.sum(w)), "sum_abs": float(np.sum(np.abs(w))),
            "sumw2": float(np.sum(w * w)),
            "neff": float(np.sum(w) ** 2 / max(np.sum(w * w), 1e-300)),
            "weight_normalization": scale_note,
        })
        log.info(
            "loaded %-62s n=%8d  neg=%6.2f%%  sum=%.4g  [%s]  (%.1fs)",
            sid, n, 100.0 * np.mean(w < 0), np.sum(w), scale_note, time.time() - t0,
        )

    X = pd.concat(X_parts, axis=0, ignore_index=True)
    y = np.concatenate(y_parts).astype("i4")
    w = np.concatenate(w_parts)
    sample_ids = np.concatenate(sid_parts)
    log.info(
        "Dataset: N=%d  signal=%d  background=%d  bkg neg.frac=%.2f%%",
        len(X), int((y == 1).sum()), int((y == 0).sum()),
        100.0 * np.mean(w[y == 0] < 0),
    )
    return X, y, w, sample_ids, summary


# ---------------------------------------------------------------------------
# Per-mode config + training
# ---------------------------------------------------------------------------

def make_mode_config(base_config: dict, mode: str, outdir: Path) -> Path:
    cfg = copy.deepcopy(base_config)
    nwh = cfg.setdefault("training", {}).setdefault("negative_weight_handling", {})
    nwh["mode"] = mode
    cfg_path = outdir / f"config_{mode}.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return cfg_path


def run_training(base_config: dict, mode: str, X, y, w, sample_ids, outdir: Path) -> Dict:
    t0 = time.time()
    cfg_path = make_mode_config(base_config, mode, outdir)
    trainer = DNNTrainer(str(cfg_path))
    model_dir = outdir / "model"
    plot_dir = outdir / "plots"
    metrics = trainer.train_from_arrays(
        X, y, w,
        sample_ids=sample_ids,
        outdir=str(model_dir),
        plot_dir=str(plot_dir),
    )
    log.info("[%s] training finished in %.1f min", mode, (time.time() - t0) / 60.0)
    return metrics


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _asimov_z_counting(s: float, b: float, sigma_rel: float) -> float:
    from darkbottomline.dnn_trainer import _asimov_significance_from_hist_syst
    return _asimov_significance_from_hist_syst(
        np.array([max(s, 0.0)]), np.array([max(b, 0.0)]), float(sigma_rel)
    )


def _cl95_upper_limit(s: float, b: float, sigma_rel: float) -> float:
    """Asimov CLs 95% expected upper limit on mu (b-only dataset).

    Counting experiment with Gaussian background nuisance (sigma_b = sigma_rel*b).
    CLs(mu) = p_mu / (1 - p_b) with p_b = 0.5 for the Asimov b-only dataset, so
    the 95% CL limit solves  q_mu(mu) = [Phi^{-1}(0.975)]^2 = 3.8415 where
      q_mu = min_theta [-2 ln L(mu, theta)] - [-2 ln L(0, 0)]
      -2 ln L(mu, theta) = 2 (pred - b ln pred) + theta^2,  pred = mu*s + b(1+sigma_rel*theta)
    (Poisson terms + Gaussian nuisance; constants cancel in the difference.)
    """
    s = float(max(s, 0.0))
    b = float(max(b, 1e-12))
    sigma_rel = max(float(sigma_rel), 0.0)

    def f_of_theta(mu: float, theta: float) -> float:
        pred = mu * s + b * (1.0 + sigma_rel * theta)
        if pred <= 1e-300:
            return 1e300
        return 2.0 * (pred - b * np.log(pred)) + theta * theta

    def q_mu(mu: float) -> float:
        # golden-section minimization of f over theta in [-10, 10]
        lo, hi = -10.0, 10.0
        phi = (np.sqrt(5.0) - 1.0) / 2.0
        c = hi - phi * (hi - lo)
        d = lo + phi * (hi - lo)
        fc, fd = f_of_theta(mu, c), f_of_theta(mu, d)
        for _ in range(200):
            if fc < fd:
                hi, d, fd = d, c, fc
                c = hi - phi * (hi - lo)
                fc = f_of_theta(mu, c)
            else:
                lo, c, fc = c, d, fd
                d = lo + phi * (hi - lo)
                fd = f_of_theta(mu, d)
        fmin = min(fc, fd)
        f00 = f_of_theta(0.0, 0.0)
        return fmin - f00

    target = 3.841458820694124  # chi2(1, 0.05) two-sided
    mu_lo, mu_hi = 0.0, 1e3
    # find bracket
    while q_mu(mu_hi) < target:
        mu_lo, mu_hi = mu_hi, mu_hi * 10.0
        if mu_hi > 1e12:
            return float("inf")
    for _ in range(100):
        mu_mid = 0.5 * (mu_lo + mu_hi)
        if q_mu(mu_mid) > target:
            mu_hi = mu_mid
        else:
            mu_lo = mu_mid
    return float(0.5 * (mu_lo + mu_hi))


def _tv(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype="f8"); b = np.asarray(b, dtype="f8")
    return float(0.5 * np.sum(np.abs(a - b)))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_scheme(
    mode: str,
    outdir: Path,
    X,
    y,
    w_signed,
    sample_ids,
    base_config: dict,
    sig_syst: float,
) -> Dict:
    """Score the test split with the trained model and compute physics metrics."""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, roc_curve

    # --- replicate the pipeline's split exactly (same seed / stratify) -------
    tc = base_config.get("training", {})
    seed = int(tc.get("seed", 7))
    val_size = float(tc.get("val_size", 0.2))
    test_size = float(tc.get("test_size", 0.3))
    indices = np.arange(len(X), dtype="i8")
    train_idx, temp_idx = train_test_split(
        indices, test_size=val_size + test_size, random_state=seed, stratify=y,
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=test_size / (val_size + test_size),
        random_state=seed,
        stratify=y[temp_idx],
    )

    # --- load model / scaler / features --------------------------------------
    model_dir = outdir / "model"
    net, spec = load_checkpoint(str(model_dir / "dnn_model.pt"), map_location="cpu")
    net.eval()
    sd = json.loads((model_dir / "scaler.json").read_text())
    scaler = StandardScaler.from_jsonable(sd)
    features = json.loads((model_dir / "features.json").read_text())

    X_test = X.iloc[test_idx][features]
    y_test = y[test_idx]
    w_test = w_signed[test_idx]
    sid_test = sample_ids[test_idx]

    Xn = scaler.transform(X_test.to_numpy(dtype="f8")).astype("float32")
    with torch.no_grad():
        scores = torch.sigmoid(net(torch.from_numpy(Xn)).squeeze(1)).numpy()

    # --- scheme weights for metric/AUC use (fitted on train rows) ------------
    # Use THIS mode's own config (not the base config), so that e.g. absolute
    # evaluates with |w|, clip with max(w,0), local with the fitted mapping.
    mode_cfg_path = outdir / f"config_{mode}.yaml"
    mode_cfg = yaml.safe_load(mode_cfg_path.read_text()) if mode_cfg_path.exists() else base_config
    wcfg = dict(mode_cfg.get("training", {}).get("negative_weight_handling", {"mode": mode}))
    wcfg["mode"] = mode  # guard against stale configs
    weight_models, w_train_local, _ = fit_dnn_weight_models(
        X.iloc[train_idx], w_signed[train_idx], sample_ids[train_idx], wcfg,
    )
    w_test_local = apply_dnn_weight_models(
        X_test, w_test, sid_test, weight_models, wcfg,
    )

    n_bins = 50
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    sig_mask = y_test == 1
    bkg_mask = y_test == 0
    hs, _ = np.histogram(scores[sig_mask], bins=bins, weights=w_test[sig_mask])
    hb, _ = np.histogram(scores[bkg_mask], bins=bins, weights=w_test[bkg_mask])
    hs2, _ = np.histogram(scores[sig_mask], bins=bins, weights=w_test[sig_mask] ** 2)
    hb2, _ = np.histogram(scores[bkg_mask], bins=bins, weights=w_test[bkg_mask] ** 2)
    hs_l, _ = np.histogram(scores[sig_mask], bins=bins, weights=w_test_local[sig_mask])
    hb_l, _ = np.histogram(scores[bkg_mask], bins=bins, weights=w_test_local[bkg_mask])

    S_tot = float(np.sum(w_test[sig_mask]))
    B_tot = float(np.sum(w_test[bkg_mask]))
    S_local = float(np.sum(w_test_local[sig_mask]))
    B_local = float(np.sum(w_test_local[bkg_mask]))

    def _norm(h):
        s = float(np.sum(h))
        return h / s if abs(s) > 1e-12 else np.zeros_like(h)

    hs_n, hb_n = _norm(hs), _norm(hb)
    hs_l_n, hb_l_n = _norm(hs_l), _norm(hb_l)

    # shape chi2 per bin (Poisson stat from sumw2, against signed shape)
    def _chi2_per_bin(h_local_n, h_signed_n, h2, total):
        err_n = np.sqrt(np.maximum(h2, 0.0)) / max(abs(total), 1e-12)
        with np.errstate(divide="ignore", invalid="ignore"):
            return float(np.nansum(((h_local_n - h_signed_n) / np.where(err_n > 0, err_n, 1.0)) ** 2) / len(h_local_n))

    auc_test = float(roc_auc_score(y_test, scores, sample_weight=w_test_local))
    fpr, tpr, _ = roc_curve(y_test, scores, sample_weight=w_test_local)

    # --- threshold scan: counting Z + binned Z above threshold ---------------
    # Mirrors dnn/compute_dnn_significance.py: per threshold compute
    #   z_cnt   = single-bin counting Asimov Z
    #   z_bin   = binned Asimov Z over the score range [thr, 1]
    # Both use SIGNED, physically-normalized weights; the working point is
    # selected with the BINNED Z (shape information), like the repo script.
    # thresholds = quantiles of the SIGNAL score distribution → fine resolution
    # at any signal efficiency (uniform-grid thresholds are too coarse near 1
    # for a peaked classifier). Plus the inclusive point 0.0.
    thresholds = np.concatenate([
        np.array([0.0]),
        np.quantile(scores[sig_mask], np.linspace(1.0, 0.0, 2000)),
    ])
    thresholds = np.unique(np.clip(thresholds, 0.0, 1.0))
    scan = []
    for thr in thresholds:
        p = scores > thr
        m_ps = p & sig_mask
        m_pb = p & bkg_mask
        s_c = float(np.sum(w_test[m_ps]))
        b_c = float(np.sum(w_test[m_pb]))
        b2_c = float(np.sum(w_test[m_pb] ** 2))
        eff_s = s_c / S_tot if S_tot > 0 else 0.0
        eff_b = b_c / B_tot if B_tot > 0 else 0.0
        z_cnt_stat = _asimov_significance_from_hist(np.array([max(s_c, 0.0)]), np.array([max(b_c, 0.0)]))
        z_cnt_syst = _asimov_significance_from_hist_syst(np.array([max(s_c, 0.0)]), np.array([max(b_c, 0.0)]), sig_syst)
        if m_ps.sum() > 0 and m_pb.sum() > 0:
            hs_c, _ = np.histogram(scores[m_ps], bins=n_bins, range=(thr, 1.0), weights=w_test[m_ps])
            hb_c, _ = np.histogram(scores[m_pb], bins=n_bins, range=(thr, 1.0), weights=w_test[m_pb])
            z_bin_stat = _asimov_significance_from_hist(hs_c, hb_c)
            z_bin_syst = _asimov_significance_from_hist_syst(hs_c, hb_c, sig_syst)
        else:
            z_bin_stat = z_bin_syst = 0.0
        neff = (b_c ** 2) / max(b2_c, 1e-300)
        rel_mc = np.sqrt(max(b2_c, 0.0)) / max(abs(b_c), 1e-12)
        scan.append({
            "threshold": float(thr), "eff_s": eff_s, "eff_b": eff_b,
            "s": s_c, "b": b_c, "sumw2_b": b2_c,
            "z_cnt_stat": z_cnt_stat, "z_cnt_syst": z_cnt_syst,
            "z_bin_stat": z_bin_stat, "z_bin_syst": z_bin_syst,
            "neff_b": neff, "rel_mc_b": rel_mc,
        })

    def _pick(rows, key, cond):
        cands = [r for r in rows if cond(r)]
        return max(cands, key=lambda r: r[key]) if cands else None

    def _closest(rows, cond, target_key, target_val):
        cands = [r for r in rows if cond(r)]
        return min(cands, key=lambda r: abs(r[target_key] - target_val)) if cands else None

    # max binned Z with a mild minimum signal-efficiency floor so the WP does
    # not land in the degenerate high-threshold tail (a few events, noisy bins).
    best = _pick(scan, "z_bin_syst", lambda r: r["s"] > 0 and r["b"] > 0 and r["eff_s"] >= 0.01)
    # TRUE fixed-efficiency working points (closest threshold, not max-Z):
    at20 = _closest(scan, lambda r: r["s"] > 0 and r["b"] > 0, "eff_s", 0.20)
    at_bkg1pct = _closest(scan, lambda r: r["s"] > 0 and r["b"] > 0, "eff_b", 0.01)

    result = {
        "mode": mode,
        "n_test": int(len(y_test)),
        "S_tot": S_tot, "B_tot": B_tot,
        "S_local": S_local, "B_local": B_local,
        "yield_bias_signal_pct": 100.0 * (S_local - S_tot) / S_tot if S_tot else float("nan"),
        "yield_bias_bkg_pct": 100.0 * (B_local - B_tot) / B_tot if B_tot else float("nan"),
        "sumw2_bkg_incl": float(np.sum(w_test[bkg_mask] ** 2)),
        "sumw2_sig_incl": float(np.sum(w_test[sig_mask] ** 2)),
        "neff_bkg_incl": B_tot ** 2 / max(float(np.sum(w_test[bkg_mask] ** 2)), 1e-300),
        "rel_mc_bkg_incl": np.sqrt(max(float(np.sum(w_test[bkg_mask] ** 2)), 0.0)) / max(abs(B_tot), 1e-12),
        "tv_signal": _tv(hs_l_n, hs_n),
        "tv_bkg": _tv(hb_l_n, hb_n),
        "chi2nb_signal": _chi2_per_bin(hs_l_n, hs_n, hs2, S_tot),
        "chi2nb_bkg": _chi2_per_bin(hb_l_n, hb_n, hb2, B_tot),
        "auc_test_local": auc_test,
        "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "best_z_syst": {k: best[k] for k in ("threshold", "eff_s", "eff_b", "s", "b", "sumw2_b", "z_cnt_stat", "z_cnt_syst", "z_bin_stat", "z_bin_syst", "neff_b", "rel_mc_b")} if best else None,
        "at_eff_s_20": {k: at20[k] for k in ("threshold", "eff_s", "eff_b", "s", "b", "sumw2_b", "z_cnt_stat", "z_cnt_syst", "z_bin_stat", "z_bin_syst", "neff_b", "rel_mc_b")} if at20 else None,
        "at_eff_b_1pct": {k: at_bkg1pct[k] for k in ("threshold", "eff_s", "eff_b", "s", "b", "sumw2_b", "z_cnt_stat", "z_cnt_syst", "z_bin_stat", "z_bin_syst", "neff_b", "rel_mc_b")} if at_bkg1pct else None,
        "score_hist": {
            "bins_low": bins[:-1].tolist(), "bins_high": bins[1:].tolist(),
            "hs_signed": hs.tolist(), "hb_signed": hb.tolist(),
            "hs_local": hs_l.tolist(), "hb_local": hb_l.tolist(),
        },
        "scan": scan,
    }

    if best:
        s_b, b_b = best["s"], best["b"]
        result["best_z_syst"]["limit_mu95"] = _cl95_upper_limit(s_b, b_b, sig_syst)
    if at20:
        result["at_eff_s_20"]["limit_mu95"] = _cl95_upper_limit(at20["s"], at20["b"], sig_syst)

    # Persist raw test scores + weights so distribution plots can use fine bins
    np.savez(
        outdir / "test_scores.npz",
        scores=scores, y=y_test, w_signed=w_test, w_local=w_test_local,
    )
    return result


# ---------------------------------------------------------------------------
# Plots / tables
# ---------------------------------------------------------------------------

def make_plots(results: Dict[str, Dict], outdir: Path, sig_syst: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"absolute": "#1f77b4", "local_cancellation": "#2ca02c", "clip_negative": "#d62728"}
    labels = {"absolute": "|w| training", "local_cancellation": "local aggregation", "clip_negative": "positive-only"}
    outdir.mkdir(parents=True, exist_ok=True)

    # ROC
    fig, ax = plt.subplots(figsize=(7, 6))
    for mode, r in results.items():
        ax.plot(r["roc"]["fpr"], r["roc"]["tpr"], color=colors[mode], lw=2,
                label=f"{labels[mode]}  AUC={r['auc_test_local']:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("Test ROC (local weights)")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(outdir / "roc_compare.png", dpi=300); plt.close(fig)

    # signed score distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, cls, key in ((axes[0], "signal", "hs_signed"), (axes[1], "background", "hb_signed")):
        bins = np.linspace(0, 1, 51)
        for mode, r in results.items():
            h = np.asarray(r["score_hist"][key])
            h = h / max(abs(h.sum()), 1e-12)
            ax.step(0.5 * (bins[:-1] + bins[1:]), h, where="mid", color=colors[mode],
                    lw=1.8, label=labels[mode])
        ax.set_xlabel("DNN score"); ax.set_ylabel("Normalized signed yield")
        ax.set_title(cls.capitalize()); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.suptitle("Signed-weight score distributions (test)")
    fig.tight_layout(); fig.savefig(outdir / "score_distributions_signed.png", dpi=300); plt.close(fig)

    # Z vs signal efficiency
    fig, ax = plt.subplots(figsize=(8, 6))
    for mode, r in results.items():
        effs = [row["eff_s"] for row in r["scan"]]
        zs = [row["z_bin_syst"] for row in r["scan"]]
        ax.plot(effs, zs, color=colors[mode], lw=2, label=f"{labels[mode]} (binned Z_syst, σ={sig_syst*100:.0f}%)")
    ax.set_xlabel("Signal efficiency"); ax.set_ylabel("Binned Asimov Z (syst-aware)")
    ax.set_title("Sensitivity vs working point (test, signed normalized weights)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(outdir / "z_vs_sigeff.png", dpi=300); plt.close(fig)

    # N_eff and relative MC uncertainty vs cut
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for mode, r in results.items():
        effs = [row["eff_s"] for row in r["scan"]]
        neff = [row["neff_b"] for row in r["scan"]]
        rel = [row["rel_mc_b"] for row in r["scan"]]
        axes[0].plot(effs, neff, color=colors[mode], lw=2, label=labels[mode])
        axes[1].plot(effs, np.asarray(rel) * 100, color=colors[mode], lw=2, label=labels[mode])
    axes[0].set_yscale("log"); axes[0].set_xlabel("Signal efficiency"); axes[0].set_ylabel(r"$N_{\rm eff}$ (background)")
    axes[0].set_title(r"Effective background statistics  $N_{\rm eff}=(\sum w)^2/\sum w^2$")
    axes[1].set_xlabel("Signal efficiency"); axes[1].set_ylabel(r"Relative MC uncertainty  $\sigma_{\rm MC}/|\sum w|$ [%]")
    axes[1].set_title(r"Background Monte Carlo uncertainty  $\sigma_{\rm MC}=\sqrt{\sum w^2}$")
    for ax in axes:
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(outdir / "neff_sigma_vs_cut.png", dpi=300); plt.close(fig)


def make_delta_plots(results: Dict[str, Dict], outdir: Path):
    """Relative-advantage diagnostics.

    The raw Z-vs-efficiency plot hides a 3-12% scheme difference on its
    linear 0-12 scale, so these panels show the difference directly:
      (1) dZ = Z_bin(mode) - Z_bin(|w|)  vs signal efficiency
      (2) background efficiency vs signal efficiency on a LOG scale
          (background rejection is where the local scheme wins)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"absolute": "#1f77b4", "local_cancellation": "#2ca02c", "clip_negative": "#d62728"}
    labels = {"absolute": "|w| training", "local_cancellation": "local aggregation", "clip_negative": "positive-only"}

    grid = np.linspace(0.02, 0.95, 400)

    def _curve(mode: str, key: str) -> np.ndarray:
        rows = sorted(results[mode]["scan"], key=lambda r: r["eff_s"])
        e = np.array([r["eff_s"] for r in rows])
        v = np.array([r[key] for r in rows])
        return np.interp(grid, e, v)

    z_ref = _curve("absolute", "z_bin_syst")
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 10.5), sharex=True)

    # (1) delta Z vs |w| training
    ax = axes[0]
    for mode, c in (("local_cancellation", colors["local_cancellation"]), ("clip_negative", colors["clip_negative"])):
        dz = _curve(mode, "z_bin_syst") - z_ref
        ax.plot(grid, dz, color=c, lw=2, label=f"{labels[mode]} − |w| training")
    ax.axhline(0.0, color="gray", lw=0.8)
    ax.axvspan(0.02, 0.15, color="gray", alpha=0.15, label="MC-noise-dominated tail (B≲20)")
    ax.set_ylabel(r"$\Delta Z_{\mathrm{binned}}$")
    ax.set_title("Binned Asimov Z advantage relative to |w| training (20% bkg syst)")
    ax.legend(loc="lower left"); ax.grid(alpha=0.3)

    # (2) background rejection, log scale
    ax = axes[1]
    for mode, c in colors.items():
        eb = 100.0 * _curve(mode, "eff_b")
        ax.semilogy(grid, eb, color=c, lw=2, label=labels[mode])
    ax.axvspan(0.02, 0.15, color="gray", alpha=0.15)
    ax.set_xlabel("Signal efficiency")
    ax.set_ylabel(r"Background efficiency $\varepsilon_b$ [%] (log)")
    ax.set_title("Background rejection vs signal efficiency — local advantage lives in the high-purity tail")
    ax.legend(); ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(outdir / "z_delta_and_rejection.png", dpi=300)
    plt.close(fig)


def make_weighted_score_plots(results: Dict[str, Dict], outdir: Path):
    """Normalized, WEIGHTED DNN-score distributions — the key physics figure.

    Panels (all normalized, all with weights):
      (0,0) signal,   SIGNED weights, linear
      (0,1) background, SIGNED weights, log y
      (1,0) background, SIGNED weights, log y, zoomed to the high-score tail
            with each scheme's eps_sig=20% working-point cut drawn
      (1,1) background, each scheme's OWN training weights (|w| / max(w,0) /
            local-cell weights) vs the signed physical shape — this is the
            density each network actually learned from, and why |w| inflates
            cancellation regions while local tracks the signed shape.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"absolute": "#1f77b4", "local_cancellation": "#2ca02c", "clip_negative": "#d62728"}
    labels = {"absolute": "|w| training", "local_cancellation": "local aggregation", "clip_negative": "positive-only"}

    def _norm_hist(x, w, bins):
        h, _ = np.histogram(x, bins=bins, weights=w)
        s = h.sum()
        return h / s if abs(s) > 1e-12 else np.zeros_like(h)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    bins_full = np.linspace(0.0, 1.0, 101)
    bins_tail = np.linspace(0.70, 1.0, 61)
    centers_full = 0.5 * (bins_full[:-1] + bins_full[1:])
    centers_tail = 0.5 * (bins_tail[:-1] + bins_tail[1:])

    # (0,0) signal, signed, linear
    ax = axes[0, 0]
    for mode in MODES:
        z = np.load(outdir / mode / "test_scores.npz")
        m = z["y"] == 1
        ax.step(centers_full, _norm_hist(z["scores"][m], z["w_signed"][m], bins_full),
                where="mid", color=colors[mode], lw=1.8, label=labels[mode])
    ax.set_xlabel("DNN score"); ax.set_ylabel("Normalized signed yield")
    ax.set_title("Signal score distribution (signed weights)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")

    # (0,1) background, signed, log y
    ax = axes[0, 1]
    for mode in MODES:
        z = np.load(outdir / mode / "test_scores.npz")
        m = z["y"] == 0
        h = _norm_hist(z["scores"][m], z["w_signed"][m], bins_full)
        ax.step(centers_full, np.maximum(h, 1e-6), where="mid", color=colors[mode], lw=1.8, label=labels[mode])
    ax.set_yscale("log")
    ax.set_xlabel("DNN score"); ax.set_ylabel("Normalized signed yield (log)")
    ax.set_title("Background score distribution (signed weights, log)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")

    # (1,0) background tail zoom, signed, with eps_sig=20% cuts
    ax = axes[1, 0]
    for mode in MODES:
        z = np.load(outdir / mode / "test_scores.npz")
        m = z["y"] == 0
        h = _norm_hist(z["scores"][m], z["w_signed"][m], bins_tail)
        ax.step(centers_tail, np.maximum(h, 1e-6), where="mid", color=colors[mode], lw=1.8, label=labels[mode])
        thr = results[mode]["at_eff_s_20"]["threshold"]
        ax.axvline(thr, color=colors[mode], ls="--", lw=1.0, alpha=0.7,
                   label=f"{labels[mode]} cut @ε_sig=20% ({thr:.3f})")
    ax.set_yscale("log")
    ax.set_xlabel("DNN score"); ax.set_ylabel("Normalized signed yield (log)")
    ax.set_title("Background tail zoom (signed weights) — the region that decides ε_bkg")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

    # (1,1) training-weight density vs signed shape (background, tail)
    ax = axes[1, 1]
    z0 = np.load(outdir / "absolute" / "test_scores.npz")
    m0 = z0["y"] == 0
    h_ref = _norm_hist(z0["scores"][m0], z0["w_signed"][m0], bins_tail)
    ax.step(centers_tail, np.maximum(h_ref, 1e-6), where="mid", color="black", lw=2.2,
            label="signed physical shape (reference)")
    for mode in MODES:
        z = np.load(outdir / mode / "test_scores.npz")
        m = z["y"] == 0
        h = _norm_hist(z["scores"][m], z["w_local"][m], bins_tail)
        ax.step(centers_tail, np.maximum(h, 1e-6), where="mid", color=colors[mode], lw=1.6,
                ls="--", label=f"{labels[mode]} training weights")
    ax.set_yscale("log")
    ax.set_xlabel("DNN score"); ax.set_ylabel("Normalized density (log)")
    ax.set_title("Background: density each network was trained on vs signed truth")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

    fig.suptitle("Normalized weighted DNN-score distributions (test split)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outdir / "score_distributions_weighted.png", dpi=300)
    plt.close(fig)


def make_scheme_vs_signed_summary_plots(outdir: Path):
    """2x3 figure, one column per scheme. Each column:

      TOP panel (log y), three area-normalized curves of THAT scheme's model:
        orange = background with the scheme's own training weights (|w| / local / max(w,0))
        blue   = background with SIGNED weights
        green  = signal with SIGNED weights
      BOTTOM panel: per-bin ratio of the normalized scheme-weight background
        histogram to the normalized signed background histogram (dashed line = 1).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    schemes = (
        ("absolute", "|w| training model"),
        ("local_cancellation", "local aggregation model"),
        ("clip_negative", "positive-only model"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), sharex=True, sharey="row")
    bins = np.linspace(0.0, 1.0, 51)
    centers = 0.5 * (bins[:-1] + bins[1:])

    def _norm_hist(x, w):
        h, _ = np.histogram(x, bins=bins, weights=w)
        tot = h.sum()
        return h / tot if abs(tot) > 1e-12 else np.zeros_like(h)

    for j, (mode, title) in enumerate(schemes):
        z = np.load(outdir / mode / "test_scores.npz")
        s = z["scores"]
        y = z["y"].astype(bool)
        w_signed = z["w_signed"]
        w_scheme = z["w_local"]

        h_sig = _norm_hist(s[y], w_signed[y])
        h_bkg_signed = _norm_hist(s[~y], w_signed[~y])
        h_bkg_scheme = _norm_hist(s[~y], w_scheme[~y])

        ax = axes[0, j]
        ax.set_yscale("log")
        ax.step(centers, np.maximum(h_bkg_scheme, 1e-6), where="mid", color="#e76300",
                lw=2.0, label=f"{mode} weights (bkg)")
        ax.step(centers, np.maximum(h_bkg_signed, 1e-6), where="mid", color="#3f90da",
                lw=1.8, label="signed (bkg)")
        ax.step(centers, np.maximum(h_sig, 1e-6), where="mid", color="#2ca02c",
                lw=1.8, label="signed (sig)")
        ax.set_ylabel("Normalized density (log)")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")

        ax = axes[1, j]
        ratio = np.divide(
            h_bkg_scheme, h_bkg_signed,
            out=np.full_like(h_bkg_signed, np.nan),
            where=np.abs(h_bkg_signed) > 1e-9,
        )
        ax.axhline(1.0, color="black", lw=0.9, ls="--")
        ax.plot(centers, ratio, color="#e76300", marker="o", ms=3, lw=1.2,
                label=f"{mode} bkg / signed bkg")
        ax.set_ylim(0.5, 1.5)
        ax.set_xlabel("DNN score")
        ax.set_ylabel("Ratio (norm.)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Per-scheme DNN-score distributions: own weights vs signed (test split)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outdir / "score_distribution_scheme_vs_signed_summary.png", dpi=300)
    plt.close(fig)


def make_yield_closure_plots(outdir: Path):
    """Cut-based yield closure curve, three schemes side by side in ONE figure:

        R(t) = [sum_{score>t} w_own] / [sum_{score>t} w_signed]   (background)

    as a function of the DNN-score threshold t. R(t) = 1 means the scheme's
    effective non-negative weights reproduce the physical signed yield of the
    cut region. Signal is trivial (no negative weights -> exactly 1) and is
    not plotted.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    schemes = (
        ("absolute", "|w| training"),
        ("local_cancellation", "local aggregation"),
        ("clip_negative", "positive-only"),
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), sharex=True, sharey=True)
    grid = np.linspace(0.0, 0.9995, 500)
    for ax, (mode, title) in zip(axes, schemes):
        z = np.load(outdir / mode / "test_scores.npz")
        s, y, w, wl = z["scores"], z["y"].astype(bool), z["w_signed"], z["w_local"]
        b = ~y
        sb, wb, wlb = s[b], w[b], wl[b]
        # ascending scores; suffix sums = yields of events with score > t
        order = np.argsort(sb)
        xs = sb[order]
        suf_own = np.cumsum(wlb[order][::-1])[::-1]
        suf_sig = np.cumsum(wb[order][::-1])[::-1]
        n_above = np.arange(len(xs), 0, -1)
        idx = np.searchsorted(xs, grid, side="right")
        ok = idx < len(xs)
        ratio = np.where(ok, suf_own[np.minimum(idx, len(xs) - 1)] / suf_sig[np.minimum(idx, len(xs) - 1)], np.nan)
        n_ab = np.where(ok, n_above[np.minimum(idx, len(xs) - 1)], 0)
        keep = n_ab >= 50
        ax.plot(grid[keep], ratio[keep], color="#1f6fb2", lw=1.6)
        ax.axhline(1.0, color="black", ls="--", lw=0.8, alpha=0.6)
        ax.set_xlabel("DNN score threshold $t$")
        ax.set_ylim(0.9, 1.6)
        ax.set_title(title)
        ax.grid(alpha=0.3, which="both")
        ax.axhline(1.0, color="black", ls="--", lw=0.8, alpha=0.6)
        ax.set_xlabel("DNN score threshold $t$")
        ax.set_ylim(0.9, 1.6)
        ax.set_title(title)
        ax.grid(alpha=0.3, which="both")

    axes[0].set_ylabel(r"$\sum_{D>t} w_{\rm own}\;/\;\sum_{D>t} w_{\rm signed}$ (background)")
    fig.suptitle("Cut-based yield closure (background, test split)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outdir / "yield_closure_summary.png", dpi=300)
    plt.close(fig)


def make_cell_alpha_plots(outdir: Path, X, y, w, sample_ids):
    """Cell-level cancellation structure of the local scheme's cells.

    Cells C_k in (njets, n_bjets, costheta_star, Recoil, JetHT), as used by
    local_cancellation. Per cell, on background TRAIN rows:

        alpha_k = sum_{i in C_k} w_i / sum_{i in C_k} |w_i|

    alpha = 1 for all-positive cells; alpha < 1 measures the local cancellation.

      left  : |w|-weighted histogram of alpha_k  (log y)
      right : alpha_k vs the cell's mean DNN score (local model, test rows),
              i.e. how strong the cancellation is in cells near the boundary

    Scheme implication per cell (annotated):
      local  : w~ = |w| * alpha_k          -> cell yield ratio = 1 (closure)
      |w|    : w~ = |w|                    -> cell yield ratio = 1/alpha_k
      positive-only: w~ = max(w,0)         -> cell yield ratio = 1 + sum|w^-|/sum w
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from darkbottomline.dnn_trainer import _local_weight_cell_codes, _DEFAULT_LOCAL_WEIGHT_BINS

    tc = yaml.safe_load(Path("configs/dnn.yaml").read_text()).get("training", {})
    seed = int(tc.get("seed", 7))
    val_size = float(tc.get("val_size", 0.2))
    test_size = float(tc.get("test_size", 0.3))
    indices = np.arange(len(X), dtype="i8")
    train_idx, temp_idx = train_test_split(
        indices, test_size=val_size + test_size, random_state=seed, stratify=y,
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=test_size / (val_size + test_size),
        random_state=seed, stratify=y[temp_idx],
    )

    required = {"njets", "n_bjets", "costheta_star", "Recoil", "JetHT"}
    bins = _DEFAULT_LOCAL_WEIGHT_BINS

    def cell_key(frame):
        codes = _local_weight_cell_codes(frame, bins, required, missing_sentinel=-9999.0)
        return np.column_stack([codes[n] for n in sorted(required - {"has_jet2"})])

    # ---- train background cells: alpha_k = sum w / sum |w| -------------------
    Xtr = X.iloc[train_idx]
    wtr = np.asarray(w, dtype="f8")[train_idx]
    mtr = (np.asarray(y, dtype="i4")[train_idx] == 0)
    key_tr = cell_key(Xtr[mtr])
    uniq, inv = np.unique(key_tr, axis=0, return_inverse=True)
    sum_w = np.bincount(inv, weights=wtr[mtr], minlength=len(uniq))
    sum_abs = np.bincount(inv, weights=np.abs(wtr[mtr]), minlength=len(uniq))
    alpha_k = np.divide(sum_w, sum_abs, out=np.zeros_like(sum_w), where=sum_abs > 0)
    wcell = sum_abs
    use = (wcell > 1e-9) & (np.bincount(inv, minlength=len(uniq)) >= 10)
    alpha_use = alpha_k[use]
    wcell_use = wcell[use]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    h, edges, _ = ax.hist(alpha_use, bins=40, range=(0.0, 1.0), weights=wcell_use,
                          color="#1f6fb2", alpha=0.8, edgecolor="white", linewidth=0.4)
    ax.set_yscale("log")
    ax.axvline(alpha_use.sum() and float(np.sum(sum_w) / max(np.sum(sum_abs), 1e-12)),
               color="#d62728", ls="--", lw=1.4,
               label=f"global $\\alpha$ = {np.sum(sum_w)/max(np.sum(sum_abs),1e-12):.3f}")
    ax.set_xlabel(r"$\alpha_{k} = \sum_{C_k}w / \sum_{C_k}|w|$")
    ax.set_ylabel(r"$\sum_{C_k}|w|$ per bin (log)")
    ax.set_title(f"Cell cancellation structure (train background cells)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    # ---- alpha vs cell mean score (local model, test rows) -------------------
    zl = np.load(outdir / "local_cancellation" / "test_scores.npz")
    s_te, y_te = zl["scores"], zl["y"].astype(bool)
    Xte = X.iloc[test_idx]
    key_te = cell_key(Xte[~y_te])
    # match test cells to train alpha
    idx_map = {tuple(int(v) for v in u.tolist()): i for i, u in enumerate(uniq)}
    a_te = np.array([alpha_k[idx_map.get(tuple(int(v) for v in k.tolist()), -1)]
                     if tuple(int(v) for v in k.tolist()) in idx_map else np.nan for k in key_te])
    ok = np.isfinite(a_te)
    ax = axes[1]
    # per-cell mean score
    key_te2, inv_te = np.unique(key_te[ok], axis=0, return_inverse=True)
    ms = np.bincount(inv_te, weights=s_te[~y_te][ok], minlength=len(key_te2))
    cnt = np.bincount(inv_te, minlength=len(key_te2))
    ms = np.divide(ms, cnt, out=np.full_like(ms, np.nan), where=cnt > 0)
    a_mean = np.array([
        alpha_k[idx_map[tuple(int(v) for v in k.tolist())]] if tuple(int(v) for v in k.tolist()) in idx_map else np.nan
        for k in key_te2])
    m_ok = np.isfinite(a_mean) & np.isfinite(ms) & (cnt >= 30)
    m_neg = m_ok & (a_mean < 0.0)
    ax.scatter(ms[m_ok & ~m_neg], a_mean[m_ok & ~m_neg], s=8, color="#1f6fb2", alpha=0.6)
    ax.scatter(ms[m_neg], a_mean[m_neg], s=18, color="#d62728", alpha=0.9)
    ax.axhline(1.0, color="#2ca02c", ls="--", lw=1.0, label=r"$\alpha=1$ (no negatives)")
    ax.set_ylim(-0.1, 1.05)
    ax.set_xlabel("Cell mean DNN score (local model, test)")
    ax.set_ylabel(r"$\alpha_{k}$ (train-fitted)")
    ax.set_title("Cancellation vs distance to decision boundary")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle(
        "Cell-level cancellation: local keeps cell yield ratio = 1; |w| inflates by 1/alpha; "
        "positive-only adds sum|w-|/sum w per cell", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(outdir / "cell_alpha_distribution.png", dpi=300)
    plt.close(fig)


def _fmt(v, nd=3):
    if v is None:
        return "—"
    if isinstance(v, float):
        if not np.isfinite(v):
            return "∞" if v > 0 else "—"
        return f"{v:.{nd}f}"
    return str(v)


def write_comparison(results: Dict[str, Dict], outdir: Path, sig_syst: float):
    outdir.mkdir(parents=True, exist_ok=True)
    row_names = [
        "Test events N", "Test AUC (local w)",
        "Score dist. agreement TV (sig / bkg)",
        "Shape chi2/Nbins: own-w shape vs signed (sig / bkg)",
        "Yield bias (sig / bkg) [%]",
        "eps_bkg @ eps_sig=20% [%]", "eps_sig @ eps_bkg=1% [%]",
        "sum w^2 background (incl / @WP)",
        "rel. sigma_MC background (incl / @WP) [%]",
        "N_eff background (incl / @WP)",
        "Z_syst counting (max)", "Z_syst binned (max)",
        "Z_syst binned @ eps_sig=20%",
        "Exp. limit mu95 (best WP / @20%)",
    ]
    cells = {}
    for mode in MODES:
        r = results[mode]
        b = r.get("best_z_syst") or {}
        a = r.get("at_eff_s_20") or {}
        eb = r.get("at_eff_b_1pct") or {}
        cells[mode] = [
            r["n_test"], r["auc_test_local"],
            f"{r['tv_signal']:.3f} / {r['tv_bkg']:.3f}",
            f"{r['chi2nb_signal']:.2f} / {r['chi2nb_bkg']:.2f}",
            f"{r['yield_bias_signal_pct']:.2f} / {r['yield_bias_bkg_pct']:.2f}",
            100.0 * a["eff_b"] if a else float("nan"),
            eb["eff_s"] * 100 if eb else float("nan"),
            f"{r['sumw2_bkg_incl']:.1f} / {b.get('sumw2_b', float('nan')):.1f}",
            f"{100.0 * r['rel_mc_bkg_incl']:.2f} / {100.0 * b.get('rel_mc_b', float('nan')):.2f}",
            f"{r['neff_bkg_incl']:.1f} / {b.get('neff_b', float('nan')):.1f}",
            b.get("z_cnt_syst", float("nan")), b.get("z_bin_syst", float("nan")),
            a.get("z_bin_syst", float("nan")),
            f"{b.get('limit_mu95', float('nan')):.4f} / {a.get('limit_mu95', float('nan')):.4f}",
        ]

    lines = ["# Weight-scheme benchmark — physics-oriented comparison (test split, SIGNED lumi*xsec-normalized weights)",
             "",
             f"Sensitivity: Asimov Z (Cowan et al. 2011), counting AND binned in the DNN score; bkg syst {sig_syst*100:.0f}%.",
             "Limits: Asimov CLs 95% expected upper limit on mu (b-only Asimov, Gaussian bkg nuisance).",
             "WP = per-scheme threshold maximizing BINNED Z_syst (with eps_sig >= 1% floor).",
             "'@20%' = closest threshold to eps_sig = 20%; '@1%' = closest threshold to eps_bkg = 1%.",
             "",
             "> NOTE (signal normalization): the merged signal file only records one file-level",
             "> `weighted_total_events` for all 29 masspoints, so per-masspoint yields use the",
             "> merged denominator (same convention as darkbottomline.cli). This is a common",
             "> overall signal scale — it does NOT affect the relative scheme comparison, but",
             "> absolute Z / limit values should be read with this caveat.",
             "",
             "| Metric | |w| training | Local aggregation | Positive-only |",
             "|---|---|---|---|"]
    for name, vals in zip(row_names, zip(cells["absolute"], cells["local_cancellation"], cells["clip_negative"])):
        lines.append(f"| {name} | " + " | ".join(_fmt(v) for v in vals) + " |")
    lines.append("")
    lines.append("## Per-scheme working points (max binned Z_syst)")
    lines.append("")
    lines.append("| Scheme | threshold | eps_sig | eps_bkg | S | B | sqrt(sum w^2)_b | rel. sigma_MC [%] | N_eff(bkg) | Z_cnt | Z_bin_stat | Z_bin_syst | mu95 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for mode in MODES:
        b = results[mode].get("best_z_syst")
        if b:
            lines.append(
                f"| {mode} | {b['threshold']:.4f} | {100*b['eff_s']:.2f} | {100*b['eff_b']:.3f} "
                f"| {b['s']:.2f} | {b['b']:.2f} | {np.sqrt(max(b['sumw2_b'],0)):.2f} "
                f"| {100*b['rel_mc_b']:.2f} | {b['neff_b']:.1f} | {b['z_cnt_syst']:.3f} | {b['z_bin_stat']:.3f} | {b['z_bin_syst']:.3f} | {b.get('limit_mu95', float('nan')):.4f} |"
            )
    lines.append("")
    table_md = "\n".join(lines) + "\n"
    (outdir / "comparison_table.md").write_text(table_md)
    log.info("Wrote %s", outdir / "comparison_table.md")


# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Benchmark 3 DNN training-weight schemes (physics-first).")
    ap.add_argument("--config", default=str(_REPO / "configs" / "dnn.yaml"))
    ap.add_argument("--events-dir", default="/home/zzq/eventsel-merged")
    ap.add_argument("--outdir", default=str(_REPO / "outputs" / "weight_scheme_benchmark"))
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    ap.add_argument("--max-events-signal", type=int, default=200000)
    ap.add_argument("--max-events-bkg", type=int, default=200000)
    ap.add_argument("--xsec-bkg", default=str(_REPO / "data" / "cross-section" / "xsection_background_run3.json"),
                    help="Background cross-section JSON (per dataset, pb)")
    ap.add_argument("--xsec-signal", default=str(_REPO / "data" / "cross-section" / "xsection_signal.json"),
                    help="Signal per-masspoint cross-section JSON (pb)")
    ap.add_argument("--lumi", type=float, default=109.82,
                    help="Integrated luminosity in 1/fb for physical normalization (configs/2024.yaml)")
    ap.add_argument("--threads", type=int, default=_THREADS)
    ap.add_argument("--sig-syst", type=float, default=0.20)
    ap.add_argument("--epochs", type=int, default=None, help="Override epochs (None = use config)")
    ap.add_argument("--scan-epochs", type=int, default=None, help="Override per-feature scan epochs (None = use config)")
    ap.add_argument("--skip-training", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    # Two scan workers x 1 torch thread each keeps the total thread budget at 2
    # (the main process is idle while the per-feature scans run).
    os.environ["DNN_SCAN_WORKERS"] = "2"

    base_config = yaml.safe_load(Path(args.config).read_text())
    if args.epochs is not None:
        base_config.setdefault("training", {})["epochs"] = int(args.epochs)
    if args.scan_epochs is not None:
        base_config.setdefault("feature_selection", {})["single_feature_epochs"] = int(args.scan_epochs)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- data (shared across schemes) --------------------------------------
    cache = outdir / "dataset_cache.json"
    X, y, w, sample_ids, summary = load_dataset(
        args.events_dir, base_config, args.max_events_signal, args.max_events_bkg,
        args.xsec_bkg, args.xsec_signal, args.lumi,
    )
    (outdir / "dataset_summary.json").write_text(
        json.dumps({"lumi": args.lumi, "xsec_bkg": args.xsec_bkg, "xsec_signal": args.xsec_signal,
                    "samples": summary}, indent=2) + "\n")

    # ---- training (identical pipeline, only mode differs) -------------------
    train_metrics = {}
    for mode in args.modes:
        mode_dir = outdir / mode
        if args.skip_training and (mode_dir / "model" / "train_metrics.json").exists():
            log.info("[%s] skipping training (models exist)", mode)
        else:
            run_training(base_config, mode, X, y, w, sample_ids, mode_dir)
        mfile = mode_dir / "model" / "train_metrics.json"
        train_metrics[mode] = json.loads(mfile.read_text()) if mfile.exists() else {}

    # ---- physics evaluation -------------------------------------------------
    results: Dict[str, Dict] = {}
    for mode in args.modes:
        log.info("[%s] evaluating physics metrics on test split", mode)
        results[mode] = evaluate_scheme(
            mode, outdir / mode, X, y, w, sample_ids, base_config, args.sig_syst,
        )

    make_plots(results, outdir, args.sig_syst)
    make_delta_plots(results, outdir)
    make_weighted_score_plots(results, outdir)
    make_scheme_vs_signed_summary_plots(outdir)
    make_cell_alpha_plots(outdir, X, y, w, sample_ids)
    write_comparison(results, outdir, args.sig_syst)

    payload = {
        "config": str(args.config),
        "events_dir": args.events_dir,
        "sig_syst": args.sig_syst,
        "lumi": args.lumi,
        "xsec_bkg": args.xsec_bkg,
        "xsec_signal": args.xsec_signal,
        "threads": args.threads,
        "modes": args.modes,
        "dataset_summary": summary,
        "results": results,
    }
    (outdir / "benchmark_summary.json").write_text(json.dumps(payload, indent=2) + "\n")

    # console digest
    print("\n" + "=" * 78)
    print("WEIGHT-SCHEME BENCHMARK — test split, signed lumi*xsec weights, binned Z_syst(%.0f%%) + counting" % (100 * args.sig_syst))
    print("=" * 78)
    hdr = f"{'metric':42s} | " + " | ".join(f"{m:>20s}" for m in args.modes)
    print(hdr)
    print("-" * len(hdr))
    for key, label, nd in [
        ("auc_test_local", "AUC (local w)", 4),
        ("tv_bkg", "TV local-vs-signed (bkg)", 3),
        ("chi2nb_bkg", "chi2/Nbins local-vs-signed (bkg)", 2),
        ("yield_bias_bkg_pct", "yield bias bkg [%]", 2),
        ("neff_bkg_incl", "N_eff bkg (incl)", 1),
        ("rel_mc_bkg_incl", "rel sigma_MC bkg incl [%]", 2),
    ]:
        vals = [results[m].get(key, float("nan")) for m in args.modes]
        if key == "rel_mc_bkg_incl":
            vals = [100.0 * v for v in vals]
        print(f"{label:42s} | " + " | ".join(f"{v:>20.{nd}f}" for v in vals))
    print("-" * len(hdr))
    for m in args.modes:
        b = results[m].get("best_z_syst") or {}
        print(f"{m:42s} best binned Z_syst={b.get('z_bin_syst', float('nan')):.3f} (cnt={b.get('z_cnt_syst', float('nan')):.3f}) "
              f"at eps_sig={100*b.get('eff_s', 0):.1f}% eps_bkg={100*b.get('eff_b', 0):.3f}%  "
              f"N_eff_b={b.get('neff_b', float('nan')):.1f}  rel sigma_MC={100*b.get('rel_mc_b', float('nan')):.2f}%  "
              f"mu95={b.get('limit_mu95', float('nan')):.4f}")
    print("=" * 78)
    print(f"Outputs: {outdir}")
    print(f"  comparison_table.md, benchmark_summary.json, plots/")


if __name__ == "__main__":
    main()
