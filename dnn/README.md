# DNN — Usage Guide

All DNN functionality runs through the `darkbottomline` CLI.

## Workflow

```
analyze (event selection) → train-dnn → apply-dnn → analyze (full, with ml_score)
```

---

## Step 1 — Event Selection Only

Run event selection and dump passing events to a flat ROOT file per sample.
No region histograms filled — output is used as DNN training input.

```bash
darkbottomline analyze \
  --config configs/2024.yaml \
  --regions-config configs/regions.yaml \
  --input outputs/samples.txt \
  --output outputs/hists/out.pkl \
  --event-selection-only true \
  --event-selection-output outputs/event_selection/mysample.root
```

> Repeat for each sample (signal + all backgrounds). Collect all output ROOT files.

---

## Step 2 — Train DNN

Train classifier from the flat event-selection ROOT files.

```bash
darkbottomline train-dnn \
  --dnn-config configs/dnn.yaml \
  --input outputs/event_selection/signal.root \
          outputs/event_selection/wjets.root \
          outputs/event_selection/ttbar.root \
  --outdir outputs/dnn \
  --plot-dir outputs/dnn/plots \
  --signal-prefix bbDM
```

Or point to a text file listing all input paths:

```bash
darkbottomline train-dnn \
  --dnn-config configs/dnn.yaml \
  --input outputs/event_selection/files.txt \
  --outdir outputs/dnn \
  --signal-prefix bbDM
```

**Signal identification** (pick one):

| Flag | Example | Notes |
|------|---------|-------|
| `--signal-prefix` | `bbDM` | Files whose name starts with this are signal |
| `--signal-pattern` | `bbDM.*1000` | Regex; repeatable |
| `--label-csv` | `labels.csv` | CSV with `path,label` columns (1=signal, 0=bkg) |

Test/placeholder samples are refused by every training path whatever these flags say:
`BBDM-2HDMa_500_EVENTSELECTION.root` carries the same `GenModel_*` signal branches as the
production sample, so it is kept out of training entirely (as signal *and* as background) and stays
available as a held-out sample. See `_is_test_sample` in `dnn/make_trees.py`.

**Key options:**

| Flag | Default | Notes |
|------|---------|-------|
| `--region` | `preselection` | Region label inside the ROOT file |
| `--weight-branch` | `full_event_weight` | Event weight branch |
| `--max-events-per-sample` | `200000` | Cap per input file |
| `--event-fraction` | `0.5` | Random fraction of each file's events used; `1.0` trains on the full data set |
| `--xsection-signal-json` | — | Signal cross sections; also the mass grid for parametric models |
| `--xsection-json` | — | Background cross sections for per-file `lumi×xsec/wte` weighting |
| `--outdir` | `data/dnn` | Model artifacts |
| `--plot-dir` | `outputs/dnn` | Training plots |

**Parametric models.** With `model.parametric_input: true` (see `configs/dnn_parametric.yaml`) the
network also takes the `(MH3, MH4)` hypothesis as input, so one model covers the whole mass grid:
signal events get their true masspoint from the `GenModel_*` flags, background events a masspoint
drawn uniformly from the grid. Both mass columns are standardized on the training split (`MH4` in
log10) and the transform is saved as `mass_scaler.json` / `<model>_mass_scaler.json` next to the
checkpoint, where inference picks it up again. Scan the trained model over mass with
`apply-dnn --dnn-mass-scan all`.

To test interpolation, restrict the training grid and hold points out:
`--masspoint-pattern 'MH3_1500_'` keeps only the `MH3=1500` points, and
`--exclude-masspoints MH3_1500_MH4_100_Mchi_1,...` drops specific ones from the signal (they are
not turned into background and never get a random mass), so they stay unseen and can be scored
afterwards to see whether the network interpolates between the masses it was trained on.

**Per-mass-point and ratio-bin models.** `scripts/dnn/train_per_masspoint.py` trains one classifier per
mass point (or per bin of `MH3/MH4` with `--ratio-bins N`), and
`scripts/dnn/run_masspoint29_full.sh` / `scripts/dnn/run_ratio3_full.sh` are ready-made wrappers. See this
file and the script `--help` output for the current training modes.

**Outputs** in `--outdir`:
- `dnn_model.pt` — trained model checkpoint
- `scaler.json`, `features.json`, `train_metrics.json`, `feature_significance.json`
- `mass_scaler.json` — parametric models only
- `plots/` — AUC, feature ranking, score distributions, loss curves

---

## Step 3 — Apply DNN to Samples

Score events in ROOT files with the trained model; writes `ml_score` branch.

```bash
darkbottomline apply-dnn \
  --input outputs/event_selection/signal.root \
          outputs/event_selection/wjets.root \
  --model outputs/dnn/dnn_model.pt \
  --config configs/dnn.yaml \
  --output-dir outputs/event_selection_scored
```

**Key options:**

| Flag | Default | Notes |
|------|---------|-------|
| `--output-dir` | (in-place) | Write scored files here; omit to overwrite inputs |
| `--score-branch` | `ml_score` | Name of the new branch |
| `--config` | — | Optional; needed if feature list not saved in checkpoint |

**On real data.** The same command scores collision data: data files are excluded from training
(`_is_data`) but scored like any MC file, so `ml_score` is directly comparable between data and MC.
The score is a discriminant in (0,1) and *not* a signal probability — its absolute scale comes from
the class balancing used in training, so compare distributions rather than reading single values.
Check the data/MC score closure in a control region before looking at a signal region.

`DNN_DEVICE_RESIDENT=0` keeps the feature tensors on the host and moves each batch to the GPU
(tiny VRAM footprint); `DNN_DIAG_MAX_ROWS` caps the rows used for the feature diagnostics
(default 5M); `DNN_MEM_TRACE=1` logs RSS at every training stage. DNN region plots use a
file-streaming histogram path by default; set `DNN_PLOT_STREAMING=0` to force the legacy all-events
path for cross-checks.

---

## Step 4 — Full Analysis with DNN Score

Run full region analysis using pre-trained model. DNN scores events in-memory before filling histograms.

```bash
darkbottomline analyze \
  --config configs/2024.yaml \
  --regions-config configs/regions.yaml \
  --input outputs/event_selection/mysample.root \
  --output outputs/hists/out_dnn.pkl \
  --dnn-model outputs/dnn/dnn_model.pt \
  --dnn-config configs/dnn.yaml
```

**Or train DNN inline** (train on current sample, inject `ml_score`, fill regions — all in one pass):

```bash
darkbottomline analyze \
  --config configs/2024.yaml \
  --regions-config configs/regions.yaml \
  --input outputs/event_selection/files.txt \
  --output outputs/hists/out_dnn.pkl \
  --train-dnn configs/dnn.yaml \
  --dnn-outdir outputs/dnn \
  --signal-prefix bbDM
```

Add `--dnn-only` to stop after DNN scoring (skips region filling — useful for quick score checks):

```bash
darkbottomline analyze \
  --config configs/2024.yaml \
  --input outputs/event_selection/files.txt \
  --train-dnn configs/dnn.yaml \
  --dnn-outdir outputs/dnn \
  --dnn-only \
  --signal-prefix bbDM
```

---

## Module Layout

| File | Purpose |
|------|---------|
| `train_classifier.py` | Standalone training script (legacy) |
| `apply_classifier.py` | Standalone inference script (legacy) |
| `make_trees.py` | Convert flat ROOT → ppbbchichi-trees.root |
| `feature_engineering.py` | Feature definitions (`REQUESTED_FEATURES_25`) |
| `common.py` | Shared utilities (sanitize, missing values) |
| `data.py` | Dataset loading helpers |
| `model.py` | Network architecture |
| `scaler.py` | Feature scaling |
| `plot_feature_comparison.py` | Feature distribution plots |

Integrated modules (in `darkbottomline/`):

| File | Purpose |
|------|---------|
| `dnn_trainer.py` | `DNNTrainer`, `ParametricDNN` — training API |
| `dnn_inference.py` | `DNNInference` — scoring API |
