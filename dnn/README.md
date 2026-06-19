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
  --config configs/dnn.yaml \
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
  --config configs/dnn.yaml \
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

**Key options:**

| Flag | Default | Notes |
|------|---------|-------|
| `--region` | `preselection` | Region label inside the ROOT file |
| `--weight-branch` | `full_event_weight` | Event weight branch |
| `--max-events-per-sample` | `200000` | Cap per input file |

**Outputs** in `--outdir`:
- `dnn_model.pt` — trained model checkpoint
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
