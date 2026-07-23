# DarkBottomLine Framework

CMS Run 3 bbMET analysis framework. Coffea-based NanoAOD processor producing event-selected flat ROOT files and stacked region plots.

## Overview

DarkBottomLine processes NanoAOD datasets through event selection, region analysis (1b/2b categories with control regions), and plot generation. Config-driven per year (2022–2024). Supports DNN scoring integration.

## Features

- Config-driven object selection, triggers, corrections (no hardcoded cuts)
- Multi-region analysis: 1b/2b categories, SR + W/Top/Z CRs
- Per-region trigger routing: MET/SingleMuon trigger for SR and muon CRs; EGamma trigger for electron CRs
- Three analysis modes: `event-selection`, `region-analysis`, `full`
- Stacked MC+data plots: PDF, PNG, ROOT TH1, TXT yield tables, TEX yield tables (5 formats)
- Cutflow plots per region with event-selection and region-cut steps on log scale
- SR blinding by default (bkg-sum as pseudo-data); `--show-data` to unblind
- DNN scoring integration (train or apply)
- Executors: iterative, futures, Dask
- Multi-core parallelization (`multiprocessing`) for DNN training/plotting and region-plot generation — see [Parallelization](#parallelization)

---

## Installation

### Local

```bash
git clone https://github.com/tiwariPC/DarkBottomLine.git
cd DarkBottomLine
source local_setup.sh      # conda env + pip install -e .
```

### Lxplus

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_109/x86_64-el9-gcc15-opt/setup.sh
git clone https://github.com/tiwariPC/DarkBottomLine.git
cd DarkBottomLine
python3 check_requirements.py --install
./lxplus_setup.sh
# future sessions:
source start.sh
```

### Condor

```bash
cd condorJobs
voms-proxy-init --voms cms --valid 192:00
cp /tmp/x509up_u$(id -u) /afs/cern.ch/user/u/username/private/
condor_submit submit.sub
```

---

## Single Command: `analyze`

All pipeline modes run via one command with `--mode`:

| `--mode`                                            | Input                      | What runs                                     |
| --------------------------------------------------- | -------------------------- | --------------------------------------------- |
| `event-selection`                                   | NanoAOD ROOT               | Event selection → EVENTSELECTION.root         |
| `event-selection` + `--make-event-selection-plots`  | EVENTSELECTION.root folder | Stacked pre-region plots only                 |
| `region-analysis` + `--make-region-plots`           | EVENTSELECTION.root folder | Region cuts → stacked region plots            |
| `full` + `--make-region-plots`                      | NanoAOD ROOT               | Event selection + region analysis + plots     |

---

## Mode 1: `event-selection` — NanoAOD → EVENTSELECTION.root

Runs trigger, filters, object building, preselection. Saves flat ROOT per sample.

```bash
# MC sample
darkbottomline analyze \
    --mode event-selection \
    --config configs/2022.yaml \
    --input /path/to/TTto2L2Nu_NanoAOD.root \
    --event-selection-output outputs/eventsel/TTto2L2Nu_EVENTSELECTION.root

# Collision data (golden JSON lumi mask applied, MC weights skipped)
darkbottomline analyze \
    --mode event-selection \
    --config configs/2022.yaml \
    --input /path/to/JetMET-Run2022C_NanoAOD.root \
    --event-selection-output outputs/eventsel/JetMET-Run2022C_EVENTSELECTION.root \
    --data
```

**Output:** `outputs/eventsel/SAMPLENAME_EVENTSELECTION.root`

Loop over all samples:

```bash
EVTSEL=outputs/eventsel
for f in /path/to/NanoAODv12_2022/*.root; do
    base=$(basename "$f"); dataset="${base%___*}"
    [[ "$dataset" == JetMET-Run* || "$dataset" == EGamma-Run* ]] && flag="--data" || flag=""
    darkbottomline analyze \
        --mode event-selection \
        --config configs/2022.yaml \
        --input "$f" \
        --event-selection-output ${EVTSEL}/${dataset}_EVENTSELECTION.root \
        $flag 2>&1 | tee -a ${EVTSEL}/run.log
done
```

---

## Mode 1 + plots: `event-selection` + `--make-event-selection-plots`

Reads EVENTSELECTION.root folder, produces stacked plots **without** applying region cuts.

```bash
darkbottomline analyze \
    --mode event-selection \
    --config configs/2022.yaml \
    --input outputs/eventsel/ \
    --output-dir outputs/ \
    --xsection-json data/cross-section/xsection_background.json \
    --plot-config configs/plotting.yaml \
    --make-event-selection-plots
```

**Output:**

```text
outputs/plots/{version}/png/event_selection/hist_event_selection_{var}.png
outputs/plots/{version}/png/event_selection/hist_event_selection_{var}_log.png
outputs/plots/{version}/pdf/event_selection/hist_event_selection_{var}.pdf
outputs/plots/{version}/text/event_selection/hist_event_selection_{var}.txt
outputs/plots/{version}/root/hist_event_selection_{var}.root
```

---

## Mode 2: `region-analysis` — EVENTSELECTION.root → region plots

Applies region cuts in-memory, produces stacked plots per region.

```bash
# All regions (SR blinded by default — bkg-sum shown as pseudo-data)
darkbottomline analyze \
    --mode region-analysis \
    --config configs/2022.yaml \
    --regions-config configs/regions.yaml \
    --input outputs/eventsel/ \
    --output-dir outputs/ \
    --xsection-json data/cross-section/xsection_background.json \
    --plot-config configs/plotting.yaml \
    --make-region-plots

# Single region
darkbottomline analyze \
    --mode region-analysis \
    --config configs/2022.yaml \
    --regions-config configs/regions.yaml \
    --input outputs/eventsel/ \
    --output-dir outputs/ \
    --xsection-json data/cross-section/xsection_background.json \
    --plot-config configs/plotting.yaml \
    --make-region-plots \
    --plot-regions "1b:SR"

# Multiple specific regions
darkbottomline analyze \
    --mode region-analysis \
    --config configs/2022.yaml \
    --regions-config configs/regions.yaml \
    --input outputs/eventsel/ \
    --output-dir outputs/ \
    --xsection-json data/cross-section/xsection_background.json \
    --plot-config configs/plotting.yaml \
    --make-region-plots \
    --plot-regions "1b:SR" "2b:SR" "1b:CR_Wmunu"

# Unblind SR (show real data in SR)
darkbottomline analyze \
    --mode region-analysis \
    --config configs/2022.yaml \
    --regions-config configs/regions.yaml \
    --input outputs/eventsel/ \
    --output-dir outputs/ \
    --xsection-json data/cross-section/xsection_background.json \
    --plot-config configs/plotting.yaml \
    --make-region-plots \
    --show-data

# Both region plots + event-selection plots
darkbottomline analyze \
    --mode region-analysis \
    --config configs/2022.yaml \
    --regions-config configs/regions.yaml \
    --input outputs/eventsel/ \
    --output-dir outputs/ \
    --xsection-json data/cross-section/xsection_background.json \
    --plot-config configs/plotting.yaml \
    --make-region-plots \
    --make-event-selection-plots

# Specific variables only
darkbottomline analyze \
    --mode region-analysis \
    --config configs/2022.yaml \
    --regions-config configs/regions.yaml \
    --input outputs/eventsel/ \
    --output-dir outputs/ \
    --xsection-json data/cross-section/xsection_background.json \
    --plot-config configs/plotting.yaml \
    --make-region-plots \
    --plot-regions "1b:SR" \
    --plot-variables Recoil PFMET_pt Jet1Pt n_bjets

# With signal overlay (×100 scale for shape visibility) + systematic plots
darkbottomline analyze \
    --mode region-analysis \
    --config configs/2022.yaml \
    --regions-config configs/regions.yaml \
    --input outputs/eventsel/ \
    --xsection-json data/cross-section/xsection_background.json \
    --xsection-signal-json data/cross-section/xsection_signal.json \
    --plot-config configs/plotting.yaml \
    --make-region-plots \
    --signal-scale 100 \
    --make-syst-plots
```

**New flags:**

| Flag | Description |
|------|-------------|
| `--xsection-signal-json JSON` | Signal cross sections: `{model: {masspoint: xsec_pb}}` e.g. `data/cross-section/xsection_signal.json` |
| `--signal-scale N` | Multiply all signal histograms by N for shape visibility (shown as `×N` in legend, default: 1) |
| `--make-syst-plots` | Also produce central+UP+DOWN comparison plots per uncertainty (same output dirs as normal plots) |

**Output:**

```text
outputs/plots/{version}/
  png/region_analysis/1b_SR/
      hist_1b_SR_{var}.png              ← stacked bkg + signal overlay
      cutflow_1b_SR.png                 ← event-sel steps + region cuts, log scale
      1b_SR_{var}_weight_pdf.png        ← syst plot: central/UP/DOWN (--make-syst-plots)
      1b_SR_{var}_weight_scale.png
      1b_SR_{var}_JES.png
      ...
  pdf/region_analysis/1b_SR/            ← same, PDF format
  text/region_analysis/{region}/
      hist_{region}_{var}.txt           ← yield tables (booktabs)
      cutflow_{region}.txt              ← cut-by-cut yield table
  root/
      hist_{cat}_{region}_{var}.root    ← all bkg TH1s + TotalBkg + data_obs + sig_* per masspoint
      hist_{cat}_{region}_{var}_weight_pdfUP.root    ← syst variations
      ...
      cutflow_{region}.root             ← TH1D, bin = cut step
```

**Cutflow plot:** blue bars = event-selection steps (OR Trigger → Noise filters → Recoil → ...) + orange bars = region cuts (MET trigger → Nbjets → Njets → Nleptons → ...), lumi×xsec weighted, log y-axis.

---

## Mode 3: `full` — NanoAOD → event-selection + region plots

Full pipeline in one command.

```bash
darkbottomline analyze \
    --mode full \
    --config configs/2022.yaml \
    --regions-config configs/regions.yaml \
    --input /path/to/sample_NanoAOD.root \
    --event-selection-output outputs/eventsel/sample_EVENTSELECTION.root \
    --output-dir outputs/ \
    --xsection-json data/cross-section/xsection_background.json \
    --plot-config configs/plotting.yaml \
    --make-region-plots \
    [--make-event-selection-plots] \
    [--show-data] \
    [--output outputs/hists/sample.pkl]   # optional: save PKL for make-plots
```

`--output` is optional in `full` mode when `--make-region-plots` is set.

---

## SR Blinding

| Flag           | SR behavior                                    | CR behavior       |
| -------------- | ---------------------------------------------- | ----------------- |
| (default)      | Bkg-sum shown as pseudo-data points, ratio = 1 | Real data points  |
| `--show-data`  | Real data points                               | Real data points  |

---

## Region Names

| Category | Regions |
| -------- | ------- |
| `1b` | `SR`, `CR_Wmunu`, `CR_Wenu`, `CR_Zmumu`, `CR_Zee` |
| `2b` | `SR`, `CR_Topmunu`, `CR_Topenu`, `CR_Zmumu`, `CR_Zee` |

1b has no Top CR. 2b has no W CR.

---

## Process Groups & Data Routing

Configured in `configs/plotting.yaml`:

```yaml
process_groups:
  WtoLNuJets:
    type: background
    patterns: ["WtoLNu-2Jets_PTLNu-40to100_2J", ...]
  MET_Data:
    type: data
    regions: ["SR", "munu", "mumu"]   # JetMET → SR + muon CRs
    patterns: ["JetMET-Run", "MET-Run"]
  EGamma_Data:
    type: data
    regions: ["enu", "Zee"]           # EGamma → electron CRs
    patterns: ["EGamma-Run"]
```

Cross sections: `data/cross-section/xsection_background.json` (keyed by filename stem).

### Signal group

```yaml
process_groups:
  Signal_2HDMa:
    type: signal
    patterns:
      - "BBDM-2HDMa"
```

PNG/PDF: first 3 masspoints as dashed lines. ROOT: all masspoints as individual TH1s (`sig_MH3_..._MH4_...`).

---

## Plotting yaml — full reference

All plot appearance controlled by `configs/plotting.yaml`. No hardcoded values in Python.

| Key | Default | Description |
|-----|---------|-------------|
| `cms_label` | `"Work in progress"` | CMS label text (e.g. `"Preliminary"`, `"Simulation"`) |
| `com_energy` | `13.6` | Centre-of-mass energy in TeV |
| `dpi` | `200` | Output PNG/PDF resolution |
| `figsize_ratio` | `[12, 12]` | Figure size with ratio panel |
| `figsize_no_ratio` | `[12, 10]` | Figure size without ratio panel |
| `figsize_cutflow` | `[12, 12]` | Cutflow figure size |
| `subplots_top/bottom/left/right` | `0.92/0.09/0.14/0.95` | Subplot margins |
| `subplots_hspace` | `0.08` | Gap between main and ratio panels |
| `main_height` / `ratio_height` | `3.0` / `1.0` | Height ratio of main:ratio panels |
| `fontsize_axis` | `22` | Axis label font size |
| `fontsize_legend` | `20` | Legend font size |
| `fontsize_xtick_cutflow` | `16` | Cutflow x-tick font size |
| `ratio_ylim` | `[0.0, 2.0]` | Data/MC ratio panel y range |
| `data_markersize` | `5.5` | Data point marker size |
| `data_elinewidth` | `1.2` | Data error bar line width |
| `data_color` | `"black"` | Data point color |
| `signal_linewidth` | `2.0` | Signal overlay line width |
| `signal_colors` | `["#000000", "#e31a1c", ...]` | Colors for signal masspoint lines |
| `legend_ncol` | `2` | Legend column count |
| `uncertainty_facecolor` | `"#bbbbbb"` | MC stat band fill color |
| `uncertainty_edgecolor` | `"#666666"` | MC stat band edge color |
| `uncertainty_hatch` | `"////"` | MC stat band hatch pattern |
| `uncertainty_alpha` | `0.5` | MC stat band opacity |
| `uncertainty_label` | `"Stat. unc."` | MC stat band legend label |
| `n_bins_default` | `40` | Auto-bin count when no `variable_bins` entry |
| `variable_bins` | — | Per-variable bin edges (linspace or explicit edges) |
| `no_log_scale_vars` | — | Variables plotted linear (not log) |
| `common_variables` | — | Variables plotted in every region |
| `region_variables` | — | Per-region additional variables |
| `event_selection_variables` | — | Variables for pre-region plots |
| `systematic_variables` | — | Variables for which syst ROOT files are produced |
| `met_suffix_patterns` | — | Branch suffixes treated as MET-type for binning |

### Bin configuration

```yaml
variable_bins:
  Recoil:   {edges: [250, 300, 400, 550, 1000]}   # explicit edges
  MET_pt:   {edges: [100, 200, 300, 400, 500, 750, 1000]}
  Jet1Pt:   {low: 30., high: 800., n: 24}          # linspace
```

Bins always applied as-is — x-axis range uses full config edges regardless of data content.

---

## Signal Samples (2HDMa multi-masspoint)

Signal NanoAOD files contain events from multiple mass-plane grid points in a single file. Each event belongs to exactly one grid point, identified by a boolean branch:

```text
GenModel_MH3_600_MH4_10_Mchi_1      # 1 if event is MH3=600, MH4=10, Mchi=1; else 0
GenModel_MH3_600_MH4_50_Mchi_1
...
GenModel_MH3_1500_MH4_1450_Mchi_1   # 29 branches total for 2HDMa Type-II
```

**Exactly one flag is 1 per event.** The framework preserves all GenModel branches through event selection automatically — no special flag needed.

### Running event selection on signal

```bash
darkbottomline analyze \
    --mode event-selection \
    --config configs/2024.yaml \
    --input /path/to/BBDM-2HDMa-fullsim_NanoAOD.root \
    --event-selection-output outputs/eventsel/BBDM-2HDMa_EVENTSELECTION.root
```

The output EVENTSELECTION.root Events TTree contains all 29 `GenModel_*` branches as `int8` alongside all standard physics branches. Use them downstream to select events for a specific masspoint:

```python
import uproot, numpy as np
f = uproot.open("BBDM-2HDMa_EVENTSELECTION.root")
ev = f["Events"].arrays(library="np")
mask = ev["GenModel_MH3_600_MH4_200_Mchi_1"] == 1
recoil_600_200 = ev["Recoil"][mask]
```

### Cross sections

Signal cross sections go in `data/cross-section/xsection_signal.json`. Nested by model name (extensible for future signal models):

```json
{
  "2HDMa": {
    "_comment": "tanb=35, sint=0.7, mchi=1 GeV",
    "MH3_600_MH4_200_Mchi_1":  0.20796,
    "MH3_1500_MH4_200_Mchi_1": 0.20593
  }
}
```

Keys match the `GenModel_*` branch suffix (strip `GenModel_` prefix). All models in the file are merged into the cross-section lookup — add future models as new top-level keys.

---

## DNN Integration

The framework supports training a binary classifier (optionally **parametric** — conditioned on the
signal mass hypothesis) and injecting per-event DNN scores (`ml_score`) into the analysis pipeline.

### Workflow

```
  outputs/eventsel/*_EVENTSELECTION.root  (signal + background + data)
       │
       ▼
  darkbottomline train-dnn  ──→  data/dnn/dnn_model.pt, scaler.json, features.json
       │
       ▼
  darkbottomline analyze --mode region-analysis --apply-dnn  ──→  region plots with ml_score
```

### Step 0 — Event selection (run once per sample)

`scripts/run_eventsel_all.sh` loops `darkbottomline analyze --mode event-selection` over every
`.root` file in a directory, auto-detecting data files (via filename pattern) and applying `--data`
to them:

```bash
scripts/run_eventsel_all.sh [INPUT_DIR] [OUTPUT_DIR] [CONFIG] [--dry-run]

# e.g.
scripts/run_eventsel_all.sh /path/to/NanoAODv15_2024 outputs/eventsel configs/2024.yaml
```

Defaults: `INPUT_DIR=../TestingSamples/NanoAODv15_2024`, `OUTPUT_DIR=outputs/eventsel`,
`CONFIG=configs/2024.yaml`. `--dry-run` prints the planned per-file commands without running them.

### Step 1 — Train

```bash
darkbottomline train-dnn \
  --dnn-config configs/dnn.yaml \
  --input outputs/eventsel \
  --weight-branch full_event_weight \
  --outdir data/dnn \
  --plot-dir outputs/dnn \
  --xsection-signal-json data/cross-section/xsection_signal.json \
  --xsection-json data/cross-section/xsection_background_run3.json
```

Signal/background split, feature list, and model architecture all come from `configs/dnn.yaml`
(`features:`, `model:`, `training:`) — no separate `--signal-prefix`/`--signal-pattern` needed when
`--input` is an `outputs/eventsel/` folder, since signal vs. background is resolved from
`configs/plotting.yaml`'s `process_groups` (`type: signal` / `type: background` entries).

Writes `data/dnn/dnn_model.pt` (+ `_scaler.json`, `features.json`, `train_metrics.json`,
`feature_significance.json`) and diagnostic plots (ROC, loss/AUC curves, score distributions,
feature correlation/significance) to `outputs/dnn/`.

**Parametric training** — set `model.parametric_input: true` in `configs/dnn.yaml` to train a single
network conditioned on the signal's `(MH3, MH4)` mass grid instead of one mass-averaged classifier:
signal events get their true masspoint (parsed from the `GenModel_MH3_*_MH4_*_Mchi_*` flag set on
that event); background events get a masspoint sampled uniformly at random from the same 29-point
grid. Same `train-dnn` command as above — no extra flags needed, the grid is derived automatically
from `--xsection-signal-json`. `parametric_input: false` (default) trains the plain non-parametric
classifier described above.

### Step 2 — Apply, in region-analysis

```bash
darkbottomline analyze \
    --mode region-analysis \
    --config configs/2024.yaml \
    --regions-config configs/regions.yaml \
    --input outputs/eventsel \
    --output-dir outputs/region_plots \
    --xsection-json data/cross-section/xsection_background_run3.json \
    --plot-config configs/plotting.yaml --make-region-plots \
    --apply-dnn --dnn-model data/dnn/dnn_model.pt --dnn-config configs/dnn.yaml \
    --xsection-signal-json data/cross-section/xsection_signal.json --signal-scale 10
```

`--apply-dnn` scores every event and adds `ml_score` as a plotted variable in every region
(`common_variables` in `configs/plotting.yaml`) — same output structure as any other variable
(PNG/PDF/ROOT/TXT per region).

**Parametric models** — whether `data/dnn/dnn_model.pt` is parametric was decided at training time by
`configs/dnn.yaml`'s `model.parametric_input` (`true`/`false`), baked into the checkpoint (`spec.parametric`)
and read from there at apply time — there's no separate switch here, and no way to apply a
non-parametric checkpoint as if it were parametric or vice versa. `--dnn-config` is only used to
resolve the feature list if `features.json` is missing next to the checkpoint; it does **not**
re-decide parametric-ness.

By default `ml_score` is scored once at the checkpoint's benchmark masspoint (`mass_grid[0]`), same
single-branch output as a non-parametric model — this is what you get with `parametric_input: false`,
and it's also the default with `parametric_input: true` if `--dnn-mass-scan` is omitted. Add
`--dnn-mass-scan` to evaluate a parametric checkpoint at other masspoints instead — produces one
`ml_score_mh3_<a>_mh4_<b>` branch (and one full set of region plots) per point scanned:

```bash
# Two specific masspoints
darkbottomline analyze \
    --mode region-analysis \
    --config configs/2024.yaml \
    --regions-config configs/regions.yaml \
    --input outputs/eventsel \
    --output-dir outputs/region_plots \
    --xsection-json data/cross-section/xsection_background_run3.json \
    --plot-config configs/plotting.yaml --make-region-plots \
    --apply-dnn --dnn-model data/dnn/dnn_model.pt --dnn-config configs/dnn.yaml \
    --xsection-signal-json data/cross-section/xsection_signal.json --signal-scale 10 \
    --dnn-mass-scan MH3_600_MH4_300_Mchi_1,MH3_1500_MH4_1000_Mchi_1

# Every grid point (29 sets of region plots)
    --dnn-mass-scan all
```

Ignored (no-op) for a non-parametric model.

### Standalone scoring (`apply-dnn`)

Score existing `EVENTSELECTION.root` files directly, writing `ml_score` (or `ml_score_mh3_*` per
scanned point) back to the ROOT file in-place or to `--output-dir`, without running region analysis:

```bash
darkbottomline apply-dnn \
    --input outputs/eventsel/sample_EVENTSELECTION.root \
    --model data/dnn/dnn_model.pt \
    --config configs/dnn.yaml \
    --score-branch ml_score \
    --output-dir scored_outputs/ \
    [--dnn-mass-scan all]   # parametric models only
```

### Key DNN flags

| Flag | Subcommand(s) | Default | Description |
| ---- | ------------- | ------- | ----------- |
| `--dnn-config CONFIG` | `train-dnn` | required | DNN config YAML (`configs/dnn.yaml`) |
| `--input FILES/DIR` | `train-dnn`, `apply-dnn` | required | `EVENTSELECTION.root` files or a folder |
| `--weight-branch BRANCH` | `train-dnn` | `full_event_weight` | Per-event weight branch to train with |
| `--outdir DIR` | `train-dnn` | `data/dnn` | Model artifacts: `dnn_model.pt`, `scaler.json`, `features.json`, `train_metrics.json` |
| `--plot-dir DIR` | `train-dnn` | `outputs/dnn` | Training plots: ROC, loss, AUC, score distributions, feature significance |
| `--xsection-signal-json JSON` | `train-dnn`, `analyze` | — | Signal cross sections / mass grid, e.g. `data/cross-section/xsection_signal.json` |
| `--xsection-json JSON` | `train-dnn`, `analyze` | — | Background cross sections for per-file lumi×xsec weighting |
| `--apply-dnn` | `analyze` | `False` | Score events with `--dnn-model`/`--dnn-config` in the region-plots path |
| `--dnn-model PATH` | `analyze`, `apply-dnn` | — | Pre-trained `.pt` checkpoint |
| `--dnn-config CONFIG` | `analyze`, `apply-dnn` | — | DNN config for inference (feature list, etc.) |
| `--dnn-mass-scan SPEC` | `analyze`, `apply-dnn` | — (single benchmark) | Parametric models only: `all`, or comma list of `MH3_<a>_MH4_<b>_Mchi_<c>` labels |
| `--signal-scale N` | `analyze` | `1` | Multiply signal histograms by N for shape visibility |

### DNN configuration (`configs/dnn.yaml`)

- **model**: architecture (hidden layers, dropout), `parametric_input` (mass-conditioned training)
- **training**: batch size, learning rate, epochs, early stopping, class balancing, seed
- **feature_selection**: top-K by Asimov significance, single-feature scans
- **topology_decorrelation**: penalty weight for score vs topology correlations
- **features**: input variables (MET, jet kinematics, angular variables, b-tag scores)
- **variable_labels**: LaTeX x-axis labels for feature/score distribution plots

---

## Parallelization

`train-dnn` and `analyze --mode region-analysis` dispatch independent per-item work
(one input ROOT file, one feature, or one region/variable plot) across CPU cores via
`multiprocessing.Pool` (`spawn` context) — not threads, since the work is CPU-bound
(numpy/torch/matplotlib) and would serialize on the GIL under `threading`. Coffea's
`--executor futures/dask` (file-chunk level parallelism) is unaffected and independent
of this.

| Stage | Command | Env var (default: `os.cpu_count()`) |
| ----- | ------- | ------------------------------------ |
| Loading input ROOT files | `train-dnn` | `DNN_LOAD_WORKERS` |
| Per-feature significance ranking | `train-dnn` | `DNN_SIGNIF_WORKERS` |
| Per-feature 1D DNN scan (post-training) | `train-dnn` | `DNN_SCAN_WORKERS` |
| Region/variable plot generation | `analyze --make-region-plots` | `PLOT_NUM_WORKERS` |

```bash
# Force serial (e.g. to compare timing, or on a memory-constrained node)
DNN_LOAD_WORKERS=1 DNN_SCAN_WORKERS=1 darkbottomline train-dnn ...
PLOT_NUM_WORKERS=1 darkbottomline analyze --mode region-analysis --make-region-plots ...

# Cap worker count explicitly
PLOT_NUM_WORKERS=4 darkbottomline analyze --mode region-analysis --make-region-plots ...
```

**Notes:**

- Each worker pins `OMP_NUM_THREADS=1`/`OPENBLAS_NUM_THREADS=1` (and `torch.set_num_threads(1)`
  for torch-based workers) to avoid N workers × M BLAS threads oversubscribing the host.
- In `train-dnn`, the per-feature DNN scan (`DNN_SCAN_WORKERS`) trains one small model per
  feature — on large datasets (10⁶+ events) this stage is bound by per-worker `DataLoader`
  mini-batch iteration, not raw core count, so speedup is sub-linear; reducing
  `single_feature_epochs` or increasing `batch_size` in `configs/dnn.yaml` helps more than
  adding workers.
- In `analyze --mode region-analysis` (`--input` an `EVENTSELECTION.root` folder), only the
  final draw+save step per region/variable is parallelized — region-cut application and DNN
  scoring on the raw event arrays happens once, serially, before dispatch.

---

## EGamma HLT Scale Factor Map

Standalone script to plot the EGamma HLT trigger scale factor (SF) as a 2D map
(electron pT vs η) from correctionlib JSON files.  Uses `mplhep` CMS style.
Outputs PNG + PDF.

```bash
# Single HLT path
python scripts/plot_egamma_hlt_sf.py \
    --json data/corrections/2022/Run3-22CDSep23-Summer22-NanoAODv12_electronHlt.json.gz \
    --path HLT_SF_Ele30_TightID \
    --lumi 8.1 --com 13.6 \
    --outdir outputs/hlt_sf/

# All HLT paths in one go
python scripts/plot_egamma_hlt_sf.py \
    --json data/corrections/2024/Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15_electronHlt.json.gz \
    --all-paths \
    --lumi 109 --com 13.6 --label Internal \
    --outdir outputs/hlt_sf/2024/

# sfup / sfdown variations
python scripts/plot_egamma_hlt_sf.py \
    --json data/corrections/2022/Run3-22EFGSep23-Summer22EE-NanoAODv12_electronHlt.json.gz \
    --path HLT_SF_Ele30_TightID \
    --valtype sfup \
    --lumi 26.7 --outdir outputs/hlt_sf/

# List available HLT paths in a file (omit --path and --all-paths)
python scripts/plot_egamma_hlt_sf.py \
    --json data/corrections/2022/Run3-22CDSep23-Summer22-NanoAODv12_electronHlt.json.gz
```

### Key options

| Flag | Default | Description |
|------|---------|-------------|
| `--json` | — | Correctionlib `.json` or `.json.gz` file |
| `--corr` | `Electron-HLT-SF` | Correction name inside the JSON |
| `--year` | auto | Year key (auto-detected when only one present) |
| `--path` | — | HLT path key, e.g. `HLT_SF_Ele30_TightID` |
| `--all-paths` | — | Plot all available HLT paths |
| `--valtype` | `sf` | `sf` \| `sfup` \| `sfdown` |
| `--lumi` | — | Luminosity in fb⁻¹ for CMS header |
| `--com` | `13.6` | Centre-of-mass energy in TeV |
| `--label` | `Internal` | CMS label: `Internal` \| `Preliminary` \| `Work in Progress` |
| `--pt-min` | `30.0` | Minimum pT to display [GeV] |
| `--pt-max` | `500.0` | Maximum pT to display [GeV] |
| `--outdir` | `plots/hlt_sf` | Output directory (PNG + PDF written here) |
| `--vmin/--vmax` | auto | Manual color scale limits |

Works with both 2022/2023 (float ±inf edges) and 2024 (string `'-inf'` edges) JSON formats.

---

## Framework Components

| File | Role |
| ---- | ---- |
| `analyzer.py` | `DarkBottomLineAnalyzer`: full pipeline, `process()`, `process_from_eventselection()` |
| `objects.py` | `build_objects()`, object selection functions |
| `selections.py` | `apply_selection()`: trigger → filters → recoil → multiplicities → jet → dphi |
| `regions.py` | `RegionManager`, `Region.apply_cuts()`, flat-branch fallbacks for CR variables |
| `variables.py` | `compute_event_variables()`: all output branch computation |
| `histograms.py` | `HistogramManager`: ~40+ histogram definitions |
| `plotting.py` | `PlotManager`: stacked plots, 5 formats (PDF/PNG/ROOT/TXT/TEX), SR blinding, process group routing |
| `corrections.py` | `CorrectionManager`: correctionlib scale factors |
| `weights.py` | `WeightCalculator` |
| `cli.py` | CLI: `analyze` (all modes), `make-plots`, `make-stacked-plots` |

---

## Configuration

```text
configs/
  2022.yaml / 2022EE.yaml / 2023.yaml / 2024.yaml   # year-specific
  regions.yaml      # region definitions and cuts
  plotting.yaml     # process groups, exclusions, bin edges, log scale vars
```

All thresholds in YAML — no hardcoded cuts in Python. Missing key → loud `KeyError`.

---

## PKL Output Structure

`--output out.pkl` saves (used by `make-plots`):

```text
pkl
├── region_histograms   {region: {var: hist.Hist}}
├── regions             {region: {n_events, variables, dnn_scores}}
├── region_cutflow      {total_events, regions: {n_events, fraction}}
├── event_selection_cutflow  {cut_name: count}
├── region_validation   {status, overlaps, warnings}
└── metadata            {n_events_processed, weighted_total_events, luminosity}
```

---

## Versioning

Scheme: `YYYYMMDD+<sha7>` (`_version.py`), `YYYYMMDD-<sha7>` (git tags).

```bash
scripts/tag_version.sh   # bumps _version.py, commits, tags, gh release
```

---

## Troubleshooting

| Problem | Fix |
| ------- | --- |
| Corrupt EVENTSELECTION.root | Re-run `--mode event-selection` for that sample |
| 0 plots for CRs | Check `plotting.yaml` process group patterns match filenames |
| `--output` required error | Add `--output` or `--make-region-plots` to `--mode full` |
| Memory issues | `--max-events N` for testing; `--chunk-size 25000` for futures/dask |
| ROOT format missing | ROOT not installed — PNG/PDF/TXT still written |

---

## Dependencies

- Core: `coffea`, `awkward`, `uproot`, `correctionlib`
- Histogramming: `hist`, `boost-histogram`
- Plotting: `matplotlib`, `mplhep`
- Output: `pyarrow`, `pandas`
- Execution: `dask`, `distributed`

```bash
source local_setup.sh
```
