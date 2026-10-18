# Condor Job Submission — Region Analysis

Runs the last stage of the chain on lxplus HTCondor: merged `<sample>_EVENTSELECTION.root`
→ per-sample region PKL with DNN scores → region plots.

```
condorJobs/event-selection   NanoAOD → per-file EVENTSELECTION.root   (EOS, .../eventsel/<sample>/)
scripts/merge_rootfiles.py   one <sample>_EVENTSELECTION.root per sample   (.../eventsel_merged/)
scripts/lumi_calculator.py   brilcalc luminosity of the data that reached the selection
condorJobs/region-analysis   region cuts + DNN → per_sample/<stem>_DNNAPPLIED.pkl → plots   (this directory)
```

- **One job per merged sample file** (`run_region.sh`). Each job reads its input over
  XRootD, runs `darkbottomline analyze --mode region-analysis --per-sample-only`, verifies the
  PKL unpickles and `xrdcp`s it to EOS with an adler32 checksum. A PKL already on EOS is skipped.
- **The plot job is the DAGMan child of all region jobs** (`run_plots.sh`, `plot.sub`): it runs
  only if every PKL job succeeded, then writes the plots to scratch and `xrdcp -r`s them to EOS.
- **Memory is set by chunk size, not file size.** The per-file driver
  (`darkbottomline.cli.process_eventselection_file`) reads the Events tree in chunks of 300,000
  events (`EVENTSEL_STEP_SIZE`), so the old whole-tree load (46 GB RSS on a 6.6 GB file) is gone.
  The per-sample PKL is identical to what the whole-tree path writes (up to float summation order
  in the cutflow weights).
- DNN: `data/dnn/dnn_model.pt` with `configs/dnn.yaml`. torch 2.10 is in LCG_109, so no extra
  environment is needed. The checkpoint has 33 input features while `dnn.yaml` lists 25; the code
  uses the checkpoint's list.

```
<OUTDIR>/per_sample/<stem>_DNNAPPLIED.pkl      job output (<stem> = <sample>_EVENTSELECTION; _DNNAPPLIED marks DNN-scored PKLs)
<OUTDIR>/plots/...                                  plot job output
<OUTDIR>/config/<year>_lumi<lumi>.yaml              config copy the jobs used (lumi set at submission)
<OUTDIR>/config/run_info.json                       git version, lumi, inputs, command
```

## Files

| File | Role |
|---|---|
| `submit_regions.py` | Plans jobs (skips PKLs on EOS), sizes memory, writes job list + DAG + config copy, runs `condor_submit_dag` |
| `submit.sub` | Region-job template (DAG node `regions`): `queue … from` the job list |
| `run_region.sh` | Region-job executable: skip / run / verify / stage-out |
| `plot.sub`, `run_plots.sh` | Plot-job template and executable (DAG node `plot`) |
| `submitfiles/<year>/` | Generated: `regions.jobs`, `regions.dag`, lumi config copy (git-ignored) |
| `logs/<year>/` | Condor stdout/stderr/event logs (git-ignored) |

## Full chain

All paths below are for the 2026-10-14 production (tag `20261014-d110783`).

```bash
RUN=/eos/cms/store/group/phys_susy/sus-23-008/run3_bbMET/20261014-d110783_14102026/2024

# 0. proxy (see ../event-selection/README.md); the submitter wants >= 24 h left
voms-proxy-init --voms cms --valid 192:00
cp /tmp/x509up_u$(id -u) /afs/cern.ch/user/${USER:0:1}/${USER}/private/

# 1. event selection: see ../event-selection/README.md  →  $RUN/eventsel/<sample>/

# 2. merge: one file per sample  →  $RUN/eventsel_merged/
python3 scripts/merge_rootfiles.py --help        # options; flat output folder, <sample>_EVENTSELECTION.root

# 3. luminosity of the data that reached the selection (brilcalc)
python3 scripts/lumi_calculator.py --input-dir $RUN/eventsel_merged --output-dir outputs/lumi/2024
#    the config uses the JetMET value (109.627 /fb for this production; EGamma is 108.460)

# 4+5. region analysis and plots, as one DAG
python3 condorJobs/region-analysis/submit_regions.py --lumi 109.627 --dry-run \
    --input-dir $RUN/eventsel_merged --outdir $RUN/region_analysis
# check the table (jobs, memory per job), then:
python3 condorJobs/region-analysis/submit_regions.py --lumi 109.627 \
    --input-dir $RUN/eventsel_merged --outdir $RUN/region_analysis
```

The last command writes the lumi config copy, `run_info.json`, the job list and the DAG, copies the
config and `run_info.json` to `$RUN/region_analysis/config/`, and runs `condor_submit_dag`. Options are
in `submit_regions.py --help`; the defaults are `configs/regions.yaml`,
`data/cross-section/xsection_background_run3.json`, `data/cross-section/xsection_signal.json`,
`configs/plotting.yaml`, `--signal-scale 10`.

The plot job runs

```bash
darkbottomline analyze --mode region-analysis --config <lumi config copy> \
    --regions-config configs/regions.yaml --input $RUN/eventsel_merged \
    --region-results $RUN/region_analysis/per_sample --output-dir <scratch> --version <v> \
    --xsection-json data/cross-section/xsection_background_run3.json \
    --plot-config configs/plotting.yaml --make-region-plots \
    --xsection-signal-json data/cross-section/xsection_signal.json --signal-scale 10
```

### Lumi

`--lumi` is required and is written into a copy of `configs/<year>.yaml` (only the `lumi:` line
changes). The copy is what every job reads and what the plot job uses, and it is recorded next to
the outputs in `<OUTDIR>/config/`. The config has one lumi value, so EGamma-stream samples are
normalised with the JetMET number.

### Memory

`request_memory = clamp(--mem-min + --mem-per-gb × input GB, …, --mem-max)` (defaults 4000, 500,
12000 MB). lxplus ties memory to cores (about 3 GB per core), so `request_cpus` is raised to
`ceil(memory / 3000)` and the table printed by the submitter shows what will be asked for.
Calibration: see "Measured" below.

### Two steps instead of one DAG

PKLs first, plots later (the plot job can be rerun on its own after changing `configs/plotting.yaml`):

```bash
# step 1: region jobs only; rerun until "Nothing to submit"
python3 condorJobs/region-analysis/submit_regions.py --lumi 109.627 --no-plot \
    --input-dir $RUN/eventsel_merged --outdir $RUN/region_analysis
# step 2: plot job only; refuses if any PKL is missing
python3 condorJobs/region-analysis/submit_regions.py --lumi 109.627 --plot-only \
    --input-dir $RUN/eventsel_merged --outdir $RUN/region_analysis
```

### Plot job without Condor

On lxplus, in `tmux` (needs about 16 GB; check `free -g`):

```bash
cd <repo>; source /cvmfs/sft.cern.ch/lcg/views/LCG_109/x86_64-el9-gcc15-opt/setup.sh
export PYTHONPATH=$PWD/.local/lib/python3.13/site-packages:$PWD:$PYTHONPATH
python3 -m darkbottomline.cli analyze --mode region-analysis \
    --config condorJobs/region-analysis/submitfiles/2024/2024_lumi109.627.yaml \
    --regions-config configs/regions.yaml --input $RUN/eventsel_merged \
    --region-results $RUN/region_analysis/per_sample --output-dir outputs/regions_2024 \
    --version <v> --xsection-json data/cross-section/xsection_background_run3.json \
    --plot-config configs/plotting.yaml --make-region-plots \
    --xsection-signal-json data/cross-section/xsection_signal.json --signal-scale 10
```

`submit_regions.py --no-plot` submits the region jobs only.

## Resubmit what is missing

When the DAG has finished (`condor_q` empty), run the same `submit_regions.py` command again.
Only samples with no PKL on EOS get a job; the plot job runs again as their child (or alone if
nothing is missing). A region job that fails is held, released after 10 min, and removed after 3
starts; a removed job fails the `regions` node, so the plot job does not run on an incomplete set.

## Monitoring

```bash
condor_q -batch
condor_q -hold -af ClusterId ProcId HoldReason
tail condorJobs/region-analysis/submitfiles/2024/region_analysis.dag.dagman.out
grep -l FAILED condorJobs/region-analysis/logs/2024/error/*.err
xrdfs eoscms.cern.ch ls $RUN/region_analysis/per_sample | wc -l      # PKLs on EOS
```

## Running one sample by hand

```bash
python3 -m darkbottomline.cli analyze --mode region-analysis --per-sample-only \
    --config configs/2024.yaml --regions-config configs/regions.yaml \
    --input <one>_EVENTSELECTION.root --output <dir> \
    --xsection-json data/cross-section/xsection_background_run3.json \
    --apply-dnn --dnn-model data/dnn/dnn_model.pt --dnn-config configs/dnn.yaml   # + --data for collision data
# → <dir>/per_sample/<stem>_DNNAPPLIED.pkl
```

Without `--per-sample-only` the same command also makes the plots from the PKLs it just wrote
(the previous behaviour). `merged.pkl` (all samples in one file, built in memory) is now opt-in
with `--write-merged`; plotting does not use it.
