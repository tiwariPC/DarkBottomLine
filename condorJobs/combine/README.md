# Condor Job Submission Guide — Combine Limit-Setting Pipeline

This directory submits `darkbottomline` Combine-pipeline subcommands
(`make-datacard`, `run-combine`, `merge-categories`, `merge-eras`, `run-all`, ...)
to Condor. Unlike `event-selection/`, there is no bespoke pipeline logic here —
`runcombine.sh` is a one-line dispatch to `python3 -m darkbottomline.cli
<subcommand> <args>`; every actual pipeline decision (categories, regions,
eras, blind, signal grid, rateParam scheme, ...) is read from
`configs/combine.yaml` by the CLI itself, per the framework's CLI-only
orchestration convention.

## Files

- `runcombine.sh`: Condor executable stub — sources the LCG environment, then
  execs `darkbottomline <subcommand> <args>` using `DBL_COMBINE_SUBCOMMAND`/
  `DBL_COMBINE_ARGS` from the job's environment.
- `submit_datacard.sub`: Template for `make-datacard` jobs (one job per
  category x mass point).
- `submit_limits.sub`: Template for `run-combine --mode AsymptoticLimits`
  jobs (one job per signal mass point on the merged Run3 card). Sized for the
  full 29-point 2HDM+a grid from `data/cross-section/xsection_signal.json`
  (combine.yaml's `signal_grid.points: null` = run all points).

## Setup

1. **Update paths**: set `x509userproxy` and `DBL_REPO_DIR` in each `.sub` file.
2. **Create log directories**:
   ```bash
   mkdir -p condorJobs/combine/logs/output condorJobs/combine/logs/error
   ```
3. **Initialize voms proxy** (if needed):
   ```bash
   voms-proxy-init --voms cms --valid 192:00
   cp /tmp/x509up_u$(id -u) /afs/cern.ch/user/u/username/private/
   ```
4. **Combine must be installed** (optional step, see repo README/`check_requirements.py
   --install-combine` on lxplus or `INSTALL_COMBINE=1 source local_setup.sh`
   locally) — datacard-generation jobs don't need it, but `run-combine`/
   `merge-categories`/`merge-eras` jobs do (they shell out to
   `combine`/`combineTool.py`/`text2workspace.py`/`combineCards.py`).

## Submitting jobs

There is currently no `submit_samples.py`-equivalent auto-enumeration script
for Combine jobs (unlike `event-selection/`). For now, either:

- Submit one `.sub` file per (category, mass_point) manually, setting
  `DBL_COMBINE_ARGS` accordingly, or
- Prefer `darkbottomline run-all --combine-config configs/combine.yaml --era 2024`
  run interactively/via a single Condor job for small grids — it loops over
  categories and the signal grid itself, so a full Condor fan-out is only
  needed once the mass-point count makes a single job too slow.

```bash
condor_submit condorJobs/combine/submit_datacard.sub
condor_submit condorJobs/combine/submit_limits.sub
```

## Monitoring jobs

```bash
condor_q
condor_q 12345
tail -f condorJobs/combine/logs/output/dbl_datacard.*.out
tail -f condorJobs/combine/logs/error/dbl_limit.*.err
```

## Troubleshooting

1. **Jobs held**: check `condorJobs/combine/logs/error/`.
2. **`combine`/`combineTool.py` not found**: Combine install is optional and
   must be done explicitly — see Setup step 4.
3. **Datacard/workspace not found**: `run-combine`/`merge-categories`/
   `merge-eras` jobs depend on `make-datacard` output existing first; check
   job ordering/dependencies if chaining these via separate submissions.
