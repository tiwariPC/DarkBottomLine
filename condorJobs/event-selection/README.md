# Condor Job Submission — Event Selection

Runs `darkbottomline analyze --mode event-selection` (NanoAOD → `EVENTSELECTION.root`)
on lxplus HTCondor for every samplelist in `data/samplelist/<year>/`.

- **One cluster per sample**, **one job per slice** of `BATCH` files (not one job per file).
- Each job processes its files **sequentially**, one CLI call per file, so a bad file
  never costs the rest of the slice.
- Inputs are read from the CERN EOS replica when one exists (checked with `xrdfs stat`),
  else the samplelist's redirector, then INFN and FNAL as fallbacks.
- Outputs are written to worker scratch, verified, then `xrdcp`'d to EOS with an adler32
  checksum (no FUSE writes, see `condorJobs/btag-efficiency/run_slimmer.sh` for why):

  ```
  <OUTDIR>/<year>/eventsel/<sample>/<sample>_I<fileIdx>C<ClusterId>P<ProcId>_EVENTSELECTION.root
  OUTDIR = <base>/<tag>_<DDMMYYYY>/     base default: /eos/cms/store/group/phys_susy/sus-23-008/run3_bbMET
  ```

  You only pass the base (`--outdir`, optional); the run directory is created from the git version
  tag (`darkbottomline/_version.py` with `+` → `-`, e.g. `20261014-d110783`) and the submission
  date. To resubmit missing files on a later day, pass the original run directory as `--outdir`
  (e.g. `.../run3_bbMET/20261014-d110783_14102026`); a path already ending in `<tag>_<DDMMYYYY>`
  is used as is. The submitter prints whether today's run directory already exists (reused, files already
  there are skipped) or will be created, and lists same-tag directories from other days.

- Files already on EOS are never resubmitted: **rerunning the same command resubmits only
  what is missing.** Samples with jobs still in the queue are skipped, so nothing runs twice.

  One subdirectory per sample. Region analysis globs `*.root` in a single folder (not
  recursive), so either merge each sample into one file in a flat folder first
  (`scripts/merge_rootfiles.py`) or point `--input` at one sample's folder.

  Output name fields: `I` = line index in the samplelist (identifies the input file),
  `C`/`P` = Condor ClusterId/ProcId of the job that wrote it. A held+released retry keeps
  its ClusterId/ProcId; across submissions the submitter dedups on (sample, `I`).

## Which samples run, and how

Classified from the samplelist name by `sample_kind()` in `submit_samples.py`:

| Kind | Rule | Run as |
|---|---|---|
| DATA | `-Run20YY` in the name **and** primary dataset in `DATA_PRIMARY_DATASETS = ("EGamma", "JetMET")` | `--data`: `is_data: true`, golden-JSON lumi mask, no MC weights |
| SKIP | `-Run20YY` but any other primary dataset (Muon0/1) | not submitted; listed as skipped on every run |
| SIG | `2hdma` in the name (BBDM-2HDMa) | MC weights; the CLI sets `is_signal` from `2hdma` in the input path, which bypasses HLT (fastsim has no trigger branches) |
| MC | everything else | `is_data: false` (the `configs/<year>.yaml` default), MC weights |

The plan table printed before submission shows the kind of every sample. To use another data
primary dataset, add it to `DATA_PRIMARY_DATASETS`.

## Files

| File | Role |
|---|---|
| `submit_samples.py` | Builds slices, checks limits, calls `condor_submit` once per sample |
| `submit.sub` | Template; every value is passed by `submit_samples.py` via `-append` |
| `runanalysis.sh` | Job executable: loop over the slice, skip/run/verify/stage-out |
| `pilot_report.py` | Timing + CPU efficiency + memory summary; writes the `--timing` JSON |
| `submitfiles/<year>/<sample>.slices` | Generated: `<sliceId> <idx>:<idx>:...` per job (git-ignored) |
| `logs/<year>/` | Condor stdout/stderr/event logs (git-ignored) |

## Setup

```bash
voms-proxy-init --voms cms --valid 192:00
cp /tmp/x509up_u$(id -u) /afs/cern.ch/user/${USER:0:1}/${USER}/private/
```

`submit_samples.py` refuses to submit with less than 48 h left on the proxy. EOS writes
use the Kerberos token HTCondor forwards, so no extra setup is needed for them.

## Workflow

Run from anywhere; all paths are absolute.

### 1. Pilot (about 1 h)

The first file of a few samples, at 2 and at 8 cores, to measure time per file, memory and CPU
efficiency. Pilot outputs are real outputs and are reused by production.

```bash
# an array, quoted: works in bash and zsh (zsh does not word-split a plain $S)
S=('WtoLNu-2Jets_Bin-2J-PTLNu-40to100_*' 'JetMET0-Run2024G-*' 'ZZ_*' 'BBDM-2HDMa-*')
python3 condorJobs/event-selection/submit_samples.py --year 2024 --samples "${S[@]}" \
    --files 0:1 --batch 1 --cpus 2 --memory 8000
python3 condorJobs/event-selection/submit_samples.py --year 2024 --samples "${S[@]}" \
    --files 1:2 --batch 1 --cpus 8 --memory 12000 --allow-duplicates

# once they finish
python3 condorJobs/event-selection/pilot_report.py --year 2024
```

Pick the core count with the lowest **core-seconds per file** (s/file × cpus). Remote reads
are usually I/O-bound, so 2 cores usually wins. Set `--memory` to the measured peak plus headroom.

2024 pilot (2026-10-10, one file per job, so a rough comparison):

| Sample | s/file at 2 cores | s/file at 8 cores |
|---|---|---|
| ZZ | 99 | 126 |
| WtoLNu-2Jets 40to100 | 188 | 137 |
| Muon0-Run2024G (data) | 216 | 77 |
| BBDM-2HDMa | 265 | 80 |

At 2 cores a file costs 200–530 core-seconds, against 600–1,100 at 8, so production uses 2 cores.

Two lxplus behaviours seen in the pilot:
- **Memory and cores are tied together.** Asking for more than about 3 GB per core raises
  `request_cpus` (2 cores + 8000 MB ran as 3 cores / 9 GB), and memory is raised to at least
  3 GB per core (2 cores + 4000 MB runs as 2 cores / 6000 MB). Keep `--memory` ≤ 2000 × cpus
  so you are charged only for the cores you use.
- **Peak memory from short jobs is unreliable.** Condor samples memory every few minutes;
  the 1.5–4.5 min pilot jobs reported 70–200 MB. Read it from production jobs instead.

### 2. Production

What was submitted for 2024 (69 clusters, 1,715 jobs, 25,191 files):

```bash
python3 condorJobs/event-selection/submit_samples.py --year 2024 --dry-run \
    --batch 15 --cpus 2 --memory 4000 \
    --outdir /eos/cms/store/group/phys_susy/sus-23-008/run3_bbMET/
# check the table, then drop --dry-run
```

The 2026-10-14 submission (tag `20261014-d110783`) used exactly this with `--dry-run` dropped
(outputs in `.../run3_bbMET/20261014-d110783_14102026/`):
79 clusters, 2,288 jobs, 33,722 files, Muon0/Muon1 skipped (not in `DATA_PRIMARY_DATASETS`).

At roughly 100–265 s per file, 15 files per job gives jobs of about 25–70 minutes, well
inside the 8 h `workday` limit. `--batch 10` gives 2,550 jobs of about 17–45 minutes.

Or size each sample from the pilot timing:

```bash
python3 condorJobs/event-selection/pilot_report.py --year 2024 --workers 2 \
    --write condorJobs/event-selection/timing_2024.json
python3 condorJobs/event-selection/submit_samples.py --year 2024 --dry-run \
    --timing condorJobs/event-selection/timing_2024.json --target-minutes 90 --cpus 2 --memory 4000
```

Files per job = `clamp(round(target_minutes / minutes_per_file), 1, --max-batch)` per sample.
Samples the pilot did not cover use the 75th percentile of the pilot MC (or data) samples.
Without `--timing`, every sample uses `--batch` (default 10).

### 3. Resubmit what is missing

When a round has finished (no jobs left in `condor_q`), run the same production command
again. Only files with no output on EOS are submitted. Do not use `--no-skip-existing` on a
directory that already has outputs: it creates a second file with the same `I` and a
different `C`/`P`, and a `*_EVENTSELECTION.root` glob would then count those events twice.

## Options (`submit_samples.py --help`)

| Option | Default | |
|---|---|---|
| `--year` | required | 2022, 2022EE, 2023, 2024 |
| `--outdir` | see above | EOS base; outputs go to `<outdir>/<year>/eventsel/<sample>/` |
| `--samples` / `--exclude` | all | sample-name globs |
| `--files START:END` | all | restrict file indices per sample (pilots/tests) |
| `--batch` | 10 | files per job without `--timing` |
| `--timing`, `--target-minutes`, `--max-batch` | –, 90, 50 | pilot-driven files per job |
| `--cpus` | 2 | `request_cpus` and executor workers |
| `--memory` | 2000 × cpus | `request_memory` in MB |
| `--flavour` | workday | 8 h limit; a job stuck on one file is killed within hours |
| `--chunk-size` | 50000 | executor chunk size or `auto` |
| `--dry-run` | | write slice files and print the plan only |

## lxplus limits (schedd, 2026-10-10)

| Limit | Value |
|---|---|
| `MAX_JOBS_PER_SUBMISSION` (one `condor_submit`) | 20,000 |
| `MAX_JOBS_PER_OWNER` (queued at once) | 100,000 |
| Idle jobs removed after | 10 days |

`submit_samples.py` checks both job limits before submitting. Throughput is set by your
fair-share core quota, not by these limits, so ask for as few cores as the pilot allows.

## Monitoring

```bash
condor_q -batch                                       # one row per sample
condor_q -hold -af ClusterId ProcId HoldReason        # why jobs are held
grep -l FAILED condorJobs/event-selection/logs/2024/error/*.err
python3 condorJobs/event-selection/pilot_report.py --year 2024   # timing/efficiency so far
xrdfs eoscms.cern.ch ls <OUTDIR>/2024/eventsel | wc -l          # outputs on EOS
```

`pilot_report.py` gets CPU efficiency and memory from `condor_history`, which can hang for
minutes on the shared lxplus schedd. The per-file timing comes from the logs and does not
depend on it. Use `ls` via `xrdfs` rather than the `/eos` FUSE mount to count outputs:
FUSE listings can lag behind renames and deletes.

Failed jobs are held, then released after 10 min, up to 3 starts. Files that reached EOS
are skipped on each retry. After 3 starts the job is removed; step 3 picks it up.
