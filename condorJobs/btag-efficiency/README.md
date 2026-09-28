# B-tag efficiency maps on Condor

**One cluster per samplelist `.txt`, one job per BATCH-sized slice of ROOT files**
(mirrors `condorJobs/met_trigger`). Each job's slice = lines
`[ProcId*BATCH+1 .. ProcId*BATCH+BATCH]` of the `.txt`; the slimmer's `--inputs`
mode sums that whole slice's raw counts into ONE output. A `.txt` with 120 files
and the default `BATCH=50` becomes a cluster of `ceil(120/50)=3` jobs. Output:

```text
<OUTDIR>/<txtfile stem>/<txtfile stem>_<ClusterId>_<ProcId>.root
```

i.e. each samplelist `.txt` gets its own subdirectory under `OUTDIR`, with one
merged-slice efficiency-map ROOT per condor job inside it (the
`_<ClusterId>_<ProcId>` suffix keeps concurrent and resubmitted jobs for the
same `.txt` from writing the same file).

## Files

| File | Role |
|------|------|
| `run_slimmer.sh`  | Job executable: env setup, slice this ProcId's BATCH of files, run the slimmer, verify the write |
| `submit.sub`      | Per-`.txt` submit template; `queue $(NJOBS)`; vars injected by `submit_all.sh` |
| `submit_all.sh`   | Loops the joblist, one `condor_submit` per `.txt` → a separate cluster each |
| `make_joblist.sh` | Build `joblist.txt` from `data/samplelist/<era>/` (MC only, no data PDs) |

## Run

```bash
# 1. Grid proxy — XRootD reads inside the jobs need it. Put it under your AFS home
#    with the default voms name (x509up_u<uid>); submit.sub ships it to the sandbox:
voms-proxy-init --voms cms --valid 192:00 \
    --out /afs/cern.ch/user/${USER:0:1}/${USER}/private/x509up_u$(id -u)

# 2. Build the joblist for an era (all non-data samplelists, including signal)
condorJobs/btag-efficiency/make_joblist.sh 2024

# 3. Edit submit.sub:
#      REPO_DIR / OUTDIR  -> replace the /CHANGE/ME/ placeholders with your paths
#                            (OUTDIR must be job-visible: AFS or EOS).
#      Proxy_filename     -> x509up_u<your uid> (matches step 1). The proxy DIR
#                            auto-resolves from $(USER_INITIAL)/$ENV(USER).

# 4. Submit — one cluster per txt, ceil(NFILES/BATCH) jobs each (BATCH=50 default)
condorJobs/btag-efficiency/submit_all.sh
#    retune batch size:  BATCH=100 condorJobs/btag-efficiency/submit_all.sh
#                   or:  condorJobs/btag-efficiency/submit_all.sh <joblist> 100
```

`submit_all.sh` counts the ROOT lines in each `.txt`, computes
`NJOBS=ceil(NFILES/BATCH)`, creates `logs/` next to the scripts, and calls
`condor_submit` once per `.txt`, passing `TXTFILE` / `BATCH` / `NJOBS` /
`USER_INITIAL` / `LOGDIR` into `submit.sub` — so each `.txt` gets its own
`ClusterId` with `NJOBS` jobs.

The job sets up the software env itself (`source LCG_109 setup.sh` + repo `.local`
on `PYTHONPATH`), same pattern as `condorJobs/met_trigger/run_skim.sh`. The proxy
is shipped via `transfer_input_files` and `run_slimmer.sh` exports
`X509_USER_PROXY` from it before any XRootD read.

### Write verification + retry

EOS (via `eosxd` FUSE) has been observed to silently corrupt a small fraction of
sibling files written concurrently by many condor jobs into the same output
directory — the slimmer's own write can report success while a subset of
histogram keys in the file on disk decompress to garbage afterwards. `run_slimmer.sh`
re-opens and verifies every histogram key after writing, retrying the whole
slimmer run (up to 3x) before giving up; a job that still fails is held and
retried by Condor itself (`periodic_release`, also up to 3x).

## After the jobs finish

Per-job outputs land in `<OUTDIR>/<txtstem>/<txtstem>_<ClusterId>_<ProcId>.root`.
**No `hadd` step** — `scripts/btag_efficiency_maker.py` sums raw counts directly
from every job's output across every sample in ONE call, writing a SINGLE merged
efficiency map (the `--inputs` glob spans every sample subdirectory at once):

```bash
python3 scripts/btag_efficiency_maker.py \
    --inputs "<OUTDIR>/*/*.root" \
    --output <OUTDIR>/heavyflavor_efficiency_maps_2024.root
```

The maker skips (and clearly reports, then exits non-zero) any input file with a
corrupted histogram key instead of crashing on the first one — re-run the
corresponding condor job(s) for any file it flags, then re-run the maker.

## Notes

- `CONFIG` is a job argument, so the same setup works for any era (2022/2023/2024).
- Bump `request_memory` / `request_cpus` / `+JobFlavour` in `submit.sub` if a
  batch is large/slow (default: 6000 MB, 4 cpus, `workday`).
- At the condor level, a failed job is held and re-run up to 3x (`periodic_release`).
