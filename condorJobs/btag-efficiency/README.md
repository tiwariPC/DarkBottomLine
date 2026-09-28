# B-tag efficiency maps on Condor

**One cluster per samplelist `.txt`, one job per input ROOT file** (the slimmer is
single-file-in/single-file-out, so unlike `met_trigger` there is no BATCH slicing:
`NJOBS = NFILES`). A `.txt` with 120 files becomes a cluster of 120 jobs. Output:

```text
<OUTDIR>/<txtfile stem>/<txtfile stem>_<ClusterId>_<ProcId>.root
```

i.e. each samplelist `.txt` gets its own subdirectory under `OUTDIR`, with one
efficiency-map ROOT per condor job inside it (the `_<ClusterId>_<ProcId>` suffix
keeps concurrent and resubmitted jobs for the same `.txt` from writing the same
file).

## Files

| File | Role |
|------|------|
| `run_slimmer.sh`  | Job executable: env setup, pick this ProcId's file, run the slimmer |
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

# 4. Submit — one cluster per txt, NFILES jobs each
condorJobs/btag-efficiency/submit_all.sh
```

`submit_all.sh` counts the ROOT lines in each `.txt`, creates `logs/` next to the
scripts, and calls `condor_submit` once per `.txt`, passing `TXTFILE` / `NJOBS` /
`USER_INITIAL` / `LOGDIR` into `submit.sub` — so each `.txt` gets its own
`ClusterId` with `NJOBS` jobs.

The job sets up the software env itself (`source LCG_109 setup.sh` + repo `.local`
on `PYTHONPATH`), same pattern as `condorJobs/met_trigger/run_skim.sh`. The proxy
is shipped via `transfer_input_files` and `run_slimmer.sh` exports
`X509_USER_PROXY` from it before any XRootD read.

## After the jobs finish

Per-`.txt` efficiency maps land in `<OUTDIR>/<txtstem>/<txtstem>_<ClusterId>_<ProcId>.root`
— one per condor job (one input file each). `hadd` per sample to get the sample-total
map before feeding it to `CorrectionManager` / the heavy-flavor efficiency loader:

```bash
for d in <OUTDIR>/*/ ; do
  s=$(basename "$d"); hadd -f <MERGED>/${s}.root "$d"/*.root
done
```

## Notes

- `CONFIG` is a job argument, so the same setup works for any era (2022/2023/2024).
- Bump `request_memory` / `+JobFlavour` in `submit.sub` if a sample's files are large/slow.
- At the condor level, a failed job is held and re-run up to 3x (`periodic_release`).
