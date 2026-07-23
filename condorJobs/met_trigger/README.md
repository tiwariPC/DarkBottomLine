# MET-trigger skim on Condor

**One cluster per samplelist `.txt`, one job per BATCH-sized slice of ROOT files.**
Each job skims up to `BATCH` files (default **50**): its slice = lines
`[ProcId*BATCH+1 .. ProcId*BATCH+BATCH]` of the `.txt`. A `.txt` with 594 files
becomes a cluster of `ceil(594/50)=12` jobs. The skim step merges every file it's
given into ONE output, so each job (one BATCH-sized slice) produces one output:

```text
<OUTDIR>/<txtfile stem>/<txtfile stem>_<ClusterId>_<ProcId>.root
```

i.e. each samplelist `.txt` gets its own subdirectory under `OUTDIR`, with one
merged skim per job inside it (the `_<ClusterId>_<ProcId>` suffix keeps concurrent
and resubmitted jobs for the same `.txt` from writing the same file). (For the 2024
joblist, BATCH=50 gives ~419 jobs total instead of ~19,915 — a ~47× reduction.)

## Files

| File | Role |
|------|------|
| `run_skim.sh`     | Job executable: env setup, extract this ProcId's BATCH-file slice, skim them |
| `submit.sub`      | Per-`.txt` submit template; `queue $(NJOBS)`; vars injected by `submit_all.sh` |
| `submit_all.sh`   | Loops the joblist, one `condor_submit` per `.txt` → a separate cluster each |
| `make_joblist.sh` | Build `joblist.txt` from `data/samplelist/<era>/` (Muon→data, all bkg→mc) |

## Run

```bash
# 1. Grid proxy — XRootD reads inside the jobs need it. Put it under your AFS home
#    with the default voms name (x509up_u<uid>); submit.sub ships it to the sandbox:
voms-proxy-init --voms cms --valid 192:00 \
    --out /afs/cern.ch/user/${USER:0:1}/${USER}/private/x509up_u$(id -u)

# 2. Build the joblist for an era (Muon* = data; all bkg samplelists = mc)
condorJobs/met_trigger/make_joblist.sh 2024

# 3. Edit submit.sub:
#      REPO_DIR / OUTDIR  -> replace the /CHANGE/ME/ placeholders with your paths
#                            (OUTDIR must be job-visible: AFS or EOS).
#      Proxy_filename     -> x509up_u<your uid> (matches step 1). The proxy DIR
#                            auto-resolves from $(USER_INITIAL)/$ENV(USER).

# 4. Submit — one cluster per txt, ceil(NFILES/BATCH) jobs each (BATCH=50 default)
condorJobs/met_trigger/submit_all.sh
#    retune batch size:  BATCH=100 condorJobs/met_trigger/submit_all.sh
#                   or:  condorJobs/met_trigger/submit_all.sh <joblist> 100
```

`submit_all.sh` counts the ROOT lines in each `.txt`, computes
`NJOBS=ceil(NFILES/BATCH)`, creates `logs/` next to the scripts, and calls
`condor_submit` once per `.txt`, passing `KIND` / `TXTFILE` / `BATCH` / `NJOBS` /
`USER_INITIAL` / `LOGDIR` into `submit.sub` — so each `.txt` gets its own `ClusterId`
with `NJOBS` jobs.

The job sets up the software env itself (`source LCG_109 setup.sh` + repo `.local`
on `PYTHONPATH`); the skim script is self-contained (no `darkbottomline` import).
The proxy is shipped via `transfer_input_files` and `run_skim.sh` exports
`X509_USER_PROXY` from it before any XRootD read.

## After the jobs finish — step 2 (analyze)

Skims land in `<OUTDIR>/<txtstem>/<txtstem>_<ClusterId>_<ProcId>.root` — one per condor
job (already a merge of that job's BATCH-sized slice). The `MetTriggerSkim` TTree has
ONE row per selected event, with orthogonal `wmu` / `zmu` flags (an event is a W→μν
or a Z→μμ candidate, never both) and these branches:

```text
recoil recoilPhi  lep1Pt lep1Eta lep1Phi  lep2Pt lep2Eta lep2Phi
Jet1Pt Jet1Eta Jet1Phi  muTrigPass metTrigPass  wmu zmu  genweight
```

`genweight` is raw `sign(genWeight)`; the file's total `sign(genWeight)` and raw event
count are stored as 1-bin `genTotalSumw` / `genTotalCount` histos. NO xsec/lumi/sumw
normalisation yet — that is applied in analyze. (Trees are written as classic TTrees,
not RNTuple, so ROOT viewers and `hadd` read them correctly.)

**`hadd` one merged ROOT per sample** (a sample's per-job skims → one file). The
histos hadd-sum, so the merged file carries the sample-total SUMW:

```bash
for d in <OUTDIR>/*/ ; do
  s=$(basename "$d"); hadd -f <MERGED>/${s}.root "$d"/*.root
done
```

Then `analyze` reads one file per sample; it looks up the **xsec from the filename**,
reads **SUMW from the histo**, and applies `weight = sign(genWeight)*xsec*lumi/SUMW`:

```bash
python3 scripts/met_trigger_efficiency.py analyze --config configs/2024.yaml \
  --data-skims "<MERGED>/Muon*.root" \
  --mc-skims   "<MERGED>/DYto2L-2Jets_*.root" "<MERGED>/WtoLNu-2Jets_*.root" \
               "<MERGED>/TT*.root" "<MERGED>/*SingleTop*.root" "<MERGED>/WW*.root" ...
```

Keep merged MC filenames matching the sample (for the xsec lookup). Data skims are
unweighted. The recoil binning is applied at analyze time (`--recoil-bins`), so it
can be changed without re-skimming.

## Notes

- `CONFIG` is a job argument, so the same setup works for any era (2022/2023/2024).
  Uses `data/cross-section/xsection_background_run3.json` for xsecs.
- Bump `request_memory` / `+JobFlavour` in `submit.sub` if a batch is large/slow.
- XRootD reads are retried **3× per file** (5s/10s backoff) before that file is
  skipped, so transient errors (`Invalid operation`, timeouts) recover. A file skipped
  after all retries doesn't kill the job — the rest of the slice still produces output.
- A file MISSING the MET / IsoMu trigger branches fails the job loudly (KeyError) —
  it would otherwise silently bias the efficiency, so fix/exclude such files.
- At the condor level, a failed job is held and re-run up to 3× (`periodic_release`).
