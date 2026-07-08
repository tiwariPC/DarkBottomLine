# MET-trigger skim on Condor

One condor job per samplelist `.txt`. Each job runs
`scripts/met_trigger_efficiency.py skim` on that single text file and writes one
skim ROOT per ROOT path listed inside it:

```
<OUTDIR>/<txtfile stem>_<unique last string of the input ROOT filename>.root
```

## Files

| File | Role |
|------|------|
| `run_skim.sh`     | Job executable: `source start.sh`, then `skim` one txt → outdir |
| `submit.sub`      | Submit description; `queue KIND, TXTFILE from joblist.txt` |
| `make_joblist.sh` | Build `joblist.txt` from `data/samplelist/<era>/` (Muon→data, DY/W→mc) |

## Run

```bash
# 1. Grid proxy (for XRootD reads inside the jobs)
voms-proxy-init --voms cms --valid 192:00

# 2. Build the joblist for an era (Muon* = data, DYto2L*/WtoLNu-2Jets* = mc)
condorJobs/met_trigger/make_joblist.sh 2024

# 3. Edit submit.sub: set REPO_DIR, CONFIG, OUTDIR (OUTDIR is yours to choose,
#    must be condor-visible — e.g. an /eos path).

# 4. Submit — one job per txt (24 for 2024: 14 data + 10 mc)
condor_submit condorJobs/met_trigger/submit.sub
```

## After the jobs finish

All skim ROOTs are in `OUTDIR`. Optionally `hadd` per samplelist, then run step 2:

```bash
python scripts/met_trigger_efficiency.py analyze --config configs/2024.yaml \
  --data-skims "<OUTDIR>/Muon*.root" \
  --mc-skims   "<OUTDIR>/DYto2L-2Jets_*.root" "<OUTDIR>/WtoLNu-2Jets_*.root"
```

`analyze` sums counts across all matched skims (hadded or per-file — either works),
so the recoil binning is applied at analyze time and can be changed without
re-skimming.

## Notes

- `CONFIG` is a job argument, so the same setup works for any era (2022/2023/2024).
- Jobs read via XRootD; `use_x509userproxy = true` ships your proxy to the node.
- Bump `request_memory` / `+JobFlavour` in `submit.sub` if a samplelist is large.
