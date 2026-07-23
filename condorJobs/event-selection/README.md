# Condor Job Submission Guide — Event Selection

This directory submits `darkbottomline analyze --mode event-selection` jobs to Condor
(NanoAOD → `EVENTSELECTION.root`). Sample lists come from `data/samplelist/<year>/`,
not a local `samplefiles/` directory.

## Files

- `runanalysis.sh`: Main job execution script (runs on condor nodes)
- `submit.sub`: Template submit file (used by submit_samples.py)
- `submit_samples.py`: **Simple Python script to submit jobs** ⭐

## Quick Start (Recommended)

```bash
# Submit all *.txt files in data/samplelist/2022/
python3 condorJobs/event-selection/submit_samples.py --year 2022
```

That's it! The script will:

1. Find all `*.txt` files in `data/samplelist/<year>/`
2. For each sample file:
   - Count the number of ROOT files in it
   - Auto-detect collision data samples (name matches `-Run20YY`) and pass `--data`
   - Submit N condor jobs (one job per file)
   - Each sample gets its own condor cluster
3. Each file from a sample runs on a separate node

**Example output:**
```
Sample: Zto2Nu-2Jets_PTNuNu-40to100_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.txt [MC]
  Files in sample: 5
  Sample type: MC
  Will submit: 5 jobs (1 job per file)
  ✓ Submitted cluster: 12345
  ✓ Jobs: 5 (one per file)

Sample: JetMET-Run2022D-22Sep2023-v1.txt [DATA]
  Files in sample: 3
  Sample type: DATA
  Will submit: 3 jobs (1 job per file)
  ✓ Submitted cluster: 12346
  ✓ Jobs: 3 (one per file)
```

## Job Structure

```
Server 1 (Cluster 12345):
  Node 1: Job 0 → processes file 1 from Zto2Nu-2Jets_....txt
  Node 2: Job 1 → processes file 2 from Zto2Nu-2Jets_....txt
  ...

Server 2 (Cluster 12346):
  Node 1: Job 0 → processes file 1 from JetMET-Run2022D-....txt
  ...
```

## Python Script Options

```bash
# Basic usage (uses data/samplelist/<year>/ by default)
python3 condorJobs/event-selection/submit_samples.py --year 2022

# Override input directory
python3 condorJobs/event-selection/submit_samples.py --year 2022 --input-dir /path/to/samplelist

# Custom configuration (default: configs/<year>.yaml)
python3 condorJobs/event-selection/submit_samples.py \
    --year 2022 \
    --config configs/2022.yaml \
    --executor futures \
    --chunk-size 50000

# Dry run (see what would be submitted)
python3 condorJobs/event-selection/submit_samples.py --year 2022 --dry-run

# Help
python3 condorJobs/event-selection/submit_samples.py --help
```

## Setup

1. **Update paths in `submit.sub`**:
   - Change `x509userproxy` to your proxy path (line 3)

2. **Create log directories** (automatically created by submit_samples.py):
   ```bash
   mkdir -p condorJobs/event-selection/logs/output condorJobs/event-selection/logs/error
   ```

3. **Initialize voms proxy** (if needed):
   ```bash
   voms-proxy-init --voms cms --valid 192:00
   cp /tmp/x509up_u$(id -u) /afs/cern.ch/user/u/username/private/
   ```

4. **Sample files**: already provided under `data/samplelist/<year>/*.txt`
   (one ROOT file path per line; comments start with `#`).

## Output Files

Each job produces an EVENTSELECTION.root file:

- Format: `outputs/eventsel/<year>/<sample_name>_<file_index>_EVENTSELECTION.root`
- Example: `outputs/eventsel/2022/Zto2Nu-2Jets_..._0_EVENTSELECTION.root`

## Monitoring Jobs

```bash
# Check all job status
condor_q

# Check specific cluster
condor_q 12345

# Check job logs
tail -f condorJobs/event-selection/logs/output/dbl.*.out
tail -f condorJobs/event-selection/logs/error/dbl.*.err
```

## Advanced: Manual Submission

If you need to submit manually (not recommended):

1. Edit `submit.sub`:
   - Set `DBL_BKG_FILE=JetMET-Run2022D-22Sep2023-v1.txt`
   - Set `DBL_YEAR=2022`
   - Set `queue 5` where 5 = number of files

2. Submit:
   ```bash
   condor_submit condorJobs/event-selection/submit.sub
   ```

## Troubleshooting

1. **Jobs held**: Check error logs in `condorJobs/event-selection/logs/error/`
2. **Proxy expired**: Re-run `voms-proxy-init` and copy to AFS
3. **File not found**: Check that input files exist and paths are correct
4. **Memory issues**: Increase `request_memory` in submit.sub or reduce `--chunk-size`
5. **No sample files found**: Check `data/samplelist/<year>/` exists and contains `*.txt` files
