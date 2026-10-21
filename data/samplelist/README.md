# Sample lists

`data/samplelist/<year>/*.txt` are the sample lists that get submitted (one file path per line).
Sample lists that are not in use are kept in `data/samplelist/backup/<year>/`.

## 2024 V+jets: binned or inclusive

W+jets, Z(νν)+jets and DY exist in two forms in 2024. They overlap, so use one form at a time:

| | binned (default) | inclusive |
|---|---|---|
| W+jets | `Bin-1J-PTLNu-*` and `Bin-2J-PTLNu-*` (40to100 ... 600) | `Bin-1J-PTLNu-40to100`, `Bin-2J-PTLNu-40to100`, `Bin-PTLNu-100` |
| Z(νν)+jets | `Bin-1J-PTNuNu-*` and `Bin-2J-PTNuNu-*` | `Bin-1J-PTNuNu-40to100`, `Bin-2J-PTNuNu-40to100`, `Bin-PTNuNu-100` |
| DY | `Bin-2J-MLL-50-PTLL-*` (40to100 ... 600) | `Bin-2J-MLL-50-PTLL-40to100`, `Bin-MLL-50-PTLL-100` |

The `*-100` inclusive samples contain everything with pT ≥ 100 GeV, which is why the 100to200, 200to400,
400to600 and 600 bins must not be used with them. The cross sections are in
`data/cross-section/xsection_background_run3.json`.

```bash
python3 scripts/switch_sample_mode.py status
python3 scripts/switch_sample_mode.py inclusive --dry-run     # show what would move
python3 scripts/switch_sample_mode.py inclusive               # or: binned
```

The script (list of samples per mode in `configs/sample_modes.yaml`) does two things:

1. moves the sample lists between `data/samplelist/2024/` and `data/samplelist/backup/2024/`,
   so event selection is only submitted for the active set;
2. sets `sample_mode:` in `configs/plotting.yaml`. In `inclusive` mode each group uses
   `patterns_inclusive` instead of `patterns`, but only if the input folder holds a file that
   only the inclusive set matches (so 2022/2022EE/2023, which have only binned samples, keep using
   the binned patterns). Which outputs get plotted therefore depends on `sample_mode` and not on
   which files are in the folder.

It refuses to move a sample that has jobs in the HTCondor queue, because the jobs read the
sample list while running. Wait for them to finish and run it again.

Event-selection outputs of both forms can sit in the same output folder; `sample_mode`
decides which ones the plots and region analysis pick up.
