# `plot_merged_region_analysis.py` Usage Guide

This document explains how to use `scripts/plot_merged_region_analysis.py` to draw stacked control-region plots from merged ROOT files.

## What the script does

The script reads one merged ROOT file per sample from an input directory, extracts the `TH1` objects stored in the file, normalizes each background sample, groups samples into canonical MC categories, overlays `JetMET` data, and writes PNG plots to an output directory.

It is designed for merged files produced by the DarkBottomLine workflow, where each file contains region histograms such as:

- `1b:CR_Wlnu_el_*`
- `1b:CR_Wlnu_mu_*`
- `1b:CR_Zll_el_*`
- `2b:CR_Top_el_*`
- `2b:CR_Top_mu_*`

## Core behavior

- Background files are treated as MC and are normalized using event counters from the ROOT file metadata.
- Data files are detected by file name prefix, defaulting to `JetMET`.
- MC is grouped into the canonical categories:
  - `DIBOSON`
  - `DYto2L-2Jets`
  - `Top`
  - `SingleTop`
  - `WtoLNu-2Jets`
  - `Zto2Nu-2Jets`
  - `SMHiggs`
- In the final plots, these are drawn with the simplified labels:
  - `DIBOSON`
  - `Drell-yan`
  - `TTbar`
  - `WLNu`
  - `Z2Nu`
  - `SMhiggs`
  - `singletop`

Important: here `Top` means the cross-section group from `xsection_results.json`, not the region name. For example, `TTto2L2Nu`, `TTtoLNu2Q`, and `TTto4Q` all belong to the `Top` group.

## Normalization logic

The script uses the following normalization priority for each MC ROOT file:

1. `weighted_total_events` at the ROOT top level
2. `h_n_events_processed` at the ROOT top level
3. `Metadata/h_n_events_processed`
4. `Metadata/weighted_total_events`

If a file has non-zero entries in the target control region but none of the counters above can be read, the script stops with an error. This is intentional so normalization problems are not hidden.

When a cross section is available, the scale is:

$$
\text{scale} = \frac{\mathcal{L} \times \sigma_{\rm pb} \times 1000}{N_{\rm weighted}}
$$

where:

- $\mathcal{L}$ is the luminosity in fb$^{-1}$
- $\sigma_{\rm pb}$ is the cross section in pb
- $N_{\rm weighted}$ is the total event count read from the ROOT metadata

If no cross section is available, the script falls back to:

$$
\text{scale} = \frac{\mathcal{L}}{N_{\rm weighted}}
$$

Data files are not scaled.

## Command-line arguments

The most important arguments are:

- `--input-root`: directory containing the merged ROOT files
- `--output-dir`: where PNG plots will be written
- `--lumi`: luminosity in fb$^{-1}$
- `--xsection-json`: JSON file with cross-section definitions
- `--year`: year key used with the cross-section JSON
- `--data-prefix`: file name prefix used to detect data files, default `JetMET`
- `--region-pattern`: substring used to select a subset of histograms, default `:CR`
- `--max-plots`: optional limit for debugging

## Typical usage

### Full control-region run

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_109/x86_64-el9-gcc15-opt/setup.sh
cd /afs/cern.ch/user/x/xdu/xdu/DarkBottomLine
source start.sh

python3 scripts/plot_merged_region_analysis.py \
  --input-root /afs/cern.ch/user/x/xdu/xdu/run3_bbMET/detached_3def8a5/19052026/2022/region_analysis_MERGED \
  --output-dir /afs/cern.ch/user/x/xdu/xdu/run3_bbMET/detached_3def8a5/19052026/2022/region_plots_lumi7p99_full_fixed \
  --lumi 7.99 \
  --xsection-json /afs/cern.ch/user/x/xdu/xdu/DarkBottomLine/scripts/xsection_results.json \
  --year 2022 \
  --region-pattern ":CR"
```

### Only Top control regions

```bash
python3 scripts/plot_merged_region_analysis.py \
  --input-root /afs/cern.ch/user/x/xdu/xdu/run3_bbMET/detached_3def8a5/19052026/2022/region_analysis_MERGED \
  --output-dir /afs/cern.ch/user/x/xdu/xdu/run3_bbMET/detached_3def8a5/19052026/2022/region_plots_top_check \
  --lumi 7.99 \
  --xsection-json /afs/cern.ch/user/x/xdu/xdu/DarkBottomLine/scripts/xsection_results.json \
  --year 2022 \
  --region-pattern "CR_Top"
```

### Small debug run

```bash
python3 scripts/plot_merged_region_analysis.py \
  --input-root /afs/cern.ch/user/x/xdu/xdu/run3_bbMET/detached_3def8a5/19052026/2022/region_analysis_MERGED \
  --output-dir /afs/cern.ch/user/x/xdu/xdu/run3_bbMET/detached_3def8a5/19052026/2022/region_plots_debug \
  --lumi 7.99 \
  --xsection-json /afs/cern.ch/user/x/xdu/xdu/DarkBottomLine/scripts/xsection_results.json \
  --year 2022 \
  --region-pattern ":CR" \
  --max-plots 10
```

## Output layout

The output directory is organized by region. Examples:

- `1b_CR_Wlnu_el/btag_deepjet.png`
- `1b_CR_Wlnu_mu/met.png`
- `1b_CR_Zll_el/jet_pt.png`
- `2b_CR_Top_el/met.png`
- `2b_CR_Top_mu/n_bjets.png`

## Common pitfalls

- If a background file has non-zero histograms but no readable event counter, the script will stop and report the file name.
- If you only see `W`, `Z`, `DY`, or `DIBOSON` in a plot, that means the other categories had no non-zero contribution for that histogram after aggregation.
- If `Top` is missing, first check whether the corresponding ROOT file contains the relevant `CR_Top_*` histograms and whether the sample was classified into the `Top` JSON group.
- `JetMET-Run2022C-22Sep2023-v1` and `JetMET-Run2022D-22Sep2023-v1` are treated as data and are not normalized like MC.

## Notes for debugging

The script prints lines like:

```text
Drawing categories for 1b:CR_Wlnu_el_met: ['DIBOSON', 'DYto2L-2Jets', 'SMHiggs', 'SingleTop', 'Top', 'WtoLNu-2Jets']
```

This is the quickest way to see which MC categories are actually contributing to a given plot.

## Related files

- `scripts/plot_merged_region_analysis.py`
- `scripts/xsection_results.json`
- `normalization_problems.txt`