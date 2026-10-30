# Plotting Configuration Guide

How `configs/plotting.yaml` controls the region plots and the histograms that feed Combine.
Pass another file with `--plot-config`.

## What the plotter does

`darkbottomline analyze --mode region-analysis --make-region-plots` replays the per-sample
region PKLs (see [analysis_structure.md](analysis_structure.md)): it bins the stored
per-event values, applies `lumi × xsec / weighted_total_events` to Monte Carlo, adds data and
signal, and writes plots plus the ROOT files Combine reads.

```bash
darkbottomline analyze --mode region-analysis \
    --config <year>.yaml --regions-config configs/regions.yaml \
    --input <eventsel_merged dir> --region-results <per_sample dir> \
    --output-dir <out> --version <name> \
    --xsection-json data/cross-section/xsection_background_run3.json \
    --xsection-signal-json data/cross-section/xsection_signal.json \
    --plot-config configs/plotting.yaml --make-region-plots --signal-scale 10 \
    [--plot-variables Recoil mll ...] [--plot-regions 1b:SR 2b:CR_Zee ...] [--make-syst-plots]
```

`--input` is only used to find signal branches and for the cutflow lookups; the background
and data values come from `--region-results`. The SR is blinded: its "data" is the total MC unless
`--show-data` is given.

## Layout of `plotting.yaml`

| Key | Purpose |
|---|---|
| `sample_mode` | `binned` or `inclusive`, selects the V+jets sample patterns (see below) |
| `common_variables` | plotted in every region |
| `region_variables` | extra variables per region (key = region name) |
| `systematic_variables` | variables for which shape-systematic histograms are written |
| `variable_labels`, `variable_bins` | axis labels and binning per variable |
| `no_log_scale_vars` | variables drawn with a linear y axis |
| `process_groups` | background, signal and data groups with their file patterns |
| style keys (`dpi`, `figsize_*`, `fontsize_*`, `ratio_ylim`, `cms_label`, ...) | appearance |
| `event_selection_variables` | variables for the event-selection (pre-region) plots |

## Choosing the variables of a region

The variables plotted in a region are a whitelist: `common_variables` followed by
`region_variables[region]` (exact key first, then any key contained in the region name),
without duplicates. A variable is plotted only if it exists in that region's PKL values.
There is no separate exclusion step: to keep a variable out of a region, leave it out of that
region's list (for example the SRs list no lepton variables, and the 1b regions list no
jet-3 variables).

Current per-region additions in the shipped file: W CRs `mt`, `Wpt` and the lepton kinematics;
Top CRs jet variables, `mt` and the lepton kinematics; Z CRs `mll`, `Zpt` and both leptons'
kinematics; SRs the dijet variables.

`PlotManager` still reads a `region_exclusions` key and contains a helper that would use it
(`_get_excluded_variables_for_region`, with built-in rules written for the legacy histogram names),
but nothing calls that helper, so the key has no effect on the plots.

## Variable names, aliases and file names

Names are the field names in the PKL (`Recoil`, `Jet1Pt`, `mll`, `Zpt`, `ml_score`,
`costheta_star`, ...). A few names are handled specially in `plotting.py`:

| Variable | Behaviour |
|---|---|
| `Wpt` | same values as `Recoil` (with one lepton, recoil = −pT of lepton + MET, so |recoil| = pT(W)); own bins; the direction differs by π |
| `PV_npvsGood_noPU` | values of `PV_npvsGood` weighted with `weight_noPileup` (data unchanged); shares the bins of `PV_npvsGood` |
| `PV_npvsGood`, `PV_npvsGood_noPU` | written as `PV` and `PV_noPU` in file names; labelled "after / before PU reweighting" |

File names always carry the output name: `hist_<cat>_<region>_<var>_log.{png,pdf,root,tex}`,
e.g. `hist_1b_Zmumu_PV_noPU_log.png`.

## Binning

```yaml
variable_bins:
  Recoil:  {edges: [250, 300, 400, 550, 1000]}
  Wpt:     {edges: [250, 275, ..., 1000]}
  ml_score: {low: 0.0, high: 1.0, n: 25}
```

With `low/high/n`, `n` is the number of **edges** (`np.linspace(low, high, n)`), so you get
`n − 1` bins; use `edges` for exact control. Values above the last edge are folded into the
last bin; values below the first edge are dropped. Variables with no entry use `n_bins_default`.
If the same variable is plotted in regions with different physical ranges (for example `MET_pt`
in W and Z regions), the first edge decides which events appear.

## Process groups, data and `sample_mode`

Each background, signal or data group lists `patterns` that are matched against the PKL file
stems after the dataset name is normalised (jet-bin tokens, DY dashes and so on).

- **`sample_mode: binned`** uses `patterns` (the pT-binned W, Z(νν) and DY samples).
- **`sample_mode: inclusive`** uses `patterns_inclusive` (the pT-inclusive samples plus the 40–100 GeV
  bin) for the groups that define it, but only when the input folder holds a file that only the
  inclusive set matches; otherwise it falls back to `patterns`.
- **Data groups** have a `regions` list of substrings: `MET_Data` for regions containing `SR`,
  `munu`, `mumu`; `EGamma_Data` for `enu`, `Zee`.
- **Signal** (`Signal_2HDMa`) is drawn in the SRs only, one histogram per mass point named
  `sig_MH3_<a>_MH4_<b>_Mchi_1`, with the weighted event yield and the per-mass-point cross section
  (`--signal-scale` multiplies it for display only). Variables that are not branches of the signal
  event-selection file (for example `ml_score`) are read from the signal PKL.

## Systematic variations

For each variable in `systematic_variables` (`Recoil`, `costheta_star`, `ml_score` by default),
the PKL path writes variation files next to the nominal one:

| Files | Variables | Content |
|---|---|---|
| `hist_<cat>_<region>_<var>_weight_<name>{UP,DOWN}_log.root` | all | 20 weight variations: `pileup`, `btag`, `btag_true`, `btag_mistag`, `muon`, `electron`, `electronHLT`, `metHLT`, `pdf`, `scale` |
| `hist_<cat>_<region>_Recoil_JER{UP,DOWN}_log.root` | `Recoil` only | JER (no effect: smearing is not implemented) |
| `hist_<cat>_<region>_<var>_JEC<source>{UP,DOWN}_log.root` | `Recoil`, `costheta_star` | the 11 reduced JEC sources (`Absolute`, `Absolute_2024`, `BBEC1`, `BBEC1_2024`, `EC2`, `EC2_2024`, `HF`, `HF_2024`, `FlavorQCD`, `RelativeBal`, `RelativeSample_2024`) |

Each file has the same keys as the nominal one (process groups, `TotalBkg`, `sig_*` in the SRs).
The JEC shift is stored as one shifted quantity, `Recoil_JEC<source>{Up,Down}`. For `Recoil` the shifted
value is filled; for `costheta_star` (the 2b fit variable) each event keeps its nominal value, since JEC changes
jet pT and MET but not eta, and the variation is the migration of events through `recoil > recoil_min`
(an event counts only if its shifted recoil passes). `ml_score` has no JEC variations and JER exists only for
`Recoil`. Region PKLs and ntuples hold only events that pass the nominal selection, so migration *into* the
region (nominal recoil below the threshold, shifted above) is not seen and the Up variations understate it.
`--make-syst-plots` also draws central/up/down comparison plots.

## Output layout

```text
<output-dir>/<version>/
  png/region_analysis/<cat>_<region>/hist_<cat>_<region>_<var>_log.png
  pdf/region_analysis/...                       same, PDF
  text/region_analysis/<cat>_<region>/*.tex     yield tables and cutflows (booktabs)
  root/hist_<cat>_<region>_<var>[_<syst>]_log.root   process groups, TotalBkg, data_obs, sig_*
  root/cutflow_<cat>_<region>.root
```

The plot annotation for a region comes from `_REGION_LABELS` in `plotting.py`; if you change what a
region selects but keep its name, edit the label too.

## Common tasks

**Add a variable to a region:** make sure it is a field in the PKL values, add it to
`common_variables` or `region_variables`, and add a `variable_labels` and `variable_bins` entry.

**Add a systematic variable:** add it to `systematic_variables` and to `fit_variable` or the
Combine configuration if it is used in a fit. Only weight variations are available unless shifted
variants of the variable are stored.

**Compare two selections:** run each with its own `--regions-config` and `--version`; the outputs do not
overwrite each other.

## Tips

- Check variable names against the PKL fields, not against the plot labels.
- Change binning in `plotting.yaml` and re-run only the plot step; the region PKLs do not change.
- Use `edges` for variables whose values are integers (for example PV counts) to avoid empty or
  double-filled bins.
- `tests/test_plot_exclusions.py` tests the `region_variables` whitelist logic.
