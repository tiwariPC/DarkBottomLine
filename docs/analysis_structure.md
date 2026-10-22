# Analysis Structure

How regions are defined, how events flow from NanoAOD to Combine inputs, and what the
per-sample region files contain.

## Pipeline

```text
NanoAOD ──event selection──▶ <sample>_EVENTSELECTION.root   (flat branches, one tree "Events")
   condorJobs/event-selection        merged per sample with scripts/merge_rootfiles.py
                                          │
                  darkbottomline analyze --mode region-analysis
                  (region cuts + DNN score, chunked over the tree)
                                          │
                                          ▼
                     per_sample/<sample>_EVENTSELECTION_DNNAPPLIED.pkl   (one PKL per sample)
                                          │
                  analyze --make-region-plots   (PKL replay, ROOT → PKL → plot)
                                          │
                                          ▼
          png/ pdf/ text/ root/   (stacked plots, yield tables, cutflows, Combine histograms)
                                          │
                       darkbottomline make-datacard / run-all   (see README, Combine section)
```

`condorJobs/region-analysis` runs the region and plot stages on HTCondor (one job per
merged sample file). Memory is set by the chunk size (300,000 events), not by file size.

## Region naming

Format: `{category}:{region}` where the category is `1b` or `2b` (the b-tag category the
signal region belongs to) and the region name encodes the process and lepton channel.

| Region | Process | Channel |
|---|---|---|
| `1b:SR`, `2b:SR` | signal | – |
| `1b:CR_Wmunu`, `1b:CR_Wenu` | W+jets | μ / e |
| `2b:CR_Topmunu`, `2b:CR_Topenu` | top | μ / e |
| `1b:CR_Zmumu`, `1b:CR_Zee`, `2b:CR_Zmumu`, `2b:CR_Zee` | Z+jets | μμ / ee |

Region names are used exactly as written in `configs/regions.yaml` everywhere: in the
PKLs, in plot directory names (`:` becomes `_`, e.g. `png/region_analysis/1b_Wmunu/`),
in ROOT file names (`hist_1b_Wmunu_Recoil_log.root`) and in the Combine configuration.
The data stream is chosen from the region name: regions containing `SR`, `munu` or `mumu`
use the JetMET (MET-trigger) data, regions containing `enu` or `Zee` use EGamma.

## Region definitions (`configs/regions.yaml`)

Each region has a `description`, a `trigger` (`met` or `ele`, applied as the first
cutflow step) and an ordered `cuts` block. Cuts are ANDed, so their order changes only the
cutflow table, not the selection.

| Region | Jets / b-tag | Leptons | Other |
|---|---|---|---|
| `1b:SR` | `Bjet1bCond == 1`, `Njets <= 2` | none | |
| `2b:SR` | `Bjet2bCond == 1`, `2 <= Njets <= 3` | none | |
| `1b:CR_W*` | `Bjet1bCond == 1`, `NAdditionalJets == 0` | 1 tight (μ or e), other flavour vetoed | `MET > 100`, `MTMin > 0`, `MT < 160` |
| `2b:CR_Top*` | `Bjet2bCond == 1`, `NAdditionalJets >= 1` | 1 tight (μ or e), other flavour vetoed | `MT < 160` |
| `1b:CR_Z*` | `Njets <= 2` (no b-tag) | 2 opposite-sign, Z-candidate pair | `70 < mll < 110` |
| `2b:CR_Z*` | `2 <= Njets <= 3` (no b-tag) | same | same |

All regions inherit the event selection (`event_selection` in the year config):
recoil ≥ 250 GeV, leading-jet pT ≥ 100 GeV, min Δφ(jet, MET) > 0.5, tau and photon vetoes.

Cut keys used in `regions.yaml`:

| Key | Meaning |
|---|---|
| `Bjet1bCond` | exactly one b-tagged jet and the leading jet is the b-tagged one |
| `Bjet2bCond` | exactly two b-tagged jets, which are the leading and sub-leading jets |
| `LeadingBjet`, `SubleadingBjet` | the leading / sub-leading jet is b-tagged |
| `Njets`, `NjetsMin` | upper and lower bound on the jet multiplicity (YAML keys are unique, so two keys) |
| `NAdditionalJets` | `n_jets − n_bjets`, i.e. the number of non-b-tagged jets (so `== 0` means every jet is b-tagged) |
| `Nmuons`, `Nelectrons`, `NmuonsTight`, `NelectronsTight` | loose / tight lepton counts |
| `NmuonsZ`, `NelectronsZ`, `MllMin`, `MllMax` | Z-candidate pair and mass window |
| `MET`, `MT`, `MTMin`, `MTMax` | MET threshold and transverse-mass window (`MTMin`/`MTMax` exist to allow both bounds) |

### Overlap between the 1b and 2b Z regions

The Z regions carry no b-tag requirement, so the 1b-type (`Njets <= 2`) and 2b-type
(`2 <= Njets <= 3`) selections share every event with exactly two jets: about 48% of the 1b-type
and 67% of the 2b-type Z regions. In a combined 1b+2b fit with separate per-bin rate
parameters, those events enter two channels and the statistical errors of the two Z
normalisations are correlated (about 0.58 per Recoil bin) while the fit treats them as
independent. Options: make the Z regions orthogonal by b-tag count (copy the SR's
`Bjet1bCond` / `Bjet2bCond`), assign `Njets == 2` to one category only, or use one shared Z region.

### Region variants

To study another definition, copy `configs/regions.yaml`, edit the cuts and pass the copy
with `--regions-config`. Keep the region names the same if the plotting and Combine
configurations should be reused unchanged, and write to a new `--version` so the nominal
output stays intact.

## Region PKL structure

One PKL per sample, written by `darkbottomline analyze --mode region-analysis --per-sample-only`
(or by the HTCondor stage). The file is named `<stem>_DNNAPPLIED.pkl`, with `<stem>` the merged
file name without `.root` (`<sample>_EVENTSELECTION`); the suffix marks PKLs in which the DNN
score `ml_score` was added to every event, so they cannot be mistaken for unscored ones. Without
`--apply-dnn`/a DNN model the name is `<stem>.pkl` and there is no `ml_score` field. The plotting
step matches samples by name patterns, so either form works as `--region-results` input.

```python
{
  "region_histograms": {
      "1b:SR": {
          "weight": ndarray,                  # nominal event weight (full_event_weight) of selected events
          "values": {                         # one 1-D array per scalar field, same length as weight
              "Recoil": ..., "Jet1Pt": ..., "ml_score": ..., "mll": ..., "Zpt": ..., "njets": ...,
              "weight_btagUP": ..., "weight_scaleDOWN": ...,   # every weight variation
              "Recoil_JERUp": ..., "Recoil_JECAbsoluteUp": ..., # shifted recoil for JER and the 11 JEC sources
              "GenModel_MH3_600_MH4_150_Mchi_1": ...,           # signal only: 0/1 flag per mass point
          },
      },
      "1b:CR_Wmunu": {...},   # ... every region in regions.yaml
  },
  "region_cutflow_steps": {region: {step: weighted yield}},    # cumulative, in cut order
  "metadata": {"weighted_total_events": ..., "xsec": ..., "luminosity": ..., "sample": ...},
}
```

Multi-point signal samples are normalised per mass point: the signal `_EVENTSELECTION.root` carries one
one-bin histogram `weighted_total_events_GenModel_<point>` per point (generated Σ sign(genWeight) of that
point, written by event selection or added to a merged file with `scripts/add_genmodel_counts.py`), copied
into `metadata["weighted_total_events_per_genmodel"]`. A point without a counter is an error: there is no
fallback to the file-level `weighted_total_events`, which covers every point.

Cross sections are applied at plot time (`lumi × xsec / weighted_total_events`), not when the
PKL is written, so changing `data/cross-section/*.json` needs no region re-run. Data PKLs
have unit weights. Because the per-event weight variations and shifted recoils are stored, the
Combine shape-systematic histograms can be produced from the PKLs alone (see
[plotting_configuration.md](plotting_configuration.md)).

## Verification

```python
from darkbottomline.regions import RegionManager
print(list(RegionManager("configs/regions.yaml").regions.keys()))
# ['1b:SR', '2b:SR', '1b:CR_Wmunu', '1b:CR_Wenu', '2b:CR_Topmunu', '2b:CR_Topenu',
#  '1b:CR_Zmumu', '1b:CR_Zee', '2b:CR_Zmumu', '2b:CR_Zee']
```

Checks worth running on a finished set of outputs:
- Every histogram of a region (except the cutflow) carries the same total, apart from
  variables that are undefined for some events (`Jet2*`, `Jet3*`: sentinel −9999), variables whose
  plot range starts above the physical minimum (`MET_pt`), and `PV_noPU`, which uses a different weight.
- The last cutflow step equals that total, and data totals equal the event count in the data PKLs.

## Troubleshooting

**A region is missing from plots or Combine inputs.** The region name in `plotting.yaml`
(`region_variables`), `combine.yaml` (`regions`) and the file names must match
`regions.yaml` exactly. A region with no events passing is skipped by the plotter.

**A process is missing from a region's ROOT file.** The plotter writes a process histogram only
if the process has events there (for example Z(νν) in the Z regions). The datacard writer treats
a missing background as empty.

**Old PKLs.** PKLs written before a change to `regions.yaml` keep the old selection; re-run the
region step after changing a cut. Changing only cross sections, labels, bins or process groups
needs only the plot step.
