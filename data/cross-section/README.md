# xsection comparison (XSDB vs Twiki)

This table compares xsection values from different sources for the same process.

| Process / Dataset | XSDB xsection (pb) | Twiki xsection (pb) | Notes |
|---|---:|---:|---|
| VBF | 0.628 | 4.078 | |
| ttH | 0.5381 | 0.5688 | |
| Top | 762.1 | 923.6 | |
| SingleTop | 275.0 | 232.2 | |


## Single top: decay-filtered 2024 samples

The 2024 single-top samples are decay-filtered (checked at generator level:
every `TBbarQtoLNu`/`TbarBQtoLNu` t-channel event has a leptonic top decay,
every `TWminustoLNu2Q`/`TbarWplustoLNu2Q` event has exactly one leptonic and
one hadronic W), so their cross sections include the W branching ratios,
with BR(W→ℓν, ℓ=e,μ,τ) = 3 × 0.1086 = 0.3258:

| Dataset | Inclusive (pb) | BR | Used (pb) |
|---|---:|---:|---:|
| TBbarQtoLNu-t-channel-4FS | 123.8 | 0.3258 | 38.6 (XSDB, Summer24) |
| TbarBQtoLNu-t-channel-4FS | 75.47 | 0.3258 | 23.34 (XSDB, Summer24) |
| TWminustoLNu2Q | 35.99 | 2 × 0.3258 × 0.6742 = 0.4393 | 15.81 |
| TbarWplustoLNu2Q | 36.05 | 0.4393 | 15.84 |

The inclusive 2022 t-channel names (`TBbarQ_t-channel_4FS`,
`TbarBQ_t-channel_4FS`) keep the inclusive values. The s-channel entries are
already leptonic-only.

## tt̄ decay-specific samples

XSDB lists 762.1 pb (inclusive tt̄) for each decay-specific 2022 `TTto…` dataset,
so the cross sections used here are 762.1 pb times the W branching ratios
(BR(W→ℓν) = 0.3258, BR(W→qq) = 0.6742):

| Dataset | BR | Used (pb) |
|---|---:|---:|
| TTto2L2Nu | 0.3258² = 0.1061 | 80.89 |
| TTtoLNu2Q | 2 × 0.3258 × 0.6742 = 0.4393 | 334.8 |
| TTto4Q | 0.6742² = 0.4545 | 346.41 |

Earlier values were 82.0 / 346.3 / 333.8, with the TTtoLNu2Q and TTto4Q numbers swapped.

## Process names and XSDB

Every `process` in `xsection_background_run3.json` is the full XSDB `process_name`, and
`xsection_xsdb.yaml` holds the matching XSDB record (id, DAS, MCM, uncertainty, ...) for each
one. The cross-section depends only on the 13.6 TeV process, so the dataset names of other
years that correspond to the same process are listed in `full_dataset` of that single entry.

- W+jets and Z→νν 2J bins: one entry per bin, named after the 2024 XSDB entry
  (`...Bin-2J-PTLNu-...`, `...Bin-2J-PTNuNu-...`), with the 2022/2022EE/2023 names
  (`..._PTLNu-..._2J`, `..._PTNuNu-..._2J`) also in `full_dataset`. The 2024 XSDB value is used
  for all years. This lowered the 2022/2022EE/2023 cross sections by 0.2-2.6 % relative to the
  separate 2022 XSDB values (e.g. W+jets PTLNu-100to200: 421.9 -> 411.1 pb).
- DY `MLL-50_PTLL-..._2J`: named after the 2022 XSDB entry, same value as before. The 2024
  samples use the inclusive `DYto2L-2Jets_Bin-MLL-50-PTLL-{100,200,400,600}` instead.
- Inclusive t-channel `TBbarQ_t-channel_4FS` / `TbarBQ_t-channel_4FS` stay separate from the
  decay-filtered `TBbarQtoLNu` / `TbarBQtoLNu` samples (different physical samples).
- tt̄ and tW decay-specific samples: XSDB lists the inclusive value, the JSON holds it times
  the W branching ratio (see the tables above and below).

## Field Notes

- `XSDB xsection (pb)`: Cross section taken from XSDB, in pb.
- `Twiki xsection (pb)`: Cross section taken from the analysis Twiki or group recommendation page, in pb.
- `Notes`: Add source links, version numbers, update dates, and applicable campaign information.
