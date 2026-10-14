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
| TBbarQtoLNu-t-channel-4FS | 123.8 | 0.3258 | 40.33 |
| TbarBQtoLNu-t-channel-4FS | 75.47 | 0.3258 | 24.59 |
| TWminustoLNu2Q | 35.99 | 2 × 0.3258 × 0.6742 = 0.4393 | 15.81 |
| TbarWplustoLNu2Q | 36.05 | 0.4393 | 15.84 |

The inclusive 2022 t-channel names (`TBbarQ_t-channel_4FS`,
`TbarBQ_t-channel_4FS`) keep the inclusive values. The s-channel entries are
already leptonic-only.

## Field Notes

- `XSDB xsection (pb)`: Cross section taken from XSDB, in pb.
- `Twiki xsection (pb)`: Cross section taken from the analysis Twiki or group recommendation page, in pb.
- `Notes`: Add source links, version numbers, update dates, and applicable campaign information.
