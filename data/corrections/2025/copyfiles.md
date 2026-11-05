# Correction Files — 2025

All files are copied from `/cvmfs/cms-griddata.cern.ch/cat/metadata/` (newest dated snapshot, checked 2026-11-05)
and renamed to `Run3-25Prompt-Summer24-NanoAODv15_<name>.json.gz`. The MC is the Summer24 campaign
(the same samples as 2024), so the Summer24 JME set is used, not Winter25.

## LUM

- `/cvmfs/cms-griddata.cern.ch/cat/metadata/LUM/Run3-25Prompt-Summer24-NanoAODv15/2026-06-05`
  - `puWeights_2025pp_Golden_Summer24_25ns_69200ub.json.gz` -> `_puWeights.json.gz` (correction `Collisions25_goldenJSON`)
  - no luminosity value is provided there

## EGM

- `/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/Run3-25Prompt-Summer24-NanoAODv15/2026-06-26`
  - `electron.json.gz` (`Electron-ID-SF`, year key `2025Prompt`), `electronSS_EtDependent.json.gz`
  - no electron HLT SF exists for 2025 (the 2024 EGM area has `electronHlt.json.gz`, this one does not)

## MUO

- `/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/Run3-25Prompt-Summer24-NanoAODv15/2026-10-27`
  - `muon_HighPt.json.gz`, `muon_JPsi.json.gz` (low-pT SFs were added in this snapshot)
  - `NUM_TightPFIso_DEN_TightID` is not in `muon_HighPt`; it has e.g. `NUM_TightID_DEN_GlobalMuonProbes`

## JME

- `/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-25Prompt-Summer24-NanoAODv15/2026-07-16`
  - `jet_jerc.json.gz` (tags `Summer24Prompt25_V3_MC` / `Summer24Prompt25_V3_DATA`), `jetid.json.gz`,
    `jetvetomaps.json.gz` (`Summer24Prompt25_RunCDEFG_V1`)
- `Run3-25Prompt-Winter25-NanoAODv15_jet_jerc.json.gz` (Winter25Prompt25_V3, from `JME/Run3-25Prompt-Winter25-NanoAODv15/2026-02-09`)
  is kept for reference only; it matches the Winter25 MC, not the Summer24 MC used here.

## BTV

- `/cvmfs/cms-griddata.cern.ch/cat/metadata/BTV/Run3-25Prompt-Summer24-NanoAODv15/2026-10-14`
  - `btagging.json.gz` with `UParTAK4_comb`, `UParTAK4_mujets`, `UParTAK4_light` (the earlier copy had working points only)

## Not available for 2025

- Electron HLT SF, MET trigger SF (`Run3-Summer24-NanoAODv15_metHlt` exists for 2024 only).

## Golden JSON

- `/eos/user/c/cmsdqm/www/CAF/certification/Collisions25/Cert_Collisions2025_391658_398903_Golden.json`
  (copied as `Cert_Collisions2025_391658_398903_Golden.json`; not on CVMFS. Replaced the earlier copy with the
  official file of 2026-07-30, 513 runs.)

## B-tag efficiency map

- `heavyflavor_efficiency_maps_2025.root` is a copy of `../2024/heavyflavor_efficiency_maps_2024.root`: the map is built from MC only
  and the MC samples are the same Summer24 ones, with identical UParT working points in the 2024 and 2025 b-tag files.
  It is the 2024 placeholder map (built from a few local test samples), to be replaced by a full-statistics one for all years.
