# Correction Files — 2026

All files are copied from `/cvmfs/cms-griddata.cern.ch/cat/metadata/` (newest dated snapshot, checked 2026-11-05)
and renamed to `Run3-26Prompt-Summer24-NanoAODv15_<name>.json.gz`. The MC is the Summer24 campaign.

## LUM

- Not yet available

## EGM

- `/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/Run3-26Prompt-Summer24-NanoAODv15/2026-06-08`
  - `electronSS_EtDependent.json.gz` only (no electron ID or HLT SF)

## MUO

- Not yet available (the area exists but is empty)

## JME

- `/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-26Prompt-Summer24-NanoAODv15/2026-07-15`
  - `jet_jerc.json.gz` (`Summer24Prompt26_V1_MC` / `_DATA`; JER is run-dependent: `RunBD_JRV1` and `RunC_JRV1`),
    `jetid.json.gz`, `jetvetomaps.json.gz` (`Summer24Prompt26_RunBCD_V1`)

## BTV

- Not yet available

## Golden JSON

- `/eos/user/c/cmsdqm/www/CAF/certification/Collisions26/Cert_Collisions2026_401624_403937_golden.json`
  (copied as `Cert_Collisions2026_401624_403937_golden.json`; not on CVMFS. Note the lowercase `golden` in the name.)

## B-tag efficiency map

- `heavyflavor_efficiency_maps_2026.root` is a copy of `../2024/heavyflavor_efficiency_maps_2024.root` (same Summer24 MC;
  the 2026 UParT working points were not checked, there is no 2026 b-tag file yet). It is the 2024 placeholder map,
  to be replaced by a full-statistics one for all years.
