# Correction Files — 2024

## LUM

-  `/cvmfs/cms-griddata.cern.ch/cat/metadata/LUM/Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/2025-12-02`

## EGM

-  `/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/2025-12-15`

## MUO

-  `/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/2026-06-18`
   (bumped from 2025-11-27: muon_HighPt.json.gz and muon_JPsi.json.gz both
   changed content between these snapshots — NUM_TightID_DEN_GlobalMuonProbes
   and NUM_probe_TightRelTkIso_DEN_MediumIDProbes (the SFs this analysis
   actually uses) went from internal `"version": 2` to `"version": 3`.
   Neither the 2026-04-28 nor 2026-06-18 changes.md explicitly names
   muon_HighPt.json.gz/muon_JPsi.json.gz (both only call out muon_Z.json.gz
   and muon_scalesmearing*.json.gz, which this analysis doesn't use) — this
   looks like an undocumented regeneration bundled with those releases.
   Took the latest available snapshot as the standing policy: always pull
   the newest POG release rather than pin to a specific dated one.)

## JME

- `/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/2026-07-16`
  (bumped from 2025-12-02/V2: that snapshot predates the V3->V4 L2L3Residual
  bug fix released as V5 on 2026-07-16 — see that directory's changes.md.
  V4 looked tagged correctly but its L2L3Residual payloads were silently
  unchanged from the older version; V5 is the first tag with the real fix.)
  Also added `jetvetomaps.json.gz` from this same snapshot (jet veto maps,
  mandatory for Run 3 analyses — see configs/2024.yaml's `jetvetomaps`
  entry; implemented in CorrectionManager.get_jet_veto_mask but not yet
  wired into event selection).

## BTV

-  `/cvmfs/cms-griddata.cern.ch/cat/metadata/BTV/Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/2026-03-10`
