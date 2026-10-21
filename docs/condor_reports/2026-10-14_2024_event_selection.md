# 14 Oct 2026 — 2024 event-selection production on lxplus HTCondor (tag 20261014-d110783)

Full 2024 NanoAOD → `EVENTSELECTION.root` run: 79 samples, 2,288 jobs, 33,722 files, all successful, 3.6 h wall-clock, 4,051 core-hours, 88 GB of output.

- **Date run:** 14 Oct 2026, first job submitted 08:58, last job finished 12:34
- **Code version:** `20261014-d110783` (`darkbottomline/_version.py`)
- **Output:** `/eos/cms/store/group/phys_susy/sus-23-008/run3_bbMET/20261014-d110783_14102026/2024/eventsel/<sample>/`
- **Command:** `python3 condorJobs/event-selection/submit_samples.py --year 2024 --batch 15 --cpus 2 --memory 4000 --outdir /eos/cms/store/group/phys_susy/sus-23-008/run3_bbMET/`
- **Source of numbers:** HTCondor event logs (`condorJobs/event-selection/logs/2024/*.log`, clusters 20910657–20910735) and `DBL_TIMING` lines in the job outputs. CPU efficiency = remote CPU time / (wall time × 2 cores). Memory is Condor's sampled value, not an exact peak.

## 1. Setup
| | |
|---|---|
| Scope | 79 samples: 32 data (EGamma0/1, JetMET0/1, Run2024C–I), 46 MC, 1 signal (BBDM-2HDMa); Muon0/1 skipped |
| Size | 79 clusters, 2,288 jobs, 33,722 NanoAOD files |
| Per job | 15 files run sequentially, 2 cpus, 4000 MB requested (lxplus allocated 6000 MB), `workday` flavour (8 h limit) |
| Output | 33,722 ROOT files, 88 GB, no duplicates, none truncated |

## 2. Outcome
- All 2,288 jobs finished with return value 0; no holds, evictions or restarts.
- 33,722 files done, 0 failed, 0 skipped.
- Total time 3.6 h; peak 2,109 jobs running at once (4,218 cores).
- Queue wait: median 0.3 h, p90 0.5 h, max 0.6 h.

## 3. Time
| | min | median | mean | p90 | max |
|---|---|---|---|---|---|
| Job wall time (min) | 1.6 | 40.8 | 53.1 | 102 | 190 |
| Seconds per file | – | 157 | – | 420 | 3,200 |

Longest job 3.2 h against the 8 h limit. Data median 321 s/file, MC 133 s/file.

## 4. Resources
| | |
|---|---|
| Wall-clock total | 2,025 job-hours = 4,051 core-hours |
| CPU used | 2,237 core-hours (55% overall efficiency; per-job median 48%, p90 72%, max 84%) |
| Memory | median 1.7 GB, p90 2.1 GB, max 6.2 GB |

| Kind | Samples | Jobs | Files | Core-hours | CPU eff. | Median s/file | Output |
|---|---|---|---|---|---|---|---|
| Data | 32 | 626 | 9,141 | 1,679 | 68% | 321 | 3.4 GB |
| MC | 46 | 1,660 | 24,557 | 2,368 | 46% | 133 | 89.4 GB |
| Signal | 1 | 2 | 24 | 3.3 | 55% | 258 | 1.0 GB |

CPU efficiency is below full because jobs wait on remote xrootd reads, so extra cores would not help. Four high-pT MC samples use close to the 6 GB allocation: Zto2Nu-1J 400to600 (median 5.9 GB), WtoLNu-1J 600 (4.7 GB), Zto2Nu-1J 600 (max 6.2 GB, above the allocation) and Zto2Nu-2J 600 (max 3.8 GB).

## 5. Reliability
862 of 2,288 jobs logged transient xrootd read errors (mostly `Operation expired`). The fallback sources (EOS replica → INFN → FNAL) handled them: every file still completed and no job was held.

## 6. Per-sample details
Sorted by core-hours. Times in seconds (per file) or minutes (per job); memory in MB.

| sample | kind | files | jobs | med s/file | p90 | max | med job min | max job min | core-h | eff% | med mem MB | max mem MB | out GB | MB/file |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| WtoLNu-2Jets_Bin-1J-PTLNu-40to100 | MC | 3384 | 226 | 133.0 | 219 | 781 | 37.7 | 58.5 | 288.4 | 35.5 | 1738.0 | 2410 | 0.5 | 0.1 |
| WtoLNu-2Jets_Bin-1J-PTLNu-100to200 | MC | 3089 | 206 | 133 | 188 | 3200 | 35.2 | 131.1 | 245.5 | 43.5 | 1797.5 | 2379 | 0.7 | 0.2 |
| WtoLNu-2Jets_Bin-2J-PTLNu-100to200 | MC | 3262 | 218 | 120.0 | 176 | 407 | 31.0 | 69.2 | 236.8 | 46.1 | 1790.5 | 2439 | 0.8 | 0.2 |
| WtoLNu-2Jets_Bin-2J-PTLNu-40to100 | MC | 3309 | 221 | 110 | 159 | 339 | 28.6 | 67.0 | 215.1 | 46.1 | 1740 | 2131 | 0.5 | 0.1 |
| EGamma0-Run2024G-MINIv6NANOv15-v2 | DATA | 850 | 57 | 404.5 | 563 | 794 | 98.1 | 150.3 | 196.9 | 75.5 | 1723 | 2268 | 0.2 | 0.2 |
| EGamma1-Run2024G-MINIv6NANOv15-v2 | DATA | 888 | 60 | 368.0 | 545 | 1200 | 90.2 | 169.4 | 190.0 | 73.9 | 1714.5 | 1980 | 0.2 | 0.2 |
| DYto2L-2Jets_Bin-2J-MLL-50-PTLL-40to100 | MC | 1787 | 120 | 151 | 228 | 1381 | 40.5 | 83.0 | 166.1 | 42.9 | 1785.0 | 2034 | 0.3 | 0.1 |
| TTto4Q | MC | 773 | 52 | 376 | 531 | 766 | 94.1 | 138.6 | 164.1 | 47.6 | 2223.0 | 2607 | 0.1 | 0.2 |
| Zto2Nu-2Jets_Bin-1J-PTNuNu-40to100 | MC | 1341 | 90 | 153 | 245 | 743 | 42.5 | 63.4 | 129.4 | 35.4 | 1732.5 | 2114 | 0.2 | 0.1 |
| JetMET0-Run2024G-MINIv6NANOv15-v2 | DATA | 443 | 30 | 508 | 665 | 997 | 126.7 | 185.0 | 124.4 | 67.3 | 2329.5 | 2660 | 0.3 | 0.8 |
| EGamma0-Run2024F-MINIv6NANOv15-v1 | DATA | 719 | 48 | 299 | 428 | 678 | 76.4 | 130.4 | 121.6 | 73.3 | 1710.5 | 1857 | 0.2 | 0.2 |
| TTtoLNu2Q | MC | 790 | 53 | 261.0 | 411 | 709 | 65.4 | 106.4 | 121.4 | 68.2 | 1910 | 2198 | 3.4 | 4.3 |
| EGamma1-Run2024F-MINIv6NANOv15-v1 | DATA | 747 | 50 | 273 | 413 | 582 | 67.7 | 125.6 | 119.5 | 72.7 | 1699.5 | 1738 | 0.2 | 0.2 |
| Zto2Nu-2Jets_Bin-2J-PTNuNu-100to200 | MC | 354 | 24 | 514.5 | 768 | 2280 | 137.1 | 190.1 | 108.6 | 44.7 | 1969.0 | 2259 | 0.3 | 0.9 |
| TTto2L2Nu | MC | 780 | 52 | 234.0 | 325 | 532 | 59.2 | 109.8 | 104.8 | 65.4 | 1807.0 | 2193 | 2.5 | 3.2 |
| JetMET1-Run2024G-MINIv6NANOv15-v2 | DATA | 451 | 31 | 413 | 520 | 685 | 100.5 | 126.9 | 99.0 | 58.0 | 1760 | 2067 | 0.3 | 0.8 |
| DYto2L-2Jets_Bin-2J-MLL-50-PTLL-100to200 | MC | 389 | 26 | 431 | 652 | 1275 | 116.0 | 161.7 | 96.4 | 43.3 | 1705.0 | 1974 | 0.2 | 0.5 |
| JetMET1-Run2024F-MINIv6NANOv15-v2 | DATA | 381 | 26 | 416 | 616 | 847 | 105.4 | 158.1 | 90.5 | 75.6 | 1751.5 | 1800 | 0.3 | 0.7 |
| JetMET0-Run2024F-MINIv6NANOv15-v2 | DATA | 381 | 26 | 413 | 612 | 773 | 102.3 | 150.7 | 90.0 | 65.6 | 2490.5 | 2622 | 0.3 | 0.7 |
| Zto2Nu-2Jets_Bin-2J-PTNuNu-40to100 | MC | 1412 | 95 | 103.0 | 145 | 295 | 25.9 | 44.6 | 85.7 | 56.8 | 1468 | 1865 | 0.2 | 0.1 |
| Zto2Nu-2Jets_Bin-1J-PTNuNu-100to200 | MC | 344 | 23 | 419.5 | 676 | 1053 | 115.5 | 155.8 | 85.6 | 42.8 | 1674 | 1904 | 0.2 | 0.6 |
| EGamma1-Run2024E-MINIv6NANOv15-v1 | DATA | 335 | 23 | 260 | 426 | 597 | 55.5 | 105.9 | 50.8 | 71.3 | 1654 | 1737 | 0.1 | 0.2 |
| JetMET1-Run2024E-MINIv6NANOv15-v1 | DATA | 228 | 16 | 388.5 | 596 | 928 | 93.6 | 129.4 | 46.7 | 44.6 | 1946.0 | 2411 | 0.1 | 0.5 |
| EGamma0-Run2024E-MINIv6NANOv15-v1 | DATA | 267 | 18 | 310 | 417 | 584 | 78.5 | 114.1 | 45.2 | 71.8 | 1710.5 | 1734 | 0.1 | 0.2 |
| TBbarQtoLNu-t-channel-4FS | MC | 391 | 27 | 153 | 205 | 776 | 38.9 | 56.8 | 34.9 | 42.1 | 1645 | 1921 | 0.1 | 0.4 |
| EGamma1-Run2024C-MINIv6NANOv15-v1 | DATA | 247 | 17 | 227 | 441 | 572 | 54.0 | 100.2 | 33.8 | 70.3 | 1701 | 1746 | 0.1 | 0.2 |
| WW | MC | 518 | 35 | 104.0 | 159 | 377 | 28.1 | 36.5 | 32.5 | 39.2 | 1522 | 2086 | 0.2 | 0.4 |
| EGamma0-Run2024I-MINIv6NANOv15_v2-v1 | DATA | 198 | 14 | 277.0 | 438 | 579 | 72.9 | 107.7 | 31.4 | 71.5 | 1685.0 | 1736 | 0.0 | 0.2 |
| JetMET0-Run2024E-MINIv6NANOv15-v1 | DATA | 231 | 16 | 239 | 345 | 649 | 58.0 | 81.8 | 29.9 | 69.9 | 1637.5 | 1733 | 0.1 | 0.5 |
| JetMET0-Run2024D-MINIv6NANOv15-v1 | DATA | 151 | 11 | 364 | 556 | 985 | 86.6 | 158.7 | 29.9 | 44.7 | 1919 | 2210 | 0.1 | 0.5 |
| EGamma0-Run2024C-MINIv6NANOv15-v1 | DATA | 167 | 12 | 317 | 595 | 770 | 80.6 | 145.8 | 29.5 | 72.1 | 1740.0 | 1769 | 0.0 | 0.2 |
| EGamma1-Run2024D-MINIv6NANOv15-v1 | DATA | 279 | 19 | 198 | 295 | 532 | 48.4 | 71.0 | 29.0 | 67.3 | 1610 | 1716 | 0.1 | 0.2 |
| EGamma1-Run2024H-MINIv6NANOv15-v1 | DATA | 191 | 13 | 246 | 451 | 583 | 62.8 | 111.9 | 28.5 | 72.9 | 1656 | 1739 | 0.0 | 0.2 |
| EGamma0-Run2024D-MINIv6NANOv15-v1 | DATA | 173 | 12 | 304 | 397 | 559 | 69.2 | 106.2 | 27.5 | 70.9 | 1724.0 | 1759 | 0.0 | 0.2 |
| EGamma1-Run2024I-MINIv6NANOv15_v2-v1 | DATA | 179 | 12 | 245 | 400 | 569 | 58.8 | 107.8 | 26.2 | 70.1 | 1675.0 | 1727 | 0.0 | 0.2 |
| EGamma1-Run2024I-MINIv6NANOv15-v1 | DATA | 176 | 12 | 251.0 | 384 | 559 | 60.9 | 114.4 | 26.0 | 71.4 | 1638.5 | 1722 | 0.0 | 0.2 |
| JetMET0-Run2024I-MINIv6NANOv15_v2-v1 | DATA | 179 | 12 | 242 | 372 | 601 | 58.4 | 100.5 | 25.3 | 52.9 | 2134.0 | 2434 | 0.1 | 0.4 |
| EGamma0-Run2024I-MINIv6NANOv15-v1 | DATA | 211 | 15 | 206 | 254 | 332 | 50.4 | 65.0 | 24.0 | 67.6 | 1599 | 1654 | 0.0 | 0.2 |
| JetMET1-Run2024C-MINIv6NANOv15-v1 | DATA | 110 | 8 | 368.5 | 691 | 792 | 79.7 | 145.8 | 23.9 | 44.9 | 1858.0 | 2611 | 0.1 | 0.6 |
| EGamma0-Run2024H-MINIv6NANOv15-v2 | DATA | 138 | 10 | 320.0 | 407 | 444 | 75.1 | 88.1 | 23.1 | 72.1 | 1717.0 | 1762 | 0.0 | 0.2 |
| WZ | MC | 307 | 21 | 120 | 188 | 312 | 33.7 | 44.5 | 22.3 | 43.5 | 1764 | 2349 | 0.1 | 0.4 |
| DYto2L-2Jets_Bin-2J-MLL-50-PTLL-200to400 | MC | 139 | 10 | 251 | 505 | 756 | 64.2 | 89.0 | 21.5 | 40.4 | 2316.0 | 2723 | 4.7 | 33.6 |
| JetMET0-Run2024C-MINIv6NANOv15-v1 | DATA | 161 | 11 | 206 | 423 | 584 | 54.8 | 114.4 | 20.6 | 57.5 | 2000 | 2350 | 0.1 | 0.5 |
| WtoLNu-2Jets_Bin-2J-PTLNu-200to400 | MC | 136 | 10 | 262.5 | 403 | 484 | 62.0 | 91.6 | 20.3 | 52.8 | 2341.5 | 2624 | 10.7 | 79.0 |
| JetMET0-Run2024H-MINIv6NANOv15-v2 | DATA | 82 | 6 | 443.0 | 663 | 740 | 111.5 | 124.4 | 20.0 | 43.0 | 2055.0 | 2485 | 0.1 | 0.6 |
| JetMET1-Run2024D-MINIv6NANOv15-v1 | DATA | 155 | 11 | 218 | 359 | 468 | 53.6 | 84.3 | 19.2 | 69.3 | 1622 | 1710 | 0.1 | 0.5 |
| TWminustoLNu2Q | MC | 252 | 17 | 133.0 | 157 | 208 | 30.6 | 38.5 | 17.7 | 55.2 | 1486 | 1590 | 0.3 | 1.3 |
| JetMET1-Run2024I-MINIv6NANOv15_v2-v2 | DATA | 116 | 8 | 235.5 | 362 | 455 | 60.8 | 65.0 | 15.6 | 65.5 | 1713.5 | 1767 | 0.1 | 0.5 |
| JetMET0-Run2024I-MINIv6NANOv15-v2 | DATA | 149 | 10 | 173 | 285 | 366 | 45.3 | 60.7 | 14.7 | 67.3 | 1605.0 | 1747 | 0.1 | 0.4 |
| TbarBQtoLNu-t-channel-4FS | MC | 249 | 17 | 95 | 137 | 442 | 26.7 | 34.2 | 14.6 | 51.0 | 1438 | 1523 | 0.1 | 0.3 |
| Zto2Nu-2Jets_Bin-2J-PTNuNu-200to400 | MC | 102 | 7 | 244.5 | 423 | 572 | 59.5 | 75.7 | 14.3 | 49.7 | 2023 | 2186 | 9.1 | 88.9 |
| TbarWplustoLNu2Q | MC | 252 | 17 | 94.5 | 141 | 197 | 23.1 | 36.7 | 13.8 | 53.8 | 1473 | 1712 | 0.3 | 1.4 |
| JetMET1-Run2024H-MINIv6NANOv15-v2 | DATA | 76 | 6 | 346.0 | 448 | 516 | 76.9 | 94.9 | 13.5 | 60.7 | 1763.0 | 2040 | 0.1 | 0.7 |
| JetMET1-Run2024I-MINIv6NANOv15-v1 | DATA | 82 | 6 | 295.5 | 446 | 484 | 67.3 | 102.5 | 13.3 | 62.8 | 1862.5 | 2009 | 0.1 | 0.7 |
| WtoLNu-2Jets_Bin-1J-PTLNu-200to400 | MC | 109 | 8 | 175 | 332 | 472 | 47.5 | 58.8 | 11.6 | 49.9 | 1758.5 | 2310 | 5.2 | 48.1 |
| Zto2Nu-2Jets_Bin-1J-PTNuNu-200to400 | MC | 70 | 5 | 222.0 | 568 | 846 | 64.3 | 103.7 | 11.3 | 33.4 | 2458 | 3449 | 3.5 | 49.4 |
| Zto2Nu-2Jets_Bin-1J-PTNuNu-400to600 | MC | 53 | 4 | 273 | 738 | 926 | 76.2 | 88.7 | 10.3 | 44.1 | 5914.5 | 6083 | 5.4 | 102.8 |
| WtoLNu-2Jets_Bin-1J-PTLNu-400to600 | MC | 62 | 5 | 194.5 | 435 | 804 | 57.2 | 80.2 | 8.6 | 31.1 | 1941 | 2447 | 4.2 | 67.7 |
| WtoLNu-2Jets_Bin-2J-PTLNu-400to600 | MC | 51 | 4 | 289 | 439 | 642 | 55.5 | 89.9 | 7.7 | 57.3 | 2716.0 | 3598 | 4.7 | 91.5 |
| WtoLNu-2Jets_Bin-2J-PTLNu-600 | MC | 67 | 5 | 143 | 325 | 422 | 42.0 | 56.9 | 7.0 | 43.6 | 1897 | 2490 | 4.8 | 71.8 |
| DYto2L-2Jets_Bin-2J-MLL-50-PTLL-400to600 | MC | 47 | 4 | 200 | 519 | 577 | 53.3 | 88.3 | 6.8 | 45.3 | 1927.5 | 2077 | 2.2 | 46.8 |
| WtoLNu-2Jets_Bin-1J-PTLNu-600 | MC | 27 | 2 | 401 | 639 | 649 | 96.9 | 118.8 | 6.5 | 54.7 | 4712.5 | 4767 | 4.5 | 165.5 |
| Zto2Nu-2Jets_Bin-1J-PTNuNu-600 | MC | 52 | 4 | 155.5 | 384 | 559 | 53.8 | 58.5 | 6.3 | 60.4 | 3683.5 | 6248 | 4.9 | 93.6 |
| Zto2Nu-2Jets_Bin-2J-PTNuNu-600 | MC | 66 | 5 | 133.5 | 277 | 547 | 42.0 | 47.2 | 6.0 | 50.8 | 1960 | 3841 | 6.2 | 93.6 |
| TBbartoLplusNuBbar-s-channel-4FS | MC | 94 | 7 | 111.0 | 128 | 158 | 26.5 | 29.7 | 5.7 | 44.3 | 1559 | 1735 | 0.0 | 0.4 |
| Zto2Nu-2Jets_Bin-2J-PTNuNu-400to600 | MC | 66 | 5 | 122.0 | 249 | 360 | 34.4 | 39.2 | 5.1 | 57.3 | 1961 | 3507 | 6.3 | 96.0 |
| DYto2L-2Jets_Bin-2J-MLL-50-PTLL-600 | MC | 50 | 4 | 114.5 | 412 | 500 | 38.8 | 60.0 | 5.1 | 58.0 | 1748.5 | 1863 | 1.8 | 36.3 |
| ZZ | MC | 76 | 6 | 92.5 | 151 | 217 | 26.1 | 30.3 | 4.5 | 27.1 | 1014.5 | 1735 | 0.0 | 0.3 |
| VBFH-Hto2B_Par-M-125 | MC | 40 | 3 | 109.5 | 425 | 582 | 45.8 | 59.9 | 4.3 | 56.6 | 1655 | 1706 | 0.0 | 0.1 |
| GluGluH-Hto2B_Par-M-125 | MC | 36 | 3 | 123.5 | 376 | 554 | 45.0 | 67.6 | 4.2 | 55.0 | 1974 | 2212 | 0.0 | 0.1 |
| TbarBtoLminusNuB-s-channel-4FS | MC | 68 | 5 | 96.0 | 129 | 185 | 23.6 | 28.1 | 3.9 | 45.4 | 1631 | 1802 | 0.0 | 0.3 |
| BBDM-2HDMa-5f | SIGNAL | 24 | 2 | 258.0 | 305 | 570 | 49.6 | 56.8 | 3.3 | 54.6 | 1593.5 | 1595 | 1.0 | 43.5 |
| WplusH-WtoLNu-Hto2B_Par-M-125 | MC | 28 | 2 | 167.5 | 319 | 400 | 42.2 | 51.5 | 2.8 | 34.9 | 1709.5 | 1757 | 0.0 | 1.3 |
| ZH-Zto2L-Hto2B_Par-M-125 | MC | 32 | 3 | 116.5 | 186 | 329 | 32.6 | 38.0 | 2.6 | 27.2 | 956 | 1511 | 0.0 | 0.3 |
| ZH-Zto2Nu-Hto2B_Par-M-125 | MC | 25 | 2 | 99 | 224 | 459 | 29.6 | 30.6 | 2.0 | 31.4 | 1634.5 | 1726 | 0.0 | 0.7 |
| GluGluZH-Zto2L-Hto2B_Par-M-125 | MC | 21 | 2 | 157 | 204 | 231 | 28.5 | 41.3 | 1.9 | 24.2 | 936.0 | 966 | 0.0 | 0.5 |
| TTH-Hto2B_Par-M-125 | MC | 22 | 2 | 106.0 | 286 | 418 | 27.7 | 29.0 | 1.8 | 53.0 | 1736.5 | 1842 | 0.0 | 1.0 |
| WminusH-WtoLNu-Hto2B_Par-M-125 | MC | 20 | 2 | 105.0 | 305 | 362 | 22.3 | 36.6 | 1.5 | 46.6 | 961.5 | 1438 | 0.0 | 1.0 |
| GluGluZH-Zto2Nu-Hto2B_Par-M-125 | MC | 15 | 1 | 90 | 147 | 250 | 27.6 | 27.6 | 0.9 | 42.6 | 974 | 974 | 0.0 | 2.3 |

Not validated here: physics content (weights, cutflow) against the golden JSON.
