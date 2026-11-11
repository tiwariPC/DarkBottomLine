"""
Deterministic disjoint year-split for MC samples reused (reweighted) across
multiple data-taking years.

Problem: when the same MC sample is processed three times — once per year,
each time with that year's own SF/pileup/JEC/lumi weighting — all three
year-channels in Combine are filled from (nearly) the same underlying events.
`autoMCStats`'s Barlow-Beeston-lite bin nuisances are computed independently
per channel, so the correlation between channels is not accounted for and
the combined MC-stat uncertainty is systematically underestimated.

Fix (same mechanism as a lumi-mask, applied to MC): hash each event's stable
identity (run, luminosityBlock, event) into a uniform u in [0, 1), and assign
it to exactly one year bucket based on cumulative luminosity fraction. Same
event -> same bucket regardless of which year's job computes it, so the
three years' kept subsets are provably disjoint. Each year then only
retains ~f_y of the original statistics and must rescale its per-event
weight by 1/f_y to keep the total predicted yield correct.
"""

from typing import Dict, Sequence

import numpy as np
import pandas as pd


def assign_year_bucket(
    run: np.ndarray,
    luminosity_block: np.ndarray,
    event: np.ndarray,
    years: Sequence[str],
    lumis: Sequence[float],
    salt: int = 12345,
) -> np.ndarray:
    """Deterministic disjoint per-event year assignment.

    Hashes (run, luminosityBlock, event, salt) into uint64, maps to a
    uniform u in [0, 1), and buckets by cumulative lumi fraction over
    `years`/`lumis` (same order). Returns an array of strings from `years`,
    one per input event.
    """
    run = np.asarray(run, dtype=np.uint64)
    luminosity_block = np.asarray(luminosity_block, dtype=np.uint64)
    event = np.asarray(event, dtype=np.uint64)

    key = (
        run * np.uint64(1000003)
        + luminosity_block * np.uint64(10007)
        + event
        + np.uint64(salt)
    )
    u = pd.util.hash_array(key).astype(np.float64) / 2.0**64

    lumis_arr = np.asarray(lumis, dtype=np.float64)
    edges = np.cumsum(lumis_arr) / np.sum(lumis_arr)
    idx = np.searchsorted(edges, u, side="right")
    idx = np.minimum(idx, len(years) - 1)
    return np.asarray(years)[idx]


def year_lumi_fraction(target_year: str, lumi_by_year: Dict[str, float]) -> float:
    """f_y = L_y / sum(L_y' for all configured years')."""
    total = sum(lumi_by_year.values())
    if total <= 0:
        raise ValueError(f"Total lumi across {list(lumi_by_year)} must be > 0, got {total}")
    return float(lumi_by_year[target_year]) / float(total)
