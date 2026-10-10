#!/usr/bin/env python3
"""
Summarise event-selection jobs (pilot or production) and write the timing JSON
that submit_samples.py --timing uses to pick files-per-job.

Reads:
  - DBL_TIMING lines in logs/<year>/output/*.out (seconds per file, per sample)
  - condor_history for the clusters in logs/<year>/*.log (CPU efficiency, memory)

Usage:
    python3 condorJobs/event-selection/pilot_report.py --year 2024
    python3 condorJobs/event-selection/pilot_report.py --year 2024 --workers 2 \\
        --write condorJobs/event-selection/timing_2024.json
"""

import argparse
import json
import re
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

CONDOR_DIR = Path(__file__).resolve().parent
TIMING_RE = re.compile(r"^DBL_TIMING (.*)$")


def read_timing(log_dir: Path):
    """{(sample, workers): [seconds, ...]} plus {sample: is_data}."""
    times, is_data = defaultdict(list), {}
    for out in sorted((log_dir / "output").glob("*.out")):
        for line in out.read_text(errors="replace").splitlines():
            m = TIMING_RE.match(line)
            if not m:
                continue
            kv = dict(item.split("=", 1) for item in m.group(1).split())
            times[(kv["sample"], int(kv["workers"]))].append(int(kv["seconds"]))
            is_data[kv["sample"]] = kv["data"] == "true"
    return times, is_data


def read_history(log_dir: Path):
    """Per-job resource usage from condor_history for the clusters in log_dir."""
    clusters = sorted({p.name.rsplit(".", 2)[-2] for p in log_dir.glob("*.log")})
    if not clusters:
        return []
    attrs = ["JobBatchName", "RequestCpus", "RemoteWallClockTime", "RemoteUserCpu",
             "RemoteSysCpu", "MemoryUsage", "RequestMemory", "ExitCode", "NumJobStarts"]
    res = subprocess.run(["condor_history", *clusters, "-af", *attrs],
                         capture_output=True, text=True, timeout=300)
    rows = []
    for line in res.stdout.splitlines():
        f = line.split()
        if len(f) != len(attrs):
            continue
        row = dict(zip(attrs, f))
        try:
            cpus, wall = int(row["RequestCpus"]), float(row["RemoteWallClockTime"])
            cpu = float(row["RemoteUserCpu"]) + float(row["RemoteSysCpu"])
        except ValueError:
            continue
        row["cpu_eff"] = cpu / (wall * cpus) if wall > 0 else 0.0
        rows.append(row)
    return rows


def pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--year", required=True)
    p.add_argument("--workers", type=int, default=2, help="Worker setting to write timing for (default: 2)")
    p.add_argument("--write", type=Path, default=None, help="Write timing JSON here")
    args = p.parse_args()

    log_dir = CONDOR_DIR / "logs" / args.year
    times, is_data = read_timing(log_dir)
    if not times:
        raise SystemExit(f"No DBL_TIMING lines in {log_dir}/output/*.out yet")

    print(f"{'sample':<70} {'wrk':>3} {'files':>5} {'median s/file':>13} {'max':>6}")
    for (sample, workers), secs in sorted(times.items()):
        print(f"{sample[:70]:<70} {workers:>3} {len(secs):>5} {statistics.median(secs):>13.0f} {max(secs):>6}")

    rows = read_history(log_dir)
    if rows:
        print(f"\n{'batch':<70} {'cpus':>4} {'wall s':>7} {'cpu eff':>7} {'mem MB':>7} {'req MB':>7} {'exit':>4}")
        for r in sorted(rows, key=lambda r: (r["JobBatchName"], int(r["RequestCpus"]))):
            print(f"{r['JobBatchName'][:70]:<70} {r['RequestCpus']:>4} {float(r['RemoteWallClockTime']):>7.0f} "
                  f"{r['cpu_eff']:>7.0%} {r['MemoryUsage']:>7} {r['RequestMemory']:>7} {r['ExitCode']:>4}")
        by_cpus = defaultdict(list)
        for r in rows:
            by_cpus[int(r["RequestCpus"])].append(r)
        print("\nBy request_cpus:")
        for cpus, rs in sorted(by_cpus.items()):
            mem = [int(r["MemoryUsage"]) for r in rs if r["MemoryUsage"].isdigit()]
            print(f"  {cpus} cpus: {len(rs)} jobs, mean CPU eff {statistics.mean(r['cpu_eff'] for r in rs):.0%}, "
                  f"peak memory {max(mem) if mem else '?'} MB")
        print("  -> core-seconds per file = s/file x cpus; pick the setting with the lowest.")

    if args.write:
        samples = {s: {"sec_per_file": statistics.median(v), "workers": w}
                   for (s, w), v in times.items() if w == args.workers}
        if not samples:
            raise SystemExit(f"No timing at --workers {args.workers}")
        default = {}
        for kind, flag in (("mc", False), ("data", True)):
            vals = [v["sec_per_file"] for s, v in samples.items() if is_data[s] == flag]
            if vals:
                # Conservative for samples the pilot did not cover: 75th percentile.
                default[kind] = {"sec_per_file": pct(vals, 0.75), "workers": args.workers}
        args.write.write_text(json.dumps({"year": args.year, "samples": samples, "default": default}, indent=2) + "\n")
        print(f"\nWrote {args.write} ({len(samples)} samples, defaults: {sorted(default)})")


if __name__ == "__main__":
    main()
