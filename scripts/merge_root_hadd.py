#!/usr/bin/env python3
"""Merge ROOT files with hadd, grouped by common prefix up to a marker token."""

import os
import argparse
import subprocess
import re


def execute(cmd, dry_run):
    output = cmd[2]
    inputs = cmd[3:]
    if dry_run:
        print(f"  merge {len(inputs)} file(s) → {os.path.basename(output)}")
        for f in inputs:
            print(f"    {os.path.basename(f)}")
    else:
        print(f"  MERGING {len(inputs)} file(s) → {os.path.basename(output)} ...")
        subprocess.run(cmd, check=True)
        print(f"  done → {os.path.basename(output)}")


def main():
    parser = argparse.ArgumentParser(description="Group and merge ROOT files by dataset name.")
    parser.add_argument("-i", "--input", required=True, help="Input directory")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument("--marker", required=True, help="Marker to filter files")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    args = parser.parse_args()

    if not os.path.exists(args.output):
        if args.dry_run:
            print(f"[DRY-RUN] Create directory: {args.output}")
        else:
            os.makedirs(args.output)

    pattern = re.compile(rf"^(.*)_{args.marker}")

    groups = {}
    files = [f for f in os.listdir(args.input) if f.endswith(".root") and args.marker in f]
    for f in files:
        match = pattern.match(f)
        if match:
            dataset_name = match.group(1)
            groups.setdefault(dataset_name, []).append(os.path.join(args.input, f))

    mode = "DRY RUN" if args.dry_run else "RUNNING"
    print(f"\n[{mode}] {len(groups)} dataset(s) to merge\n")

    batch_size = 500
    for dataset, file_list in sorted(groups.items()):
        print(f"[{dataset}]")
        output_file = os.path.join(args.output, f"{dataset}.root")

        if len(file_list) > batch_size:
            intermediates = []
            for i in range(0, len(file_list), batch_size):
                batch = file_list[i : i + batch_size]
                tmp = os.path.join(args.output, f"tmp_{dataset}_{i // batch_size}.root")
                intermediates.append(tmp)
                execute(["hadd", "-f", tmp] + batch, args.dry_run)
            execute(["hadd", "-f", output_file] + intermediates, args.dry_run)
            if not args.dry_run:
                for tmp in intermediates:
                    os.remove(tmp)
        else:
            execute(["hadd", "-f", output_file] + file_list, args.dry_run)


if __name__ == "__main__":
    main()
