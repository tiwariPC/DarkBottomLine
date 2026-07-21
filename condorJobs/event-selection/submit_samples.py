#!/usr/bin/env python3
"""
Simple script to submit condor jobs for event-selection sample files.

For each sample file (*.txt in data/samplelist/<year>/):
  - Counts the number of ROOT files in it
  - Submits N condor jobs (one job per file)
  - Each sample gets its own condor cluster
  - Runs `darkbottomline analyze --mode event-selection` (NanoAOD -> EVENTSELECTION.root)

Usage:
    python3 submit_samples.py --year 2022 [options]

Example:
    python3 submit_samples.py --year 2022 --config configs/2022.yaml
"""

import argparse
import logging
import os
import pwd
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Union

# Parse chunk size argument (simple inline function - no external dependency needed)
def parse_chunk_size_arg(chunk_size_str: str) -> Optional[int]:
    """
    Parse chunk size argument, accepting 'auto' or an integer.

    Args:
        chunk_size_str: String value from command line

    Returns:
        None if 'auto', otherwise integer chunk size

    Raises:
        ValueError: If chunk_size_str is neither 'auto' nor a valid integer
    """
    if chunk_size_str.lower() == 'auto':
        return None
    try:
        return int(chunk_size_str)
    except ValueError:
        raise ValueError(f"Invalid chunk-size value: {chunk_size_str}. Must be 'auto' or an integer.")


def get_request_memory_from_template(template_file: Path) -> int:
    """
    Extract request_memory value from Condor submit template file.

    Args:
        template_file: Path to submit.sub template

    Returns:
        Memory request in MB (default: 8000)
    """
    try:
        with open(template_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('request_memory'):
                    # Parse: request_memory = 8000
                    parts = line.split('=')
                    if len(parts) == 2:
                        try:
                            return int(parts[1].strip())
                        except ValueError:
                            pass
    except Exception:
        pass

    # Default fallback
    return 8000


def get_request_cpus_from_template(template_file: Path) -> int:
    """
    Extract request_cpus value from Condor submit template file.

    Args:
        template_file: Path to submit.sub template

    Returns:
        Number of CPUs/workers (default: 4)
    """
    try:
        with open(template_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('request_cpus'):
                    # Parse: request_cpus = 4
                    parts = line.split('=')
                    if len(parts) == 2:
                        try:
                            return int(parts[1].strip())
                        except ValueError:
                            pass
    except Exception:
        pass

    # Default fallback
    return 4


def count_files_in_sample(sample_file: Path) -> int:
    """Count non-empty, non-comment lines in a sample file."""
    count = 0
    with open(sample_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                count += 1
    return count


def find_sample_files(input_dir: Path) -> List[Path]:
    """Find all *.txt files in the input directory."""
    sample_files = sorted(input_dir.glob('*.txt'))
    return sample_files


def create_submit_file(
    template_file: Path,
    output_file: Path,
    sample_file: str,
    num_jobs: int,
    config: str,
    year: str,
    executor: str,
    chunk_size: Union[int, str],
    workers: int,
    is_data: bool,
    max_events: str = "",
) -> None:
    """Create a submit file from template with specific settings."""
    with open(template_file, 'r') as f:
        lines = f.readlines()

    # Get repository directory (absolute path from submission location)
    condor_dir = output_file.parent.parent  # condorJobs/event-selection directory
    repo_dir = condor_dir.parent.parent  # Repository root directory
    repo_dir_abs = repo_dir.resolve()

    # Build environment variable string
    # Config paths are relative to repository (script will cd to repo directory)
    # Workers are automatically derived from request_cpus in the template
    env_parts = [
        f'DBL_REPO_DIR={repo_dir_abs}',
        f'DBL_CONFIG={config}',
        f'DBL_BKG_FILE={sample_file}',
        f'DBL_YEAR={year}',
        f'DBL_EXECUTOR={executor}',
        f'DBL_CHUNK_SIZE={chunk_size}',
        f'DBL_WORKERS={workers}',
        f'DBL_DATA={"true" if is_data else "false"}',
    ]
    if max_events:
        env_parts.append(f'DBL_MAX_EVENTS={max_events}')

    env_vars = ' \\\n'.join(env_parts)

    # Background file is in repository, accessible from condor nodes
    # No need to transfer - file is read from DBL_REPO_DIR/data/samplelist/<year>/
    # So we don't need transfer_input_files for the background file

    # Get username and user ID for x509userproxy path
    try:
        username = os.getenv('USER') or os.getenv('LOGNAME') or pwd.getpwuid(os.getuid()).pw_name
        uid = os.getuid()
        first_letter = username[0].lower()
        x509_path = f'/afs/cern.ch/user/{first_letter}/{username}/private/x509up_u{uid}'
    except Exception:
        # Fallback to template value if detection fails
        x509_path = '/afs/cern.ch/user/u/username/private/x509up_u98885'

    # Replace lines
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # Replace x509userproxy line
        if line.strip().startswith('x509userproxy'):
            new_lines.append(f'x509userproxy = {x509_path}\n')
            i += 1
            continue

        # Replace environment block
        if line.strip().startswith('environment ='):
            new_lines.append(f'environment = "{env_vars}"\n')
            # Skip continuation lines until we find the closing quote
            i += 1
            while i < len(lines) and ('\\' in lines[i] or lines[i].strip().endswith('"') or not lines[i].strip()):
                i += 1
            continue

        # Remove transfer_input_files line (background file is in repository, not transferred)
        if line.strip().startswith('transfer_input_files'):
            # Skip this line - background file is read from repository directory
            i += 1
            continue

        # Replace queue line
        if line.strip().startswith('queue'):
            new_lines.append(f'queue {num_jobs}\n')
            i += 1
            continue

        new_lines.append(line)
        i += 1

    with open(output_file, 'w') as f:
        f.writelines(new_lines)


def is_data_sample(sample_name: str) -> bool:
    """Detect collision data samples by name, e.g. JetMET-Run2022D-22Sep2023-v1."""
    return re.search(r'-Run20\d{2}', sample_name) is not None


def submit_sample(
    sample_file: Path,
    condor_dir: Path,
    template_file: Path,
    config: str,
    year: str,
    executor: str,
    chunk_size: Optional[int],
    workers: int,
    max_events: str,
    request_memory_mb: int = 8000,
    dry_run: bool = False,
) -> Tuple[str, int]:
    """Submit condor jobs for a single sample file."""
    sample_name = sample_file.stem  # e.g., "Zto2Nu-2Jets_..." from "Zto2Nu-2Jets_....txt"
    sample_filename = sample_file.name  # e.g., "Zto2Nu-2Jets_....txt"

    # Count files in sample
    num_files = count_files_in_sample(sample_file)

    if num_files == 0:
        print(f"⚠  Skipping {sample_filename}: No files found")
        return sample_name, 0

    print(f"\n{'='*60}")
    print(f"Sample: {sample_filename}")
    print(f"{'='*60}")
    print(f"  Files in sample: {num_files}")
    print(f"  Will submit: {num_files} jobs (1 job per file)")

    is_data = is_data_sample(sample_name)
    print(f"  Sample type: {'DATA' if is_data else 'MC'}")

    # Handle chunk size: if None (auto), pass "auto" string to let each worker optimize
    # Otherwise, use the provided value (int or "auto" string)
    if chunk_size is None:
        chunk_size_str = "auto"
        print(f"  Using chunk size: auto (will optimize per job based on input file)")
    else:
        chunk_size_str = str(chunk_size)
        if chunk_size_str == "auto":
            print(f"  Using chunk size: auto (will optimize per job based on input file)")
        else:
            print(f"  Using chunk size: {chunk_size_str}")

    # Create temporary submit file in submitfiles directory
    submitfiles_dir = condor_dir / 'submitfiles'
    submitfiles_dir.mkdir(parents=True, exist_ok=True)
    temp_submit = submitfiles_dir / f"submit_{sample_name}.sub"

    create_submit_file(
        template_file=template_file,
        output_file=temp_submit,
        sample_file=sample_filename,
        num_jobs=num_files,
        config=config,
        year=year,
        executor=executor,
        chunk_size=chunk_size_str,
        workers=workers,
        is_data=is_data,
        max_events=max_events,
    )

    if dry_run:
        print(f"  [DRY RUN] Would submit: condor_submit {temp_submit}")
        print(f"  [DRY RUN] Submit file created: {temp_submit}")
        return sample_name, num_files

    # Submit to condor
    try:
        result = subprocess.run(
            ['condor_submit', str(temp_submit)],
            capture_output=True,
            text=True,
            check=True,
        )

        # Extract cluster ID from output
        cluster_id = None
        for line in result.stdout.split('\n'):
            if 'submitted to cluster' in line.lower():
                parts = line.split()
                for i, part in enumerate(parts):
                    if part.isdigit():
                        cluster_id = part
                        break

        print(f"  ✓ Submitted cluster: {cluster_id}")
        print(f"  ✓ Jobs: {num_files} (one per file)")
        print(f"  ✓ Each file will run on a separate node")

        return sample_name, num_files

    except subprocess.CalledProcessError as e:
        print(f"  ✗ Error submitting: {e}")
        print(f"  stderr: {e.stderr}")
        return sample_name, 0
    except FileNotFoundError:
        print(f"  ✗ Error: condor_submit not found. Are you on a condor-enabled system?")
        return sample_name, 0


def main():
    parser = argparse.ArgumentParser(
        description='Submit condor jobs for event-selection sample files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run from repo root
  python3 condorJobs/event-selection/submit_samples.py --year 2022

  # Submit all *.txt files in data/samplelist/2022/
  python3 submit_samples.py --year 2022

  # Override input directory
  python3 submit_samples.py --year 2022 --input-dir /path/to/samplelist

  # Dry run (don't actually submit)
  python3 submit_samples.py --year 2022 --dry-run

  # Custom configuration
  python3 submit_samples.py --year 2022 --config configs/2022.yaml --workers 8

  # Auto-optimize chunk size per sample
  python3 submit_samples.py --year 2022 --chunk-size auto

  # Use fixed chunk size
  python3 submit_samples.py --year 2022 --chunk-size 100000
        """
    )

    parser.add_argument(
        '--year',
        type=str,
        required=True,
        choices=['2022', '2022EE', '2023', '2024'],
        help='Data-taking year: selects data/samplelist/<year>/ and configs/<year>.yaml by default',
    )

    # Default to data/samplelist/<year>/ relative to repo root (set after --year is known)
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=None,
        help='Directory containing *.txt sample files (default: data/samplelist/<year>/)',
    )

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Configuration file (default: configs/<year>.yaml)',
    )

    parser.add_argument(
        '--executor',
        type=str,
        default='futures',
        choices=['iterative', 'futures', 'dask'],
        help='Executor type (default: futures)',
    )

    parser.add_argument(
        '--chunk-size',
        type=str,
        default='50000',
        help='Chunk size for futures/dask. Use "auto" for automatic optimization based on file characteristics, or specify an integer (default: 50000)',
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of workers (default: 4)',
    )

    parser.add_argument(
        '--max-events',
        type=str,
        default='',
        help='Maximum events to process (optional)',
    )

    parser.add_argument(
        '--template',
        type=Path,
        default=Path(__file__).parent / 'submit.sub',
        help='Template submit file (default: submit.sub)',
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be submitted without actually submitting',
    )

    args = parser.parse_args()

    # Setup logging (warnings level to show optimization messages)
    logging.basicConfig(
        level=logging.WARNING,
        format='%(levelname)s: %(message)s'
    )

    # Get script directory and repository root (condorJobs/event-selection/../..)
    condor_dir = Path(__file__).parent
    repo_dir = condor_dir.parent.parent

    # Default config: configs/<year>.yaml
    if args.config is None:
        args.config = f'configs/{args.year}.yaml'

    # Handle input directory path: default to data/samplelist/<year>/ at repo root
    if args.input_dir is None:
        input_dir = (repo_dir / 'data' / 'samplelist' / args.year).resolve()
    elif args.input_dir.is_absolute():
        input_dir = args.input_dir
    else:
        input_dir = (repo_dir / args.input_dir).resolve()

    # Check input directory exists
    if not input_dir.exists():
        print(f"✗ Error: Input directory not found: {input_dir}")
        sys.exit(1)

    # Check template file exists
    if not args.template.exists():
        print(f"✗ Error: Template file not found: {args.template}")
        sys.exit(1)

    # Parse chunk size argument
    try:
        chunk_size = parse_chunk_size_arg(args.chunk_size)
    except ValueError as e:
        print(f"✗ Error: {e}")
        sys.exit(1)

    # Get request_memory and request_cpus from template file
    request_memory_mb = get_request_memory_from_template(args.template)
    request_cpus = get_request_cpus_from_template(args.template)

    # Use request_cpus as workers (automatically derived from Condor template)
    # This ensures workers match the allocated CPUs
    workers = request_cpus

    # Create log directories inside condorJobs
    logs_dir = condor_dir / 'logs'
    logs_output_dir = logs_dir / 'output'
    logs_error_dir = logs_dir / 'error'

    for log_dir in [logs_dir, logs_output_dir, logs_error_dir]:
        log_dir.mkdir(parents=True, exist_ok=True)

    # Create submitfiles directory (will create if doesn't exist)
    submitfiles_dir = condor_dir / 'submitfiles'
    submitfiles_dir.mkdir(parents=True, exist_ok=True)

    # Delete existing submit files from previous jobs (automatic cleanup)
    existing_submit_files = []
    for submit_file in submitfiles_dir.glob('submit_*.sub'):
        if submit_file.is_file():
            existing_submit_files.append(submit_file)

    if existing_submit_files:
        if not args.dry_run:
            deleted_count = 0
            for submit_file in existing_submit_files:
                try:
                    submit_file.unlink()
                    deleted_count += 1
                except Exception as e:
                    print(f"  ✗ Error deleting {submit_file}: {e}")
            if deleted_count > 0:
                print(f"✓ Deleted {deleted_count} old submit file(s) from previous jobs")
        else:
            print(f"  [DRY RUN] Would delete {len(existing_submit_files)} old submit file(s)")

    # Check for existing log files and prompt to delete
    existing_logs = []
    for log_dir in [logs_output_dir, logs_error_dir, logs_dir]:
        if log_dir.exists():
            # Find all log files (dbl.*.out, dbl.*.err, dbl.*.log)
            for log_file in log_dir.glob('dbl.*'):
                if log_file.is_file():
                    existing_logs.append(log_file)

    if existing_logs and not args.dry_run:
        print(f"\n⚠  Found {len(existing_logs)} existing log file(s) from previous jobs:")
        # Show first few files
        for log_file in existing_logs[:5]:
            print(f"  - {log_file}")
        if len(existing_logs) > 5:
            print(f"  ... and {len(existing_logs) - 5} more")

        response = input("\nDo you want to delete these log files? [y/N]: ").strip().lower()
        if response in ['y', 'yes']:
            deleted_count = 0
            for log_file in existing_logs:
                try:
                    log_file.unlink()
                    deleted_count += 1
                except Exception as e:
                    print(f"  ✗ Error deleting {log_file}: {e}")
            print(f"✓ Deleted {deleted_count} log file(s)")
        else:
            print("⚠  Keeping existing log files (new logs will be appended/overwritten)")
        print()

    # Find sample files
    sample_files = find_sample_files(input_dir)

    if not sample_files:
        print(f"✗ No *.txt files found in {input_dir}")
        sys.exit(1)

    print("="*60)
    print("DarkBottomLine Condor Job Submission (event-selection)")
    print("="*60)
    print(f"Year: {args.year}")
    print(f"Config: {args.config}")
    print(f"Input directory: {input_dir}")
    print(f"Log directories: {logs_dir}/")
    print(f"Request memory: {request_memory_mb} MB")
    print(f"Workers per job: {workers} (auto-derived from request_cpus={request_cpus})")
    print(f"Executor: {args.executor}")
    if chunk_size is None:
        print(f"Chunk size: AUTO (will optimize per sample)")
    else:
        print(f"Chunk size: {chunk_size:,} events")
    print(f"Found {len(sample_files)} sample file(s):")
    for sample in sample_files:
        num_files = count_files_in_sample(sample)
        tag = "DATA" if is_data_sample(sample.stem) else "MC"
        print(f"  - {sample.name} [{tag}]: {num_files} files")
    print()

    if args.dry_run:
        print("⚠  DRY RUN MODE - No jobs will be submitted")
        print()

    # Submit each sample
    results = []
    for sample_file in sample_files:
        sample_name, num_jobs = submit_sample(
            sample_file=sample_file,
            condor_dir=condor_dir,
            template_file=args.template,
            config=args.config,
            year=args.year,
            executor=args.executor,
            chunk_size=chunk_size,
            workers=workers,  # Use auto-derived workers from request_cpus
            max_events=args.max_events,
            request_memory_mb=request_memory_mb,
            dry_run=args.dry_run,
        )
        results.append((sample_name, num_jobs))

    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    total_jobs = 0
    for sample_name, num_jobs in results:
        if num_jobs > 0:
            print(f"  {sample_name}: {num_jobs} jobs")
            total_jobs += num_jobs
        else:
            print(f"  {sample_name}: Failed or skipped")

    print(f"\nTotal jobs submitted: {total_jobs}")
    print("\nEach sample runs on separate condor cluster.")
    print("Each file from a sample runs on a separate node.")
    print("\nMonitor jobs with: condor_q")


if __name__ == '__main__':
    main()
