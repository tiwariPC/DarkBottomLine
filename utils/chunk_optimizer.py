"""
Chunk size optimization utilities for DarkBottomLine framework.

This module provides functions to automatically optimize chunk sizes based on
file characteristics, memory constraints, and parallelization needs.
"""

import logging
import os
from typing import Tuple, Optional, List, Union
from pathlib import Path


def estimate_event_size_and_total_events(root_file_path: str) -> Tuple[float, int]:
    """
    Estimate event size (bytes) and total events from a ROOT file.

    Args:
        root_file_path: Path to ROOT file (can be xrootd URL or local path)

    Returns:
        (event_size_bytes, total_events)
    """
    try:
        try:
            import uproot
        except ImportError:
            logging.warning("uproot not available, using default event size estimate")
            return 5000, 100000

        # Open file and get Events tree
        with uproot.open(root_file_path) as f:
            tree = f.get("Events")
            if tree is None:
                # Fallback: try to estimate from file size
                # For xrootd URLs, we can't get file size easily
                # Assume 10k events if we can't read the tree
                return 5000, 10000

            total_events = tree.num_entries

            # Try to get file size (works for local files, not xrootd)
            try:
                if not root_file_path.startswith(('root://', 'http://', 'https://')):
                    file_size = os.path.getsize(root_file_path)
                    # Estimate event size (file size / total events)
                    # This is approximate as ROOT files have overhead
                    event_size = file_size / total_events if total_events > 0 else 0
                else:
                    # For xrootd URLs, use default estimate
                    event_size = 5000  # Typical NanoAOD event size
            except (OSError, AttributeError):
                # Fallback: assume typical NanoAOD event size
                event_size = 5000

            return event_size, total_events

    except Exception as e:
        logging.warning(f"Could not estimate event size for {root_file_path}: {e}")
        # Fallback: assume 5KB per event (typical NanoAOD)
        return 5000, 100000


def calculate_optimal_chunk_size(
    event_size_bytes: float,
    total_events: int,
    available_memory_mb: int,
    num_workers: int,
    executor: str = "futures",
    min_chunk_size: int = 10000,
    max_chunk_size: int = 1000000,
) -> int:
    """
    Calculate optimal chunk size based on event size, total events, and available memory.

    Args:
        event_size_bytes: Estimated size per event in bytes
        total_events: Total number of events in file
        available_memory_mb: Available memory per worker in MB
        num_workers: Number of workers
        executor: Executor type ("futures" or "dask")
        min_chunk_size: Minimum chunk size (default: 10k)
        max_chunk_size: Maximum chunk size (default: 1M)

    Returns:
        Optimal chunk size (number of events)
    """
    # Convert memory to bytes
    available_memory_bytes = available_memory_mb * 1024 * 1024

    # Safety factor: use only 70% of available memory to leave headroom
    safe_memory_bytes = available_memory_bytes * 0.7

    # Calculate chunk size based on memory constraint
    memory_based_chunk_size = int(safe_memory_bytes / event_size_bytes) if event_size_bytes > 0 else min_chunk_size

    # Calculate chunk size based on parallelization (want 2-4 chunks per worker)
    # Ideal: total_events / chunk_size = 2 * num_workers
    parallelization_based_chunk_size = total_events // (2 * num_workers) if num_workers > 0 else min_chunk_size

    # Take the minimum to satisfy both constraints
    optimal_chunk_size = min(memory_based_chunk_size, parallelization_based_chunk_size)

    # Apply min/max bounds
    optimal_chunk_size = max(min_chunk_size, min(optimal_chunk_size, max_chunk_size))

    # Round to nearest 10k for cleaner numbers
    optimal_chunk_size = int(round(optimal_chunk_size / 10000) * 10000)

    # Executor-specific adjustments
    if executor == "dask":
        # Dask can handle larger chunks more efficiently
        optimal_chunk_size = max(optimal_chunk_size, 100000)
    elif executor == "futures":
        # Futures works better with smaller chunks
        optimal_chunk_size = min(optimal_chunk_size, 200000)

    return optimal_chunk_size


def optimize_chunk_size_for_files(
    input_files: List[str],
    available_memory_mb: int,
    num_workers: int,
    executor: str = "futures",
    max_sample_files: int = 5,
) -> int:
    """
    Calculate optimal chunk size by analyzing input ROOT files.

    Args:
        input_files: List of ROOT file paths (can be xrootd URLs)
        available_memory_mb: Available memory per worker in MB
        num_workers: Number of workers
        executor: Executor type (default: "futures")
        max_sample_files: Maximum number of files to sample for estimation (default: 5)

    Returns:
        Optimal chunk size for these files
    """
    if not input_files:
        # Fallback to default
        return 50000 if executor == "futures" else 200000

    # Analyze first few files to estimate (sample up to max_sample_files)
    sample_files = input_files[:min(max_sample_files, len(input_files))]

    total_event_size = 0
    total_events_sum = 0
    valid_samples = 0

    for root_file in sample_files:
        try:
            event_size, num_events = estimate_event_size_and_total_events(root_file)
            if event_size > 0 and num_events > 0:
                total_event_size += event_size
                total_events_sum += num_events
                valid_samples += 1
        except Exception as e:
            logging.warning(f"Could not analyze {root_file}: {e}")
            continue

    if valid_samples == 0:
        # Fallback to default
        return 50000 if executor == "futures" else 200000

    # Average event size and events
    avg_event_size = total_event_size / valid_samples
    avg_total_events = total_events_sum / valid_samples

    # Available memory per worker
    available_memory_per_worker = available_memory_mb / num_workers if num_workers > 0 else available_memory_mb

    # Calculate optimal chunk size
    optimal_chunk_size = calculate_optimal_chunk_size(
        event_size_bytes=avg_event_size,
        total_events=avg_total_events,
        available_memory_mb=available_memory_per_worker,
        num_workers=num_workers,
        executor=executor,
    )

    return optimal_chunk_size


def optimize_chunk_size_for_sample_file(
    sample_file: Union[str, Path],
    available_memory_mb: int = 8000,
    num_workers: int = 4,
    executor: str = "futures",
) -> int:
    """
    Calculate optimal chunk size for a sample file by analyzing its ROOT files.

    Args:
        sample_file: Path to sample file containing ROOT file paths (one per line)
        available_memory_mb: Available memory per worker in MB (default: 8000)
        num_workers: Number of workers per job (default: 4)
        executor: Executor type (default: "futures")

    Returns:
        Optimal chunk size for this sample
    """
    sample_file = Path(sample_file)

    # Read ROOT files from sample file
    root_files = []
    with open(sample_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                root_files.append(line)

    return optimize_chunk_size_for_files(
        input_files=root_files,
        available_memory_mb=available_memory_mb,
        num_workers=num_workers,
        executor=executor,
    )


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
