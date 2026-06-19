"""
Utility to count total events in ROOT files.
"""

import logging
from typing import List, Union


def count_total_events(input_files: Union[str, List[str]], tree_name: str = "Events") -> int:
    """
    Count the total number of events in one or more ROOT files.
    
    This function counts events BEFORE any selection or processing,
    which is useful for tracking the initial event count.
    
    Args:
        input_files: Single ROOT file path or list of ROOT file paths
        tree_name: Name of the TTree to count events from (default: "Events")
        
    Returns:
        Total number of events across all files
        
    Raises:
        ImportError: If uproot is not available
        FileNotFoundError: If input files don't exist
        ValueError: If tree not found in file
    """
    try:
        import uproot
    except ImportError:
        raise ImportError("uproot is required for counting events. Please install it.")
    
    # Handle single file input
    if isinstance(input_files, str):
        input_files = [input_files]
    
    total_events = 0
    
    for file_path in input_files:
        try:
            # Open the ROOT file and get the tree
            with uproot.open(file_path) as file:
                if tree_name not in file:
                    raise ValueError(f"Tree '{tree_name}' not found in {file_path}")
                
                tree = file[tree_name]
                n_events = tree.num_entries
                total_events += n_events
                logging.debug(f"File {file_path}: {n_events} events")
                
        except Exception as e:
            logging.error(f"Error reading {file_path}: {e}")
            raise
    
    logging.info(f"Total events counted: {total_events} from {len(input_files)} file(s)")
    return total_events


def count_total_events_from_chunk(
    input_files: Union[str, List[str]], 
    tree_name: str = "Events",
    max_events: int = None,
    chunksize: int = None
) -> int:
    """
    Count total events considering chunk-based processing parameters.
    
    When using chunked processing, this function helps determine the actual
    number of events that will be processed given the chunk size and max_events.
    
    Args:
        input_files: Single ROOT file path or list of ROOT file paths
        tree_name: Name of the TTree (default: "Events")
        max_events: Maximum number of events to process (None = all events)
        chunksize: Size of each chunk (used to calculate maxchunks)
        
    Returns:
        Total number of events that will be processed
        
    Note:
        If max_events is set and chunking is used, the actual processed events
        may be slightly more than max_events due to chunk boundaries.
    """
    # First, count the actual total events in the files
    total_available = count_total_events(input_files, tree_name)
    
    # If no max_events, return the full count
    if max_events is None:
        return total_available
    
    # If max_events is less than available, consider chunking behavior
    if max_events < total_available:
        if chunksize is not None:
            # With chunking, events are processed in complete chunks
            # Calculate number of chunks that will be processed
            maxchunks = (max_events + chunksize - 1) // chunksize
            estimated_events = min(maxchunks * chunksize, total_available)
            logging.info(
                f"With max_events={max_events} and chunksize={chunksize}: "
                f"~{estimated_events} events will be processed ({maxchunks} chunks)"
            )
            return estimated_events
        else:
            # Without chunking, exact max_events will be used
            return max_events
    
    return total_available


if __name__ == "__main__":
    """
    Test the event counter with a sample file.
    
    Usage:
        python -m darkbottomline.utils.event_counter input/tester.root
    """
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if len(sys.argv) < 2:
        print("Usage: python -m darkbottomline.utils.event_counter <root_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    try:
        n_events = count_total_events(file_path)
        print(f"\n{'='*60}")
        print(f"Total events in {file_path}: {n_events}")
        print(f"{'='*60}\n")
        
        # Test with chunking parameters
        print("Testing with chunked parameters:")
        for max_events in [None, 1000, 5000]:
            for chunksize in [None, 500, 1000]:
                if chunksize is not None or max_events is not None:
                    n = count_total_events_from_chunk(
                        file_path, max_events=max_events, chunksize=chunksize
                    )
                    print(f"  max_events={max_events}, chunksize={chunksize} -> {n} events")
        
    except Exception as e:
        logging.error(f"Failed to count events: {e}", exc_info=True)
        sys.exit(1)
