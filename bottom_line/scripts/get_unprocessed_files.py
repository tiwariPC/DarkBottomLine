#!/usr/bin/env python
from bottom_line.utils.logger_utils import setup_logger
from bottom_line.utils.runner_utils import get_proxy
from XRootD import client
from concurrent.futures import ThreadPoolExecutor
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.live import Live
from rich.console import Group
import argparse
import json
import subprocess
import os


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# - This script creates a json file containing the unprocessed samples, based on a provided sample.json and parquet directory. ----------------------------------------------------#
# - It can work with the DAS UUID naming convention or the Legacy one, where the UUID is contained in the ROOT file header. -------------------------------------------------------#
# - It will check for missing UUIDs in the parquet file names & unprocessed event in individual parquets, and will produce a json with the updated list of unprocessed samples. ---#
# - EXAMPLE USAGE: ----------------------------------------------------------------------------------------------------------------------------------------------------------------#
# - python3 get_unprocessed_files.py --convention <naming_convention> --source <dir_to_HiggsDNA_dump> --json <sample.json> --output <some_path/unprocessed_samples.json>
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# --------------------------------------------------------------------------------------------------------------------#
# If you are just interested in the number of files that were processed, you can use the following command: ----------#
# - find <path_to_parquet_files> -type f -regextype posix-extended -regex '.*_Events_0-[0-9]+\.parquet' | wc -l
# --------------------------------------------------------------------------------------------------------------------#

def get_fetcher_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Obtain the list of unprocessed root files from the associated samples list."
    )
    parser.add_argument(
        "--log",
        dest="log",
        type=str,
        default="INFO",
        help="Logger info level"
    )
    parser.add_argument(
        "-s",
        "--source",
        help="Directory containing the datasets.",
        required=True,
    )
    parser.add_argument(
        "-j",
        "--json",
        help="Json containing the lists of samples.",
        required=True,
    )
    parser.add_argument(
        "-c",
        "--convention",
        help="Parquet files naming convention: DAS or Legacy.",
        required=True,
        choices=["DAS", "Legacy"]
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Path in which to save the unprocessed samples, eg 'Thisdir/myoutput.json'.",
        default="unprocessed_samples.json",
    )
    parser.add_argument(
        "--limit",
        type=int,
        metavar="N",
        help="Limit to the first N files of each dataset in sample JSON",
        default=None,
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="Timeout for dasgoclient/xrootd query (default: 30s)",
        default=30,
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of workers (threads) to use for multi-worker executors (default: 8)",
        default=1,
    )
    parser.add_argument(
        "--skipbadfiles",
        help="Skip xrootd bad files when retrieving Legacy UUID",
        default=False,
        action='store_true'
    )

    return parser.parse_args()


# This function takes a list of event chunks associated with one parquet, eg [0-127,127-255,255-511]
# It will return the total number of events if the chunks are continuous (eg 511) and 0 otherwise.
def check_range(range_list):

    # Create list of event chunks and sort it
    chunks = [int(val) for rg in range_list for val in rg.split('-')]
    chunks.sort()

    total_range = chunks[-1] - chunks[0]
    # Verify that chunks in the list are successive.
    for i in range(1,len(chunks)-1,2):
        if chunks[i] != chunks[i+1]:
            total_range = 0
            break

    return total_range


# Use xrootd to read the header of a root file.
# Retrieve the associated UUID from the file header.
def get_root_uuid_using_xrootd(fpath, timeout, skipbadfiles):

    logger.debug(f"Opening {fpath} with xrootd")

    try:
        with client.File() as f:
            # Try to open the file
            status, _ = f.open(fpath, timeout=timeout)
            if not status.ok:
                raise TimeoutError(f"Failed to open file {fpath}: {status.message}")

            # Read version bytes (offset 4, size 4)
            status, version_bytes = f.read(offset=4, size=4)
            if not status.ok:
                raise RuntimeError(f"Failed to read version from file {fpath}: {status.message}")
            # Convert version to an integer
            version = int.from_bytes(version_bytes, "big")

            # Determine the offset for UUID based on the version
            uuid_offset = 59 if version >= 1000000 else 47
            # Read the UUID bytes
            status, uuid_bytes = f.read(offset=uuid_offset, size=16)
            if not status.ok:
                raise RuntimeError(f"Failed to read UUID from file {fpath}: {status.message}")
            # Convert UUID bytes to standard UUID format
            root_uuid = '-'.join([
                uuid_bytes[0:4].hex(),
                uuid_bytes[4:6].hex(),
                uuid_bytes[6:8].hex(),
                uuid_bytes[8:10].hex(),
                uuid_bytes[10:16].hex()
            ])
    except Exception as e:
        if skipbadfiles:
            root_uuid = "00000000-0000-0000-0000-000000000000"
        else:
            raise e

    return root_uuid


# Create a dict of form {'dataset':{'uuid':nevent}} from the source directory.
def create_pq_dict(path, root_dict):
    tree =  {}
    parquetinfo = {}
    source_dict = {}

    dataset_from_json = list(root_dict.keys())

    # List all subdirectories in provided path. Subdirs name are expected to be the same as datasets in sample.json
    subdir = [
            d for d in os.listdir(path) if (os.path.isdir(os.path.join(path, d)) and d in dataset_from_json)
        ]

    logger.info(f"Starting inspection of source directory {path}.")

    if not subdir:
        logger.warning(f"No datasets from the original JSON were found in the source directory.")

    # Create the dict containing the parquet files: 'dataset':(parquet1,parquet2,...)
    for dataset in subdir:
        logger.info(f"Starting inspection of directory {dataset}")
        # We look into the "nominal" subdirectory
        pq_path = os.path.join(path, dataset+"/nominal/")
        if not os.path.exists(pq_path):
            logger.info(f"{pq_path} does not exist, continue.")
            continue

        file_list = [
                f for f in os.listdir(pq_path) if os.path.isfile(os.path.join(pq_path, f))
            ]
        tree[dataset] = file_list
        logger.debug(f"Successfully read parquet files for dataset {dataset}")

    # Retrieve the uuid and event chunks from the parquet name.
    for dataset, file_list in tree.items():
        for filename in file_list:
            index = 1 if filename.startswith("_") else 0
            uuid = filename.split('_')[index]
            ev_range = (filename.split('_')[index+2]).replace(".parquet", "").replace(".txt", "")

            if dataset not in parquetinfo:
                parquetinfo[dataset] = {}

            if uuid not in parquetinfo[dataset]:
                parquetinfo[dataset][uuid] = [ev_range]
            else:
                parquetinfo[dataset][uuid].extend([ev_range])

    # Finally, we create the dict 'dataset':{'uuid':nevent}
    for dataset, uuidinfo in parquetinfo.items():
        source_dict[dataset] = {}
        for uuid, ranges in uuidinfo.items():
            total_range = check_range(ranges)
            source_dict[dataset][uuid] = total_range

        logger.debug(f"Successfully retrieved parquet information for dataset {dataset}")

    return(source_dict)

def process_dataset(name: str, sample_files: list, convention: str, limit, timeout, skipbadfiles):
    """
    Process one dataset from the sample json.
    Returns a dict mapping root file uuid -> (number of events, file location)
    """
    # Apply limit and remove redirector prefix
    rootf_location = sample_files[:limit]
    rootf_name = ["/store" + f.split("store")[-1] for f in rootf_location]

    # Get unique directory names
    # Retrieve the file location, eg "/store/data/Run2022C/EGamma/NANOAOD/16Dec2023-v1"
    rootf_directory = ["/".join(f.split("/")[:-2]) for f in rootf_name]
    unique_rootf_directory = []
    for directory in rootf_directory:
        if directory not in unique_rootf_directory:
            unique_rootf_directory.append(directory)

    # Get the first file for each unique directory
    files_from_unique_directories = []
    for directory in unique_rootf_directory:
        for rf in rootf_name:
            if directory in rf:
                files_from_unique_directories.append(rf)
                break

    # Here we retrieve the original dataset names and status based on the root file list we just retrieved
    dataset_info = []
    # use the cvmfs source for dasgoclient because it works for everyone
    # Both local infrastructures with cvmfs and lxplus!
    for rootf in files_from_unique_directories:
        cmd = ("/cvmfs/cms.cern.ch/common/dasgoclient -query='dataset file={} status=* | "
               "grep dataset.name | grep dataset.status'").format(rootf)
        out = subprocess.check_output(cmd, shell=True, universal_newlines=True, timeout=timeout if timeout != 0 else None).strip()
        dataset_info.append(out)

    # Construct the list of original dataset & print a warning if the dataset we are processing is INVALID
    dataset_list = []
    for dinfo in dataset_info:
        if not dinfo:
            continue
        parts = dinfo.split()
        dataset_location, dataset_status = parts[0], parts[1]
        dataset_list.append(dataset_location)
        if dataset_status not in ["PRODUCTION", "VALID"]:
            logger.warning(f"{dataset_location} status is {dataset_status}, which is neither VALID nor PRODUCTION. Make sure this is intentional")

    # Get all root files in the datasets.
    nested_file_list = []
    for dataset in dataset_list:
        cmd = ("/cvmfs/cms.cern.ch/common/dasgoclient -query='file status=* dataset={} | "
               "grep file.name | grep file.nevents'").format(dataset.strip())
        out = subprocess.check_output(cmd, shell=True, universal_newlines=True, timeout=timeout if timeout != 0 else None).splitlines()
        nested_file_list.append(out)
    # Flatten the list of all root files
    file_list = [rootf for file_list in nested_file_list for rootf in file_list]

    # Retrieve UUIDs for the files
    if convention == "DAS":
        rootf_uuid = [f.split("/")[-1].replace(".root", "") for f in rootf_name]
    elif convention == "Legacy":
        rootf_uuid = [get_root_uuid_using_xrootd(f, timeout, skipbadfiles) for f in rootf_location]
    else:
        raise ValueError("Invalid naming convention")

    # Build the output dictionary: uuid -> (number of events, file location)
    dataset_dict = {}
    for f_line in file_list:
        parts = f_line.split()
        if len(parts) < 2:
            continue
        fname, nev_str = parts[0], parts[1]
        nev = int(nev_str)
        if fname in rootf_name:
            idx = rootf_name.index(fname)
            associated_rootuuid = rootf_uuid[idx]
            if associated_rootuuid == "00000000-0000-0000-0000-000000000000":
                logger.debug(f"{fname} could not be accessed by xrootd.")
                associated_rootuuid = "failed_" + fname.split("/")[-1].replace(".root", "")
                nev = -999
            associated_rootf_location = rootf_location[idx]
            dataset_dict[associated_rootuuid] = (nev, associated_rootf_location)

    return dataset_dict


# Create a dict of form {'dataset':{'uuid':(nevent,physical_location)}} from the sample.json file.
def parse_sample_json(samples_json: str, convention: str, limit, timeout, workers, skipbadfiles):
    """
    Create a dict of form {'dataset':{'uuid': (nevent, physical_location)}}
    from the provided sample.json file, processing each dataset in parallel.
    If processing a dataset takes longer than 'timeout' seconds, its worker is restarted.
    """
    root_dict = {}

    with open(samples_json) as f:
        samples = json.load(f)

    logger.info(f"Retrieving information on root files from {samples_json}.")

    # Process each dataset in parallel.
    def submit_dataset(name, sample_files, progress, global_task_id, spinner_progress):
        spinner_task_id = spinner_progress.add_task("Individual progress", dataset=name, start=True)
        try:
            while True:
                try:
                    result = process_dataset(name, sample_files, convention, limit, timeout, skipbadfiles)
                    logger.info(f"Successfully checked dataset '{name}'")
                    progress.update(global_task_id, advance=1)
                    return result
                except subprocess.TimeoutExpired as e:
                    logger.warning(f"DAS query timeout in dataset '{name}': {e}. Retrying...")
                except TimeoutError as e:
                    logger.warning(f"XRootD timeout in dataset '{name}': {e}. Retrying...")
                except Exception as e:
                    logger.error(f"Unexpected error while while processing dataset '{name}': {e}")
                    raise e
        # remove the spinner gracefully at the end of the subtask
        finally:
            spinner_progress.remove_task(spinner_task_id)

    # Define global progress bar (overall)
    global_progress = Progress(
        TextColumn("[bold green]Retrieving datasets information -"),
        TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
    )

    # Define spinner progress bar (for subtasks)
    spinner_progress = Progress(
        TextColumn("    "),
        SpinnerColumn(spinner_name="dots"),
        TextColumn("[cyan]Checking dataset {task.fields[dataset]}", justify="left"),
    )

    with Live(Group(global_progress, spinner_progress), refresh_per_second=2):
        total_samples = len(samples)
        global_task_id = global_progress.add_task("Global progress", total=total_samples)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_results = {}
            spinner_task_ids = {}

            # Submit each dataset task
            for name in samples:
                future_results[name] = executor.submit(submit_dataset, name, samples[name], global_progress, global_task_id, spinner_progress)

            # Handle results and update progress
            for name, fut in future_results.items():
                try:
                    root_dict[name] = fut.result()
                    logger.debug(f"Successfully retrieved file information for dataset {name}")
                except Exception as e:
                    logger.error(f"Failed to process dataset {name}: {e}")

    # Log warning if any dataset encountered xrootd access issues.
    for name, d in root_dict.items():
        if any("failed" in uuid for uuid in d.keys()):
            number_of_xrootd_fails = sum('failed' in uuid for uuid in d.keys())
            logger.warning(f"{number_of_xrootd_fails} file(s) in {name} could not be accessed by xrootd and have been automatically marked as unprocessed.")

    return root_dict

# Compare two dataset (uuid and nevents) and return the list of unprocessed samples.
def compare_data(root_dict, source_dict):

    unprocessed_files = []

    root_uuid = list(root_dict.keys())
    parquet_uuid = list(source_dict.keys())

    for uuid in root_uuid:
        # Retrieve number of events and physical file location
        rt_nevent, file_location = root_dict.get(uuid, 0)

        # Check if uuid is found in parquet list
        if uuid not in parquet_uuid:
            logger.debug(f"Missing file with uuid: {uuid}")
            unprocessed_files.append(file_location)
            continue

        # Check if all events have been processed
        pq_nevent = source_dict.get(uuid, 0)

        if not rt_nevent == pq_nevent:
            logger.debug(f"Missing event for uuid: {uuid}. Expected {rt_nevent} and got {pq_nevent}")
            unprocessed_files.append(file_location)
            continue

    return unprocessed_files


def main():
    output_dict = {}

    args = get_fetcher_args()

    global logger
    logger = setup_logger(level=args.log)

    if ".json" not in args.output:
        raise Exception("Output file must have '.json' extension and be a json file!")
    if ".json" not in args.json:
        raise Exception("Input json must have '.json' extension and be a json file!")
    if not os.path.isdir(args.source):
        raise Exception("Source directory does not exist. Make sure you provided a valid path.")

    # Check for valid proxy (required to use dasgoclient)
    get_proxy()

    # Create dicts from sample.json and parquet directory
    root_dict = parse_sample_json(args.json, args.convention, args.limit, args.timeout, args.workers, args.skipbadfiles)
    pq_dict = create_pq_dict(args.source, root_dict)

    logger.info("Starting creation of output file.")
    for dataset_name in root_dict:
        rt = root_dict[dataset_name]
        # Check if dataset in sample.json exists in parquet dir.
        if dataset_name not in pq_dict:
            logger.info(f"Dataset {dataset_name} could not be found in source directory and will be marked as unprocessed.")
            pq = {}
        else :
            pq = pq_dict[dataset_name]

        # Construct the dict of unprocessed root files
        unprocessed_files = compare_data(rt,pq)

        if unprocessed_files:
            output_dict[dataset_name] = unprocessed_files

        expected = len(rt)
        fully_processed = len(rt)-len(unprocessed_files)
        logger.info(f"Out of the {expected} files specified for dataset {dataset_name} in json file, {fully_processed} were fully processed. ")

    # If dict is empty (all samples fully processed), do not create output json
    if not output_dict:
        logger.info("All samples were fully processed. Output file will not be created.")

    else:
        logger.info(f"Output file will be saved in {args.output}.")
        with open(args.output, 'w') as f:
            json.dump(output_dict, f, indent=4)

if __name__ == "__main__":
    main()
