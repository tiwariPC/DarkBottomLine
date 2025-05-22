import argparse
import json
import os
import subprocess
import sys
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed


def is_directory_empty(directory):
    """
    Check if a given directory is empty.

    Args:
        directory (str): Path to the directory.

    Returns:
        bool: True if the directory is empty, False otherwise.
    """
    return not any(os.scandir(directory))


def handle_target_directory(target_dir, force_recreate):
    """
    Manage the target directory based on its current state and user preferences.

    - If the directory exists and is empty, proceed with downloading.
    - If the directory exists and is not empty:
        - If `force_recreate` is True, delete its contents and proceed.
        - Otherwise, exit the script to prevent accidental data loss.
    - If the directory does not exist, create it.

    Args:
        target_dir (str): Path to the target directory.
        force_recreate (bool): Flag indicating whether to force recreate the directory.

    Raises:
        SystemExit: If unable to handle the directory based on the above conditions.
    """
    if os.path.exists(target_dir):
        if is_directory_empty(target_dir):
            print(f"Directory '{target_dir}' exists and is empty. Proceeding with download.")
        else:
            if force_recreate:
                print(f"Directory '{target_dir}' exists and is not empty. Force recreating by deleting its contents.")
                try:
                    # Iterate over all items in the directory and remove them
                    for filename in os.listdir(target_dir):
                        file_path = os.path.join(target_dir, filename)
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)  # Remove file or symbolic link
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)  # Remove directory and its contents
                except Exception as e:
                    print(f"Failed to clear directory '{target_dir}': {e}")
                    sys.exit(1)
            else:
                print(f"Directory '{target_dir}' exists and is not empty. Use --force-recreate to overwrite.")
                sys.exit(1)
    else:
        try:
            os.makedirs(target_dir, exist_ok=True)  # Create the directory if it doesn't exist
            print(f"Created directory '{target_dir}'.")
        except Exception as e:
            print(f"Failed to create directory '{target_dir}': {e}")
            sys.exit(1)


def download_file(file_url, destination_folder, key):
    """
    Download a single file using xrdcp.

    Args:
        file_url (str): URL of the file to download.
        destination_folder (str): Base directory where files are downloaded.
        key (str): The key associated with the file, used to create subdirectories.

    Returns:
        tuple: (file_url, success, message)
            - file_url (str): The URL of the file attempted to download.
            - success (bool): True if download was successful, False otherwise.
            - message (str): Success or error message.
    """
    try:
        filename = os.path.basename(file_url)  # Extract the filename from the URL
        destination_path = os.path.join(destination_folder, os.path.basename(file_url))

        # Skip downloading if the file already exists
        if os.path.exists(destination_path):
            message = f"File already exists: {filename}. Skipping download."
            print(message)
            return (file_url, True, message)

        # Define the xrdcp command
        command = ['xrdcp', '-f', file_url, destination_path]

        # Execute the xrdcp command to copy the file
        subprocess.run(command, check=True)

        message = f"Downloaded: {filename} to '{destination_folder}'"
        print(message)
        return (file_url, True, message)
    except subprocess.CalledProcessError as e:
        message = f"Failed to download {file_url}. Error: {e}"
        print(message)
        return (file_url, False, message)
    except Exception as e:
        message = f"Unexpected error downloading {file_url}. Error: {e}"
        print(message)
        return (file_url, False, message)


def download_files_in_parallel(file_dict, target_dir, num_files, num_threads):
    """
    Download files in parallel using multiple threads.

    Args:
        file_dict (dict): Dictionary where each key is a string representing a category
                          (e.g., dataset name) and the value is a list of file URLs to download.
        target_dir (str): Base directory where files will be downloaded. Subdirectories for each key
                          will be created within this directory.
        num_files (int or None): Number of files to download per key. If None, download all files.
        num_threads (int): Number of concurrent threads to use for downloading.

    Returns:
        None
    """
    # Iterate over each key and its corresponding list of file URLs
    for key, files in file_dict.items():
        # Determine the actual number of files to download for the current key
        actual_num_files = num_files if num_files is not None else len(files)
        if actual_num_files > len(files):
            print(f"Requested {actual_num_files} files for key '{key}', but only {len(files)} available. Proceeding to download all available files.")
            actual_num_files = len(files)

        # Slice the list of files based on the determined number
        files_to_download = files[:actual_num_files]

        print(f"\nStarting download of {actual_num_files} files for key '{key}' with {num_threads} threads.")

        # Use ThreadPoolExecutor to manage parallel downloads
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit download tasks for each file and map futures to their file URLs
            future_to_file = {
                executor.submit(download_file, file_url, target_dir, key): file_url
                for file_url in files_to_download
            }

            # Process each completed future as they finish
            for future in as_completed(future_to_file):
                file_url = future_to_file[future]
                try:
                    _, success, message = future.result()
                    if not success:
                        print(f"Error downloading {file_url}: {message}")
                except Exception as exc:
                    print(f"Generated an exception while downloading {file_url}: {exc}")


def main():
    """
    Main function to parse arguments and initiate the file download process.

    Workflow:
        1. Parse command-line arguments.
        2. Read and parse the JSON file containing file URLs.
        3. Determine whether to process a specific key or all keys.
        4. Handle the target directory based on its state and user flags.
        5. Download the specified number of files using xrdcp, either sequentially or in parallel.
    """
    # Initialize the argument parser with a description of the script
    parser = argparse.ArgumentParser(description="Download files from a JSON file using xrdcp")

    # Define required and optional command-line arguments
    parser.add_argument('--json-file', required=True, help='Path to the JSON file')
    parser.add_argument('--key', required=True, help='Key in the JSON file to process or "all" to process all keys')
    parser.add_argument('--num-files', type=int, default=None, help='Number of files to process per key')
    parser.add_argument('--target-dir', required=True, help='Target directory to copy files into')
    parser.add_argument('--force-recreate', '-f', action='store_true', help='Force recreate the target directory by deleting its contents if it exists')
    parser.add_argument('--num-threads', type=int, default=8, help='Number of parallel threads to use for downloading')

    # Parse the arguments provided by the user
    args = parser.parse_args()

    # Read and load the JSON file
    try:
        with open(args.json_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        sys.exit(1)

    # Check if the user wants to process all keys in the JSON
    if args.key == 'all':
        keys = data.keys()  # Retrieve all keys from the JSON
        if not keys:
            print("No keys found in the JSON file.")
            sys.exit(1)
        # Iterate over each key to process its associated files
        for key in keys:
            print(f"\nProcessing key '{key}':")
            # Define a subdirectory for the current key within the target directory
            sub_target_dir = os.path.join(args.target_dir, key)
            # Handle the subdirectory based on existing content and force flag
            handle_target_directory(sub_target_dir, args.force_recreate)
            file_list = data[key]  # Retrieve the list of files for the current key

            # Validate that file_list is a list
            if not isinstance(file_list, list):
                print(f"Data for key '{key}' is not a list. Skipping this key.")
                continue

            # Download the files in parallel
            download_files_in_parallel({key: file_list}, sub_target_dir, args.num_files, args.num_threads)
    else:
        # If a specific key is provided, process only that key
        if args.key not in data:
            print(f"Key '{args.key}' not found in JSON file.")
            sys.exit(1)

        file_list = data[args.key]  # Retrieve the list of files for the specified key

        # Validate that file_list is a list
        if not isinstance(file_list, list):
            print(f"Data for key '{args.key}' is not a list. Exiting.")
            sys.exit(1)

        # Handle the main target directory based on existing content and force flag
        handle_target_directory(args.target_dir, args.force_recreate)

        # Download the files in parallel
        download_files_in_parallel({args.key: file_list}, args.target_dir, args.num_files, args.num_threads)


if __name__ == '__main__':
    main()
