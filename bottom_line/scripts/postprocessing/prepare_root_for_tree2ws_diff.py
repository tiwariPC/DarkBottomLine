import os
import shutil
import argparse
import uproot
from bottom_line.utils.logger_utils import setup_logger


def get_trees_in_directory(file_path, directory_name):
    """
    Retrieves the list of trees and the original directory from a ROOT file.

    Parameters:
    - file_path (str): Path to the ROOT file.
    - directory_name (str): Name of the directory containing TTrees.

    Returns:
    - original_trees (list): List of tree names.
    - original_directory (uproot.reading.ReadOnlyDirectory): Original directory object.
    """
    input_file = uproot.open(file_path)
    original_directory = input_file[directory_name]
    original_trees = original_directory.keys()
    return original_trees, original_directory


def process_root_file(file_path, output_directory):
    """
    Processes the original TTrees, applies fiducial region selection, and saves the selected data to the output directory.

    Parameters:
    - file_path (str): Path to the input ROOT file.
    - output_directory (uproot.write.TDirectory): Output directory for saving processed TTrees.
    """
    original_trees, original_directory = get_trees_in_directory(file_path, "DiphotonTree")

    # List of branch names to save
    branches_to_save = ["CMS_hgg_mass", "weight", "dZ"]

    # Loop through the original TTrees
    for tree_name in original_trees:
        for fiducial_region in ['In', 'Out']:
            tree = original_directory[tree_name]
            process_name = tree_name.split("_", 1)[0]  # Get the substring before the first "_"
            output_tree_name = tree_name.replace(f"{process_name}_", f"{process_name}{fiducial_region}_")
            if fiducial_region == 'In':
                mask = tree.arrays()["fiducialTagger_20"] > 20.5  # Events in the fiducial region
            elif fiducial_region == 'Out':
                mask = tree.arrays()["fiducialTagger_20"] < 20.5  # Events out of the fiducial region
            selected_data = {branch: tree.arrays()[branch][mask] for branch in branches_to_save}
            output_directory[output_tree_name] = selected_data


def main():
    # Setup logger
    logger = setup_logger(level="INFO")

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Prepare the root output of prepare_output_file.py for differential measurements by splitting in- and out-events (fiducial particle level selection).")
    parser.add_argument("--input", help="Path to the input ROOT directory as prepared by prepare_output_file.py.")
    args = parser.parse_args()

    # Create a new directory "root_fid_split" at the same level as the input directory
    output_root_split_dir = os.path.join(args.input, "root_fid_split")
    os.makedirs(output_root_split_dir, exist_ok=True)

    # Traverse the input directory and process each .root file
    for root, dirs, files in os.walk(os.path.join(args.input, 'root')):
        for directory in dirs:
            input_directory_path = os.path.join(root, directory)
            output_directory_path = os.path.join(output_root_split_dir, directory)
            os.makedirs(output_directory_path, exist_ok=True)
            # Copy the root file to the new location
            for file in os.listdir(input_directory_path):
                if file.endswith(".root"):
                    input_file_path = os.path.join(input_directory_path, file)
                    output_file_path = os.path.join(output_directory_path, file)
                    logger.info(f"Processing file: {input_file_path}")
                    if directory == "Data":
                        shutil.copyfile(input_file_path, output_file_path)
                    else:
                        with uproot.recreate(output_file_path) as output_file:
                            process_root_file(input_file_path, output_file)


if __name__ == "__main__":
    main()