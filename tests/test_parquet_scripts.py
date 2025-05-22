import os
import json

import requests
import zipfile

import pytest
import subprocess

import uproot

file_dict = {
    "GluGluHtoGG_M-125": "output_GluGluHToGG_M125_13TeV_amcatnloFXFX_pythia8.root",
    "VBFHtoGG_M-125": "output_VBFHToGG_M125_13TeV_amcatnlo_pythia8.root",
    "VHtoGG_M-125": "output_VHToGG_M125_13TeV_amcatnlo_pythia8.root",
    "ttHtoGG_M-125": "output_TTHToGG_M125_13TeV_amcatnlo_pythia8.root",
}

tree_dict = {
    "GluGluHtoGG_M-125": "ggh_125",
    "VBFHtoGG_M-125": "vbf_125",
    "VHtoGG_M-125": "vh_125",
    "ttHtoGG_M-125": "tth_125",
}


def test_merge_parquet_and_convert():
    """
    Test if merging parquet files and converting to ROOT format with the helper scripts works
    """
    if not os.path.exists("./tests/samples/parquet_files/merged/"):
        os.makedirs("./tests/samples/parquet_files/merged/")
    os.system("cp ./tests/test_cat.json ./bottom_line/category.json")
    x = os.system("merge_parquet.py --source ./tests/samples/parquet_files/singles/ --target ./tests/samples/parquet_files/merged/ --cats test_cat.json --skip-normalisation")
    x += os.system("convert_parquet_to_root.py ./tests/samples/parquet_files/merged/merged.parquet ./tests/samples/parquet_files/merged/merged.root mc --cats category.json")

    # clean up
    subprocess.run("rm -r ./tests/samples/parquet_files/merged", shell=True)
    subprocess.run("rm ./bottom_line/category.json", shell=True)

    assert x==0


def test_prepare_output_file():
    """
    Test if preparation of output file with the helper script works
    """
    # Download and unzip without relying on wget or unzip for slim python CI
    url = "https://cernbox.cern.ch/remote.php/dav/public-files/J7s2HFTJ0WrdVRg/example_HiggsDNA_output.zip"
    output_file = "example_HiggsDNA_output.zip"
    response = requests.get(url)
    with open(output_file, "wb") as f:
        f.write(response.content)
    with zipfile.ZipFile(output_file, 'r') as zip_ref:
        zip_ref.extractall(".")

    os.remove(output_file) # Clean up

    location = os.path.abspath("example_HiggsDNA_output")
    x = os.system("prepare_output_file.py --input example_HiggsDNA_output/data --merge --root --cats --catDict cat_dict_inclusive_data.json")
    x += os.system("prepare_output_file.py --input example_HiggsDNA_output/signal --merge --root --cats --catDict cat_dict_inclusive_MC.json --syst --varDict var_dict.json")
    assert x == 0, "prepare_output_file.py didn't run properly"

    # check merged and root directories
    assert os.path.exists(f"{location}/data/merged")
    assert os.path.exists(f"{location}/data/root")
    assert os.path.exists(f"{location}/signal/merged")
    assert os.path.exists(f"{location}/signal/root")

    with open("cat_dict_inclusive_data.json") as f:
        categories = list(json.load(f).keys())

    # check merged parquet files for data
    for era in ["C", "D", "E", "F", "G"]:
        for cat in categories:
            assert os.path.exists(f"{location}/data/merged/Data_2022/Data{era}_2022_{cat}_merged.parquet")
            assert os.path.exists(f"{location}/data/merged/Data_2022/allData_{cat}_merged.parquet")

    # check root files for data
    assert os.path.exists(f"{location}/data/root/Data/allData_2022.root")

    with uproot.open(f"{location}/data/root/Data/allData_2022.root") as f:
        assert f.keys() == ["DiphotonTree;1"] + [f"DiphotonTree/Data_13TeV_{cat};1" for cat in categories]

    # check merged parquet files for MC
    with open("cat_dict_inclusive_MC.json") as f:
        categories = list(json.load(f).keys())

    with open("var_dict.json") as f:
        variations = list(json.load(f).values())

    for sample in ["GluGluHtoGG_M-125", "VBFHtoGG_M-125", "VHtoGG_M-125", "ttHtoGG_M-125"]:
        for leak in ["preEE", "postEE"]:
            for variation in variations:
                assert os.path.exists(f"{location}/signal/merged/{sample}_{leak}/{variation}")
                for cat in categories:
                    assert os.path.exists(f"{location}/signal/merged/{sample}_{leak}/{variation}/{cat}_merged.parquet")

            assert os.path.exists(f"{location}/signal/root/{sample}_{leak}")
            assert os.path.exists(f"{location}/signal/root/{sample}_{leak}/{file_dict[sample]}")

    # check root files for MC
    with open("var_dict.json") as f:
        variations = list(json.load(f).keys())

    for sample in ["GluGluHtoGG_M-125", "VBFHtoGG_M-125", "VHtoGG_M-125", "ttHtoGG_M-125"]:
        for leak in ["preEE", "postEE"]:
            expected_trees = ["DiphotonTree;1"]
            for cat in categories:
                for variation in variations:
                    if variation == "NOMINAL":
                        expected_trees.append(f"DiphotonTree/{tree_dict[sample]}_13TeV_{cat};1")
                    else:
                        expected_trees.append(f"DiphotonTree/{tree_dict[sample]}_13TeV_{cat}_{variation}01sigma;1")

            with uproot.open(f"{location}/signal/root/{sample}_{leak}/{file_dict[sample]}") as f:
                assert f.keys() == expected_trees

    # clean up
    os.system(f"rm -r {location}")
    os.system("rm cat_dict_inclusive_data.json")
    os.system("rm cat_dict_inclusive_MC.json")
    os.system("rm var_dict.json")
