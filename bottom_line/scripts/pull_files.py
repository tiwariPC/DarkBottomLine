#!/usr/bin/env python
import argparse
import os
from bottom_line.utils.logger_utils import setup_logger
import requests
import urllib.request
import pathlib
import shutil
import subprocess
from distutils.dir_util import copy_tree
import importlib.resources as resources

resource_dir = resources.files("bottom_line")


# ---------------------- A few helping functions  ----------------------


def unzip_gz_with_gunzip(logger, input_path, output_path=None):
    try:
        # Check if gunzip is available in the system
        subprocess.check_call(
            ["gunzip", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        if os.path.isdir(input_path):
            # If input_path is a directory, process all .gz files in the directory
            for root, _, files in os.walk(input_path):
                for file in files:
                    if file.endswith(".gz"):
                        input_file = os.path.join(root, file)
                        output_file = os.path.join(root, file[:-3])  # Remove .gz extension
                        with open(output_file, "wb") as output:
                            with open(input_file, "rb") as input_gz:
                                subprocess.check_call(["gunzip", "-c"], stdin=input_gz, stdout=output)
                        logger.info(f"File '{input_file}' successfully unzipped to '{output_file}'.")
                        # Remove the gz file after extraction
                        os.remove(input_file)
                        logger.info(f"File '{input_file}' deleted.")
        else:
            # If input_path is a file, process the single file
            if output_path is None:
                output_path = input_path[:-3]  # Remove .gz extension
            with open(output_path, "wb") as output:
                with open(input_path, "rb") as input_gz:
                    subprocess.check_call(["gunzip", "-c"], stdin=input_gz, stdout=output)
            logger.info(f"File '{input_path}' successfully unzipped to '{output_path}'.")
            # Remove the gz file after extraction
            os.remove(input_path)
            logger.info(f"File '{input_path}' deleted.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error: {e}")
    else:
        pass


def copy_xrdcp(logger, target_name, ikey, from_path, to_path):
    fs = "root://eoscms.cern.ch/"
    try:
        # Check the file exists
        res = subprocess.run(
            ["xrdfs", fs, "stat", from_path], text=True, capture_output=True
        )
        if res.returncode != 0:
            logger.error(res)
            raise Exception(f"Could not stat {from_path}, {res.stderr}")
        is_dir = "IsDir" in res.stdout

        # copy
        if is_dir:
            # Create the base directory as we stripped the last part
            p = pathlib.Path(to_path)
            p.mkdir(parents=True, exist_ok=True)

            # Copy everything
            res = subprocess.run(["xrdcp", "-r", "-f", "-s", fs + from_path, to_path])

            # Emulate the copy_tree function for remote directories
            items = os.listdir(to_path)
            top_dirs = [
                os.path.join(to_path, item)
                for item in items
                if os.path.isdir(os.path.join(to_path, item))
            ]
            if len(top_dirs) == 0:
                logger.debug(f"No top directories found in {to_path}")
            else:
                # Move all the files from the top directories to the base directory
                for top_dir in top_dirs:
                    for item in os.listdir(top_dir):
                        src_path = os.path.join(top_dir, item)
                        dest_path = os.path.join(to_path, item)
                        # Check if destination exists and remove it if it does
                        if os.path.exists(dest_path):
                            if os.path.isdir(dest_path):
                                shutil.rmtree(dest_path)
                            else:
                                os.remove(dest_path)
                        shutil.move(src_path, dest_path)
                    os.rmdir(top_dir)
        else:
            res = subprocess.run(["xrdcp", "-f", "-s", fs + from_path, to_path])
        if res.returncode != 0:
            logger.error(res)
            raise Exception(f"Could not copy {from_path} to {to_path}, {res.stderr}")
        logger.info(
            "[ {} ] {}: xrdcp from {} to {}".format(
                target_name,
                ikey,
                from_path,
                to_path,
            )
        )
    except Exception as e:
        logger.error(
            "[ {} ] {}: Can't xrdcp from {}: {}".format(target_name, ikey, from_path, e)
        )


def fetch_file(target_name, logger, from_to_dict, use_xrdcp=False, type="url"):
    for ikey in from_to_dict.keys():
        # Wrap the source and destination into lists if needed.
        src = from_to_dict[ikey]["from"]
        dest = from_to_dict[ikey]["to"]
        srcs = src if isinstance(src, list) else [src]
        dests = dest if isinstance(dest, list) else [dest]

        if len(srcs) != len(dests):
            logger.error(
                f"[ {target_name} ] {ikey}: Mismatch in number of sources and destinations."
            )
            continue

        if type == "url":
            # For each pair of source URL and destination file.
            for s, d in zip(srcs, dests):
                try:
                    with urllib.request.urlopen(s) as f:
                        json_object = f.read().decode("utf-8")
                except Exception:
                    logger.info(
                        "INFO: urllib did not work, falling back to requests to fetch file from URL..."
                    )
                    pass
                try:
                    response = requests.get(s, verify=True)
                    response.raise_for_status()  # Raise an exception for HTTP errors
                    json_object = response.text
                    # Create the destination directory
                    p = pathlib.Path(d).parent
                    p.mkdir(parents=True, exist_ok=True)
                    with open(d, "w") as f:
                        f.write(json_object)
                    logger.info(
                        f"[ {target_name} ] {ikey}: Download from {s} to {d}"
                    )
                except Exception as ex:
                    logger.error(
                        f"[ {target_name} ] {ikey}: Can't download from {s}: {ex}"
                    )
        elif type == "copy":
            for s, d in zip(srcs, dests):
                try:
                    # Check that the type of the file system is specified
                    assert from_to_dict[ikey].get("type") is not None, "Type of file system must be specified"
                    # Create the destination directory
                    p = pathlib.Path(d).parent
                    p.mkdir(parents=True, exist_ok=True)
                    # Proceed with copy
                    if from_to_dict[ikey]["type"] == "eos" and use_xrdcp:
                        copy_xrdcp(logger, target_name, ikey, s, d)
                    else:
                        if os.path.isdir(s):
                            copy_tree(s, d)
                        else:
                            shutil.copy(s, d)
                    logger.info(
                        f"[ {target_name} ] {ikey}: Copy from {s} to {d}"
                    )
                except Exception as ex:
                    logger.error(
                        f"[ {target_name} ] {ikey}: Can't copy from {s}: {ex}"
                    )


def get_jec_files(logger, target_dir, use_xrdcp=False):
    if target_dir is not None:
        to_prefix = target_dir
    else:
        to_prefix = os.path.join(resource_dir, "../bottom_line/systematics/data/")

    from_to_dict = {
        "2017": {
            "from": "/eos/cms/store/group/phys_higgs/cmshgg/tbevilac/JECDatabase/textFiles/Summer19UL17_V5_MC",
            "to": f"{to_prefix}/Summer19UL17_MC/JEC/",
            "type": "eos",
        },
        "2022postEE": {
            "from": "/eos/cms/store/group/phys_higgs/cmshgg/tbevilac/JECDatabase/textFiles/Winter22Run3_V2_MC",
            "to": f"{to_prefix}/Winter22Run3_MC/JEC/",
            "type": "eos",
        },
    }
    fetch_file("JEC", logger, from_to_dict, use_xrdcp=use_xrdcp, type="copy")
    os.system(f"rename .txt .junc.txt {to_prefix}/*/JEC/*Uncertainty*.txt")


def get_jer_files(logger, target_dir, use_xrdcp=False):
    if target_dir is not None:
        to_prefix = target_dir
    else:
        to_prefix = os.path.join(resource_dir, "../bottom_line/systematics/data/")

    from_to_dict = {
        "2017": {
            "from": "/eos/cms/store/group/phys_higgs/cmshgg/tbevilac/JRDatabase/textFiles/Summer19UL17_JRV2_MC",
            "to": f"{to_prefix}/Summer19UL17_MC/JER/",
            "type": "eos",
        },
        "2022postEE": {
            "from": "/eos/cms/store/group/phys_higgs/cmshgg/tbevilac/JRDatabase/textFiles/JR_Winter22Run3_V1_MC",
            "to": f"{to_prefix}/Winter22Run3_MC/JER/",
            "type": "eos",
        },
    }
    fetch_file("JER", logger, from_to_dict, use_xrdcp=use_xrdcp, type="copy")
    os.system(
        f"rename PtResolution_AK4PFchs.txt PtResolution_AK4PFchs.jr.txt {to_prefix}/*/JER/*PtResolution_AK4PFchs.txt"
    )
    os.system(
        f"rename PtResolution_AK4PFPuppi.txt PtResolution_AK4PFPuppi.jr.txt {to_prefix}/*/JER/*PtResolution_AK4PFPuppi.txt"
    )
    os.system(
        f"rename SF_AK4PFchs.txt SF_AK4PFchs.jersf.txt {to_prefix}/*/JER/*SF_AK4PFchs.txt"
    )
    os.system(
        f"rename SF_AK4PFPuppi.txt SF_AK4PFPuppi.jersf.txt {to_prefix}/*/JER/*SF_AK4PFPuppi.txt"
    )


def get_material_json(logger, target_dir, use_xrdcp=False):
    if target_dir is not None:
        to_prefix = target_dir
    else:
        to_prefix = os.path.join(
            resource_dir, "../bottom_line/systematics/JSONs/Material"
        )

    from_to_dict = {
        "2016": {
            "from": "/eos/cms/store/group/phys_higgs/cmshgg/tbevilac/JSONs/2016/Material_2016.json",
            "to": f"{to_prefix}/2016/Material_2016.json",
            "type": "eos",
        },
        "2017": {
            "from": "/eos/cms/store/group/phys_higgs/cmshgg/tbevilac/JSONs/2017/Material_2017.json",
            "to": f"{to_prefix}/2017/Material_2017.json",
            "type": "eos",
        },
        "2018": {
            "from": "/eos/cms/store/group/phys_higgs/cmshgg/tbevilac/JSONs/2018/Material_2018.json",
            "to": f"{to_prefix}/2018/Material_2018.json",
            "type": "eos",
        },
    }
    fetch_file("Material", logger, from_to_dict, use_xrdcp=use_xrdcp, type="copy")


def get_fnuf_json(logger, target_dir, use_xrdcp=False):
    if target_dir is not None:
        to_prefix = target_dir
    else:
        to_prefix = os.path.join(resource_dir, "../bottom_line/systematics/JSONs/FNUF")

    from_to_dict = {
        "2016": {
            "from": "/eos/cms/store/group/phys_higgs/cmshgg/tbevilac/JSONs/2016/FNUF_2016.json",
            "to": f"{to_prefix}/2016/FNUF_2016.json",
            "type": "eos",
        },
        "2017": {
            "from": "/eos/cms/store/group/phys_higgs/cmshgg/tbevilac/JSONs/2017/FNUF_2017.json",
            "to": f"{to_prefix}/2017/FNUF_2017.json",
            "type": "eos",
        },
        "2018": {
            "from": "/eos/cms/store/group/phys_higgs/cmshgg/tbevilac/JSONs/2018/FNUF_2018.json",
            "to": f"{to_prefix}/2018/FNUF_2018.json",
            "type": "eos",
        },
        "2022": {
            "from": "/eos/cms/store/group/phys_higgs/cmshgg/earlyRun3Hgg/JSONs/FNUF_2022.json",
            "to": f"{to_prefix}/2022/FNUF_2022.json",
            "type": "eos",
        },
    }
    fetch_file("FNUF", logger, from_to_dict, use_xrdcp=use_xrdcp, type="copy")


def get_shower_shape_json(logger, target_dir, use_xrdcp=False):
    if target_dir is not None:
        to_prefix = target_dir
    else:
        to_prefix = os.path.join(
            resource_dir, "../bottom_line/systematics/JSONs/ShowerShape"
        )

    from_to_dict = {
        "2016": {
            "from": "/eos/cms/store/group/phys_higgs/cmshgg/tbevilac/JSONs/2016/ShowerShape_2016.json",
            "to": f"{to_prefix}/2016/ShowerShape_2016.json",
            "type": "eos",
        },
        "2017": {
            "from": "/eos/cms/store/group/phys_higgs/cmshgg/tbevilac/JSONs/2017/ShowerShape_2017.json",
            "to": f"{to_prefix}/2017/ShowerShape_2017.json",
            "type": "eos",
        },
        "2018": {
            "from": "/eos/cms/store/group/phys_higgs/cmshgg/tbevilac/JSONs/2018/ShowerShape_2018.json",
            "to": f"{to_prefix}/2018/ShowerShape_2018.json",
            "type": "eos",
        },
    }
    fetch_file("ShowerShape", logger, from_to_dict, use_xrdcp=use_xrdcp, type="copy")


def get_loose_mva_json(logger, target_dir, use_xrdcp=False):
    if target_dir is not None:
        to_prefix = target_dir
    else:
        to_prefix = os.path.join(
            resource_dir, "../bottom_line/systematics/JSONs/LooseMvaSF"
        )

    from_to_dict = {
        "2016": {
            "from": "/eos/cms/store/group/phys_higgs/cmshgg/tbevilac/JSONs/2016/LooseMvaSF_2016.json",
            "to": f"{to_prefix}/2016/LooseMvaSF_2016.json",
            "type": "eos",
        },
        "2017": {
            "from": "/eos/cms/store/group/phys_higgs/cmshgg/tbevilac/JSONs/2017/LooseMvaSF_2017.json",
            "to": f"{to_prefix}/2017/LooseMvaSF_2017.json",
            "type": "eos",
        },
        "2018": {
            "from": "/eos/cms/store/group/phys_higgs/cmshgg/tbevilac/JSONs/2018/LooseMvaSF_2018.json",
            "to": f"{to_prefix}/2018/LooseMvaSF_2018.json",
            "type": "eos",
        },
    }
    fetch_file("LooseMva", logger, from_to_dict, use_xrdcp=use_xrdcp, type="copy")


def get_trigger_json(logger, target_dir, use_xrdcp=False):
    if target_dir is not None:
        to_prefix = target_dir
    else:
        to_prefix = os.path.join(
            resource_dir, "../bottom_line/systematics/JSONs/TriggerSF"
        )

    from_to_dict = {
        "2016_lead": {
            "from": "/eos/cms/store/group/phys_higgs/cmshgg/tbevilac/JSONs/2016/TriggerSF_lead_2016.json",
            "to": f"{to_prefix}/2016/TriggerSF_lead_2016.json",
            "type": "eos",
        },
        "2016_sublead": {
            "from": "/eos/cms/store/group/phys_higgs/cmshgg/tbevilac/JSONs/2016/TriggerSF_sublead_2016.json",
            "to": f"{to_prefix}/2016/TriggerSF_sublead_2016.json",
            "type": "eos",
        },
        "2017_lead": {
            "from": "/eos/cms/store/group/phys_higgs/cmshgg/tbevilac/JSONs/2017/TriggerSF_lead_2017.json",
            "to": f"{to_prefix}/2017/TriggerSF_lead_2017.json",
            "type": "eos",
        },
        "2017_sublead": {
            "from": "/eos/cms/store/group/phys_higgs/cmshgg/tbevilac/JSONs/2017/TriggerSF_sublead_2017.json",
            "to": f"{to_prefix}/2017/TriggerSF_sublead_2017.json",
            "type": "eos",
        },
        "2018_lead": {
            "from": "/eos/cms/store/group/phys_higgs/cmshgg/tbevilac/JSONs/2018/TriggerSF_lead_2018.json",
            "to": f"{to_prefix}/2018/TriggerSF_lead_2018.json",
            "type": "eos",
        },
        "2018_sublead": {
            "from": "/eos/cms/store/group/phys_higgs/cmshgg/tbevilac/JSONs/2018/TriggerSF_sublead_2018.json",
            "to": f"{to_prefix}/2018/TriggerSF_sublead_2018.json",
            "type": "eos",
        },
        "2022preEE_lead": {
            "from": "/eos/cms/store/group/phys_higgs/cmshgg/fmausolf/HiggsDNA_JSONs/TriggerSF_lead_2022_preEE.json",
            "to": f"{to_prefix}/2022preEE/TriggerSF_lead_2022preEE.json",
            "type": "eos",
        },
        "2022preEE_sublead": {
            "from": "/eos/cms/store/group/phys_higgs/cmshgg/fmausolf/HiggsDNA_JSONs/TriggerSF_sublead_2022_preEE.json",
            "to": f"{to_prefix}/2022preEE/TriggerSF_sublead_2022preEE.json",
            "type": "eos",
        },
        "2022postEE_lead": {
            "from": "/eos/cms/store/group/phys_higgs/cmshgg/fmausolf/HiggsDNA_JSONs/TriggerSF_lead_2022_postEE.json",
            "to": f"{to_prefix}/2022postEE/TriggerSF_lead_2022postEE.json",
            "type": "eos",
        },
        "2022postEE_sublead": {
            "from": "/eos/cms/store/group/phys_higgs/cmshgg/fmausolf/HiggsDNA_JSONs/TriggerSF_sublead_2022_postEE.json",
            "to": f"{to_prefix}/2022postEE/TriggerSF_sublead_2022postEE.json",
            "type": "eos",
        },
    }
    fetch_file("TriggerSF", logger, from_to_dict, use_xrdcp=use_xrdcp, type="copy")


def get_presel_json(logger, target_dir, use_xrdcp=False):
    if target_dir is not None:
        to_prefix = target_dir
    else:
        to_prefix = os.path.join(
            resource_dir, "../bottom_line/systematics/JSONs/Preselection"
        )
    # Old ones with puely restricted probe and non-conservative uncertainties: "/eos/cms/store/group/phys_higgs/cmshgg/earlyRun3Hgg/SFs/preselection/restrictedProbe"
    # Old ones with restricted probe and conservative uncertainties: /eos/cms/store/group/phys_higgs/cmshgg/earlyRun3Hgg/SFs/preselection/restrictedProbeConservativeUncs

    from_to_dict = {
        "2016": {
            "from": "/eos/cms/store/group/phys_higgs/cmshgg/tbevilac/JSONs/2016/PreselSF_2016.json",
            "to": f"{to_prefix}/2016/PreselSF_2016.json",
            "type": "eos",
        },
        "2017": {
            "from": "/eos/cms/store/group/phys_higgs/cmshgg/tbevilac/JSONs/2017/PreselSF_2017.json",
            "to": f"{to_prefix}/2017/PreselSF_2017.json",
            "type": "eos",
        },
        "2018": {
            "from": "/eos/cms/store/group/phys_higgs/cmshgg/tbevilac/JSONs/2018/PreselSF_2018.json",
            "to": f"{to_prefix}/2018/PreselSF_2018.json",
            "type": "eos",
        },
        "2022preEE": {
            "from": "/eos/cms/store/group/phys_higgs/cmshgg/fmausolf/HiggsDNA_JSONs/HggSFsSuman19Apr2024/Preselection_2022PreEE_Final.json",
            "to": f"{to_prefix}/2022/Preselection_2022PreEE.json",
            "type": "eos",
        },
        "2022postEE": {
            "from": "/eos/cms/store/group/phys_higgs/cmshgg/fmausolf/HiggsDNA_JSONs/HggSFsSuman19Apr2024/Preselection_2022PostEE_Final.json",
            "to": f"{to_prefix}/2022/Preselection_2022PostEE.json",
            "type": "eos",
        },
    }

    fetch_file("PreselSF", logger, from_to_dict, use_xrdcp=use_xrdcp, type="copy")


def get_eveto_json(logger, target_dir, use_xrdcp=False):
    if target_dir is not None:
        to_prefix = target_dir
    else:
        to_prefix = os.path.join(
            resource_dir, "../bottom_line/systematics/JSONs/ElectronVetoSF"
        )

    from_to_dict = {
        "2016": {
            "from": "/eos/cms/store/group/phys_higgs/cmshgg/tbevilac/JSONs/2016/eVetoSF_2016.json",
            "to": f"{to_prefix}/2016/eVetoSF_2016.json",
            "type": "eos",
        },
        "2017": {
            "from": "/eos/cms/store/group/phys_higgs/cmshgg/tbevilac/JSONs/2017/eVetoSF_2017.json",
            "to": f"{to_prefix}/2017/eVetoSF_2017.json",
            "type": "eos",
        },
        "2018": {
            "from": "/eos/cms/store/group/phys_higgs/cmshgg/tbevilac/JSONs/2018/eVetoSF_2018.json",
            "to": f"{to_prefix}/2018/eVetoSF_2018.json",
            "type": "eos",
        },
        "2022preEE": {
            "from": "/eos/cms/store/group/phys_higgs/cmshgg/fmausolf/HiggsDNA_JSONs/preEE_CSEV_SFcorrections.json",
            "to": f"{to_prefix}/2022/preEE_CSEV_SFcorrections.json",
            "type": "eos",
        },
        "2022postEE": {
            "from": "/eos/cms/store/group/phys_higgs/cmshgg/fmausolf/HiggsDNA_JSONs/postEE_CSEV_SFcorrections.json",
            "to": f"{to_prefix}/2022/postEE_CSEV_SFcorrections.json",
            "type": "eos",
        },
    }
    fetch_file("eVetoSF", logger, from_to_dict, use_xrdcp=use_xrdcp, type="copy")


def get_btag_json(logger, target_dir, use_xrdcp=False):
    if target_dir is not None:
        to_prefix = target_dir
    else:
        to_prefix = os.path.join(resource_dir, "../bottom_line/systematics/JSONs/bTagSF/")

    from_to_dict = {
        "2016preVFP": {
            "from": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/BTV/2016preVFP_UL/btagging.json.gz",
            "to": f"{to_prefix}/2016preVFP_UL/btagging.json.gz",
            "type": "cvmfs",
        },
        "2016postVFP": {
            "from": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/BTV/2016postVFP_UL/btagging.json.gz",
            "to": f"{to_prefix}/2016postVFP_UL/btagging.json.gz",
            "type": "cvmfs",
        },
        "2017": {
            "from": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/BTV/2017_UL/btagging.json.gz",
            "to": f"{to_prefix}/2017_UL/btagging.json.gz",
            "type": "cvmfs",
        },
        "2018": {
            "from": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/BTV/2018_UL/btagging.json.gz",
            "to": f"{to_prefix}/2018_UL/btagging.json.gz",
            "type": "cvmfs",
        },
        "2022preEE": {
            "from": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/BTV/2022_Summer22/btagging.json.gz",
            "to": f"{to_prefix}/2022_Summer22/btagging.json.gz",
            "type": "cvmfs",
        },
        "2022postEE": {
            "from": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/BTV/2022_Summer22EE/btagging.json.gz",
            "to": f"{to_prefix}/2022_Summer22EE/btagging.json.gz",
            "type": "cvmfs",
        },
        "2023preBPix": {
            "from": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/BTV/2023_Summer23/btagging.json.gz",
            "to": f"{to_prefix}/2023_Summer23/btagging.json.gz",
            "type": "cvmfs",
        },
        "2023postBPix": {
            "from": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/BTV/2023_Summer23BPix/btagging.json.gz",
            "to": f"{to_prefix}/2023_Summer23BPix/btagging.json.gz",
            "type": "cvmfs",
        },
    }
    fetch_file("bTag", logger, from_to_dict, use_xrdcp=use_xrdcp, type="copy")


def get_ctag_json(logger, target_dir, use_xrdcp=False):
    if target_dir is not None:
        to_prefix = target_dir
    else:
        to_prefix = os.path.join(resource_dir, "../bottom_line/systematics/JSONs/cTagSF/")

    from_to_dict = {
        "2016preVFP": {
            "from": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/BTV/2016preVFP_UL/ctagging.json.gz",
            "to": f"{to_prefix}/2016/ctagging_2016preVFP.json.gz",
            "type": "cvmfs",
        },
        "2016postVFP": {
            "from": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/BTV/2016postVFP_UL/ctagging.json.gz",
            "to": f"{to_prefix}/2016/ctagging_2016postVFP.json.gz",
            "type": "cvmfs",
        },
        "2017": {
            "from": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/BTV/2017_UL/ctagging.json.gz",
            "to": f"{to_prefix}/2017/ctagging_2017.json.gz",
            "type": "cvmfs",
        },
        "2018": {
            "from": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/BTV/2018_UL/ctagging.json.gz",
            "to": f"{to_prefix}/2018/ctagging_2018.json.gz",
            "type": "cvmfs",
        },
    }
    fetch_file("cTag", logger, from_to_dict, use_xrdcp=use_xrdcp, type="copy")


def get_photonid_json(logger, target_dir, use_xrdcp=False):
    if target_dir is not None:
        to_prefix = target_dir
    else:
        to_prefix = os.path.join(
            resource_dir, "../bottom_line/systematics/JSONs/SF_photon_ID"
        )

    from_to_dict = {
        "2016preVFP": {
            "from": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/EGM/2016preVFP_UL/photon.json.gz",
            "to": f"{to_prefix}/2016/photon_preVFP.json.gz",
            "type": "cvmfs",
        },
        "2016postVFP": {
            "from": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/EGM/2016postVFP_UL/photon.json.gz",
            "to": f"{to_prefix}/2016/photon_postVFP.json.gz",
            "type": "cvmfs",
        },
        "2017": {
            "from": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/EGM/2017_UL/photon.json.gz",
            "to": f"{to_prefix}/2017/photon.json.gz",
            "type": "cvmfs",
        },
        "2018": {
            "from": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/EGM/2018_UL/photon.json.gz",
            "to": f"{to_prefix}/2018/photon.json.gz",
            "type": "cvmfs",
        },
        "2022preEE": {
            "from": "/eos/cms/store/group/phys_higgs/cmshgg/fmausolf/HiggsDNA_JSONs/HggSFsSuman19Apr2024/PhotonIDMVA_2022PreEE_Final.json",
            "to": f"{to_prefix}/2022/PhotonIDMVA_2022PreEE.json",
            "type": "eos",
        },
        "2022postEE": {
            "from": "/eos/cms/store/group/phys_higgs/cmshgg/fmausolf/HiggsDNA_JSONs/HggSFsSuman19Apr2024/PhotonIDMVA_2022PostEE_Final.json",
            "to": f"{to_prefix}/2022/PhotonIDMVA_2022PostEE.json",
            "type": "eos",
        },
    }
    fetch_file("PhotonID", logger, from_to_dict, use_xrdcp=use_xrdcp, type="copy")


def get_scale_and_smearing(logger, target_dir, use_xrdcp=False):
    # see https://twiki.cern.ch/twiki/bin/viewauth/CMS/EgammSFandSSRun3#Scale_And_Smearings_Correctionli for Run 3
    # see https://cms-talk.web.cern.ch/t/pnoton-energy-corrections-in-nanoaod-v11/34327/2 for Run 2, jsons are from https://github.com/cms-egamma/ScaleFactorsJSON/tree/master
    if target_dir is not None:
        to_prefix = target_dir
    else:
        to_prefix = os.path.join(
            resource_dir, "../bottom_line/systematics/JSONs/scaleAndSmearing"
        )

    from_to_dict = {
        "2016preVFP": {
            "from": "/eos/cms/store/group/phys_higgs/cmshgg/tbevilac/JSONs/SandS/EGM_ScaleUnc_2016preVFP.json",
            "to": f"{to_prefix}/EGM_ScaleUnc_2016preVFP.json",
            "type": "eos",
        },
        "2016postVFP": {
            "from": "/eos/cms/store/group/phys_higgs/cmshgg/tbevilac/JSONs/SandS/EGM_ScaleUnc_2016postVFP.json",
            "to": f"{to_prefix}/EGM_ScaleUnc_2016postVFP.json",
            "type": "eos",
        },
        "2017": {
            "from": "/eos/cms/store/group/phys_higgs/cmshgg/tbevilac/JSONs/SandS/EGM_ScaleUnc_2017.json",
            "to": f"{to_prefix}/EGM_ScaleUnc_2017.json",
            "type": "eos",
        },
        "2018": {
            "from": "/eos/cms/store/group/phys_higgs/cmshgg/tbevilac/JSONs/SandS/EGM_ScaleUnc_2018.json",
            "to": f"{to_prefix}/EGM_ScaleUnc_2018.json",
            "type": "eos",
        },
        "2022preEE": {
            "from": "/eos/cms/store/group/phys_egamma/ScaleFactors/Data2022/ForRe-recoBCD/SS/photonSS.json.gz",
            "to": f"{to_prefix}/SS_Rereco2022BCD.json.gz",
            "type": "eos",
        },
        "2022postEE": {
            "from": "/eos/cms/store/group/phys_egamma/ScaleFactors/Data2022/ForRe-recoE+PromptFG/SS/photonSS.json.gz",
            "to": f"{to_prefix}/SS_RerecoE_PromptFG_2022.json.gz",
            "type": "eos",
        },
        "2022preEE_Electrons": {
            "from": "/eos/cms/store/group/phys_egamma/ScaleFactors/Data2022/ForRe-recoBCD/SS/electronSS.json.gz",
            "to": f"{to_prefix}/SS_Electron_Rereco2022BCD.json.gz",
            "type": "eos",
        },
        "2022postEE_Electrons": {
            "from": "/eos/cms/store/group/phys_egamma/ScaleFactors/Data2022/ForRe-recoE+PromptFG/SS/electronSS.json.gz",
            "to": f"{to_prefix}/SS_Electron_RerecoE_PromptFG_2022.json.gz",
            "type": "eos",
        },
        "2023preBPix": {
            "from": "/eos/cms/store/group/phys_egamma/ScaleFactors/Data2023/ForPrompt23C/SS/photonSS.json.gz",
            "to": f"{to_prefix}/SS_Prompt23C.json.gz",
            "type": "eos",
        },
        "2023postBPix": {
            "from": "/eos/cms/store/group/phys_egamma/ScaleFactors/Data2023/ForPrompt23D/SS/photonSS.json.gz",
            "to": f"{to_prefix}/SS_Prompt23D.json.gz",
            "type": "eos",
        },
        "2023preBPix_Electrons": {
            "from": "/eos/cms/store/group/phys_egamma/ScaleFactors/Data2023/ForPrompt23C/SS/electronSS.json.gz",
            "to": f"{to_prefix}/SS_Electron_Prompt23C.json.gz",
            "type": "eos",
        },
        "2023postBPix_Electrons": {
            "from": "/eos/cms/store/group/phys_egamma/ScaleFactors/Data2023/ForPrompt23D/SS/electronSS.json.gz",
            "to": f"{to_prefix}/SS_Electron_Prompt23D.json.gz",
            "type": "eos",
        },
    }
    fetch_file(
        "Scale and Smearing", logger, from_to_dict, use_xrdcp=use_xrdcp, type="copy"
    )
    # Unzip everything everywhere, all at once (did you understand that reference?)
    unzip_gz_with_gunzip(logger, to_prefix)



def get_scale_and_smearing_IJazZ(logger, target_dir, use_xrdcp=False):
    # see https://twiki.cern.ch/twiki/bin/viewauth/CMS/EgammSFandSSRun3#Scale_And_Smearings_Correctionli for Run 3
    # jsons are not taken from the jsonpog-integration repo because the 2G jsons are not there and the PRNG corrections are removed.
    if target_dir is not None:
        to_prefix = target_dir
    else:
        to_prefix = os.path.join(
            resource_dir, "../bottom_line/systematics/JSONs/scaleAndSmearing"
        )

    from_to_dict = {
        "2022preEE": {
            "from": ["/eos/cms/store/group/phys_higgs/cmshgg/ingredients/2022/SandS_IJazZ/preEE/EGMScalesSmearing_Pho_2022preEE.v1.json.gz",
                     "/eos/cms/store/group/phys_higgs/cmshgg/ingredients/2022/SandS_IJazZ/preEE/EGMScalesSmearing_Pho_2022preEE2G.v1.json.gz"],
            "to":   [f"{to_prefix}/EGMScalesSmearing_Pho_2022preEE.v1.json.gz",
                     f"{to_prefix}/EGMScalesSmearing_Pho_2022preEE2G.v1.json.gz"],
            "type": "eos",
        },
        "2022postEE": {
            "from": ["/eos/cms/store/group/phys_higgs/cmshgg/ingredients/2022/SandS_IJazZ/postEE/EGMScalesSmearing_Pho_2022postEE.v1.json.gz",
                     "/eos/cms/store/group/phys_higgs/cmshgg/ingredients/2022/SandS_IJazZ/postEE/EGMScalesSmearing_Pho_2022postEE2G.v1.json.gz"],
            "to": [f"{to_prefix}/EGMScalesSmearing_Pho_2022postEE.v1.json.gz",
                   f"{to_prefix}/EGMScalesSmearing_Pho_2022postEE2G.v1.json.gz"],
            "type": "eos",
        },
        "2022preEE_Electrons": {
            "from": ["/eos/cms/store/group/phys_higgs/cmshgg/ingredients/2022/SandS_IJazZ_for_electrons/preEE/EGMScalesSmearing_Ele_2022preEE.v1.json.gz",
                     "/eos/cms/store/group/phys_higgs/cmshgg/ingredients/2022/SandS_IJazZ_for_electrons/preEE/EGMScalesSmearing_Ele_2022preEE2G.v1.json.gz"],
            "to":   [f"{to_prefix}/EGMScalesSmearing_Ele_2022preEE.v1.json.gz",
                     f"{to_prefix}/EGMScalesSmearing_Ele_2022preEE2G.v1.json.gz"],
            "type": "eos",
        },
        "2022postEE_Electrons": {
            "from": ["/eos/cms/store/group/phys_higgs/cmshgg/ingredients/2022/SandS_IJazZ_for_electrons/postEE/EGMScalesSmearing_Ele_2022postEE.v1.json.gz",
                     "/eos/cms/store/group/phys_higgs/cmshgg/ingredients/2022/SandS_IJazZ_for_electrons/postEE/EGMScalesSmearing_Ele_2022postEE2G.v1.json.gz"],
            "to": [f"{to_prefix}/EGMScalesSmearing_Ele_2022postEE.v1.json.gz",
                   f"{to_prefix}/EGMScalesSmearing_Ele_2022postEE2G.v1.json.gz"],
            "type": "eos",
        },
        "2023preBPix": {
            "from": ["/eos/cms/store/group/phys_higgs/cmshgg/ingredients/2023/SandS_IJazZ/preBPix/EGMScalesSmearing_Pho_2023preBPIX.v1.json.gz",
                     "/eos/cms/store/group/phys_higgs/cmshgg/ingredients/2023/SandS_IJazZ/preBPix/EGMScalesSmearing_Pho_2023preBPIX2G.v1.json.gz"],
            "to": [f"{to_prefix}/EGMScalesSmearing_Pho_2023preBPIX.v1.json.gz",
                   f"{to_prefix}/EGMScalesSmearing_Pho_2023preBPIX2G.v1.json.gz"],
            "type": "eos",
        },
        "2023postBPix": {
            "from": ["/eos/cms/store/group/phys_higgs/cmshgg/ingredients/2023/SandS_IJazZ/postBPix/EGMScalesSmearing_Pho_2023postBPIX.v1.json.gz",
                     "/eos/cms/store/group/phys_higgs/cmshgg/ingredients/2023/SandS_IJazZ/postBPix/EGMScalesSmearing_Pho_2023postBPIX2G.v1.json.gz"],
            "to": [f"{to_prefix}/EGMScalesSmearing_Pho_2023postBPIX.v1.json.gz",
                   f"{to_prefix}/EGMScalesSmearing_Pho_2023postBPIX2G.v1.json.gz"],
            "type": "eos",
        },

        "2023preBPix_Electrons": {
            "from": ["/eos/cms/store/group/phys_higgs/cmshgg/ingredients/2023/SandS_IJazZ_for_electrons/preBPix/EGMScalesSmearing_Ele_2023preBPIX.v1.json.gz",
                     "/eos/cms/store/group/phys_higgs/cmshgg/ingredients/2023/SandS_IJazZ_for_electrons/preBPix/EGMScalesSmearing_Ele_2023preBPIX2G.v1.json.gz"],
            "to": [f"{to_prefix}/EGMScalesSmearing_Ele_2023preBPIX.v1.json.gz",
                   f"{to_prefix}/EGMScalesSmearing_Ele_2023preBPIX2G.v1.json.gz"],
            "type": "eos",
        },
        "2023postBPix_Electrons": {
            "from": ["/eos/cms/store/group/phys_higgs/cmshgg/ingredients/2023/SandS_IJazZ_for_electrons/postBPix/EGMScalesSmearing_Ele_2023postBPIX.v1.json.gz",
                     "/eos/cms/store/group/phys_higgs/cmshgg/ingredients/2023/SandS_IJazZ_for_electrons/postBPix/EGMScalesSmearing_Ele_2023postBPIX2G.v1.json.gz"],
            "to": [f"{to_prefix}/EGMScalesSmearing_Ele_2023postBPIX.v1.json.gz",
                   f"{to_prefix}/EGMScalesSmearing_Ele_2023postBPIX2G.v1.json.gz"],
            "type": "eos",
        },

    }
    fetch_file(
        "Scale and Smearing", logger, from_to_dict, use_xrdcp=use_xrdcp, type="copy"
    )

    # Unzip everything everywhere, all at once (did you understand that reference?)
    unzip_gz_with_gunzip(logger, to_prefix)


def get_mass_decorrelation_CDF(logger, target_dir, use_xrdcp=False):
    if target_dir is not None:
        to_prefix = target_dir
    else:
        to_prefix = os.path.join(resource_dir, "../bottom_line/tools")

    from_to_dict = {
        "2022": {
            "from": "/eos/cms/store/group/phys_higgs/cmshgg/ingredients/2022/decorrelation_CDFs/",
            "to": f"{to_prefix}/decorrelation_CDFs",
            "type": "eos",
        },
        "2023": {
            "from": "/eos/cms/store/group/phys_higgs/cmshgg/ingredients/2023/decorrelation_CDFs/",
            "to": f"{to_prefix}/decorrelation_CDFs",
            "type": "eos",
        },
    }
    fetch_file("CDFs", logger, from_to_dict, use_xrdcp=use_xrdcp, type="copy")


def get_Flow_files(logger, target_dir, use_xrdcp=False):
    if target_dir is not None:
        to_prefix = target_dir
    else:
        to_prefix = os.path.join(resource_dir, "../bottom_line/tools/flows")

    from_to_dict = {
        "Run3": {
            "from": "/eos/cms/store/group/phys_higgs/cmshgg/ingredients/Run3/",
            "to": f"{to_prefix}/run3_mvaID_models/",
            "type": "eos",
        },
        "2022CD": {
            "from": "/eos/cms/store/group/phys_higgs/cmshgg/ingredients/2022/Flows/preEE/",
            "to": f"{to_prefix}/preEE/",
            "type": "eos",
        },
        "2022EFG": {
            "from": "/eos/cms/store/group/phys_higgs/cmshgg/ingredients/2022/Flows/postEE/",
            "to": f"{to_prefix}/postEE/",
            "type": "eos",
        },
        "2023": {
            "from": "/eos/cms/store/group/phys_higgs/cmshgg/ingredients/2023/Flows/",
            "to": f"{to_prefix}/2023_model/",
            "type": "eos",
        },
    }
    fetch_file("Flows", logger, from_to_dict, use_xrdcp=use_xrdcp, type="copy")


def get_goldenjson(logger, target_dir, use_xrdcp=False):
    # References:
    # https://twiki.cern.ch/twiki/bin/view/CMS/PdmVRun3Analysis#Data
    # This is not really a correction JSON, so we only allow saving to a specific location
    # Commnenting out the code below, this was the previous method
    # if target_dir is not None:
    #    to_prefix = target_dir
    # else:
    #    to_prefix = os.path.join(
    #        resource_dir, "../metaconditions/pileup"
    #    )

    prefix = os.path.join(
        resource_dir, "../bottom_line/metaconditions/CAF/certification/"
    )

    from_to_dict = {
        "2016": {
            "from": "https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions16/13TeV/Legacy_2016/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt",
            "to": os.path.join(
                prefix,
                "Collisions16/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt",
            ),
        },
        "2017": {
            "from": "https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions17/13TeV/Legacy_2017/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt",
            "to": os.path.join(
                prefix,
                "Collisions17/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt",
            ),
        },
        "2018": {
            "from": "https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions18/13TeV/Legacy_2018/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt",
            "to": os.path.join(
                prefix,
                "Collisions18/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt",
            ),
        },
        "2022": {
            "from": "https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions22/Cert_Collisions2022_355100_362760_Golden.json",
            "to": os.path.join(
                prefix,
                "Collisions22/Cert_Collisions2022_355100_362760_Golden.json",
            ),
        },
        "2023": {
            "from": "https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions23/Cert_Collisions2023_366442_370790_Golden.json",
            "to": os.path.join(
                prefix,
                "Collisions23/Cert_Collisions2023_366442_370790_Golden.json",
            ),
        },
        "2024": {
            "from": "https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions24/Cert_Collisions2024_378981_386951_Golden.json",
            "to": os.path.join(
                prefix,
                "Collisions24/Cert_Collisions2024_378981_386951_Golden.json",
            ),
        },
    }

    fetch_file("GoldenJSON", logger, from_to_dict, type="url")


def get_jetmet_json(logger, target_dir, use_xrdcp=False):
    # References:
    # json pog of JME: https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/tree/master/POG/JME
    # jetmapveto: https://twiki.cern.ch/twiki/bin/viewauth/CMS/PdmVRun3Analysis#From_JME
    base_path = "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME"
    # Temporary directory until JME puts them somewhere centrally,
    # currently copied from https://indico.cern.ch/event/1450094/#1-corrections-for-the-pnet-pt
    eos_path_PNet = "/eos/cms/store/user/evourlio/JMEPNet_forHiggsDNA"
    if target_dir is not None:
        to_prefix = target_dir
    else:
        to_prefix = resource_dir

    from_to_dict = {
        "2016preVFP": {
            "from": os.path.join(base_path, "2016preVFP_UL"),
            "to": os.path.join(
                to_prefix,
                "../bottom_line/systematics/JSONs/POG/JME/2016preVFP_UL",
            ),
            "type": "cvfms",
        },
        "2016postVFP": {
            "from": os.path.join(base_path, "2016postVFP_UL"),
            "to": os.path.join(
                to_prefix,
                "../bottom_line/systematics/JSONs/POG/JME/2016postVFP_UL",
            ),
            "type": "cvmfs",
        },
        "2017": {
            "from": os.path.join(base_path, "2017_UL"),
            "to": os.path.join(
                to_prefix,
                "../bottom_line/systematics/JSONs/POG/JME/2017_UL",
            ),
            "type": "cvmfs",
        },
        "2018": {
            "from": os.path.join(base_path, "2018_UL"),
            "to": os.path.join(
                to_prefix,
                "../bottom_line/systematics/JSONs/POG/JME/2018_UL",
            ),
            "type": "cvmfs",
        },
        "2022Summer22": {
            "from": os.path.join(base_path, "2022_Summer22"),
            "to": os.path.join(
                to_prefix,
                "../bottom_line/systematics/JSONs/POG/JME/2022_Summer22",
            ),
            "type": "cvmfs",
        },
        "2022Summer22_PNet": {
            "from": os.path.join(eos_path_PNet, "2022_Summer22"),
            "to": os.path.join(
                to_prefix,
                "../bottom_line/systematics/JSONs/POG/JME/2022_Summer22",
            ),
            "type": "eos",
        },
        "2022Summer22EE": {
            "from": os.path.join(base_path, "2022_Summer22EE"),
            "to": os.path.join(
                to_prefix,
                "../bottom_line/systematics/JSONs/POG/JME/2022_Summer22EE",
            ),
            "type": "cvmfs",
        },
        "2022Summer22EE_PNet": {
            "from": os.path.join(eos_path_PNet, "2022_Summer22EE"),
            "to": os.path.join(
                to_prefix,
                "../bottom_line/systematics/JSONs/POG/JME/2022_Summer22EE",
            ),
            "type": "eos",
        },
        "2023_Summer23": {
            "from": os.path.join(base_path, "2023_Summer23"),
            "to": os.path.join(
                to_prefix,
                "../bottom_line/systematics/JSONs/POG/JME/2023_Summer23",
            ),
            "type": "cvmfs",
        },
        "2023_Summer23_PNet": {
            "from": os.path.join(eos_path_PNet, "2023_Summer23"),
            "to": os.path.join(
                to_prefix,
                "../bottom_line/systematics/JSONs/POG/JME/2023_Summer23",
            ),
            "type": "eos",
        },
        "2023_Summer23BPix": {
            "from": os.path.join(base_path, "2023_Summer23BPix"),
            "to": os.path.join(
                to_prefix,
                "../bottom_line/systematics/JSONs/POG/JME/2023_Summer23BPix",
            ),
            "type": "cvmfs",
        },
        "2023_Summer23BPix_PNet": {
            "from": os.path.join(eos_path_PNet, "2023_Summer23BPix"),
            "to": os.path.join(
                to_prefix,
                "../bottom_line/systematics/JSONs/POG/JME/2023_Summer23BPix",
            ),
            "type": "eos",
        },
    }

    fetch_file("JetMET", logger, from_to_dict, use_xrdcp=use_xrdcp, type="copy")


def get_pileup(logger, target_dir, use_xrdcp=False):
    # Base URL for pileup JSONs
    base_path = "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/LUM"

    if target_dir is not None:
        to_prefix = target_dir
    else:
        to_prefix = os.path.join(resource_dir, "../bottom_line/systematics/JSONs/pileup/")

    from_to_dict = {
        "2016preVFP": {
            "from": f"{base_path}/2016preVFP_UL/puWeights.json.gz",
            "to": f"{to_prefix}/pileup_2016preVFP.json.gz",
            "type": "cvmfs",
        },
        "2016postVFP": {
            "from": f"{base_path}/2016postVFP_UL/puWeights.json.gz",
            "to": f"{to_prefix}/pileup_2016postVFP.json.gz",
            "type": "cvmfs",
        },
        "2017": {
            "from": f"{base_path}/2017_UL/puWeights.json.gz",
            "to": f"{to_prefix}/pileup_2017.json.gz",
            "type": "cvmfs",
        },
        "2018": {
            "from": f"{base_path}/2018_UL/puWeights.json.gz",
            "to": f"{to_prefix}/pileup_2018.json.gz",
            "type": "cvmfs",
        },
        "2022_preEE": {
            "from": f"{base_path}/2022_Summer22/puWeights.json.gz",
            "to": f"{to_prefix}/pileup_2022preEE.json.gz",
            "type": "cvmfs",
        },
        "2022_postEE": {
            "from": f"{base_path}/2022_Summer22EE/puWeights.json.gz",
            "to": f"{to_prefix}/pileup_2022postEE.json.gz",
            "type": "cvmfs",
        },
        "2023_preBPix": {
            "from": f"{base_path}/2023_Summer23/puWeights.json.gz",
            "to": f"{to_prefix}/pileup_2023preBPix.json.gz",
            "type": "cvmfs",
        },
        "2023_postBPix": {
            "from": f"{base_path}/2023_Summer23BPix/puWeights.json.gz",
            "to": f"{to_prefix}/pileup_2023postBPix.json.gz",
            "type": "cvmfs",
        },
    }

    fetch_file("Pileup", logger, from_to_dict, use_xrdcp=use_xrdcp, type="copy")


def get_lowmass_diphotonmva_model(logger, target_dir, use_xrdcp=False):
    if target_dir is not None:
        to_prefix = target_dir
    else:
        to_prefix = resource_dir
        to_prefix = os.path.join(resource_dir, "../bottom_line/tools")

    from_to_dict = {
        "2022postEE": {
            "from": "/eos/cms/store/group/phys_higgs/cmshgg/jixiao/lowmass_diphoton/DiphotonXGboost_LM2022_postEE.onnx",
            "to": os.path.join(
                to_prefix,
                "lowmass_diphoton_mva/2022postEE/DiphotonXGboost_LM.onnx",
            ),
            "type": "eos",
        },
        "2022preEE": {
            "from": "/eos/cms/store/group/phys_higgs/cmshgg/jixiao/lowmass_diphoton/DiphotonXGboost_LM2022_postEE.onnx",
            "to": os.path.join(
                to_prefix,
                "lowmass_diphoton_mva/2022preEE/DiphotonXGboost_LM.onnx",
            ),
            "type": "eos",
        },
    }

    fetch_file(
        "LowMass-DiPhotonMVA", logger, from_to_dict, use_xrdcp=use_xrdcp, type="copy"
    )


def get_muon_SFs(logger, target_dir, use_xrdcp=False):
    if target_dir is not None:
        to_prefix = target_dir
    else:
        to_prefix = resource_dir

    from_to_dict = {
        "2022preEE": {
            "from": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/MUO/2022_Summer22/muon_Z.json.gz",
            "to": os.path.join(
                to_prefix,
                "../bottom_line/systematics/JSONs/POG/MUO/2022_Summer22/muon_Z.json.gz",
            ),
            "type": "cvmfs",
        },
        "2022postEE": {
            "from": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/MUO/2022_Summer22EE/muon_Z.json.gz",
            "to": os.path.join(
                to_prefix,
                "../bottom_line/systematics/JSONs/POG/MUO/2022_Summer22EE/muon_Z.json.gz",
            ),
            "type": "cvmfs",
        },
        "2023preBPix": {
            "from": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/MUO/2023_Summer23/muon_Z.json.gz",
            "to": os.path.join(
                to_prefix,
                "../bottom_line/systematics/JSONs/POG/MUO/2023_Summer23/muon_Z.json.gz",
            ),
            "type": "cvmfs",
        },
        "2023postBPix": {
            "from": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/MUO/2023_Summer23BPix/muon_Z.json.gz",
            "to": os.path.join(
                to_prefix,
                "../bottom_line/systematics/JSONs/POG/MUO/2023_Summer23BPix/muon_Z.json.gz",
            ),
            "type": "cvmfs",
        },
    }

    fetch_file("muonSF", logger, from_to_dict, use_xrdcp=use_xrdcp, type="copy")


def get_lowmass_dykiller_model(logger, target_dir, use_xrdcp=False):
    if target_dir is not None:
        to_prefix = target_dir
    else:
        to_prefix = resource_dir
        to_prefix = os.path.join(resource_dir, "../bottom_line/tools")

    from_to_dict = {
        "2022postEE": {
            "from": "/eos/cms/store/group/phys_higgs/cmshgg/jixiao/lowmass_dykiller/NN.onnx",
            "to": os.path.join(
                to_prefix,
                "lowmass_dykiller/2022postEE/NN.onnx",
            ),
            "type": "eos",
        },
        "2022preEE": {
            "from": "/eos/cms/store/group/phys_higgs/cmshgg/jixiao/lowmass_dykiller/NN.onnx",
            "to": os.path.join(
                to_prefix,
                "lowmass_dykiller/2022preEE/NN.onnx",
            ),
            "type": "eos",
        },
    }

    fetch_file(
        "LowMass-DYKilller", logger, from_to_dict, use_xrdcp=use_xrdcp, type="copy"
    )


def get_cqr_weights(logger, target_dir, use_xrdcp=False):
    if target_dir is not None:
        to_prefix = target_dir
    else:
        to_prefix = resource_dir
        to_prefix = os.path.join(
            resource_dir, "../bottom_line/metaconditions/corrections"
        )

    from_to_dict = {
        "2017": {
            "from": "/eos/cms/store/group/phys_higgs/cmshgg/ingredients/2017/cqr_weights/",
            "to": to_prefix,
            "type": "eos",
        }
    }

    fetch_file("CQR", logger, from_to_dict, use_xrdcp=use_xrdcp, type="copy")


def get_hgg_photon_id_mva_weights(logger, target_dir, use_xrdcp=False):
    if target_dir is not None:
        to_prefix = target_dir
    else:
        to_prefix = resource_dir
        to_prefix = os.path.join(
            resource_dir, "../bottom_line/metaconditions/photon_id_mva_weights"
        )

    from_to_dict = {
        "2017": {
            "from": "/eos/cms/store/group/phys_higgs/cmshgg/ingredients/2017/hgg_photon_id_mva_weights/",
            "to": to_prefix,
            "type": "eos",
        }
    }

    fetch_file("PhotonIDMVA", logger, from_to_dict, use_xrdcp=use_xrdcp, type="copy")


def get_diphoton_id_mva_weights(logger, target_dir, use_xrdcp=False):
    if target_dir is not None:
        to_prefix = target_dir
    else:
        to_prefix = resource_dir
        to_prefix = os.path.join(resource_dir, "../bottom_line/metaconditions/diphoton")

    from_to_dict = {
        "2017": {
            "from": "/eos/cms/store/group/phys_higgs/cmshgg/ingredients/2017/diphoton_id_mva_weights/",
            "to": to_prefix,
            "type": "eos",
        }
    }

    fetch_file("DiphotonIDMVA", logger, from_to_dict, use_xrdcp=use_xrdcp, type="copy")


def get_hpc_bdt_weights(logger, target_dir, use_xrdcp=False):
    if target_dir is not None:
        to_prefix = target_dir
    else:
        to_prefix = resource_dir
        to_prefix = os.path.join(resource_dir, "../bottom_line/metaconditions/hpc_bdt")

    from_to_dict = {
        "2016": {
            "from": "/eos/cms/store/group/phys_higgs/cmshgg/ingredients/2016/hpc_bdt_weights/",
            "to": to_prefix,
            "type": "eos",
        },
        "2017": {
            "from": "/eos/cms/store/group/phys_higgs/cmshgg/ingredients/2017/hpc_bdt_weights/",
            "to": to_prefix,
            "type": "eos",
        },
        "2018": {
            "from": "/eos/cms/store/group/phys_higgs/cmshgg/ingredients/2018/hpc_bdt_weights/",
            "to": to_prefix,
            "type": "eos",
        },
    }

    fetch_file("HPCBDT", logger, from_to_dict, use_xrdcp=use_xrdcp, type="copy")



def get_HHbbgg_btag_WPs_json(logger, target_dir, use_xrdcp=False):
    if target_dir is not None:
        to_prefix = target_dir
    else:
        to_prefix = os.path.join(
            resource_dir, "../bottom_line/tools/"
        )

    from_to_dict = {
        "WPs_PNet": {
            "from": "/eos/cms/store/group/phys_b2g/HHbbgg/nkasarag/HiggsDNA_JSONs/WPs_btagging.json",
            "to": f"{to_prefix}/WPs_btagging_HHbbgg.json",
            "type": "eos",
        },
    }
    fetch_file("HHbbgg_bTag_WPs", logger, from_to_dict, use_xrdcp=use_xrdcp, type="copy")


def get_HHbbgg_weight_interference_json(logger, target_dir, use_xrdcp=False):
    if target_dir is not None:
        to_prefix = target_dir
    else:
        to_prefix = os.path.join(
            resource_dir, "../bottom_line/tools/"
        )

    from_to_dict = {
        "HHbbgg_weight_interference": {
            "from": "/eos/cms/store/group/phys_b2g/HHbbgg/nkasarag/HiggsDNA_JSONs/Weights_interference.json",
            "to": f"{to_prefix}/Weights_interference_HHbbgg.json",
            "type": "eos",
        },
    }
    fetch_file("HHbbgg_weight_interference", logger, from_to_dict, use_xrdcp=use_xrdcp, type="copy")

def get_muon_scale_smearing(logger, target_dir, use_xrdcp=False):
    # References (not in Central jsonPOG repo yet):
    # https://gitlab.cern.ch/cms-muonPOG/muonscarekit

    if target_dir is not None:
        to_prefix = target_dir
    else:
        to_prefix = os.path.join(
            resource_dir, "../bottom_line/systematics/JSONs/MuonScaRe"
        )

    from_to_dict = {
        "2022postEE": {
            "from": "/eos/cms/store/group/phys_higgs/cmshgg/jixiao/backup_MuonScaRe/muonscarekit/corrections/2022_Summer22EE.json",
            "to": f"{to_prefix}/2022_Summer22EE.json",
            "type": "eos",
        },
        "2022preEE": {
            "from": "/eos/cms/store/group/phys_higgs/cmshgg/jixiao/backup_MuonScaRe/muonscarekit/corrections/2022_Summer22.json",
            "to": f"{to_prefix}/2022_Summer22.json",
            "type": "eos",
        },
        "2023postBPix": {
            "from": "/eos/cms/store/group/phys_higgs/cmshgg/jixiao/backup_MuonScaRe/muonscarekit/corrections/2023_Summer23BPix.json",
            "to": f"{to_prefix}/2023_Summer23BPix.json",
            "type": "eos",
        },
        "2023preBPix": {
            "from": "/eos/cms/store/group/phys_higgs/cmshgg/jixiao/backup_MuonScaRe/muonscarekit/corrections/2023_Summer23.json",
            "to": f"{to_prefix}/2023_Summer23.json",
            "type": "eos",
        },
    }

    fetch_file("MuonScaRe", logger, from_to_dict, use_xrdcp=use_xrdcp, type="copy")

def main():
    parser = argparse.ArgumentParser(
        description="Simple utility script to retrieve the needed files for corections, luminostiy mask, systematics uncertainties ..."
    )

    parser.add_argument(
        "-t",
        "--target",
        dest="target",
        help="Choose the target to download (default: %(default)s)",
        default="GoldenJSON",
        choices=[
            "GoldenJSON",
            "cTag",
            "bTag",
            "PhotonID",
            "PU",
            "SS",
            "SS-IJazZ",
            "JetMET",
            "CDFs",
            "JEC",
            "JER",
            "Material",
            "TriggerSF",
            "PreselSF",
            "eVetoSF",
            "Flows",
            "FNUF",
            "ShowerShape",
            "LooseMva",
            "LowMass-DiPhotonMVA",
            "muonSF",
            "LowMass-DYKilller",
            "CQR",
            "HggPhotonIDMVA",
            "DiphotonIDMVA",
            "HPCBDT",
            "HHbbgg_bTag_WPs",
            "HHbbgg_weight_interference",
            "MuonScaRe"
        ],
    )

    parser.add_argument(
        "-a",
        "--all",
        dest="all",
        action="store_true",
        help="Download all the targets (default: %(default)s)",
        default=False,
    )
    parser.add_argument(
        "--log", dest="log", type=str, default="INFO", help="Logger info level"
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default=None,
        help="directory to place the correction jsons, default: ../higgs-dna/systematics/JSONs",
    )
    parser.add_argument(
        "--analysis",
        type=str,
        default="higgs-dna-test",
        help="Name of the analysis you're perfoming, ideally it would match the output directory in which you're analysis parquet will end up, default: higgs-dna-test.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./json-log/",
        help="Log file summarising the json will end up here, default: ./json-log/",
    )
    parser.add_argument(
        "--use-xrdcp",
        action="store_true",
        help="Use xrdcp to copy the files, default: %(default)s",
        default=False,
    )

    args = parser.parse_args()

    # log output
    logfile = os.path.join(args.log_dir, f"{args.analysis}_jsons.log")
    p = pathlib.Path(logfile)
    p = pathlib.Path(*p.parts[:-1])  # remove file name
    p.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(level=args.log, logfile=logfile)

    if args.all:
        get_goldenjson(logger, args.target_dir, use_xrdcp=args.use_xrdcp)
        get_pileup(logger, args.target_dir, use_xrdcp=args.use_xrdcp)
        get_scale_and_smearing(logger, args.target_dir, use_xrdcp=args.use_xrdcp)
        get_scale_and_smearing_IJazZ(logger, args.target_dir, use_xrdcp=args.use_xrdcp)
        get_mass_decorrelation_CDF(logger, args.target_dir, use_xrdcp=args.use_xrdcp)
        get_Flow_files(logger, args.target_dir, use_xrdcp=args.use_xrdcp)
        get_ctag_json(logger, args.target_dir, use_xrdcp=args.use_xrdcp)
        get_btag_json(logger, args.target_dir, use_xrdcp=args.use_xrdcp)
        get_photonid_json(logger, args.target_dir, use_xrdcp=args.use_xrdcp)
        get_jetmet_json(logger, args.target_dir, use_xrdcp=args.use_xrdcp)
        get_jec_files(logger, args.target_dir, use_xrdcp=args.use_xrdcp)
        get_jer_files(logger, args.target_dir, use_xrdcp=args.use_xrdcp)
        get_material_json(logger, args.target_dir, use_xrdcp=args.use_xrdcp)
        get_fnuf_json(logger, args.target_dir, use_xrdcp=args.use_xrdcp)
        get_loose_mva_json(logger, args.target_dir, use_xrdcp=args.use_xrdcp)
        get_shower_shape_json(logger, args.target_dir, use_xrdcp=args.use_xrdcp)
        get_trigger_json(logger, args.target_dir, use_xrdcp=args.use_xrdcp)
        get_presel_json(logger, args.target_dir, use_xrdcp=args.use_xrdcp)
        get_eveto_json(logger, args.target_dir, use_xrdcp=args.use_xrdcp)
        get_lowmass_diphotonmva_model(logger, args.target_dir, use_xrdcp=args.use_xrdcp)
        get_muon_SFs(logger, args.target_dir, use_xrdcp=args.use_xrdcp)
        get_lowmass_dykiller_model(logger, args.target_dir, use_xrdcp=args.use_xrdcp)
        get_cqr_weights(logger, args.target_dir, use_xrdcp=args.use_xrdcp)
        get_hgg_photon_id_mva_weights(logger, args.target_dir, use_xrdcp=args.use_xrdcp)
        get_diphoton_id_mva_weights(logger, args.target_dir, use_xrdcp=args.use_xrdcp)
        get_hpc_bdt_weights(logger, args.target_dir, use_xrdcp=args.use_xrdcp)
        get_HHbbgg_btag_WPs_json(logger, args.target_dir, use_xrdcp=args.use_xrdcp)
        get_HHbbgg_weight_interference_json(logger, args.target_dir, use_xrdcp=args.use_xrdcp)
        get_muon_scale_smearing(logger, args.target_dir, use_xrdcp=args.use_xrdcp)
    elif args.target == "GoldenJSON":
        get_goldenjson(logger, args.target_dir, use_xrdcp=args.use_xrdcp)
    elif args.target == "PU":
        get_pileup(logger, args.target_dir, use_xrdcp=args.use_xrdcp)
    elif args.target == "SS":
        get_scale_and_smearing(logger, args.target_dir, use_xrdcp=args.use_xrdcp)
    elif args.target == "SS-IJazZ":
        get_scale_and_smearing_IJazZ(logger, args.target_dir, use_xrdcp=args.use_xrdcp)
    elif args.target == "CDFs":
        get_mass_decorrelation_CDF(logger, args.target_dir, use_xrdcp=args.use_xrdcp)
    elif args.target == "Flows":
        get_Flow_files(logger, args.target_dir, use_xrdcp=args.use_xrdcp)
    elif args.target == "cTag":
        get_ctag_json(logger, args.target_dir, use_xrdcp=args.use_xrdcp)
    elif args.target == "bTag":
        get_btag_json(logger, args.target_dir, use_xrdcp=args.use_xrdcp)
    elif args.target == "PhotonID":
        get_photonid_json(logger, args.target_dir, use_xrdcp=args.use_xrdcp)
    elif args.target == "JetMET":
        get_jetmet_json(logger, args.target_dir, use_xrdcp=args.use_xrdcp)
    elif args.target == "JEC":
        get_jec_files(logger, args.target_dir, use_xrdcp=args.use_xrdcp)
    elif args.target == "JER":
        get_jer_files(logger, args.target_dir, use_xrdcp=args.use_xrdcp)
    elif args.target == "Material":
        get_material_json(logger, args.target_dir, use_xrdcp=args.use_xrdcp)
    elif args.target == "FNUF":
        get_fnuf_json(logger, args.target_dir, use_xrdcp=args.use_xrdcp)
    elif args.target == "ShowerShape":
        get_shower_shape_json(logger, args.target_dir, use_xrdcp=args.use_xrdcp)
    elif args.target == "LooseMva":
        get_loose_mva_json(logger, args.target_dir, use_xrdcp=args.use_xrdcp)
    elif args.target == "TriggerSF":
        get_trigger_json(logger, args.target_dir, use_xrdcp=args.use_xrdcp)
    elif args.target == "PreselSF":
        get_presel_json(logger, args.target_dir, use_xrdcp=args.use_xrdcp)
    elif args.target == "eVetoSF":
        get_eveto_json(logger, args.target_dir, use_xrdcp=args.use_xrdcp)
    elif args.target == "LowMass-DiPhotonMVA":
        get_lowmass_diphotonmva_model(logger, args.target_dir, use_xrdcp=args.use_xrdcp)
    elif args.target == "muonSF":
        get_muon_SFs(logger, args.target_dir, use_xrdcp=args.use_xrdcp)
    elif args.target == "LowMass-DYKilller":
        get_lowmass_dykiller_model(logger, args.target_dir, use_xrdcp=args.use_xrdcp)
    elif args.target == "MuonScaRe":
        get_muon_scale_smearing(logger, args.target_dir, use_xrdcp=args.use_xrdcp)
    elif args.target == "CQR":
        get_cqr_weights(logger, args.target_dir, use_xrdcp=args.use_xrdcp)
    elif args.target == "HggPhotonIDMVA":
        get_hgg_photon_id_mva_weights(logger, args.target_dir, use_xrdcp=args.use_xrdcp)
    elif args.target == "DiphotonIDMVA":
        get_diphoton_id_mva_weights(logger, args.target_dir, use_xrdcp=args.use_xrdcp)
    elif args.target == "HPCBDT":
        get_hpc_bdt_weights(logger, args.target_dir, use_xrdcp=args.use_xrdcp)
    elif args.target == "HHbbgg_bTag_WPs":
        get_HHbbgg_btag_WPs_json(logger, args.target_dir, use_xrdcp=args.use_xrdcp)
    elif args.target == "HHbbgg_weight_interference":
        get_HHbbgg_weight_interference_json(logger, args.target_dir, use_xrdcp=args.use_xrdcp)
    else:
        logger.info("Unknown target, exit now!")
        exit(0)

    logger.info(" " * 60)
    logger.info("Done")
    logger.info("-" * 60)


if __name__ == "__main__":
    main()

# example commands:
# python pull_files.py --all
# python pull_files.py --target GoldenJSON
# python pull_files.py --target cTag
# python pull_files.py --target GoldenJSON --target-dir ./test_json --log-dir ./json-log --analysis goldenjson_test
