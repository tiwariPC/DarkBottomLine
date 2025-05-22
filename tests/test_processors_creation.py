from bottom_line.utils.logger_utils import setup_logger
from bottom_line.workflows import DYStudiesProcessor, TagAndProbeProcessor, ZmmyProcessor
import os
import subprocess
from coffea import processor
import json
from importlib import resources
import pytest

# This is a dummy function to test the processor creation from run_analyses.py
def test_processors_creation():

    # Pulling golden json
    subprocess.run("pull_files.py --target GoldenJSON", shell=True)

    # Pull the btagging SF files
    subprocess.run("pull_files.py --target bTag", shell=True)

    # Lets test tag and probe processor
    command = [
        "run_analysis.py",
        "--json-analysis", "tests/config_files/runner_v13_tagandprobe.json",
        "--dump", "./EE_leak",
        "--skipJetVetoMap",
        "--executor", "iterative",
        "--limit", "1",
        "--workers", "1",
        "--nano-version", "13"
    ]

    #subprocess.run(command)
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        print("Error running the script:")
        print(result.stderr)
        pytest.fail(f"Script failed with stderr: {result.stderr}")

    # Asserting directly if you prefer a more pytest-like approach
    assert result.returncode == 0, f"Script failed with stderr: {result.stderr}"

    # Now the base processor
    command = [
        "run_analysis.py",
        "--json-analysis", "tests/config_files/runner_v13_base.json",
        "--dump", "./EE_leak",
        "--skipJetVetoMap",
        "--executor", "iterative",
        "--limit", "1",
        "--workers", "1",
        "--nano-version", "13"
    ]

    #subprocess.run(command)
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # clean up
    subprocess.run("rm -r EE_leak", shell=True)

    if result.returncode != 0:
        print("Error running the script:")
        print(result.stderr)
        pytest.fail(f"Script failed with stderr: {result.stderr}")

    # Asserting directly if you prefer a more pytest-like approach
    assert result.returncode == 0, f"Script failed with stderr: {result.stderr}"
