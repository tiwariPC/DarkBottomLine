import os
import subprocess
import json
import pytest
from importlib import resources
from bottom_line.workflows import DYStudiesProcessor, TagAndProbeProcessor, HHbbggProcessor, HplusCharmProcessor, lowmassProcessor, ParticleLevelProcessor, TopProcessor, ZeeProcessor, ZmmyProcessor, STXSProcessor, BTaggingEfficienciesProcessor
from coffea import processor


# Not tested by default since does not start with "test_"
def run_processor(processor_instance, fileset):
    """
    Helper function to run a given processor instance on the provided fileset.
    """
    iterative_run = processor.Runner(
        executor=processor.IterativeExecutor(compression=None),
        schema=processor.NanoAODSchema,
    )
    out = iterative_run(
        fileset,
        treename="Events",
        processor_instance=processor_instance,
    )
    return out


@pytest.mark.parametrize("processor_class", [
    DYStudiesProcessor,
    TagAndProbeProcessor,
    HHbbggProcessor,
    # Hpc Cannot be included in a simple way here since the arguments are not defaulted
    #HplusCharmProcessor,
    # Unclear to me why low mass does not work here, unit test should also be designed for this processor
    #lowmassProcessor,
    ParticleLevelProcessor,
    TopProcessor,
    ZeeProcessor,
    #ZmmyProcessor,
    BTaggingEfficienciesProcessor,
    STXSProcessor,
])
def test_processors(processor_class):
    """
    Test that each processor can run over a basic nanoAOD data v11 file without errors.
    """
    # Pull the golden JSON file
    subprocess.run("pull_files.py --target GoldenJSON", shell=True)

    # Pull the btagging SF files
    subprocess.run("pull_files.py --target bTag", shell=True)

    # Need to pull some JSONs for HHbbggProcessor
    # wget is used to download the files from the cernbox public link at the moment, should be updated with xrdcp
    subprocess.run("wget -O bottom_line/tools/WPs_btagging_HHbbgg.json https://cernbox.cern.ch/remote.php/dav/public-files/hOAABXExhgfL5AW/WPs_btagging.json", shell=True)
    subprocess.run("wget -O bottom_line/tools/Weights_interference_HHbbgg.json https://cernbox.cern.ch/remote.php/dav/public-files/ouOPOW9xLuJ4gM5/Weights_interference.json", shell=True)

    # Choose datasets to run over appropriately
    # In the future, should specify datasets on eos instead of local files
    # These should be appropriate for the processor being tested (e.g. muon for Zmmy or DY for T&P)
    # The skeleton below should be adjusted

    if processor_class == DYStudiesProcessor:
        MC = "./tests/samples/skimmed_nano/ggH_M125_amcatnlo_v13.root"
        Data = "./tests/samples/skimmed_nano/EGamma_2022E_v13.root"
    elif processor_class == TagAndProbeProcessor:
        MC = None
        Data = "./tests/samples/skimmed_nano/EGamma_2022E_v13.root"
    elif processor_class == HHbbggProcessor:
        MC = "./tests/samples/skimmed_nano/ggH_M125_amcatnlo_v13.root"
        Data = "./tests/samples/skimmed_nano/EGamma_2022E_v13.root"
    elif processor_class == HplusCharmProcessor:
        MC = "./tests/samples/skimmed_nano/ggH_M125_amcatnlo_v13.root"
        Data = "./tests/samples/skimmed_nano/EGamma_2022E_v13.root"
    elif processor_class == lowmassProcessor:
        MC = "./tests/samples/skimmed_nano/ggH_M125_amcatnlo_v13.root"
        Data = "./tests/samples/skimmed_nano/EGamma_2022E_v13.root"
    elif processor_class == ParticleLevelProcessor:
        MC = "./tests/samples/skimmed_nano/ggH_M125_amcatnlo_v13.root"
        Data = None
    elif processor_class == TopProcessor:
        MC = "./tests/samples/skimmed_nano/ggH_M125_amcatnlo_v13.root"
        Data = "./tests/samples/skimmed_nano/EGamma_2022E_v13.root"
    elif processor_class == ZeeProcessor:
        MC = "./tests/samples/skimmed_nano/ggH_M125_amcatnlo_v13.root"
        Data = "./tests/samples/skimmed_nano/EGamma_2022E_v13.root"
    elif processor_class == ZmmyProcessor:
        MC = None
        Data = None
    elif processor_class == BTaggingEfficienciesProcessor:
        MC = "./tests/samples/skimmed_nano/ggH_M125_amcatnlo_v13.root"
        Data = None
    elif processor_class == STXSProcessor:
        MC = "./tests/samples/skimmed_nano/ggH_M125_amcatnlo_v13.root"
        Data = "./tests/samples/skimmed_nano/EGamma_2022E_v13.root"

    fileset = {}

    if Data is not None:
        fileset["Data"] = [Data]

    if MC is not None:
        fileset["MC"] = [MC]

    with resources.open_text("bottom_line.metaconditions", "Era2017_legacy_v1.json") as f:
        metaconditions = json.load(f)

    processor_instance = processor_class(
        year={"Data": ["2022postEE"], "MC": ["2022postEE"]},
        metaconditions=metaconditions,
        nano_version=13,
        apply_trigger=True,
        skipJetVetoMap=True,
        output_location="output/basics"
    )

    # Run the processor and verify output
    run_processor(processor_instance, fileset)

    # Clean up
    subprocess.run("rm -r output", shell=True)
