from coffea.lumi_tools import LumiMask
import awkward as ak
import os
import logging


def select_lumis(
    year,
    events: ak.highlevel.Array,
    logger: logging.Logger,
) -> ak.highlevel.Array:
    goldenJson_dict = {
        "2016": os.path.join(
            os.path.dirname(__file__),
            "../metaconditions/CAF/certification/Collisions16/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt",
        ),
        "2017": os.path.join(
            os.path.dirname(__file__),
            "../metaconditions/CAF/certification/Collisions17/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt",
        ),
        "2018": os.path.join(
            os.path.dirname(__file__),
            "../metaconditions/CAF/certification/Collisions18/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt",
        ),
        "2022": os.path.join(
            os.path.dirname(__file__),
            "../metaconditions/CAF/certification/Collisions22/Cert_Collisions2022_355100_362760_Golden.json",
        ),
        "2023": os.path.join(
            os.path.dirname(__file__),
            "../metaconditions/CAF/certification/Collisions23/Cert_Collisions2023_366442_370790_Golden.json",
        ),
        "2024": os.path.join(
            os.path.dirname(__file__),
            "../metaconditions/CAF/certification/Collisions24/Cert_Collisions2024_378981_386951_Golden.json",
        ),
    }
    # Reference
    # https://github.com/CoffeaTeam/coffea/blob/f8a4eb97137e84dd52474d26b8100174da196b57/tests/test_lumi_tools.py
    # https://twiki.cern.ch/twiki/bin/view/CMS/PdmVRun3Analysis#Data

    # goldenjson for 2016 and 2022 are inclusive
    if "2016" in year:
        year = "2016"
    elif "2022" in year:
        year = "2022"
    elif "2023" in year:
        year = "2023"
    else:
        pass
    lumimask = LumiMask(goldenJson_dict[year])
    logger.info("Year: {} GoldenJson: {}".format(year, goldenJson_dict[year]))
    return lumimask(events.run, events.luminosityBlock)
