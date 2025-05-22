#!/usr/bin/env python
import correctionlib
import correctionlib.schemav2 as cs
import rich
from optparse import OptionParser
import pydantic

print(pydantic.__version__)
print(correctionlib.__version__)

# This script is meant to create json files containing the correction and systematic variations as were used in flashgg
# the files are created using correctionlib, copying the binning, corrections and errors from flashgg

# The script works with the most recent version of correctionlib, to be compatible with older releases one should
# change the istances of "cset.model_dump_json" to "cset.json" to not stumble into errors


def main():
    usage = "Usage: python %prog [options]"
    parser = OptionParser(usage=usage)
    parser.add_option(
        "--read",
        dest="read",
        default="",
        help="file to read from, if different from empty it will skip the creation step",
    )
    parser.add_option(
        "--year",
        dest="year",
        default="2017",
        help="year to select correction values from",
    )
    parser.add_option(
        "--do-fnuf",
        dest="do_fnuf",
        action="store_true",
        default=False,
        help="create FNUF correction json",
    )
    parser.add_option(
        "--do-showsh",
        dest="do_showsh",
        action="store_true",
        default=False,
        help="create shower shape correction json",
    )
    parser.add_option(
        "--do-idmva",
        dest="do_IDMVA",
        action="store_true",
        default=False,
        help="create PhotonID MVA correction json",
    )
    parser.add_option(
        "--do-Mat",
        dest="do_Material",
        action="store_true",
        default=False,
        help="create Material correction json",
    )
    parser.add_option(
        "--do-eVeto",
        dest="do_eVeto",
        action="store_true",
        default=False,
        help="create Electron Veto correction json",
    )
    parser.add_option(
        "--do-presel",
        dest="do_presel",
        action="store_true",
        default=False,
        help="create Preselection Scale Factor json",
    )
    parser.add_option(
        "--do-trigger",
        dest="do_trigger",
        action="store_true",
        default=False,
        help="create Trigger Scale Factor jsons",
    )
    parser.add_option(
        "--do-trigger-lead",
        dest="do_trigger_lead",
        action="store_true",
        default=False,
        help="create lead photon Trigger Scale Factor json",
    )
    parser.add_option(
        "--do-trigger-sublead",
        dest="do_trigger_sublead",
        action="store_true",
        default=False,
        help="create sublead photon Trigger Scale Factor json",
    )
    parser.add_option(
        "--do-all",
        dest="do_all",
        action="store_true",
        default=False,
        help="create all json for given year",
    )
    (opt, args) = parser.parse_args()


    def multibinning(inputs_: list, edges_: list, content_, flow_: str):
        return cs.MultiBinning(
            nodetype="multibinning",
            inputs=inputs_,
            edges=edges_,
            content=content_,
            flow=flow_,
        )


# the code creates the CorrectionSet using correctionlib, write it to a .json and reads it back
# in principle one can dump more than one Correction in the same .json adding it to the CorrectionSet
    if opt.do_fnuf or opt.do_all:
        inputs_ = ["eta", "r9"]
        edges_ = [[0.0, 1.5, 6.0], [0.0, 0.94, 999.0]]

        if opt.year == "2016":
            # FNUF variations fromflashgg: https://github.com/cms-analysis/flashgg/blob/dev_legacy_runII/Systematics/python/flashggDiPhotonSystematics2016_Legacy_postVFP_cfi.py#L37-L46
            # Based on the updated derivation by N. Schroeder for HIG-19-004. Re-evaluated in coarse Eta-R9 bins
            # 4 bins (2 in eta 2 in r9)
            content_ = {
                "nominal": [1.0, 1.0, 1.0, 1.0],
                "up": [1.00044, 1.00156, 1.00003, 1.00165],
                "down": [0.99956, 0.99844, 0.99997,0.99835],
            }
            flow_ = "clamp"
        elif opt.year == "2017":
            # FNUF variations fromflashgg: https://github.com/cms-analysis/flashgg/blob/dev_legacy_runII/Systematics/python/flashggDiPhotonSystematics2017_Legacy_cfi.py#L73-L82
            # Based on the updated derivation by N. Schroeder for HIG-19-004. Re-evaluated in coarse Eta-R9 bins
            # 4 bins (2 in eta 2 in r9)
            content_ = {
                "nominal": [1.0, 1.0, 1.0, 1.0],
                "up": [1.00062, 1.00208, 1.00005, 1.00227],
                "down": [0.99938, 0.99792, 0.99995, 0.99773],
            }
            flow_ = "clamp"
        elif opt.year == "2018":
            # FNUF variations fromflashgg: https://github.com/cms-analysis/flashgg/blob/dev_legacy_runII/Systematics/python/flashggDiPhotonSystematics2018_Legacy_cfi.py#L73-L81
            # Based on the updated derivation by N. Schroeder for HIG-19-004. Re-evaluated in coarse Eta-R9 bins
            # 4 bins (2 in eta 2 in r9)
            content_ = {
                "nominal": [1.0, 1.0, 1.0, 1.0],
                "up": [1.0007, 1.0022, 1.00005, 1.00251],
                "down": [0.9993, 0.9978, 0.99995, 0.99749],
            }
            flow_ = "clamp"

        FNUF = cs.Correction(
            name="FNUF",
            version=1,
            inputs=[
                cs.Variable(
                    name="systematic", type="string", description="Systematic variation"
                ),
                cs.Variable(name="eta", type="real", description="Photon eta"),
                cs.Variable(
                    name="r9",
                    type="real",
                    description="Photon full 5x5 R9, ratio E3x3/ERAW, where E3x3 is the energy sum of the 3 by 3 crystals surrounding the supercluster seed crystal and ERAW is the raw energy sum of the supercluster",
                ),
            ],
            output=cs.Variable(
                name="Ecorr",
                type="real",
                description="Multiplicative correction to photon energy",
            ),
            data=cs.Category(
                nodetype="category",
                input="systematic",
                content=[
                    {
                        "key": "nominal",
                        "value": multibinning(inputs_, edges_, content_["nominal"], flow_),
                    },
                    {
                        "key": "up",
                        "value": multibinning(inputs_, edges_, content_["up"], flow_),
                    },
                    {
                        "key": "down",
                        "value": multibinning(inputs_, edges_, content_["down"], flow_),
                    },
                ],
            ),
        )

        rich.print(FNUF)

        # test that everything is fine with some mockup values
        etas = [1.0, 2.0, 1.0, 2.0]
        rs = [0.9, 0.99, 0.99, 0.9]

        print("-" * 120)
        print("nominal --->","eta:",etas,"r9:",rs,"result:",FNUF.to_evaluator().evaluate("nominal", etas, rs))
        print("-" * 120)
        print("up      --->","eta:",etas,"r9:",rs,"result:",FNUF.to_evaluator().evaluate("up", etas, rs))
        print("-" * 120)
        print("down    --->","eta:",etas,"r9:",rs,"result:",FNUF.to_evaluator().evaluate("down", etas, rs))
        print("-" * 120)
        print()

        cset = cs.CorrectionSet(
            schema_version=2,
            description="FNUF",
            corrections=[
                FNUF,
            ],
        )

        # if we're not just checking an existing json we create a new one
        if opt.read == "":
            with open(f"FNUF_{opt.year}.json", "w") as fout:
                fout.write(cset.model_dump_json(exclude_unset=True))

            import gzip

            with gzip.open(f"FNUF_{opt.year}.json.gz", "wt") as fout:
                fout.write(cset.model_dump_json(exclude_unset=True))

        else:
            print("Reading back...")

            file = opt.read
            ceval = correctionlib.CorrectionSet.from_file(file)
            for corr in ceval.values():
                print(f"Correction {corr.name} has {len(corr.inputs)} inputs")
                for ix in corr.inputs:
                    print(f"   Input {ix.name} ({ix.type}): {ix.description}")

            print("-" * 120)
            print("nominal --->","eta:",etas,"r9:",rs,"result:",ceval["FNUF"].evaluate("nominal", etas, rs))
            print("-" * 120)
            print("up      --->","eta:",etas,"r9:",rs,"result:",ceval["FNUF"].evaluate("up", etas, rs))
            print("-" * 120)
            print("down    --->","eta:",etas,"r9:",rs,"result:",ceval["FNUF"].evaluate("down", etas, rs))
            print("-" * 120)

# Shower Shape variations fromflashgg: https://github.com/cms-analysis/flashgg/blob/58c243d9d1f794d7dca8a94fdcd390aed91cb49c/Systematics/python/flashggDiPhotonSystematics2017_Legacy_cfi.py#L73-L82
# 8 bins (4 in eta 2 in r9)
# Flashgg correction is from Martina: https://indico.cern.ch/event/628676/contributions/2546615/attachments/1440085/2216643/20170405_martina_regrEchecksUpdate.pdf

    if opt.do_showsh or opt.do_all:
        inputs_ = ["eta", "r9"]
        edges_ = [[0.0, 1.0, 1.5, 2.0, 6.0], [0.0, 0.94, 999.0]]

        if "2016" in opt.year:
            # Exactly the same as 2017
            # Shower Shape variations fromflashgg: https://github.com/higgs-charm/flashgg/blob/17621e1d0f032e38c20294555359ed4682bd3f3b/Systematics/python/flashggDiPhotonSystematics2016_Legacy_postVFP_cfi.py#L48C1-L65C6
            content_ = {
                "nominal": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                "up": [0.9999, 0.9994, 1.0002, 0.9989, 1.0015, 1.0002, 1.0004, 1.0003],
                "down": [1.0001, 1.0006, 0.9998, 1.0011, 0.9985, 0.9998, 0.9996, 0.9997],
            }
            flow_ = "clamp"
        elif opt.year == "2017":
            # Shower Shape variations fromflashgg: https://github.com/cms-analysis/flashgg/blob/58c243d9d1f794d7dca8a94fdcd390aed91cb49c/Systematics/python/flashggDiPhotonSystematics2017_Legacy_cfi.py#L73-L82
            # 8 bins (4 in eta 2 in r9)
            # Flashgg correction is from Martina: https://indico.cern.ch/event/628676/contributions/2546615/attachments/1440085/2216643/20170405_martina_regrEchecksUpdate.pdf
            content_ = {
                "nominal": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                "up": [0.9999, 0.9994, 1.0002, 0.9989, 1.0015, 1.0002, 1.0004, 1.0003],
                "down": [1.0001, 1.0006, 0.9998, 1.0011, 0.9985, 0.9998, 0.9996, 0.9997],
            }
            flow_ = "clamp"
        elif opt.year == "2018":
            # Exactly the same as 2017
            # Shower Shape variations fromflashgg: https://github.com/higgs-charm/flashgg/blob/17621e1d0f032e38c20294555359ed4682bd3f3b/Systematics/python/flashggDiPhotonSystematics2018_Legacy_cfi.py#L83C1-L100
            content_ = {
                "nominal": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                "up": [0.9999, 0.9994, 1.0002, 0.9989, 1.0015, 1.0002, 1.0004, 1.0003],
                "down": [1.0001, 1.0006, 0.9998, 1.0011, 0.9985, 0.9998, 0.9996, 0.9997],
            }
            flow_ = "clamp"

        SS = cs.Correction(
            name="ShowerShape",
            version=1,
            inputs=[
                cs.Variable(
                    name="systematic", type="string", description="Systematic variation"
                ),
                cs.Variable(name="eta", type="real", description="Photon eta"),
                cs.Variable(
                    name="r9",
                    type="real",
                    description="Photon full 5x5 R9, ratio E3x3/ERAW, where E3x3 is the energy sum of the 3 by 3 crystals surrounding the supercluster seed crystal and ERAW is the raw energy sum of the supercluster",
                ),
            ],
            output=cs.Variable(
                name="Ecorr",
                type="real",
                description="Multiplicative correction to photon energy and pt",
            ),
            data=cs.Category(
                nodetype="category",
                input="systematic",
                content=[
                    {
                        "key": "nominal",
                        "value": multibinning(inputs_, edges_, content_["nominal"], flow_),
                    },
                    {
                        "key": "up",
                        "value": multibinning(inputs_, edges_, content_["up"], flow_),
                    },
                    {
                        "key": "down",
                        "value": multibinning(inputs_, edges_, content_["down"], flow_),
                    },
                ],
            ),
        )

        rich.print(SS)

        etas = [0.8,1.2,1.7,2.5,0.8,1.2,1.7,2.0]
        rs = [0.9, 0.9, 0.9, 0.9, 0.99, 0.99, 0.99, 0.99]

        print("-" * 120)
        print("nominal --->","eta:",etas,"r9:",rs,"result:",SS.to_evaluator().evaluate("nominal", etas, rs))
        print("-" * 120)
        print("up      --->","eta:",etas,"r9:",rs,"result:",SS.to_evaluator().evaluate("up", etas, rs))
        print("-" * 120)
        print("down    --->","eta:",etas,"r9:",rs,"result:",SS.to_evaluator().evaluate("down", etas, rs))
        print("-" * 120)
        print()

        cset = cs.CorrectionSet(
            schema_version=2,
            description="ShowerShape",
            corrections=[
                SS,
            ],
        )

        if opt.read == "":
            with open(f"ShowerShape_{opt.year}.json", "w") as fout:
                fout.write(cset.model_dump_json(exclude_unset=True))

            import gzip

            with gzip.open(f"ShowerShape_{opt.year}.json.gz", "wt") as fout:
                fout.write(cset.model_dump_json(exclude_unset=True))

        if opt.read != "":
            print("Reading back...")
            file = opt.read
            ceval = correctionlib.CorrectionSet.from_file(file)
            for corr in ceval.values():
                print(f"Correction {corr.name} has {len(corr.inputs)} inputs")
                for ix in corr.inputs:
                    print(f"   Input {ix.name} ({ix.type}): {ix.description}")

            print("-" * 120)
            print("nominal --->","eta:",etas,"r9:",rs,"result:",ceval["ShowerShape"].evaluate("nominal", etas, rs))
            print("-" * 120)
            print("up      --->","eta:",etas,"r9:",rs,"result:",ceval["ShowerShape"].evaluate("up", etas, rs))
            print("-" * 120)
            print("down    --->","eta:",etas,"r9:",rs,"result:",ceval["ShowerShape"].evaluate("down", etas, rs))
            print("-" * 120)


    if opt.do_IDMVA or opt.do_all:
        inputs_ = ["eta", "r9"]
        edges_ = [[0.0, 1.5, 6.0], [0.0, 0.85, 0.9, 999.0]]

        if "2016" in opt.year:
            # PhotonID MVA variations fromflashgg: https://github.com/higgs-charm/flashgg/blob/17621e1d0f032e38c20294555359ed4682bd3f3b/Systematics/python/flashggDiPhotonSystematics2016_Legacy_postVFP_cfi.py#L256C1-L265C6
            # 6 bins (2 in eta 3 in r9)
            # LooseIDMVA [-0.9] SF and uncertainty for UL2016 from Arnab via Martina 10/03/2016
            content_ = {
                "nominal": [0.9999, 1.0003, 1.0003, 1.0003, 1.0003, 1.0004],
                "up": [1.0000, 1.0003, 1.0003, 1.0003, 1.0003, 1.0004],
                "down": [0.9998, 1.0003, 1.0003, 1.0003, 1.0003, 1.0004],
            }
            flow_ = "clamp"
        elif opt.year == "2017":
            # PhotonID MVA variations fromflashgg: https://github.com/cms-analysis/flashgg/blob/ae6563050722bd168545eac2b860ef56cdda7be4/Systematics/python/flashggDiPhotonSystematics2017_Legacy_cfi.py#L280-286
            # 6 bins (2 in eta 3 in r9)
            # LooseIDMVA [-0.9] SF and uncertainty for UL2017. Dt: 17/11/2020
            # link to the presentation: https://indico.cern.ch/event/963617/contributions/4103623/attachments/2141570/3608645/Zee_Validation_UL2017_Update_09112020_Prasant.pdf
            content_ = {
                "nominal": [1.0021, 1.0001, 1.0001, 1.0061, 1.0061, 1.0016],
                "up": [1.0035, 1.0039, 1.0039, 1.0075, 1.0075, 1.0039],
                "down": [1.0007, 0.9963, 0.9963, 1.0047, 1.0047, 0.9993],
            }
            flow_ = "clamp"
        elif opt.year == "2018":
            # PhotonID MVA variations fromflashgg: https://github.com/higgs-charm/flashgg/blob/17621e1d0f032e38c20294555359ed4682bd3f3b/Systematics/python/flashggDiPhotonSystematics2018_Legacy_cfi.py#L263C1-L271C1
            # Prasant Kumar Rout: UL18 Loose Photon IDMVA cut [-0.9] SF and total uncertainties.
            # 6 bins (2 in eta 3 in r9)
            content_ = {
                "nominal": [1.0022, 1.0005, 1.0005, 1.0058, 1.0058, 1.0013],
                "up": [1.0034, 1.0038, 1.0038, 1.0078, 1.0078, 1.0041],
                "down": [1.001, 0.9972, 0.9972, 1.0038, 1.0038, 0.9985],
            }
            flow_ = "clamp"

        MVA = cs.Correction(
            name="LooseMvaSF",
            version=1,
            inputs=[
                cs.Variable(
                    name="systematic", type="string", description="Systematic variation"
                ),
                cs.Variable(name="eta", type="real", description="Photon eta"),
                cs.Variable(
                    name="r9",
                    type="real",
                    description="Photon full 5x5 R9, ratio E3x3/ERAW, where E3x3 is the energy sum of the 3 by 3 crystals surrounding the supercluster seed crystal and ERAW is the raw energy sum of the supercluster",
                ),
            ],
            output=cs.Variable(
                name="Ecorr",
                type="real",
                description="Multiplicative correction to photon energy and pt",
            ),
            data=cs.Category(
                nodetype="category",
                input="systematic",
                content=[
                    {
                        "key": "nominal",
                        "value": multibinning(inputs_, edges_, content_["nominal"], flow_),
                    },
                    {
                        "key": "up",
                        "value": multibinning(inputs_, edges_, content_["up"], flow_),
                    },
                    {
                        "key": "down",
                        "value": multibinning(inputs_, edges_, content_["down"], flow_),
                    },
                ],
            ),
        )

        rich.print(MVA)

        etas = [0.8, 0.8, 0.8, 2.5, 2.5, 2.5]
        rs = [0.6, 0.87, 0.95, 0.6, 0.87, 0.99]

        print("-" * 120)
        print("nominal --->","eta:",etas,"r9:",rs,"result:",MVA.to_evaluator().evaluate("nominal", etas, rs))
        print("-" * 120)
        print("up      --->","eta:",etas,"r9:",rs,"result:",MVA.to_evaluator().evaluate("up", etas, rs))
        print("-" * 120)
        print("down    --->","eta:",etas,"r9:",rs,"result:",MVA.to_evaluator().evaluate("down", etas, rs))
        print("-" * 120)
        print()

        cset = cs.CorrectionSet(
            schema_version=2,
            description="LooseMvaSF",
            corrections=[
                MVA,
            ],
        )

        if opt.read == "":
            with open(f"LooseMvaSF_{opt.year}.json", "w") as fout:
                fout.write(cset.model_dump_json(exclude_unset=True))

            import gzip

            with gzip.open(f"LooseMvaSF_{opt.year}.json.gz", "wt") as fout:
                fout.write(cset.model_dump_json(exclude_unset=True))

        if opt.read != "":
            print("Reading back...")
            file = opt.read
            ceval = correctionlib.CorrectionSet.from_file(file)
            for corr in ceval.values():
                print(f"Correction {corr.name} has {len(corr.inputs)} inputs")
                for ix in corr.inputs:
                    print(f"   Input {ix.name} ({ix.type}): {ix.description}")

            print("-" * 120)
            print("nominal --->","eta:",etas,"r9:",rs,"result:",ceval["LooseMvaSF"].evaluate("nominal", etas, rs))
            print("-" * 120)
            print("up      --->","eta:",etas,"r9:",rs,"result:",ceval["LooseMvaSF"].evaluate("up", etas, rs))
            print("-" * 120)
            print("down    --->","eta:",etas,"r9:",rs,"result:",ceval["LooseMvaSF"].evaluate("down", etas, rs))
            print("-" * 120)


    if opt.do_Material or opt.do_all:
        inputs_ = ["SCEta", "r9"]
        edges_ = [[0.0, 1., 1.5, 999.0], [0.0, 0.94, 999.0]]

        if "2016" in opt.year:
            # Material  variations fromflashgg: https://github.com/cms-analysis/flashgg/blob/dev_legacy_runII/Systematics/python/flashggDiPhotonSystematics2016_Legacy_postVFP_cfi.py#L370-L381
            # 6 bins (3 in SCeta 2 in r9)
            content_ = {
                "nominal": [1., 1., 1., 1., 1., 1.],
                "up": [1.000455, 1.000233, 1.002089, 1.002089, 1.001090, 1.002377],
                "down": [0.999545, 0.999767, 0.9979911, 0.9979911, 0.99891, 0.997623],
            }
            flow_ = "clamp"
        if opt.year == "2017":
            # Material  variations fromflashgg: https://github.com/cms-analysis/flashgg/blob/dev_legacy_runII/Systematics/python/flashggDiPhotonSystematics2017_Legacy_cfi.py#L388-L399
            # 6 bins (3 in SCeta 2 in r9)
            content_ = {
                "nominal": [1., 1., 1., 1., 1., 1.],
                "up": [1.000455, 1.000233, 1.002089, 1.002089, 1.001090, 1.002377],
                "down": [0.999545, 0.999767, 0.9979911, 0.9979911, 0.99891, 0.997623],
            }
            flow_ = "clamp"
        elif opt.year == "2018":
            # Material  variations fromflashgg: https://github.com/cms-analysis/flashgg/blob/dev_legacy_runII/Systematics/python/flashggDiPhotonSystematics2018_Legacy_cfi.py#L372-L383
            # 6 bins (3 in SCeta 2 in r9)
            # same as 2017
            content_ = {
                "nominal": [1., 1., 1., 1., 1., 1.],
                "up": [1.000455, 1.000233, 1.002089, 1.002089, 1.001090, 1.002377],
                "down": [0.999545, 0.999767, 0.9979911, 0.9979911, 0.99891, 0.997623],
            }
            flow_ = "clamp"


        MAT = cs.Correction(
            name="Material",
            description="Material correction",
            generic_formulas=None,
            version=1,
            inputs=[
                cs.Variable(
                    name="systematic", type="string", description="Systematic variation"
                ),
                cs.Variable(name="SCEta", type="real", description="Photon Super Cluster eta"),
                cs.Variable(
                    name="r9",
                    type="real",
                    description="Photon full 5x5 R9, ratio E3x3/ERAW, where E3x3 is the energy sum of the 3 by 3 crystals surrounding the supercluster seed crystal and ERAW is the raw energy sum of the supercluster",
                ),
            ],
            output=cs.Variable(
                name="Ecorr",
                type="real",
                description="Multiplicative correction to photon energy and pt",
            ),
            data=cs.Category(
                nodetype="category",
                input="systematic",
                content=[
                    cs.CategoryItem(
                        key="up",
                        value=multibinning(inputs_, edges_, content_["up"], flow_)
                    ),
                    cs.CategoryItem(
                        key="down",
                        value=multibinning(inputs_, edges_, content_["down"], flow_)
                    ),
                ],
                default=multibinning(inputs_, edges_, content_["nominal"], flow_),
            ),
        )

        rich.print(MAT)

        etas = [0.8, 0.8, 1.2, 1.2, 2.5, 2.5]
        rs = [0.6, 0.95, 0.6, 0.95, 0.6, 0.99]

        print(MAT.to_evaluator())
        print("-" * 120)
        print("nominal --->","eta:",etas,"r9:",rs,"result:",MAT.to_evaluator().evaluate("nominal", etas, rs))
        print("-" * 120)
        print("up      --->","eta:",etas,"r9:",rs,"result:",MAT.to_evaluator().evaluate("up", etas, rs))
        print("-" * 120)
        print("down    --->","eta:",etas,"r9:",rs,"result:",MAT.to_evaluator().evaluate("down", etas, rs))
        print("-" * 120)
        print()

        cset = cs.CorrectionSet(
            schema_version=2,
            description="Material",
            compound_corrections=None,
            corrections=[
                MAT,
            ],
        )

        if opt.read == "":
            with open(f"Material_{opt.year}.json", "w") as fout:
                fout.write(cset.model_dump_json(exclude_unset=True))

            import gzip

            with gzip.open(f"Material_{opt.year}.json.gz", "wt") as fout:
                fout.write(cset.model_dump_json(exclude_unset=True))

        if opt.read != "":
            print("Reading back...")
            file = opt.read
            ceval = correctionlib.CorrectionSet.from_file(file)
            for corr in ceval.values():
                print(f"Correction {corr.name} has {len(corr.inputs)} inputs")
                for ix in corr.inputs:
                    print(f"   Input {ix.name} ({ix.type}): {ix.description}")

            print("-" * 120)
            print("nominal --->","eta:",etas,"r9:",rs,"result:",ceval["Material"].evaluate("nominal", etas, rs))
            print("-" * 120)
            print("up      --->","eta:",etas,"r9:",rs,"result:",ceval["Material"].evaluate("up", etas, rs))
            print("-" * 120)
            print("down    --->","eta:",etas,"r9:",rs,"result:",ceval["Material"].evaluate("down", etas, rs))
            print("-" * 120)


    if opt.do_eVeto or opt.do_all:
        inputs_ = ["SCEta", "r9"]
        edges_ = [[0.0, 1.5, 6.0], [0.0, 0.85, 0.9, 999.0]]

        if "2016" in opt.year:
            # elecronVetoSF  variations fromflashgg: https://github.com/higgs-charm/flashgg/blob/17621e1d0f032e38c20294555359ed4682bd3f3b/Systematics/python/flashggDiPhotonSystematics2016_Legacy_postVFP_cfi.py#L25C1-L34C6
            # 6 bins (2 in SCeta 3 in r9)
            # JTao: slide 24 of https://indico.cern.ch/event/1201810/contributions/5103257/attachments/2532365/4357414/UL2016_Zmmg_ForHgg.pdf with preVFP+postVFP combined as agreed, since the difference < 1 sigma of the unc
            content_ = {
                "nominal": [1.0004, 0.9976, 0.9976, 0.9882, 0.9882, 0.9971],
                "up": [1.0025, 0.9981, 0.9981, 0.9954, 0.9954, 0.9987],
                "down": [0.9983, 0.9971, 0.9971, 0.981, 0.981, 0.9955],
            }
            flow_ = "clamp"
        elif opt.year == "2017":
            # elecronVetoSF  variations fromflashgg: https://github.com/cms-analysis/flashgg/blob/4edea8897e2a4b0518dca76ba6c9909c20c40ae7/Systematics/python/flashggDiPhotonSystematics2017_Legacy_cfi.py#L27
            # 6 bins (2 in SCeta 3 in r9)
            # link to the presentation: JTao: slide 13 of the updated UL2017 results, https://indico.cern.ch/event/961164/contributions/4089584/attachments/2135019/3596299/Zmmg_UL2017%20With%20CorrMC_Hgg%20%2802.11.2020%29.pdf, presented by Aamir on 2nd Nov. 2020
            content_ = {
                "nominal": [0.9838, 0.9913, 0.9913, 0.9777, 0.9777, 0.9784],
                "up": [0.9862, 0.9922, 0.9922, 0.9957, 0.9957, 0.981],
                "down": [0.9814, 0.9904, 0.9904, 0.9597, 0.9597, 0.9758],
            }
            flow_ = "clamp"
        elif opt.year == "2018":
            # JTao: slide 11 of https://indico.cern.ch/event/996926/contributions/4264412/attachments/2203745/3728187/202103ZmmgUL2018_ForHggUpdate.pdf for UL18
            # 6 bins (2 in SCeta 3 in r9)
            content_ = {
                "nominal": [0.983, 0.9939, 0.9939, 0.9603, 0.9603, 0.9754],
                "up": [0.9851, 0.9944, 0.9944, 0.9680, 0.9680, 0.9771],
                "down": [0.9809, 0.9934, 0.9934, 0.9526, 0.9526, 0.9737],
            }
            flow_ = "clamp"


        EVETO = cs.Correction(
            name="ElectronVetoSF",
            description="Electron Veto Scale Factor",
            generic_formulas=None,
            version=1,
            inputs=[
                cs.Variable(
                    name="systematic", type="string", description="Systematic variation"
                ),
                cs.Variable(name="SCEta", type="real", description="Photon Super Cluster eta absolute value"),
                cs.Variable(
                    name="r9",
                    type="real",
                    description="Photon full 5x5 R9, ratio E3x3/ERAW, where E3x3 is the energy sum of the 3 by 3 crystals surrounding the supercluster seed crystal and ERAW is the raw energy sum of the supercluster",
                ),
            ],
            output=cs.Variable(
                name="Ecorr",
                type="real",
                description="Multiplicative correction to event weight (per-photon)",
            ),
            data=cs.Category(
                nodetype="category",
                input="systematic",
                content=[
                    cs.CategoryItem(
                        key="up",
                        value=multibinning(inputs_, edges_, content_["up"], flow_)
                    ),
                    cs.CategoryItem(
                        key="down",
                        value=multibinning(inputs_, edges_, content_["down"], flow_)
                    ),
                ],
                default=multibinning(inputs_, edges_, content_["nominal"], flow_),
            ),
        )

        rich.print(EVETO)

        etas = [0.8, 0.8, 0.8, 1.7, 1.7, 1.7]
        rs = [0.6, 0.85, 0.95, 0.6, 0.85, 0.99]

        print(EVETO.to_evaluator())
        print("-" * 120)
        print("nominal --->","eta:",etas,"r9:",rs,"result:",EVETO.to_evaluator().evaluate("nominal", etas, rs))
        print("-" * 120)
        print("up      --->","eta:",etas,"r9:",rs,"result:",EVETO.to_evaluator().evaluate("up", etas, rs))
        print("-" * 120)
        print("down    --->","eta:",etas,"r9:",rs,"result:",EVETO.to_evaluator().evaluate("down", etas, rs))
        print("-" * 120)
        print()

        cset = cs.CorrectionSet(
            schema_version=2,
            description="Electron Veto SF",
            compound_corrections=None,
            corrections=[
                EVETO,
            ],
        )

        if opt.read == "":
            with open(f"eVetoSF_{opt.year}.json", "w") as fout:
                fout.write(cset.model_dump_json(exclude_unset=True))

            import gzip

            with gzip.open(f"eVetoSF_{opt.year}.json.gz", "wt") as fout:
                fout.write(cset.model_dump_json(exclude_unset=True))

        if opt.read != "":
            print("Reading back...")
            file = opt.read
            ceval = correctionlib.CorrectionSet.from_file(file)
            for corr in ceval.values():
                print(f"Correction {corr.name} has {len(corr.inputs)} inputs")
                for ix in corr.inputs:
                    print(f"   Input {ix.name} ({ix.type}): {ix.description}")

            print("-" * 120)
            print("nominal --->","eta:",etas,"r9:",rs,"result:",ceval["ElectronVetoSF"].evaluate("nominal", etas, rs))
            print("-" * 120)
            print("up      --->","eta:",etas,"r9:",rs,"result:",ceval["ElectronVetoSF"].evaluate("up", etas, rs))
            print("-" * 120)
            print("down    --->","eta:",etas,"r9:",rs,"result:",ceval["ElectronVetoSF"].evaluate("down", etas, rs))
            print("-" * 120)


    if opt.do_presel or opt.do_all:
        inputs_ = ["SCEta", "r9"]
        edges_ = [[0.0, 1.5, 6.0], [0.0, 0.85, 0.9, 999.0]]

        if "2016" in opt.year:
            # from Arnab via Martina 10/03/2017
            # Presel SF and uncertainty for UL2016. fromflashgg: https://github.com/higgs-charm/flashgg/blob/17621e1d0f032e38c20294555359ed4682bd3f3b/Systematics/python/flashggDiPhotonSystematics2016_Legacy_postVFP_cfi.py#L14C1-L23C6
            # same for pre and post VFP
            # 6 bins (2 in SCeta 3 in r9)
            content_ = {
                "nominal": [1.0057, 0.9988, 0.9988, 0.9443, 0.9443, 0.9447],
                "up": [1.0067, 0.9997, 0.9997, 0.9515, 0.9515, 0.9998],
                "down": [1.0047, 0.9979, 0.9979, 0.9371, 0.9371, 0.9896],
            }
            flow_ = "clamp"
        elif opt.year == "2017":
            # Presel SF and uncertainty for UL2017. Dt:17/11/2020 fromflashgg: https://github.com/cms-analysis/flashgg/blob/4edea8897e2a4b0518dca76ba6c9909c20c40ae7/Systematics/python/flashggDiPhotonSystematics2017_Legacy_cfi.py#L27
            # 6 bins (2 in SCeta 3 in r9)
            # link to the presentation: https://indico.cern.ch/event/963617/contributions/4103623/attachments/2141570/3608645/Zee_Validation_UL2017_Update_09112020_Prasant.pdf
            content_ = {
                "nominal": [0.9961, 0.9981, 0.9981, 1.0054, 1.0054, 1.0061],
                "up": [1.0268, 1.0038, 1.0038, 1.0183, 1.0183, 1.0079],
                "down": [0.9654, 0.9924, 0.9924, 0.9925, 0.9925, 1.0043],
            }
            flow_ = "clamp"
        elif opt.year == "2018":
            # Presel SF and uncertainty for UL2018. Dt:17/11/2020 fromflashgg: https://github.com/higgs-charm/flashgg/blob/17621e1d0f032e38c20294555359ed4682bd3f3b/Systematics/python/flashggDiPhotonSystematics2018_Legacy_cfi.py#L15-23
            # 6 bins (2 in SCeta 3 in r9)
            # Prasant Kumar Rout : UL18 Preselection SF and total uncertainties.
            content_ = {
                "nominal": [1.0017, 0.9973, 0.9973, 1.0030, 1.0030, 1.0031],
                "up": [1.0254, 1.004, 1.004, 1.024, 1.024, 1.0055],
                "down": [0.978, 0.9906, 0.9906, 0.982, 0.982, 1.0007],
            }
            flow_ = "clamp"

        PRES = cs.Correction(
            name="PreselSF",
            description="Preselection Scale Factor",
            generic_formulas=None,
            version=1,
            inputs=[
                cs.Variable(
                    name="systematic", type="string", description="Systematic variation"
                ),
                cs.Variable(name="SCEta", type="real", description="Photon Super Cluster eta absolute value"),
                cs.Variable(
                    name="r9",
                    type="real",
                    description="Photon full 5x5 R9, ratio E3x3/ERAW, where E3x3 is the energy sum of the 3 by 3 crystals surrounding the supercluster seed crystal and ERAW is the raw energy sum of the supercluster",
                ),
            ],
            output=cs.Variable(
                name="Ecorr",
                type="real",
                description="Multiplicative correction to event weight (per-photon)",
            ),
            data=cs.Category(
                nodetype="category",
                input="systematic",
                content=[
                    cs.CategoryItem(
                        key="up",
                        value=multibinning(inputs_, edges_, content_["up"], flow_)
                    ),
                    cs.CategoryItem(
                        key="down",
                        value=multibinning(inputs_, edges_, content_["down"], flow_)
                    ),
                ],
                default=multibinning(inputs_, edges_, content_["nominal"], flow_),
            ),
        )

        rich.print(PRES)

        etas = [0.8, 0.8, 0.8, 1.7, 1.7, 1.7]
        rs = [0.6, 0.85, 0.95, 0.6, 0.85, 0.99]

        print(PRES.to_evaluator())
        print("-" * 120)
        print("nominal --->","eta:",etas,"r9:",rs,"result:",PRES.to_evaluator().evaluate("nominal", etas, rs))
        print("-" * 120)
        print("up      --->","eta:",etas,"r9:",rs,"result:",PRES.to_evaluator().evaluate("up", etas, rs))
        print("-" * 120)
        print("down    --->","eta:",etas,"r9:",rs,"result:",PRES.to_evaluator().evaluate("down", etas, rs))
        print("-" * 120)
        print()

        cset = cs.CorrectionSet(
            schema_version=2,
            description="Preselection SF",
            compound_corrections=None,
            corrections=[
                PRES,
            ],
        )

        if opt.read == "":
            with open(f"PreselSF_{opt.year}.json", "w") as fout:
                fout.write(cset.model_dump_json(exclude_unset=True))

            import gzip

            with gzip.open(f"PreselSF_{opt.year}.json.gz", "wt") as fout:
                fout.write(cset.model_dump_json(exclude_unset=True))

        if opt.read != "":
            print("Reading back...")
            file = opt.read
            ceval = correctionlib.CorrectionSet.from_file(file)
            for corr in ceval.values():
                print(f"Correction {corr.name} has {len(corr.inputs)} inputs")
                for ix in corr.inputs:
                    print(f"   Input {ix.name} ({ix.type}): {ix.description}")

            print("-" * 120)
            print("nominal --->","eta:",etas,"r9:",rs,"result:",ceval["PreselSF"].evaluate("nominal", etas, rs))
            print("-" * 120)
            print("up      --->","eta:",etas,"r9:",rs,"result:",ceval["PreselSF"].evaluate("up", etas, rs))
            print("-" * 120)
            print("down    --->","eta:",etas,"r9:",rs,"result:",ceval["PreselSF"].evaluate("down", etas, rs))
            print("-" * 120)


    if opt.do_trigger_lead or opt.do_trigger or opt.do_all:
        inputs_ = ["SCEta", "r9", "pt"]
        edges_ = [[0.0, 1.5, 3, 999.0], [0.0, 0.56, 0.85, 0.9, 999.0], [0., 35., 37., 40., 45., 50., 60., 70., 90., 999999.0]]

        if "2016" in opt.year:
            edges_ = [[0.0, 1.5, 3, 999.0], [0.0, 0.54, 0.84, 0.85, 0.9, 999.0], [0., 35., 37., 40., 45., 50., 60., 70., 90., 999999.0]]
            content_ = {
                # trigger SF and uncertainty for UL2016 lead photon. Dt:17/11/2020 fromflashgg: https://github.com/higgs-charm/flashgg/blob/17621e1d0f032e38c20294555359ed4682bd3f3b/Systematics/python/flashggDiPhotonSystematics2016_Legacy_postVFP_cfi.py#L70C1-L145C5
                # a lot of bins (108, 3 in SCeta, 4 in r9 and 9 in Pt)
                # for full 2016 Legacy dataset.  Trigger scale factors for use without HLT applied in MC
                "nominal": [ #  33.3333       35            40            45            50            60            70            90             inf    pt
                    0.5982581241, 0.6537562616, 0.6834316643, 0.7100225895, 0.7305707075, 0.7516032845, 0.7881683245, 0.8279418571, 0.7830515018, # 0 < eta < 1.5, R9 < 0.54
                    0.8959984137, 0.9530533041, 0.9630432423, 0.9684613903, 0.9720401914, 0.9747626651, 0.9804445251, 0.9852156542, 0.9904418090, # 0 < eta < 1.5, 0.54 < R9 < 0.84
                    0.8959984137, 0.9530533041, 0.9630432423, 0.9684613903, 0.9720401914, 0.9747626651, 0.9804445251, 0.9852156542, 0.9904418090, # 0 < eta < 1.5, 0.84 < R9 < 0.85
                    0.9135648779, 0.9709473313, 0.9765279806, 0.9814195189, 0.9858544584, 0.9868334951, 0.9857502128, 0.9898608187, 0.9923759192, # 0 < eta < 1.5, 0.85 < R9 < 0.90
                    0.9135648779, 0.9709473313, 0.9765279806, 0.9814195189, 0.9858544584, 0.9868334951, 0.9857502128, 0.9898608187, 0.9923759192, # 0 < eta < 1.5, R9 > 0.90
                    0.4207263489, 0.5026636765, 0.5403756130, 0.5839972324, 0.6342410185, 0.6817455395, 0.6910253465, 0.6931545518, 0.7812379641, # 1.5 < eta < 3, R9 < 0.54
                    0.4207263489, 0.5026636765, 0.5403756130, 0.5839972324, 0.6342410185, 0.6817455395, 0.6910253465, 0.6931545518, 0.7812379641, # 1.5 < eta < 3, 0.54 < R9 < 0.84
                    0.8554153064, 0.9418683293, 0.9702138178, 0.9805935096, 0.9822542720, 0.9801241443, 0.9919474260, 0.9930837737, 0.9969498602, # 1.5 < eta < 3, 0.84 < R9 < 0.85
                    0.8554153064, 0.9418683293, 0.9702138178, 0.9805935096, 0.9822542720, 0.9801241443, 0.9919474260, 0.9930837737, 0.9969498602, # 1.5 < eta < 3, 0.85 < R9 < 0.90
                    0.8209652722, 0.9572389912, 0.9788842022, 0.9831094578, 0.9830138772, 0.9815419342, 0.9907973736, 0.9939137095, 0.9953388630, # 1.5 < eta < 3, R9 > 0.90
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, R9 < 0.54
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, 0.54 < R9 < 0.84
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, 0.84 < R9 < 0.85
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, 0.85 < R9 < 0.90
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000  # eta > 3, R9 > 0.90
                    ],
                "up": [ #       33.3333       35            40            45            50            60            70            90             inf    pt
                    0.6022853765, 0.6636617936, 0.6889797184, 0.7153430313, 0.7342830221, 0.7612109518, 0.8106445706, 0.8531980531, 0.8415640136, # 0 < eta < 1.5, R9 < 0.54
                    0.8969984137, 0.9540533041, 0.9640432423, 0.9694613903, 0.9730401914, 0.9757626651, 0.9815382416, 0.9862156542, 0.9916074908, # 0 < eta < 1.5, 0.54 < R9 < 0.84
                    0.8969984137, 0.9540533041, 0.9640432423, 0.9694613903, 0.9730401914, 0.9757626651, 0.9815382416, 0.9862156542, 0.9916074908, # 0 < eta < 1.5, 0.84 < R9 < 0.85
                    0.9145648779, 0.9719473313, 0.9775279806, 0.9824195189, 0.9868544584, 0.9878334951, 0.9867502128, 0.9908608187, 0.9933759192, # 0 < eta < 1.5, 0.85 < R9 < 0.90
                    0.9145648779, 0.9719473313, 0.9775279806, 0.9824195189, 0.9868544584, 0.9878334951, 0.9867502128, 0.9908608187, 0.9933759192, # 0 < eta < 1.5, R9 > 0.90
                    0.4240251641, 0.5111207522, 0.5484274312, 0.5863584737, 0.6390215202, 0.6926200966, 0.7063201499, 0.7087525900, 0.8002229660, # 1.5 < eta < 3, R9 < 0.54
                    0.4240251641, 0.5111207522, 0.5484274312, 0.5863584737, 0.6390215202, 0.6926200966, 0.7063201499, 0.7087525900, 0.8002229660, # 1.5 < eta < 3, 0.54 < R9 < 0.84
                    0.8570609300, 0.9432339636, 0.9712138178, 0.9815935096, 0.9832542720, 0.9818142693, 0.9937718541, 0.9967886468, 0.9981388247, # 1.5 < eta < 3, 0.84 < R9 < 0.85
                    0.8570609300, 0.9432339636, 0.9712138178, 0.9815935096, 0.9832542720, 0.9818142693, 0.9937718541, 0.9967886468, 0.9981388247, # 1.5 < eta < 3, 0.85 < R9 < 0.90
                    0.8230144564, 0.9582389912, 0.9798842022, 0.9841094578, 0.9840138772, 0.9827189922, 0.9922049697, 0.9949529952, 0.9963388630, # 1.5 < eta < 3, R9 > 0.90
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, R9 < 0.54
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, 0.54 < R9 < 0.84
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, 0.84 < R9 < 0.85
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, 0.85 < R9 < 0.90
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, R9 > 0.90
                ],
                "down": [ #     33.3333       35            40            45            50            60            70            90             inf    pt
                    0.5942308718, 0.6438507297, 0.6778836102, 0.7047021478, 0.7268583929, 0.7419956172, 0.7656920784, 0.8026856611, 0.7245389898, # 0 < eta < 1.5, R9 < 0.54
                    0.8949984137, 0.9520533041, 0.9620432423, 0.9674613903, 0.9710401914, 0.9737626651, 0.9793508086, 0.9842156542, 0.9892761272, # 0 < eta < 1.5, 0.54 < R9 < 0.84
                    0.8949984137, 0.9520533041, 0.9620432423, 0.9674613903, 0.9710401914, 0.9737626651, 0.9793508086, 0.9842156542, 0.9892761272, # 0 < eta < 1.5, 0.84 < R9 < 0.85
                    0.9125648779, 0.9699473313, 0.9755279806, 0.9804195189, 0.9848544584, 0.9858334951, 0.9847502128, 0.9888608187, 0.9913759192, # 0 < eta < 1.5, 0.85 < R9 < 0.90
                    0.9125648779, 0.9699473313, 0.9755279806, 0.9804195189, 0.9848544584, 0.9858334951, 0.9847502128, 0.9888608187, 0.9913759192, # 0 < eta < 1.5, R9 > 0.90
                    0.4174275337, 0.4942066008, 0.5323237948, 0.5816359910, 0.6294605168, 0.6708709825, 0.6757305432, 0.6775565137, 0.7622529622, # 1.5 < eta < 3, R9 < 0.54
                    0.4174275337, 0.4942066008, 0.5323237948, 0.5816359910, 0.6294605168, 0.6708709825, 0.6757305432, 0.6775565137, 0.7622529622, # 1.5 < eta < 3, 0.54 < R9 < 0.84
                    0.8537696828, 0.9405026950, 0.9692138178, 0.9795935096, 0.9812542720, 0.9784340192, 0.9901229979, 0.9893789006, 0.9957608958, # 1.5 < eta < 3, 0.84 < R9 < 0.85
                    0.8537696828, 0.9405026950, 0.9692138178, 0.9795935096, 0.9812542720, 0.9784340192, 0.9901229979, 0.9893789006, 0.9957608958, # 1.5 < eta < 3, 0.85 < R9 < 0.90
                    0.8189160881, 0.9562389912, 0.9778842022, 0.9821094578, 0.9820138772, 0.9803648762, 0.9893897775, 0.9928744238, 0.9943388630, # 1.5 < eta < 3, R9 > 0.90
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, R9 < 0.54
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, 0.54 < R9 < 0.84
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, 0.84 < R9 < 0.85
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, 0.85 < R9 < 0.90
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, R9 > 0.90
                ]
            }
            flow_ = "clamp"
        elif opt.year == "2017":
            content_ = {
                # trigger SF and uncertainty for UL2017 lead photon. Dt:17/11/2020 fromflashgg: https://github.com/cms-analysis/flashgg/blob/4edea8897e2a4b0518dca76ba6c9909c20c40ae7/Systematics/python/flashggDiPhotonSystematics2017_Legacy_cfi.py#L27
                # a lot of bins (81, 3 in SCeta, 3 in r9 and 9 in Pt)
                # link to the presentation: https://indico.cern.ch/event/963617/contributions/4103623/attachments/2141570/3608645/Zee_Validation_UL2017_Update_09112020_Prasant.pdf
                "nominal": [ #  35            37            40            45            50            60            70            90             inf    pt
                    0.5962935611, 0.7301764945, 0.7481329080, 0.7715837809, 0.7885084282, 0.8106800401, 0.8403040278, 0.8403394687, 0.9294116662, # 0 < eta < 1.5, R9 < 0.56
                    0.8185203301, 0.9430487014, 0.9502315420, 0.9569881714, 0.9607761449, 0.9667723989, 0.9735556426, 0.9779985569, 0.9846204583, # 0 < eta < 1.5, 0.56 < R9 < 0.85
                    0.8487571377, 0.9553428958, 0.9624070784, 0.9693638237, 0.9755313942, 0.9777963391, 0.9812161610, 0.9845542680, 0.9902588867, # 0 < eta < 1.5, 0.85 < R9 < 0.90
                    0.8487571377, 0.9553428958, 0.9624070784, 0.9693638237, 0.9755313942, 0.9777963391, 0.9812161610, 0.9845542680, 0.9902588867, # 0 < eta < 1.5, R9 > 0.90
                    0.6297968504, 0.8049372754, 0.8314952358, 0.8544229767, 0.8875746672, 0.9033407955, 0.9207605401, 0.9410420565, 0.9586907211, # 1.5 < eta < 3, R9 < 0.56
                    0.6297968504, 0.8049372754, 0.8314952358, 0.8544229767, 0.8875746672, 0.9033407955, 0.9207605401, 0.9410420565, 0.9586907211, # 1.5 < eta < 3, 0.56 < R9 < 0.85
                    0.7816089799, 0.9601546944, 0.9728943976, 0.9787293111, 0.9836865868, 0.9845440645, 0.9863780801, 0.9913050524, 0.9969391106, # 1.5 < eta < 3, 0.85 < R9 < 0.90
                    0.7758811194, 0.9592830235, 0.9692856214, 0.9763703079, 0.9814613177, 0.9825431442, 0.9857720941, 0.9904181104, 0.9923572396, # 1.5 < eta < 3, R9 > 0.90
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, R9 < 0.56
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, 0.56 < R9 < 0.85
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, 0.85 < R9 < 0.90
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000  # eta > 3, R9 > 0.90
                    ],
                "up": [ #  35            37            40            45            50            60            70            90             inf    pt
                    0.6005737887, 0.7328812488, 0.7503826099, 0.7742611691, 0.7919288403, 0.8258226645, 0.8589755363, 0.8629941224, 0.9561358245, # 0 < eta < 1.5, R9 < 0.56
                    0.8203873280, 0.9441263182, 0.9512360353, 0.9580805517, 0.9617771241, 0.9688333341, 0.9754818208, 0.9794625844, 0.9859530068, # 0 < eta < 1.5, 0.56 < R9 < 0.85
                    0.8578682281, 0.9577316838, 0.9653803117, 0.9703944882, 0.9765665213, 0.9799697256, 0.9858818495, 0.9855805535, 0.9913562563, # 0 < eta < 1.5, 0.85 < R9 < 0.90
                    0.8578682281, 0.9577316838, 0.9653803117, 0.9703944882, 0.9765665213, 0.9799697256, 0.9858818495, 0.9855805535, 0.9913562563, # 0 < eta < 1.5, R9 > 0.90
                    0.6324307687, 0.8076131776, 0.8336690619, 0.8559467838, 0.8904000969, 0.9114569193, 0.9278738793, 0.9485921217, 0.9650381006, # 1.5 < eta < 3, R9 < 0.56
                    0.6324307687, 0.8076131776, 0.8336690619, 0.8559467838, 0.8904000969, 0.9114569193, 0.9278738793, 0.9485921217, 0.9650381006, # 1.5 < eta < 3, 0.56 < R9 < 0.85
                    0.7838966125, 0.9615399325, 0.9739169322, 0.9798026624, 0.9848531139, 0.9867883788, 0.9890037354, 0.9930929945, 0.9990857728, # 1.5 < eta < 3, 0.85 < R9 < 0.90
                    0.7785424680, 0.9615285198, 0.9703116690, 0.9774045805, 0.9825372871, 0.9841343722, 0.9872986285, 0.9914973187, 0.9960040747, # 1.5 < eta < 3, R9 > 0.90
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, R9 < 0.56
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, 0.56 < R9 < 0.85
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, 0.85 < R9 < 0.90
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, R9 > 0.90
                ],
                "down": [ #  35            37            40            45            50            60            70            90             inf    pt
                    0.5920133335, 0.7274717402, 0.7458832061, 0.7689063927, 0.7850880161, 0.7955374157, 0.8216325193, 0.8176848150, 0.9026875079, # 0 < eta < 1.5, R9 < 0.56
                    0.8166533322, 0.9419710846, 0.9492270487, 0.9558957911, 0.9597751657, 0.9647114637, 0.9716294644, 0.9765345294, 0.9832879098, # 0 < eta < 1.5, 0.56 < R9 < 0.85
                    0.8396460473, 0.9529541078, 0.9594338451, 0.9683331592, 0.9744962671, 0.9756229526, 0.9765504725, 0.9835279825, 0.9891615171, # 0 < eta < 1.5, 0.85 < R9 < 0.90
                    0.8396460473, 0.9529541078, 0.9594338451, 0.9683331592, 0.9744962671, 0.9756229526, 0.9765504725, 0.9835279825, 0.9891615171, # 0 < eta < 1.5, R9 > 0.90
                    0.6271629321, 0.8022613732, 0.8293214097, 0.8528991696, 0.8847492375, 0.8952246717, 0.9136472009, 0.9334919913, 0.9523433416, # 1.5 < eta < 3, R9 < 0.56
                    0.6271629321, 0.8022613732, 0.8293214097, 0.8528991696, 0.8847492375, 0.8952246717, 0.9136472009, 0.9334919913, 0.9523433416, # 1.5 < eta < 3, 0.56 < R9 < 0.85
                    0.7793213473, 0.9587694563, 0.9718718630, 0.9776559598, 0.9825200597, 0.9822997502, 0.9837524248, 0.9895171103, 0.9947924484, # 1.5 < eta < 3, 0.85 < R9 < 0.90
                    0.7732197708, 0.9570375272, 0.9682595738, 0.9753360353, 0.9803853483, 0.9809519162, 0.9842455597, 0.9893389021, 0.9887104045, # 1.5 < eta < 3, R9 > 0.90
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, R9 < 0.56
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, 0.56 < R9 < 0.85
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, 0.85 < R9 < 0.90
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, R9 > 0.90
                ]
            }
            flow_ = "clamp"
        elif opt.year == "2018":
            #for full 2018 UL (Era2018_RR-17Sep2018_v2-legacyRun2FullV2-v0, rereco A-D dataset).  Trigger scale factors for use without HLT applied in MC
            # trigger SF and uncertainty for UL2017 lead photon. Dt:17/11/2020 fromflashgg: https://github.com/higgs-charm/flashgg/blob/17621e1d0f032e38c20294555359ed4682bd3f3b/Systematics/python/flashggDiPhotonSystematics2018_Legacy_cfi.py#L103C1-L177C6
            # a lot of bins (81, 3 in SCeta, 3 in r9 and 9 in Pt)
            content_ = {
                "nominal": [ #  35            37            40            45            50            60            70            90             inf    pt
                    0.6196012864, 0.7761489544, 0.8002387986, 0.8231650667, 0.8453928052, 0.8617735570, 0.8712109006, 0.8865923420, 0.8891188478, # 0 < eta < 1.5, R9 < 0.56
                    0.7976408633, 0.9473792086, 0.9578851687, 0.9606883609, 0.9638132287, 0.9653299268, 0.9708310992, 0.9811906983, 0.9858675076, # 0 < eta < 1.5, 0.56 < R9 < 0.85
                    0.8564371935, 0.9594766157, 0.9663358811, 0.9724797256, 0.9763581853, 0.9764434357, 0.9802652970, 0.9840348160, 0.9899930822, # 0 < eta < 1.5, 0.85 < R9 < 0.90
                    0.8564371935, 0.9594766157, 0.9663358811, 0.9724797256, 0.9763581853, 0.9764434357, 0.9802652970, 0.9840348160, 0.9899930822, # 0 < eta < 1.5, R9 > 0.90
                    0.6595211482, 0.7445913750, 0.7721763963, 0.8107180165, 0.8439171883, 0.8596941548, 0.8831719219, 0.9172864768, 0.9417547952, # 1.5 < eta < 3, R9 < 0.56
                    0.6595211482, 0.7445913750, 0.7721763963, 0.8107180165, 0.8439171883, 0.8596941548, 0.8831719219, 0.9172864768, 0.9417547952, # 1.5 < eta < 3, 0.56 < R9 < 0.85
                    0.8923336432, 0.9732925196, 0.9778957733, 0.9824372073, 0.9867443152, 0.9875279678, 0.9905722669, 0.9875423811, 0.9939461454, # 1.5 < eta < 3, 0.85 < R9 < 0.90
                    0.8832344791, 0.9780127017, 0.9798867348, 0.9842874232, 0.9884889713, 0.9865944794, 0.9874199891, 0.9901687777, 0.9939249798, # 1.5 < eta < 3, R9 > 0.90
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, R9 < 0.56
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, 0.56 < R9 < 0.85
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, 0.85 < R9 < 0.90
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000  # eta > 3, R9 > 0.90
                    ],
                "up": [ #  35            37            40            45            50            60            70            90             inf    pt
                    0.6269280252, 0.7786603059, 0.8021586608, 0.8256731151, 0.8532551021, 0.8714833691, 0.8845111534, 0.9045388617, 0.9316145532, # 0 < eta < 1.5, R9 < 0.56
                    0.7988508680, 0.9483834440, 0.9613930556, 0.9618245586, 0.9649903848, 0.9671620001, 0.9721773295, 0.9838389359, 0.9870879101, # 0 < eta < 1.5, 0.56 < R9 < 0.85
                    0.8577275149, 0.9607508732, 0.9675449292, 0.9735744591, 0.9778869611, 0.9775755936, 0.9812860390, 0.9850658733, 0.9910035641, # 0 < eta < 1.5, 0.85 < R9 < 0.90
                    0.8577275149, 0.9607508732, 0.9675449292, 0.9735744591, 0.9778869611, 0.9775755936, 0.9812860390, 0.9850658733, 0.9910035641, # 0 < eta < 1.5, R9 > 0.90
                    0.6630887286, 0.7468945264, 0.7786973060, 0.8121334187, 0.8475339625, 0.8656011319, 0.8908619596, 0.9248609370, 0.9486917425, # 1.5 < eta < 3, R9 < 0.56
                    0.6630887286, 0.7468945264, 0.7786973060, 0.8121334187, 0.8475339625, 0.8656011319, 0.8908619596, 0.9248609370, 0.9486917425, # 1.5 < eta < 3, 0.56 < R9 < 0.85
                    0.8941870572, 0.9743306509, 0.9795351094, 0.9834373265, 0.9884008901, 0.9891315475, 0.9929901049, 0.9893635741, 0.9954143446, # 1.5 < eta < 3, 0.85 < R9 < 0.90
                    0.8847612299, 0.9790583409, 0.9808965076, 0.9852957931, 0.9900267600, 0.9875997349, 0.9886412751, 0.9925053982, 0.9958427811, # 1.5 < eta < 3, R9 > 0.90
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, R9 < 0.56
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, 0.56 < R9 < 0.85
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, 0.85 < R9 < 0.90
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, R9 > 0.90
                ],
                "down": [ #  35            37            40            45            50            60            70            90             inf    pt
                    0.6122745476, 0.7736376029, 0.7983189364, 0.8206570183, 0.8375305083, 0.8520637449, 0.8579106478, 0.8686458223, 0.8466231424, # 0 < eta < 1.5, R9 < 0.56
                    0.7964308586, 0.9463749732, 0.9543772818, 0.9595521632, 0.9626360726, 0.9634978535, 0.9694848689, 0.9785424607, 0.9846471051, # 0 < eta < 1.5, 0.56 < R9 < 0.85
                    0.8551468721, 0.9582023582, 0.9651268330, 0.9713849921, 0.9748294095, 0.9753112778, 0.9792445550, 0.9830037587, 0.9889826003, # 0 < eta < 1.5, 0.85 < R9 < 0.90
                    0.8551468721, 0.9582023582, 0.9651268330, 0.9713849921, 0.9748294095, 0.9753112778, 0.9792445550, 0.9830037587, 0.9889826003, # 0 < eta < 1.5, R9 > 0.90
                    0.6559535678, 0.7422882236, 0.7656554866, 0.8093026143, 0.8403004141, 0.8537871777, 0.8754818842, 0.9097120166, 0.9348178479, # 1.5 < eta < 3, R9 < 0.56
                    0.6559535678, 0.7422882236, 0.7656554866, 0.8093026143, 0.8403004141, 0.8537871777, 0.8754818842, 0.9097120166, 0.9348178479, # 1.5 < eta < 3, 0.56 < R9 < 0.85
                    0.8904802292, 0.9722543883, 0.9762564372, 0.9814370881, 0.9850877403, 0.9859243881, 0.9881544289, 0.9857211881, 0.9924779462, # 1.5 < eta < 3, 0.85 < R9 < 0.90
                    0.8817077283, 0.9769670625, 0.9788769620, 0.9832790533, 0.9869511826, 0.9855892239, 0.9861987031, 0.9878321572, 0.9920071785, # 1.5 < eta < 3, R9 > 0.90
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, R9 < 0.56
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, 0.56 < R9 < 0.85
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, 0.85 < R9 < 0.90
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, R9 > 0.90
                ]
            }
            flow_ = "clamp"

        TRIG = cs.Correction(
            name="TriggerSF",
            description="Lead Photon Trigger Scale Factor",
            generic_formulas=None,
            version=1,
            inputs=[
                cs.Variable(
                    name="systematic", type="string", description="Systematic variation"
                ),
                cs.Variable(name="SCEta", type="real", description="Photon Super Cluster eta absolute value"),
                cs.Variable(
                    name="r9",
                    type="real",
                    description="Photon full 5x5 R9, ratio E3x3/ERAW, where E3x3 is the energy sum of the 3 by 3 crystals surrounding the supercluster seed crystal and ERAW is the raw energy sum of the supercluster",
                ),
                cs.Variable(
                    name="pt",
                    type="real",
                    description="Photon pt",
                ),
            ],
            output=cs.Variable(
                name="Wcorr",
                type="real",
                description="Multiplicative correction to event weight (per-photon)",
            ),
            data=cs.Category(
                nodetype="category",
                input="systematic",
                content=[
                    cs.CategoryItem(
                        key="up",
                        value=multibinning(inputs_, edges_, content_["up"], flow_)
                    ),
                    cs.CategoryItem(
                        key="down",
                        value=multibinning(inputs_, edges_, content_["down"], flow_)
                    ),
                ],
                default=multibinning(inputs_, edges_, content_["nominal"], flow_),
            ),
        )

        rich.print(TRIG)

        etas = [
            0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8,
            0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8,
            0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8,
            0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8,
            1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7,
            1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7,
            1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7,
            1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7,
            10., 10., 10., 10., 10., 10., 10., 10., 10.,
            10., 10., 10., 10., 10., 10., 10., 10., 10.,
            10., 10., 10., 10., 10., 10., 10., 10., 10.,
            10., 10., 10., 10., 10., 10., 10., 10., 10.
            ]
        rs = [
            0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4,
            0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6,
            0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88,
            0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99,
            0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4,
            0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6,
            0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88,
            0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99,
            0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4,
            0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6,
            0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88,
            0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99
            ]
        pts = [
            15., 36., 38., 42., 47., 55., 65., 75., 200.,
            15., 36., 38., 42., 47., 55., 65., 75., 200.,
            15., 36., 38., 42., 47., 55., 65., 75., 200.,
            15., 36., 38., 42., 47., 55., 65., 75., 200.,
            15., 36., 38., 42., 47., 55., 65., 75., 200.,
            15., 36., 38., 42., 47., 55., 65., 75., 200.,
            15., 36., 38., 42., 47., 55., 65., 75., 200.,
            15., 36., 38., 42., 47., 55., 65., 75., 200.,
            15., 36., 38., 42., 47., 55., 65., 75., 200.,
            15., 36., 38., 42., 47., 55., 65., 75., 200.,
            15., 36., 38., 42., 47., 55., 65., 75., 200.,
            15., 36., 38., 42., 47., 55., 65., 75., 200.,
            ]

        print(TRIG.to_evaluator())
        print("-" * 120)
        print("nominal --->","eta:",etas,"r9:",rs,"pt:",pts,"result:",TRIG.to_evaluator().evaluate("nominal", etas, rs, pts))
        print("-" * 120)
        print("up      --->","eta:",etas,"r9:",rs,"pt:",pts,"result:",TRIG.to_evaluator().evaluate("up", etas, rs, pts))
        print("-" * 120)
        print("down    --->","eta:",etas,"r9:",rs,"pt:",pts,"result:",TRIG.to_evaluator().evaluate("down", etas, rs, pts))
        print("-" * 120)
        print()

        cset = cs.CorrectionSet(
            schema_version=2,
            description="Trigger SF",
            compound_corrections=None,
            corrections=[
                TRIG,
            ],
        )

        if opt.read == "":
            with open(f"TriggerSF_lead_{opt.year}.json", "w") as fout:
                fout.write(cset.model_dump_json(exclude_unset=True))

            import gzip

            with gzip.open(f"TriggerSF_lead_{opt.year}.json.gz", "wt") as fout:
                fout.write(cset.model_dump_json(exclude_unset=True))

        if opt.read != "":
            print("Reading back...")
            file = opt.read
            ceval = correctionlib.CorrectionSet.from_file(file)
            for corr in ceval.values():
                print(f"Correction {corr.name} has {len(corr.inputs)} inputs")
                for ix in corr.inputs:
                    print(f"   Input {ix.name} ({ix.type}): {ix.description}")

            print("-" * 120)
            print("nominal --->","eta:",etas,"r9:",rs,"result:",ceval["TriggerSF"].evaluate("nominal", etas, rs, pts))
            print("-" * 120)
            print("up      --->","eta:",etas,"r9:",rs,"result:",ceval["TriggerSF"].evaluate("up", etas, rs, pts))
            print("-" * 120)
            print("down    --->","eta:",etas,"r9:",rs,"result:",ceval["TriggerSF"].evaluate("down", etas, rs, pts))
            print("-" * 120)


    if opt.do_trigger_sublead or opt.do_trigger or opt.do_all:
        inputs_ = ["SCEta", "r9", "pt"]
        edges_ = [[0.0, 1.5, 3, 999.0], [0.0, 0.56, 0.85, 0.9, 999.0], [0., 28., 31., 35., 40., 45., 50., 60., 70., 90., 999999.0]]

        if "2016" in opt.year:
            edges_ = [[0.0, 1.5, 3, 999.0], [0.0, 0.54, 0.84, 0.85, 0.9, 999.0], [0., 22.5, 25.0, 27.5, 30.0, 32.5, 35.0, 40.0, 45.0, 50.0, 60.0, 70.0, 90.0, 999999.0]]
            # trigger SF and uncertainty for UL2016 sublead photon. fromflashgg: https://github.com/higgs-charm/flashgg/blob/17621e1d0f032e38c20294555359ed4682bd3f3b/Systematics/python/flashggDiPhotonSystematics2016_Legacy_postVFP_cfi.py#L148C1-L251C6
            # a lot of bins (180, 3 in SCeta, 5 in r9 and 12 in Pt)
            content_ = {
                "nominal": [ #  22.5          25            27.5          30            32.5          35            40            45            50            60            70            90           inf    pt
                    0.6030673118, 0.6244928501, 0.6487889796, 0.6541334399, 0.6765893143, 0.6832028631, 0.7046064998, 0.7329237982, 0.7554470113, 0.7707511552, 0.7901985113, 0.8336943751, 0.7957466347, # 0 < eta < 1.5, R9 < 0.54
                    0.9856936985, 0.9872078495, 0.9901368510, 0.9907036804, 0.9914011080, 0.9911396064, 0.9905948342, 0.9922644763, 0.9921542895, 0.9916576153, 0.9923769373, 0.9913275240, 0.9923277119, # 0 < eta < 1.5, 0.54 < R9 < 0.84
                    0.9856936985, 0.9872078495, 0.9901368510, 0.9907036804, 0.9914011080, 0.9911396064, 0.9905948342, 0.9922644763, 0.9921542895, 0.9916576153, 0.9923769373, 0.9913275240, 0.9923277119, # 0 < eta < 1.5, 0.84 < R9 < 0.85
                    0.9917610961, 0.9920930158, 0.9956508629, 0.9973259559, 0.9977762307, 0.9980340212, 0.9985046491, 0.9991978664, 0.9994846808, 0.9995256956, 0.9994891484, 0.9995869001, 0.9997687611, # 0 < eta < 1.5, 0.85 < R9 < 0.90
                    0.9917610961, 0.9920930158, 0.9956508629, 0.9973259559, 0.9977762307, 0.9980340212, 0.9985046491, 0.9991978664, 0.9994846808, 0.9995256956, 0.9994891484, 0.9995869001, 0.9997687611, # 0 < eta < 1.5, R9 > 0.90
                    0.3979046597, 0.4763382806, 0.4767600795, 0.4924567423, 0.4995746589, 0.5148837667, 0.5469283538, 0.5862182671, 0.6349481291, 0.6833440120, 0.6878934573, 0.6973386674, 0.7769101961, # 1.5 < eta < 3, R9 < 0.54
                    0.3979046597, 0.4763382806, 0.4767600795, 0.4924567423, 0.4995746589, 0.5148837667, 0.5469283538, 0.5862182671, 0.6349481291, 0.6833440120, 0.6878934573, 0.6973386674, 0.7769101961, # 1.5 < eta < 3, 0.54 < R9 < 0.84
                    0.9144405468, 0.9462403094, 0.9575900857, 0.9667399720, 0.9696445662, 0.9720551970, 0.9799502033, 0.9863179135, 0.9890668352, 0.9892349065, 0.9875474671, 0.9931794090, 0.9928048404, # 1.5 < eta < 3, 0.84 < R9 < 0.85
                    0.9144405468, 0.9462403094, 0.9575900857, 0.9667399720, 0.9696445662, 0.9720551970, 0.9799502033, 0.9863179135, 0.9890668352, 0.9892349065, 0.9875474671, 0.9931794090, 0.9928048404, # 1.5 < eta < 3, 0.85 < R9 < 0.90
                    0.9797427643, 0.9957876524, 0.9969538286, 0.9976282327, 0.9961524910, 0.9979294872, 0.9985427859, 0.9988906137, 0.9990152331, 0.9985998442, 0.9994856155, 0.9996963038, 0.9995293124, # 1.5 < eta < 3, R9 > 0.90
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, R9 < 0.54
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, 0.54 < R9 < 0.84
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, 0.84 < R9 < 0.85
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, 0.85 < R9 < 0.90
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000  # eta > 3, R9 > 0.90
                    ],
                "up": [ #       22.5          25            27.5          30            32.5          35            40            45            50            60            70            90           inf    pt
                    0.6110289715, 0.6337617851, 0.6552264796, 0.6597407700, 0.6813719310, 0.6892305340, 0.7095170024, 0.7381906621, 0.7590551732, 0.7801625746, 0.8123594186, 0.8582588856, 0.8571775346, # 0 < eta < 1.5, R9 < 0.54
                    0.9866936985, 0.9882078495, 0.9911368510, 0.9917036804, 0.9924011080, 0.9921396064, 0.9915948342, 0.9932644763, 0.9931542895, 0.9926576153, 0.9933769373, 0.9923275240, 0.9933277119, # 0 < eta < 1.5, 0.54 < R9 < 0.84
                    0.9866936985, 0.9882078495, 0.9911368510, 0.9917036804, 0.9924011080, 0.9921396064, 0.9915948342, 0.9932644763, 0.9931542895, 0.9926576153, 0.9933769373, 0.9923275240, 0.9933277119, # 0 < eta < 1.5, 0.84 < R9 < 0.85
                    0.9943127049, 0.9930930158, 0.9966508629, 0.9983259559, 0.9987762307, 0.9990340212, 0.9995046491, 1.0001978664, 1.0004846808, 1.0005256956, 1.0004891484, 1.0005869001, 1.0007687611, # 0 < eta < 1.5, 0.85 < R9 < 0.90
                    0.9943127049, 0.9930930158, 0.9966508629, 0.9983259559, 0.9987762307, 0.9990340212, 0.9995046491, 1.0001978664, 1.0004846808, 1.0005256956, 1.0004891484, 1.0005869001, 1.0007687611, # 0 < eta < 1.5, R9 > 0.90
                    0.4082747853, 0.4835849117, 0.4833890527, 0.4980506255, 0.5044670297, 0.5194943687, 0.5536924596, 0.5886069800, 0.6398089550, 0.6942964994, 0.7033232218, 0.7130522192, 0.7961727755, # 1.5 < eta < 3, R9 < 0.54
                    0.4082747853, 0.4835849117, 0.4833890527, 0.4980506255, 0.5044670297, 0.5194943687, 0.5536924596, 0.5886069800, 0.6398089550, 0.6942964994, 0.7033232218, 0.7130522192, 0.7961727755, # 1.5 < eta < 3, 0.54 < R9 < 0.84
                    0.9181393241, 0.9490024306, 0.9596238737, 0.9683841408, 0.9707412875, 0.9730551970, 0.9809502033, 0.9873179135, 0.9900668352, 0.9907428256, 0.9897084998, 0.9951082200, 0.9945945342, # 1.5 < eta < 3, 0.84 < R9 < 0.85
                    0.9181393241, 0.9490024306, 0.9596238737, 0.9683841408, 0.9707412875, 0.9730551970, 0.9809502033, 0.9873179135, 0.9900668352, 0.9907428256, 0.9897084998, 0.9951082200, 0.9945945342, # 1.5 < eta < 3, 0.85 < R9 < 0.90
                    0.9809987886, 0.9967876524, 0.9979538286, 0.9989888149, 0.9971524910, 0.9989294872, 0.9995427860, 0.9998906137, 1.0000152331, 0.9995998442, 1.0004856155, 1.0006963038, 1.0005293124, # 1.5 < eta < 3, R9 > 0.90
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, R9 < 0.54
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, 0.54 < R9 < 0.84
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, 0.84 < R9 < 0.85
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, 0.85 < R9 < 0.90
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000  # eta > 3, R9 > 0.90
                ],
                "down": [ #     22.5          25            27.5          30            32.5          35            40            45            50            60            70            90           inf    pt
                    0.5951056521, 0.6152239152, 0.6423514795, 0.6485261098, 0.6718066975, 0.6771751923, 0.6996959973, 0.7276569342, 0.7518388494, 0.7613397358, 0.7680376039, 0.8091298647, 0.7343157349, # 0 < eta < 1.5, R9 < 0.54
                    0.9846936985, 0.9862078495, 0.9891368510, 0.9897036804, 0.9904011080, 0.9901396064, 0.9895948342, 0.9912644763, 0.9911542895, 0.9906576153, 0.9913769373, 0.9903275240, 0.9913277119, # 0 < eta < 1.5, 0.54 < R9 < 0.84
                    0.9846936985, 0.9862078495, 0.9891368510, 0.9897036804, 0.9904011080, 0.9901396064, 0.9895948342, 0.9912644763, 0.9911542895, 0.9906576153, 0.9913769373, 0.9903275240, 0.9913277119, # 0 < eta < 1.5, 0.84 < R9 < 0.85
                    0.9892094873, 0.9910930158, 0.9946508629, 0.9963259559, 0.9967762307, 0.9970340212, 0.9975046491, 0.9981978664, 0.9984846808, 0.9985256956, 0.9984891484, 0.9985869001, 0.9987687611, # 0 < eta < 1.5, 0.85 < R9 < 0.90
                    0.9892094873, 0.9910930158, 0.9946508629, 0.9963259559, 0.9967762307, 0.9970340212, 0.9975046491, 0.9981978664, 0.9984846808, 0.9985256956, 0.9984891484, 0.9985869001, 0.9987687611, # 0 < eta < 1.5, R9 > 0.90
                    0.3875345340, 0.4690916496, 0.4701311063, 0.4868628590, 0.4946822880, 0.5102731648, 0.5401642481, 0.5838295541, 0.6300873032, 0.6723915246, 0.6724636927, 0.6816251156, 0.7576476167, # 1.5 < eta < 3, R9 < 0.54
                    0.3875345340, 0.4690916496, 0.4701311063, 0.4868628590, 0.4946822880, 0.5102731648, 0.5401642481, 0.5838295541, 0.6300873032, 0.6723915246, 0.6724636927, 0.6816251156, 0.7576476167, # 1.5 < eta < 3, 0.54 < R9 < 0.84
                    0.9107417696, 0.9434781884, 0.9555562978, 0.9650958032, 0.9685478450, 0.9710551970, 0.9789502033, 0.9853179135, 0.9880668352, 0.9877269874, 0.9853864344, 0.9912505981, 0.9910151466, # 1.5 < eta < 3, 0.84 < R9 < 0.85
                    0.9107417696, 0.9434781884, 0.9555562978, 0.9650958032, 0.9685478450, 0.9710551970, 0.9789502033, 0.9853179135, 0.9880668352, 0.9877269874, 0.9853864344, 0.9912505981, 0.9910151466, # 1.5 < eta < 3, 0.85 < R9 < 0.90
                    0.9784867400, 0.9947876524, 0.9959538286, 0.9962676506, 0.9951524910, 0.9969294872, 0.9975427859, 0.9978906137, 0.9980152331, 0.9975998442, 0.9984856155, 0.9986963038, 0.9985293124, # 1.5 < eta < 3, R9 > 0.90
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, R9 < 0.54
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, 0.54 < R9 < 0.84
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, 0.84 < R9 < 0.85
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, 0.85 < R9 < 0.90
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000 # eta > 3, R9 > 0.90
                ]
            }
            flow_ = "clamp"
        elif opt.year == "2017":
            # trigger SF and uncertainty for UL2017 sublead photon. Dt:17/11/2020 fromflashgg: https://github.com/cms-analysis/flashgg/blob/4edea8897e2a4b0518dca76ba6c9909c20c40ae7/Systematics/python/flashggDiPhotonSystematics2017_Legacy_cfi.py#L27
            # a lot of bins (90, 3 in SCeta, 3 in r9 and 10 in Pt)
            # link to the presentation: https://indico.cern.ch/event/963617/contributions/4103623/attachments/2141570/3608645/Zee_Validation_UL2017_Update_09112020_Prasant.pdf
            content_ = {
                "nominal": [ #  28            31            35            40            45            50            60            70            90             inf    pt
                    0.6155988939, 0.7165819087, 0.7381962831, 0.7671925006, 0.7999358222, 0.8254675016, 0.8297030540, 0.8451584417, 0.8522482004, 0.8871193652, # 0 < eta < 1.5, R9 < 0.56
                    0.9028486970, 0.9739174387, 0.9756211698, 0.9785859435, 0.9814869681, 0.9836603606, 0.9808533747, 0.9788313651, 0.9766053770, 0.9667617117, # 0 < eta < 1.5, 0.56 < R9 < 0.85
                    0.8933536854, 0.9973958724, 0.9980479262, 0.9987289490, 0.9992636424, 0.9994686970, 0.9995552559, 0.9992541003, 0.9996086647, 0.9996779894, # 0 < eta < 1.5, 0.85 < R9 < 0.90
                    0.8933536854, 0.9973958724, 0.9980479262, 0.9987289490, 0.9992636424, 0.9994686970, 0.9995552559, 0.9992541003, 0.9996086647, 0.9996779894, # 0 < eta < 1.5, R9 > 0.90
                    0.6100544113, 0.7427840769, 0.7761341323, 0.8117452882, 0.8319088440, 0.8583582498, 0.8736432627, 0.8907409748, 0.9046665266, 0.9190711276, # 1.5 < eta < 3, R9 < 0.56
                    0.6100544113, 0.7427840769, 0.7761341323, 0.8117452882, 0.8319088440, 0.8583582498, 0.8736432627, 0.8907409748, 0.9046665266, 0.9190711276, # 1.5 < eta < 3, 0.56 < R9 < 0.85
                    0.8283101205, 0.9538552575, 0.9597166341, 0.9617373097, 0.9624428298, 0.9581303007, 0.9621293579, 0.9670262230, 0.9721855102, 0.9753380476, # 1.5 < eta < 3, 0.85 < R9 < 0.90
                    0.8582078363, 0.9911788518, 0.9961663139, 0.9974520554, 0.9983872590, 0.9988958563, 0.9987919975, 0.9992790060, 0.9994720350, 0.9995989436, # 1.5 < eta < 3, R9 > 0.90
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, R9 < 0.56
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, 0.56 < R9 < 0.85
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, 0.85 < R9 < 0.90
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000  # eta > 3, R9 > 0.90
                    ],
                "up": [ #       28            31            35            40            45            50            60            70            90             inf    pt
                    0.6214049860, 0.7200945062, 0.7414391429, 0.7702228640, 0.8020985193, 0.8290374613, 0.8409052024, 0.8613078310, 0.8717261113, 0.9167756020, # 0 < eta < 1.5, R9 < 0.56
                    0.9038765514, 0.9757861251, 0.9766339755, 0.9796238611, 0.9825145230, 0.9848254173, 0.9818546188, 0.9812502236, 0.9778944586, 0.9689814201, # 0 < eta < 1.5, 0.56 < R9 < 0.85
                    0.8999503055, 0.9991625217, 0.9991339772, 0.9997289596, 1.0002637084, 1.0004687794, 1.0005552634, 1.0003072356, 1.0006087172, 1.0006779895, # 0 < eta < 1.5, 0.85 < R9 < 0.90
                    0.8999503055, 0.9991625217, 0.9991339772, 0.9997289596, 1.0002637084, 1.0004687794, 1.0005552634, 1.0003072356, 1.0006087172, 1.0006779895, # 0 < eta < 1.5, R9 > 0.90
                    0.6145174735, 0.7469034105, 0.7784538864, 0.8135725031, 0.8336459809, 0.8612610880, 0.8803866839, 0.8996338556, 0.9129986758, 0.9283278540, # 1.5 < eta < 3, R9 < 0.56
                    0.6145174735, 0.7469034105, 0.7784538864, 0.8135725031, 0.8336459809, 0.8612610880, 0.8803866839, 0.8996338556, 0.9129986758, 0.9283278540, # 1.5 < eta < 3, 0.56 < R9 < 0.85
                    0.8326968341, 0.9563037293, 0.9610060490, 0.9634283585, 0.9634432295, 0.9597145494, 0.9651548567, 0.9709326862, 0.9753379325, 0.9785131581, # 1.5 < eta < 3, 0.85 < R9 < 0.90
                    0.8592510525, 0.9930106233, 0.9972930973, 0.9984525944, 0.9995239916, 0.9998959452, 0.9997922552, 1.0003196950, 1.0004723952, 1.0005989451, # 1.5 < eta < 3, R9 > 0.90
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, R9 < 0.56
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, 0.56 < R9 < 0.85
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, 0.85 < R9 < 0.90
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000  # eta > 3, R9 > 0.90
                ],
                "down": [ #     28            31            35            40            45            50            60            70            90             inf    pt
                    0.6097928018, 0.7130693112, 0.7349534233, 0.7641621372, 0.7977731251, 0.8218975419, 0.8185009056, 0.8290090524, 0.8327702895, 0.8574631284, # 0 < eta < 1.5, R9 < 0.56
                    0.9018208426, 0.9720487523, 0.9746083641, 0.9775480259, 0.9804594132, 0.9824953039, 0.9798521306, 0.9764125066, 0.9753162954, 0.9645420033, # 0 < eta < 1.5, 0.56 < R9 < 0.85
                    0.8867570653, 0.9956292231, 0.9969618752, 0.9977289384, 0.9982635764, 0.9984686146, 0.9985552484, 0.9982009650, 0.9986086122, 0.9986779893, # 0 < eta < 1.5, 0.85 < R9 < 0.90
                    0.8867570653, 0.9956292231, 0.9969618752, 0.9977289384, 0.9982635764, 0.9984686146, 0.9985552484, 0.9982009650, 0.9986086122, 0.9986779893, # 0 < eta < 1.5, R9 > 0.90
                    0.6055913491, 0.7386647433, 0.7738143782, 0.8099180733, 0.8301717071, 0.8554554116, 0.8668998415, 0.8818480940, 0.8963343774, 0.9098144012, # 1.5 < eta < 3, R9 < 0.56
                    0.6055913491, 0.7386647433, 0.7738143782, 0.8099180733, 0.8301717071, 0.8554554116, 0.8668998415, 0.8818480940, 0.8963343774, 0.9098144012, # 1.5 < eta < 3, 0.56 < R9 < 0.85
                    0.8239234069, 0.9514067857, 0.9584272192, 0.9600462609, 0.9614424301, 0.9565460520, 0.9591038591, 0.9631197598, 0.9690330879, 0.9721629371, # 1.5 < eta < 3, 0.85 < R9 < 0.90
                    0.8571646201, 0.9893470803, 0.9950395305, 0.9964515164, 0.9972505264, 0.9978957674, 0.9977917398, 0.9982383170, 0.9984716748, 0.9985989421, # 1.5 < eta < 3, R9 > 0.90
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, R9 < 0.56
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, 0.56 < R9 < 0.85
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, 0.85 < R9 < 0.90
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, R9 > 0.90
                ]
            }
            flow_ = "clamp"
        elif opt.year == "2018":
            # trigger SF and uncertainty for UL2018 sublead photon. fromflashgg: https://github.com/higgs-charm/flashgg/blob/17621e1d0f032e38c20294555359ed4682bd3f3b/Systematics/python/flashggDiPhotonSystematics2018_Legacy_cfi.py#L180C1-L261C6
            # a lot of bins (90, 3 in SCeta, 3 in r9 and 10 in Pt)
            content_ = {
                "nominal": [ #  28            31            35            40            45            50            60            70            90             inf    pt
                    0.6700566984, 0.7965842168, 0.8081304828, 0.8330735890, 0.8651728271, 0.8911488537, 0.9011784801, 0.9076775812, 0.9314218547, 0.8688854629, # 0 < eta < 1.5, R9 < 0.56
                    0.8894513966, 0.9884665549, 0.9917398218, 0.9933705148, 0.9942898736, 0.9942864841, 0.9944882129, 0.9957957201, 0.9968300712, 0.9955940296, # 0 < eta < 1.5, 0.56 < R9 < 0.85
                    0.8954487977, 0.9994025520, 0.9984810955, 0.9994189377, 0.9997816787, 0.9998486516, 0.9998481269, 0.9999123320, 0.9998477863, 0.9998191870, # 0 < eta < 1.5, 0.85 < R9 < 0.90
                    0.8954487977, 0.9994025520, 0.9984810955, 0.9994189377, 0.9997816787, 0.9998486516, 0.9998481269, 0.9999123320, 0.9998477863, 0.9998191870, # 0 < eta < 1.5, R9 > 0.90
                    0.6077667927, 0.7029913014, 0.7348248590, 0.7751918056, 0.8177996767, 0.8498648292, 0.8663397128, 0.8949914647, 0.9172053758, 0.9422693522, # 1.5 < eta < 3, R9 < 0.56
                    0.6077667927, 0.7029913014, 0.7348248590, 0.7751918056, 0.8177996767, 0.8498648292, 0.8663397128, 0.8949914647, 0.9172053758, 0.9422693522, # 1.5 < eta < 3, 0.56 < R9 < 0.85
                    0.9353581676, 0.9899824821, 0.9871141083, 0.9904200761, 0.9918951123, 0.9917714091, 0.9928356527, 0.9938830616, 0.9943289581, 0.9953508596, # 1.5 < eta < 3, 0.85 < R9 < 0.90
                    0.9363550975, 0.9971266544, 0.9982863440, 0.9990711368, 0.9994109362, 0.9995441995, 0.9995876301, 0.9994394094, 0.9997564478, 0.9997131724, # 1.5 < eta < 3, R9 > 0.90
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, R9 < 0.56
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, 0.56 < R9 < 0.85
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, 0.85 < R9 < 0.90
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000  # eta > 3, R9 > 0.90
                    ],
                "up": [ #       28            31            35            40            45            50            60            70            90             inf    pt
                    0.6734158227, 0.7994929832, 0.8102793402, 0.8345336235, 0.8669689631, 0.8944236255, 0.9065509655, 0.9179074207, 0.9431302205, 0.8961095526, # 0 < eta < 1.5, R9 < 0.56
                    0.8930603665, 0.9899673505, 0.9929081426, 0.9943718040, 0.9953035523, 0.9952891707, 0.9956401148, 0.9968574344, 0.9981631524, 0.9965954479, # 0 < eta < 1.5, 0.56 < R9 < 0.85
                    0.8971737330, 1.0013244562, 0.9995405609, 1.0004232612, 1.0007818680, 1.0008486720, 1.0008491591, 1.0009125943, 1.0008480545, 1.0008191914, # 0 < eta < 1.5, 0.85 < R9 < 0.90
                    0.8971737330, 1.0013244562, 0.9995405609, 1.0004232612, 1.0007818680, 1.0008486720, 1.0008491591, 1.0009125943, 1.0008480545, 1.0008191914, # 0 < eta < 1.5, R9 > 0.90
                    0.6123790623, 0.7072961515, 0.7373426886, 0.7774541185, 0.8193940112, 0.8541887278, 0.8717266023, 0.9030029336, 0.9245046028, 0.9488556616, # 1.5 < eta < 3, R9 < 0.56
                    0.6123790623, 0.7072961515, 0.7373426886, 0.7774541185, 0.8193940112, 0.8541887278, 0.8717266023, 0.9030029336, 0.9245046028, 0.9488556616, # 1.5 < eta < 3, 0.56 < R9 < 0.85
                    0.9371614988, 0.9917862179, 0.9884707338, 0.9915397799, 0.9928963781, 0.9927714091, 0.9945757127, 0.9953015132, 0.9955621544, 0.9965539708, # 1.5 < eta < 3, 0.85 < R9 < 0.90
                    0.9375424957, 0.9982362405, 0.9992864290, 1.0000712007, 1.0004110567, 1.0005458278, 1.0005876458, 1.0004873262, 1.0007565322, 1.000713567, # 1.5 < eta < 3, R9 > 0.90
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, R9 < 0.56
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, 0.56 < R9 < 0.85
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, 0.85 < R9 < 0.90
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000  # eta > 3, R9 > 0.90
                ],
                "down": [ #     28            31            35            40            45            50            60            70            90             inf    pt
                    0.6666975741, 0.7936754504, 0.8059816254, 0.8316135545, 0.8633766911, 0.8878740819, 0.8958059947, 0.8974477417, 0.9197134889, 0.8416613732, # 0 < eta < 1.5, R9 < 0.56
                    0.8858424267, 0.9869657593, 0.9905715010, 0.9923692256, 0.9932761949, 0.9932837975, 0.9933363110, 0.9947340058, 0.9954969900, 0.9945926113, # 0 < eta < 1.5, 0.56 < R9 < 0.85
                    0.8937238624, 0.9974806478, 0.9974216301, 0.9984146142, 0.9987814894, 0.9988486312, 0.9988470947, 0.9989120697, 0.9988475181, 0.9988191826, # 0 < eta < 1.5, 0.85 < R9 < 0.90
                    0.8937238624, 0.9974806478, 0.9974216301, 0.9984146142, 0.9987814894, 0.9988486312, 0.9988470947, 0.9989120697, 0.9988475181, 0.9988191826, # 0 < eta < 1.5, R9 > 0.90
                    0.6031545231, 0.6986864513, 0.7323070294, 0.7729294927, 0.8162053422, 0.8455409306, 0.8609528233, 0.8869799958, 0.9099061488, 0.9356830428, # 1.5 < eta < 3, R9 < 0.56
                    0.6031545231, 0.6986864513, 0.7323070294, 0.7729294927, 0.8162053422, 0.8455409306, 0.8609528233, 0.8869799958, 0.9099061488, 0.9356830428, # 1.5 < eta < 3, 0.56 < R9 < 0.85
                    0.9335548364, 0.9881787463, 0.9857574828, 0.9893003723, 0.9908938465, 0.9907714091, 0.9910955927, 0.9924646100, 0.9930957618, 0.9941477484, # 1.5 < eta < 3, 0.85 < R9 < 0.90
                    0.9351676993, 0.9960170683, 0.9972862590, 0.9980710729, 0.9984108157, 0.9985425712, 0.9985876144, 0.9983914926, 0.9987563634, 0.9987127778, # 1.5 < eta < 3, R9 > 0.90
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, R9 < 0.56
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, 0.56 < R9 < 0.85
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, 0.85 < R9 < 0.90
                    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, # eta > 3, R9 > 0.90
                ]
            }
            flow_ = "clamp"

        TRIG = cs.Correction(
            name="TriggerSF",
            description="Sublead Photon Trigger Scale Factor",
            generic_formulas=None,
            version=1,
            inputs=[
                cs.Variable(
                    name="systematic", type="string", description="Systematic variation"
                ),
                cs.Variable(name="SCEta", type="real", description="Photon Super Cluster eta absolute value"),
                cs.Variable(
                    name="r9",
                    type="real",
                    description="Photon full 5x5 R9, ratio E3x3/ERAW, where E3x3 is the energy sum of the 3 by 3 crystals surrounding the supercluster seed crystal and ERAW is the raw energy sum of the supercluster",
                ),
                cs.Variable(
                    name="pt",
                    type="real",
                    description="Photon pt",
                ),
            ],
            output=cs.Variable(
                name="Wcorr",
                type="real",
                description="Multiplicative correction to event weight (per-photon)",
            ),
            data=cs.Category(
                nodetype="category",
                input="systematic",
                content=[
                    cs.CategoryItem(
                        key="up",
                        value=multibinning(inputs_, edges_, content_["up"], flow_)
                    ),
                    cs.CategoryItem(
                        key="down",
                        value=multibinning(inputs_, edges_, content_["down"], flow_)
                    ),
                ],
                default=multibinning(inputs_, edges_, content_["nominal"], flow_),
            ),
        )

        rich.print(TRIG)

        etas = [
            0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8,
            0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8,
            0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8,
            0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8,
            1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7,
            1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7,
            1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7,
            1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7,
            10., 10., 10., 10., 10., 10., 10., 10., 10., 10.,
            10., 10., 10., 10., 10., 10., 10., 10., 10., 10.,
            10., 10., 10., 10., 10., 10., 10., 10., 10., 10.,
            10., 10., 10., 10., 10., 10., 10., 10., 10., 10.
            ]
        rs = [
            0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4,
            0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6,
            0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88,
            0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99,
            0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4,
            0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6,
            0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88,
            0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99,
            0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4,
            0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6,
            0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88,
            0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99
            ]
        pts = [
            15., 30., 34., 38., 42., 47., 55., 65., 75., 200.,
            15., 30., 34., 38., 42., 47., 55., 65., 75., 200.,
            15., 30., 34., 38., 42., 47., 55., 65., 75., 200.,
            15., 30., 34., 38., 42., 47., 55., 65., 75., 200.,
            15., 30., 34., 38., 42., 47., 55., 65., 75., 200.,
            15., 30., 34., 38., 42., 47., 55., 65., 75., 200.,
            15., 30., 34., 38., 42., 47., 55., 65., 75., 200.,
            15., 30., 34., 38., 42., 47., 55., 65., 75., 200.,
            15., 30., 34., 38., 42., 47., 55., 65., 75., 200.,
            15., 30., 34., 38., 42., 47., 55., 65., 75., 200.,
            15., 30., 34., 38., 42., 47., 55., 65., 75., 200.,
            15., 30., 34., 38., 42., 47., 55., 65., 75., 200.,
            ]

        print(TRIG.to_evaluator())
        print("-" * 120)
        print("nominal --->","eta:",etas,"r9:",rs,"pt:",pts,"result:",TRIG.to_evaluator().evaluate("nominal", etas, rs, pts))
        print("-" * 120)
        print("up      --->","eta:",etas,"r9:",rs,"pt:",pts,"result:",TRIG.to_evaluator().evaluate("up", etas, rs, pts))
        print("-" * 120)
        print("down    --->","eta:",etas,"r9:",rs,"pt:",pts,"result:",TRIG.to_evaluator().evaluate("down", etas, rs, pts))
        print("-" * 120)
        print()

        cset = cs.CorrectionSet(
            schema_version=2,
            description="Trigger SF",
            compound_corrections=None,
            corrections=[
                TRIG,
            ],
        )

        if opt.read == "":
            with open(f"TriggerSF_sublead_{opt.year}.json", "w") as fout:
                fout.write(cset.model_dump_json(exclude_unset=True))

            import gzip

            with gzip.open(f"TriggerSF_sublead_{opt.year}.json.gz", "wt") as fout:
                fout.write(cset.model_dump_json(exclude_unset=True))

        if opt.read != "":
            print("Reading back...")
            file = opt.read
            ceval = correctionlib.CorrectionSet.from_file(file)
            for corr in ceval.values():
                print(f"Correction {corr.name} has {len(corr.inputs)} inputs")
                for ix in corr.inputs:
                    print(f"   Input {ix.name} ({ix.type}): {ix.description}")

            print("-" * 120)
            print("nominal --->","eta:",etas,"r9:",rs,"result:",ceval["TriggerSF"].evaluate("nominal", etas, rs, pts))
            print("-" * 120)
            print("up      --->","eta:",etas,"r9:",rs,"result:",ceval["TriggerSF"].evaluate("up", etas, rs, pts))
            print("-" * 120)
            print("down    --->","eta:",etas,"r9:",rs,"result:",ceval["TriggerSF"].evaluate("down", etas, rs, pts))
            print("-" * 120)


if __name__ == "__main__":
    main()
