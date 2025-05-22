import coffea.util
import gzip
import argparse
import correctionlib.schemav2 as cs
import rich


parser = argparse.ArgumentParser(
    description="Example for creating Zpt reweighting json"
)

parser.add_argument(
    "--hist",
    dest="hist",
    help="coffea file of the input histograms",
    default="hist_Zpt.coffea",
)

parser.add_argument(
    "--norm",
    dest="norm",
    help="coffea file of event counting that contain genWeightSum for MC normalization",
    default="count.coffea",
)

parser.add_argument(
    "-o",
    "--output",
    dest="output",
    help="output json file",
    default="my_Zpt_reweighting.json.gz",
)
args = parser.parse_args()


def get_hist(hists, samples=None, norm=None, isdata=False):
    hdict = {}
    for i in hists.keys():
        if samples != None and len(samples) > 0:
            hdict[i] = (hists[i][{"dataset": samples[0]}]).copy() * 0
            for j in samples:
                hdict[i] += (hists[i][{"dataset": j}]).copy()
            if norm != None and isdata == False:
                hdict[i] = (hdict[i] * 6733 * 1000 * 21) / norm
    return hdict


if __name__ == "__main__":
    """_summary_
    Example cmd:
    python ../script/Zpt_rwgt_json.py --norm count.coffea --hist hist_Zpt.coffea -o my_Zpt_reweighting.json.gz

    Testing json example:
    import correctionlib

    ceval = correctionlib.CorrectionSet.from_file("my_Zpt_reweighting.json.gz")
    print(list(ceval.keys()))
    ceval["Zpt_reweight"].evaluate([1,2,30])
    """

    hists = coffea.util.load(args.hist)

    info_dict = {"DY": coffea.util.load(args.norm), "Data": coffea.util.load(args.norm)}

    meta_dict = {
        "DY": {
            "sample": ["DYto2L_EE"],
            "norm": info_dict["DY"]["DYto2L_EE"]["genWeightSum"],
            "isdata": False,
        },
        "Data": {
            "sample": ["MuonF"],
            "norm": 1,
            "isdata": True,
        },
    }

    hdict = {
        "DY": get_hist(
            hists,
            meta_dict["DY"]["sample"],
            meta_dict["DY"]["norm"],
            isdata=meta_dict["DY"]["isdata"],
        ),
        "Data": get_hist(
            hists,
            meta_dict["Data"]["sample"],
            meta_dict["Data"]["norm"],
            isdata=meta_dict["Data"]["isdata"],
        ),
    }

    yield_dy = (hdict["DY"]["h_mmy_pt"].values()).sum()
    yield_data = (hdict["Data"]["h_mmy_pt"].values()).sum()
    norm_to_data = yield_data / yield_dy
    print(yield_dy, yield_data, norm_to_data)

    for i in hdict["DY"]:
        hdict["DY"][i] = (hdict["DY"][i]).copy() * norm_to_data

    Zpt_edge = ((hdict["Data"]["h_mmy_pt"]).axes.edges)[0]
    Zpt_sf = (hdict["Data"]["h_mmy_pt"]).values() / (hdict["DY"]["h_mmy_pt"]).values()

    # generate json
    # please notice that version "pydantic < 2"
    corr = cs.Correction(
        name="Zpt_reweight",
        version=0,
        inputs=[
            cs.Variable(
                name="Zpt", type="real", description="Z boson transverse momentum"
            ),
        ],
        output=cs.Variable(
            name="weight", type="real", description="Multiplicative event weight"
        ),
        data=cs.Binning(
            nodetype="binning",
            input="Zpt",
            edges=list(Zpt_edge),
            content=list(Zpt_sf),
            flow="clamp",
        ),
    )

    cset = cs.CorrectionSet(
        schema_version=2,
        description="Z pt reweighting",
        corrections=[
            corr,
        ],
    )
    rich.print(cset)

    # save json
    with gzip.open(args.output, "wt") as fout:
        fout.write(cset.json(exclude_unset=True))
