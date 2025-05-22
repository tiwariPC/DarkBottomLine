import argparse
import os
import subprocess
import logging

logger = logging.getLogger(__name__)


def get_main_parser():
    parser = argparse.ArgumentParser(
        description="Run Hgg Workflows on NanoAOD using coffea processor files"
    )
    # Analysis inputs
    parser.add_argument(
        "--json-analysis",
        dest="json_analysis_file",
        type=str,
        help="JSON analysis file where workflow, taggers, metaconditions, samples and systematics are defined.\n"
        + "It has to look like this:\n"
        + "{\n"
        + '\t"samplejson": "path to sample JSON",\n'
        + '\t"workflow": one of the implemented workflows, e.g., "base",\n'
        + '\t"metaconditions": one of available metaconditions, e.g., "Era2022_v1",\n'
        + '\t"taggers": one of the implemented taggers, e.g., "HHWWggTagger",\n'
        + '\t"systematics": path to systematics JSON or systematics in JSON style,\n'
        + '\t"corrections": path to corrections JSON or corrections in JSON sytle\n'
        + "}",
        required=True,
    )

    # File handling information
    parser.add_argument(
        "--no-trigger",
        dest="use_trigger",
        default=True,
        action="store_false",
        help="Turn off trigger selection",
    )
    parser.add_argument(
        "-d",
        "--dump",
        default=None,
        help="Path to dump parquet outputs to (default: None)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=r"output.coffea",
        help="Output filename (default: %(default)s)",
    )
    parser.add_argument(
        "--schema",
        default="nano",
        help="input file format schema(default: %(default)s)",
        choices=("nano", "base"),
    )
    parser.add_argument(
        "-f",
        "--format",
        default="root",
        help="input file format (default: %(default)s)",
        choices=("root", "parquet"),
    )
    parser.add_argument(
        "--triggerGroup",
        default=".*DoubleEG.*",
        help="trigger group to be selected",
        choices=(
            ".*DoubleEG.*",
            ".*EGamma.*2018.*",
            ".*EGamma.*",
            ".*SingleEle.*",
            ".*DoubleMuon.*",
        ),
    )
    parser.add_argument(
        "--analysis",
        default="mainAnalysis",
        help="analysis to run",
        choices=("mainAnalysis", "tagAndProbe", "ZmmyAnalysis"),
    )
    parser.add_argument(
        "--save",
        default=None,
        help="If not None, save the coffea output, e.g., --save run_summary.coffea",
    )
    parser.add_argument(
        "--nano-version",
        dest="nano_version",
        type=int,
        default=None,
        required=True,
        help="NanoAOD version used in the analysis.",
    )

    # Scale out
    parser.add_argument(
        "--executor",
        choices=[
            "iterative",
            "futures",
            "parsl/slurm",
            "parsl/condor",
            "dask/local",
            "dask/condor",
            "dask/slurm",
            "dask/lpc",
            "dask/lxplus",
            "dask/casa",  # Use for coffea-casa
            "vanilla_lxplus",
        ],
        default="futures",  # Local executor (named after concurrent futures package)
        help="The type of executor to use (default: %(default)s). Other options can be implemented. "
        "For example see https://parsl.readthedocs.io/en/stable/userguide/configuring.html"
        "- `parsl/slurm` - tested at DESY/Maxwell"
        "- `parsl/condor` - tested at DESY, RWTH"
        "- `dask/local` - tested at local machines"
        "- `dask/slurm` - tested at DESY/Maxwell"
        "- `dask/condor` - tested at DESY, RWTH"
        "- `dask/lpc` - custom lpc/condor setup (due to write access restrictions)"
        "- `dask/lxplus` - custom lxplus/condor setup (due to port restrictions)"
        "- `vanilla_lxplus` - custom plain lxplus submitter",
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        help="Number of workers (cores/threads) to use for multi-worker executors "
        "(e.g. futures or condor) (default: None for dask/local and 12 for others)",
    )
    parser.add_argument(
        "-m",
        "--memory",
        type=str,
        help="Memory to use for each job in distributed executors (default: 'auto' for dask/local and '10GB' for others)",
    )
    parser.add_argument(
        "--walltime",
        type=str,
        default="01:00:00",
        help="Walltime to use for each job in distributed executors (default: %(default)s)",
    )
    parser.add_argument(
        "--disk",
        type=str,
        default="20GB",
        help="Disk space to use for each job in distributed executors (default: %(default)s)",
    )
    parser.add_argument(
        "-s",
        "--scaleout",
        type=int,
        help="Number of nodes to scale out to if using slurm/condor. Total number of "
        "concurrent threads is ``workers x scaleout`` (default: None for dask/local and 6 for others)",
    )
    parser.add_argument(
        "--max-scaleout",
        dest="max_scaleout",
        type=int,
    )
    parser.add_argument(
        "--timeout",
        dest="timeout",
        type=int,
        default=60,
        help="Timeout for file opening with xrootd in seconds. (default: %(default)s)",
    )
    parser.add_argument(
        "-q",
        "--queue",
        type=str,
        default=None,
        help="Queue to submit jobs to if using slurm/condor (default: %(default)s)",
    )
    parser.add_argument(
        "--voms",
        default=None,
        type=str,
        help="Path to voms proxy, accessible to worker nodes. Note that when this is specified "
        "the environment variable X509_CERT_DIR must be set to the certificates directory location",
    )

    # Debugging
    parser.add_argument(
        "--validate",
        action="store_true",
        default=False,
        help="Do not process, just check all files are accessible",
    )
    parser.add_argument("--skipbadfiles", action="store_true", help="Skip bad files.")
    parser.add_argument(
        "--only", type=str, default=None, help="Only process specific dataset or file"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Limit to the first N files of each dataset in sample JSON",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=500000,
        metavar="N",
        help="Number of events per process chunk",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        metavar="N",
        help="Max number of chunks to run in total",
    )
    parser.add_argument(
        "--applyCQR",
        action="store_true",
        default=False,
        help="Apply chained quantile regression (CQR) corrections",
    )
    parser.add_argument(
        "--skipJetVetoMap",
        default=False,
        action="store_true",
        help="Do not apply jet vetomap selections",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Print debug information with a logger",
    )
    parser.add_argument(
        "--fiducialCuts",
        default="classical",
        choices=["classical", "geometric", "classical_noIso", "store_flag", "none"],
        help="Apply fiducial cuts at detector level according to standard CMS approach (classical with 1/3 and 1/4 thresholds for scaled pT of lead and sublead), geometric cuts (proposed in 2106.08329), 'store_flag' (stores flag for both classical and geometric instead of selection according to a specific one) or none at all. Fiducial flags for particle level are handled with utily functions and are unrelated to this argument.",
    )
    parser.add_argument(
        "--doDeco",
        default=False,
        action="store_true",
        help="Perform the mass resolution decorrelation",
    )
    parser.add_argument(
        "--Smear-sigma-m",
        default=False,
        action="store_true",
        help="Perform the mass resolution Smearing",
    )
    parser.add_argument(
        "--doFlow-corrections",
        default=False,
        action="store_true",
        help="Perform the mvaID and energyErr corrections with normalizing flows",
    )
    parser.add_argument(
        "--output-format",
        choices=[
            "root",
            "parquet",
        ],
        default="parquet",
        help="Output format (default: %(default)s).",
    )
    return parser


def get_proxy():
    """
    Use voms-proxy-info to check if a proxy is available.
    If so, copy it to $HOME/.proxy and return the path.
    An exception is raised in the following cases:
    - voms-proxy-info is not installed
    - the proxy is not valid

    :return: Path to proxy
    :rtype: str
    """
    if subprocess.getstatusoutput("voms-proxy-info")[0] != 0:
        raise RuntimeError("voms-proxy-init not found. Please install it.")

    stat, out = subprocess.getstatusoutput("voms-proxy-info -e -p")
    # stat is 0 the proxy is valid
    if stat != 0:
        raise RuntimeError("No valid proxy found. Please create one.")

    _x509_localpath = out
    _x509_path = os.environ["HOME"] + f'/.{_x509_localpath.split("/")[-1]}'
    os.system(f"cp {_x509_localpath} {_x509_path}")

    logger.debug(f"Copied proxy to {_x509_path}")

    return _x509_path
