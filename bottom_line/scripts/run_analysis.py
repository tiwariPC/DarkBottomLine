#!/usr/bin/env python
import json
from importlib import resources
import os
import sys
import time
from typing import List

import numpy as np
import uproot
from coffea import processor
from coffea.util import save
from coffea.nanoevents import NanoAODSchema, BaseSchema
from dask.distributed import Client, Worker, WorkerPlugin

from bottom_line.utils.runner_utils import get_main_parser, get_proxy
from bottom_line.workflows import workflows
from bottom_line.workflows import taggers as all_taggers
from bottom_line.metaconditions import metaconditions as all_metaconditions
from bottom_line.utils.logger_utils import setup_logger
from bottom_line.systematics import check_corr_syst_combinations


def validate(file):
    try:
        fin = uproot.open(file)
        return fin["Events"].num_entries
    except Exception:
        print(f"Corrupted file: {file}")
        return


def check_port(port):
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("0.0.0.0", port))
        available = True
    except Exception:
        available = False
    sock.close()
    return available


class DependencyInstaller(WorkerPlugin):
    def __init__(self, dependencies: List[str]):
        self._depencendies = " ".join(f"'{dep}'" for dep in dependencies)

    def setup(self, worker: Worker):
        os.system(f"pip install {self._depencendies}")


dependency_installer = DependencyInstaller(
    [
        "git+https://github.com/lgray/hgg-coffea.git@master",  # if develop a tagger, needs to be in this repo or in a fork
    ]
)  # pointing to this repo for now to install dependencies. Doesn't work like this for now on LPC and lxplus


def _worker_upload(dask_worker, data, fname):
    dask_worker.loop.add_callback(
        callback=dask_worker.upload_file,
        comm=None,
        filename=fname,
        data=data,
        load=True,
    )


def run_executor(_args, executor, sample_dict, processor_instance):
    schema_list = {
        "nano": NanoAODSchema,
        "base": BaseSchema,
    }
    run = processor.Runner(
        executor=executor,
        schema=schema_list[_args.schema],
        chunksize=_args.chunk,
        maxchunks=_args.max,
        format=_args.format,
        skipbadfiles=_args.skipbadfiles,
        xrootdtimeout=_args.timeout,
    )

    output = run(
        sample_dict,
        treename="Events",
        processor_instance=processor_instance,
    )

    # save event number to json
    if _args.save is not None:
        save(output, _args.save)


def main():
    parser = get_main_parser()
    args = parser.parse_args()

    # Setup logger
    if args.debug:
        log_level = "DEBUG"
    else:
        log_level = "INFO"
    logger = setup_logger(level=log_level)

    if args.executor != "dask/local":
        if args.workers is None:
            args.workers = 12
        if args.scaleout is None:
            args.scaleout = 6
        if args.max_scaleout is None:
            args.max_scaleout = 250
        if args.memory is None:
            args.memory = "10GB"
    if args.executor == "dask/local":
        if args.memory is None:
            args.memory = "auto"

    logger.info("Start production")

    # Here we assume that all the keys are there, otherwise an exception will be raised
    analysis_name = args.json_analysis_file.split("/")[-1].split(".")[0]
    with open(args.json_analysis_file) as f:
        analysis = json.load(f)
    workflow = analysis["workflow"]
    taggers = analysis.get("taggers", None)
    metaconditions = analysis["metaconditions"]
    samplejson = analysis["samplejson"]
    systematics = analysis["systematics"]
    corrections = analysis["corrections"]
    year = analysis["year"]
    bTagEffFileName = analysis.get("bTagEffFileName", None)
    logger.info(f"Corrections: {corrections}")
    logger.info(f"Systematics: {systematics}")
    logger.info(f"Year: {year}")
    check_corr_syst_combinations(corrections, systematics, logger)

    if args.output == parser.get_default("output"):
        args.output = (
            f'hists_{workflow}_{(samplejson).replace("/","_").rstrip(".json")}.coffea'
        )

    # load dataset
    xrootd_pfx = "root://"
    xrd_pfx_len = len(xrootd_pfx)
    with open(samplejson) as f:
        sample_dict = json.load(f)

    # log if a dataset is not in the sample_dict
    bad_datasets = [key for key in year if key not in sample_dict]
    if bad_datasets:
        logger.info(
            f"The following datasets are present in the analysis but not in the sample json and will be ignored: {bad_datasets}"
        )

    # drop datasets from sample_dict if they are not in the analysis
    keys_to_delete = [key for key in sample_dict if key not in year]
    for key in keys_to_delete:
        del sample_dict[key]

    for key in sample_dict.keys():
        sample_dict[key] = sample_dict[key][: args.limit]
    if args.executor == "dask/casa":
        for key in sample_dict.keys():
            sample_dict[key] = [
                path.replace(
                    path[xrd_pfx_len : xrd_pfx_len + path[xrd_pfx_len:].find("/")],
                    "xcache",
                )
                for path in sample_dict[key]
            ]

    # For debugging
    if args.only is not None:
        if args.only in sample_dict.keys():  # is dataset
            sample_dict = dict([(args.only, sample_dict[args.only])])
        if "*" in args.only:  # wildcard for datasets
            _new_dict = {}
            logger.debug("Will only process the following datasets:")
            for k, v in sample_dict.items():
                if k.lstrip("/").startswith(args.only.rstrip("*")):
                    logger.debug(f"    {k}")
                    _new_dict[k] = v
            sample_dict = _new_dict
        else:  # is file
            for key in sample_dict.keys():
                if args.only in sample_dict[key]:
                    sample_dict = dict([(key, [args.only])])

    # Scan if files can be opened
    if args.validate:
        logger.info("Performing sanity check on files")
        start = time.time()
        from p_tqdm import p_map

        all_invalid = []
        for sample in sample_dict.keys():
            _rmap = p_map(
                validate,
                sample_dict[sample],
                num_cpus=args.workers,
                desc=f"Validating {sample[:20]}...",
            )
            _results = list(_rmap)
            counts = np.sum([r for r in _results if r is not None])
            all_invalid += [r for r in _results if type(r) == str]
            logger.debug(f"Events: {np.sum(counts)}")
        logger.debug("Bad files:")
        for fi in all_invalid:
            logger.debug(f"  {fi}")
        end = time.time()
        logger.debug(f'TIME: {time.strftime("%H:%M:%S", time.gmtime(end - start))}')
        if input("Remove bad files? (y/n)") == "y":
            logger.debug("Removing:")
            for fi in all_invalid:
                logger.debug(f"Removing: {fi}")
                os.system(f"rm {fi}")
        sys.exit(0)

    # load workflow
    if workflow in workflows:
        wf_taggers = None
        if taggers is not None:
            for tagger in taggers:
                if tagger not in all_taggers.keys():
                    raise NotImplementedError(f"Tagger '{tagger}' not implemented")
            wf_taggers = [all_taggers[tagger]() for tagger in taggers]
        with resources.open_text(
            "bottom_line.metaconditions", all_metaconditions[metaconditions]
        ) as f:
            processor_instance = workflows[workflow](
                metaconditions=json.load(f),
                systematics=systematics,
                corrections=corrections,
                nano_version=args.nano_version,
                bTagEffFileName=bTagEffFileName,
                apply_trigger=args.use_trigger,
                output_location=args.dump,
                analysis=args.analysis,
                trigger_group=args.triggerGroup,
                taggers=wf_taggers,
                applyCQR=args.applyCQR,
                skipJetVetoMap=args.skipJetVetoMap,
                year=year,
                fiducialCuts=args.fiducialCuts,
                doDeco=args.doDeco,
                Smear_sigma_m=args.Smear_sigma_m,
                doFlow_corrections=args.doFlow_corrections,
                output_format=args.output_format,
            )  # additional args can go here to configure a processor
    else:
        if workflow == "dystudies":
            logger.error("You selected the workflow 'dystudies', but it has been renamed to base. Exiting.")
        raise NotImplementedError(f"Workflow '{workflow}' not implemented")

    if args.executor not in [
        "futures",
        "iterative",
        "dask/local",
        "dask/lpc",
        "dask/casa",
    ]:
        """
        dask/parsl needs to export x509 to read over xrootd
        dask/lpc uses custom jobqueue provider that handles x509
        """
        if args.voms is not None:
            _x509_path = args.voms
        else:
            _x509_path = get_proxy()

        env_extra = [
            "export XRD_RUNFORKHANDLER=1",
            f"export X509_USER_PROXY={_x509_path}",
            f"export PYTHONPATH=$PYTHONPATH:{os.getcwd()}",
        ]
        try:
            env_extra.append(f'export X509_CERT_DIR={os.environ["X509_CERT_DIR"]}')
        except KeyError:
            pass
        condor_extra = [
            f'source {os.environ["HOME"]}/.bashrc',
        ]

    # Execute
    if args.executor in ["futures", "iterative"]:
        if args.executor == "iterative":
            executor = processor.IterativeExecutor()
        else:
            executor = processor.FuturesExecutor(workers=args.workers)
        run_executor(args, executor, sample_dict, processor_instance)
    elif "parsl" in args.executor:
        import parsl
        from parsl.addresses import address_by_hostname, address_by_query
        from parsl.channels import LocalChannel
        from parsl.config import Config
        from parsl.executors import HighThroughputExecutor
        from parsl.launchers import SrunLauncher
        from parsl.providers import CondorProvider, SlurmProvider

        if "slurm" in args.executor:
            htex_config = Config(
                executors=[
                    HighThroughputExecutor(
                        label="coffea_parsl_slurm",
                        address=address_by_hostname(),
                        prefetch_capacity=0,
                        provider=SlurmProvider(
                            channel=LocalChannel(script_dir="logs_parsl"),
                            launcher=SrunLauncher(),
                            max_blocks=(args.scaleout) + 10,
                            init_blocks=args.scaleout,
                            partition="all",
                            worker_init="\n".join(env_extra),
                            walltime="00:120:00",
                        ),
                    )
                ],
                retries=20,
            )
        elif "condor" in args.executor:
            htex_config = Config(
                executors=[
                    HighThroughputExecutor(
                        label="coffea_parsl_condor",
                        address=address_by_query(),
                        # max_workers=1,
                        provider=CondorProvider(
                            nodes_per_block=1,
                            init_blocks=1,
                            max_blocks=1,
                            worker_init="\n".join(env_extra + condor_extra),
                            walltime="00:20:00",
                        ),
                    )
                ]
            )
        else:
            raise NotImplementedError(f"Executor '{args.executor}' not implemented")

        dfk = parsl.load(htex_config)

        executor = processor.ParslExecutor()
        run_executor(args, executor, sample_dict, processor_instance)
    elif "dask" in args.executor:
        from dask.distributed import LocalCluster, performance_report
        from dask_jobqueue import HTCondorCluster, SLURMCluster

        if "local" in args.executor:
            cluster = LocalCluster(
                n_workers=args.scaleout,
                threads_per_worker=args.workers,
                memory_limit=args.memory,
            )

        elif "lpc" in args.executor:
            env_extra = [
                f"export PYTHONPATH=$PYTHONPATH:{os.getcwd()}",
            ]
            from lpcjobqueue import LPCCondorCluster

            cluster = LPCCondorCluster(
                transfer_input_files="/srv/workflows/",
                ship_env=True,
                job_script_prologue=env_extra,
            )
        elif "lxplus" in args.executor:
            from dask_lxplus import CernCluster

            n_port = 8786
            if not check_port(8786):
                raise RuntimeError(
                    "Port '8786' is now occupied on this node. Try another one."
                )
            import socket

            cluster = CernCluster(
                cores=1,
                memory=args.memory,
                disk=args.disk,
                image_type="singularity",
                worker_image="/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/higgsdna:lxplus-el9-latest",
                death_timeout="3600",
                scheduler_options={"port": n_port, "host": socket.gethostname()},
                job_extra={
                    "log": "dask_job_output.log",
                    "output": "dask_job_output.out",
                    "error": "dask_job_output.err",
                    "should_transfer_files": "Yes",
                    "when_to_transfer_output": "ON_EXIT",
                    "+JobFlavour": '"workday"'
                    if args.queue is None
                    else f'"{args.queue}"',
                },
                job_script_prologue=[
                    "export XRD_RUNFORKHANDLER=1",
                    f"export X509_USER_PROXY={_x509_path}",
                    "export PYTHONPATH=$PYTHONPATH:$_CONDOR_SCRATCH_DIR",
                ],
            )
        elif "slurm" in args.executor:
            cluster = SLURMCluster(
                queue=args.queue,
                cores=args.workers,
                processes=args.workers,
                memory=args.memory,
                walltime=args.walltime,
                job_script_prologue=env_extra,
            )
        elif "condor" in args.executor:
            cluster = HTCondorCluster(
                cores=args.workers,
                memory=args.memory,
                disk="4GB",
                job_script_prologue=env_extra,
            )

        if args.executor == "dask/casa":
            client = Client("tls://localhost:8786")
            logger.info("Waiting for at least one worker...")
            # client.wait_for_workers(1)
            client.register_worker_plugin(dependency_installer)
        elif args.executor != "dask/local" and args.executor != "dask/casa":
            cluster.adapt(minimum=args.scaleout, maximum=args.max_scaleout)
            client = Client(cluster)
            logger.info("Waiting for at least one worker...")
            client.wait_for_workers(1)
        else:
            client = Client(cluster)
        with performance_report(filename="dask-report.html"):
            executor = processor.DaskExecutor(client=client, retries=50)
            run_executor(args, executor, sample_dict, processor_instance)
    elif args.executor == "vanilla_lxplus":
        from bottom_line.submission.lxplus import LXPlusVanillaSubmitter

        for cmd_str in env_extra:
            g_var = cmd_str.split()[-1]
            var, var_value = g_var.split("=")
            os.environ[var] = var_value
        args_string = " ".join(sys.argv[1:])
        vanilla_submitter = LXPlusVanillaSubmitter(
            analysis_name,
            analysis,
            args.json_analysis_file,
            sample_dict,
            args_string,
            queue=args.queue if args.queue is not None else "workday",
            memory=args.memory,
        )
        output = vanilla_submitter.submit()


if __name__ == "__main__":
    main()
