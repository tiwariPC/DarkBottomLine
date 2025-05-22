import subprocess
import sys
import json
from pathlib import Path
import os
from copy import deepcopy
import logging

logger = logging.getLogger(__name__)


class LXPlusVanillaSubmitter:
    """
    A class for submitting jobs on the CERN's LXPlus cluster using HTCondor, one job per file in a sample list of an analysis.
    All jobs for a given samples are submitted to the same cluster.
    The constructor creates a directory .bottom_line_vanilla_lxplus if it does not exist and another one called .bottom_line_vanilla_lxplus/<analysis_name>.
    The data and time (YMD_HMS) is appended to the end of <analysis_name> to avoid overwriting previous submissions.
    Inside this directory two subdirectories called <inputs> and <jobs> will be created.
    In the former the split JSON files will be stored, in the latter the HTCondor related job files will be stored.
    To send each job to a separate cluster then set cluster_per_sample=False in the class constructor.

    Parameters:
        :param analysis_name: Name of the analysis.
        :type analysis_name: str
        :param analysis_dict: Dictionary containing the parameters of the analysis.
        :type analysis_dict: dict
        :param original_analysis_path: Path of the original analysis to be replaced with the new ones.
        :type original_analysis_path: str
        :param sample_dict: Dictionary containing the samples and their respective files.
        :type sample_dict: dict
        :param args_string: String containing the command line arguments.
        :type args_string: str
        :param queue: HTCondor queue to submit the job to. Defaults to "longlunch".
        :type queue: str, optional
        :param memory: Memory request for the job. Defaults to "10GB".
        :type memory: str, optional
        :param cluster_per_sample: Option to submit each job from a given sample to the same cluster.
        :type cluster_per_sample: bool, optional
    """

    def __init__(
        self,
        analysis_name,
        analysis_dict,
        original_analysis_path,
        sample_dict,
        args_string,
        queue="longlunch",
        memory="10GB",
        cluster_per_sample=True,
    ):
        self.datetime_extension = subprocess.getoutput("date +%Y%m%d_%H%M%S")
        self.analysis_name = f"{analysis_name}_{self.datetime_extension}"
        self.analysis_dict = analysis_dict
        self.sample_dict = sample_dict
        self.args_string = args_string
        self.queue = queue
        self.memory = memory
        self.cluster_per_sample = cluster_per_sample
        self.current_dir = os.getcwd()
        self.base_dir = os.path.join(self.current_dir, ".bottom_line_vanilla_lxplus")
        self.analysis_dir = os.path.join(self.base_dir, self.analysis_name)

        self.input_dir = os.path.join(self.analysis_dir, "inputs")
        Path(self.input_dir).mkdir(parents=True, exist_ok=True)

        # split analysis_dict and sample_dict in different JSON files
        self.json_analysis_files = {}
        self.json_sample_files = {}
        for sample in sample_dict:
            self.json_analysis_files[sample] = []
            self.json_sample_files[sample] = []
            for fl in sample_dict[sample]:
                sample_to_dump = {}
                sample_to_dump[sample] = [fl]
                root_file_name = fl.split("/")[-1].split(".")[0]
                sample_file_name = os.path.join(
                    self.input_dir, f"{sample}-{root_file_name}.json"
                )
                with open(sample_file_name, "w") as jf:
                    json.dump(sample_to_dump, jf, indent=4)
                self.json_sample_files[sample].append(sample_file_name)
                an_file_name = os.path.join(
                    self.input_dir, f"AN-{sample}-{root_file_name}.json"
                )
                an_to_dump = deepcopy(self.analysis_dict)
                an_to_dump["samplejson"] = sample_file_name
                with open(an_file_name, "w") as jf:
                    json.dump(an_to_dump, jf, indent=4)
                self.json_analysis_files[sample].append(an_file_name)

        # create job submission directory
        self.jobs_dir = os.path.join(self.analysis_dir, "jobs")
        Path(self.jobs_dir).mkdir(parents=True, exist_ok=True)
        self.job_files = []

        # write job files if running on single cluster
        if self.cluster_per_sample:
            # Get proxy information (required in executable script for this method of running)
            try:
                stat, out = subprocess.getstatusoutput("voms-proxy-info -e --valid 5:00")
            except:
                logger.exception(
                    "voms proxy not found or validity less that 5 hours:\n%s",
                    out
                )
                raise
            try:
                stat, out = subprocess.getstatusoutput("voms-proxy-info -p")
                out = out.strip().split("\n")[-1]
            except:
                logger.exception(
                    "Unable to voms proxy:\n%s",
                    out
                )
                raise
            proxy = out

            for sample in sample_dict:
                base_name = f"AN-{sample}"
                jobs_dir = os.path.realpath(self.jobs_dir)
                # replacing /eos/home- with /eos/user/ to prevent problems with output_destination
                jobs_dir = jobs_dir.replace("/eos/home-", "/eos/user/")
                job_file_executable = os.path.join(jobs_dir, f"{base_name}.sh")
                job_file_submit = os.path.join(jobs_dir, f"{base_name}.sub")
                job_file_out = f"{base_name}.$(ClusterId).$(ProcId).out"
                job_file_err = f"{base_name}.$(ClusterId).$(ProcId).err"
                job_file_log = os.path.join(jobs_dir, f"{base_name}.$(ClusterId).log")
                if ("/eos/user" in jobs_dir):
                    job_file_dir = "root://eosuser.cern.ch/" + jobs_dir
                elif ("/eos/cms" in jobs_dir):
                    job_file_dir = "root://eoscms.cern.ch/" + jobs_dir
                else:
                    job_file_dir = jobs_dir
                n_jobs = len(self.json_analysis_files[sample])

                with open(job_file_executable, "w") as executable_file:
                    executable_file.write("#!/bin/sh\n")
                    executable_file.write(f"export X509_USER_PROXY={proxy}\n")
                    for i, json_file in enumerate(self.json_analysis_files[sample]):
                        arguments = self.args_string.replace(
                            original_analysis_path, json_file
                        ).replace(" vanilla_lxplus", " iterative")
                        executable_file.write(f"if [ $1 -eq {i} ]; then\n")
                        executable_file.write(f"    /usr/bin/env {sys.prefix}/bin/run_analysis.py {arguments} || exit 107\n")
                        executable_file.write("exit 0\n")
                        executable_file.write("fi\n")
                os.system(f"chmod 775 {job_file_executable}")
                with open(job_file_submit, "w") as submit_file:
                    submit_file.write(f"executable = {job_file_executable}\n")
                    submit_file.write("arguments = $(ProcId)\n")
                    submit_file.write(f"output = {job_file_out}\n")
                    submit_file.write(f"error = {job_file_err}\n")
                    submit_file.write(f"log = {job_file_log}\n")
                    submit_file.write(f"output_destination = {job_file_dir}\n")
                    submit_file.write(f"request_memory = {self.memory}\n")
                    submit_file.write("getenv = True\n")
                    submit_file.write(f'+JobFlavour = "{self.queue}"\n')
                    submit_file.write('on_exit_remove = (ExitBySignal == False) && (ExitCode == 0)\n')
                    submit_file.write('on_exit_hold = (ExitBySignal == True) && (ExitCode != 0)\n')
                    submit_file.write('periodic_release = (NumJobStarts < 3) && ((CurrentTime - EnteredCurrentStatus) > 600)\n')
                    submit_file.write('max_retries = 3\n')
                    submit_file.write('requirements = Machine =!= LastRemoteHost\n')
                    submit_file.write(f"queue {n_jobs}\n")
                self.job_files.append(job_file_submit)

        # write job files for separate clusters
        else:
            for sample in sample_dict:
                for json_file in self.json_analysis_files[sample]:
                    base_name = json_file.split("/")[-1].split(".")[0]
                    jobs_dir = os.path.realpath(self.jobs_dir)
                    # replacing /eos/home- with /eos/user/ to prevent problems with output_destination
                    jobs_dir = jobs_dir.replace("/eos/home-", "/eos/user/")
                    job_file_name = os.path.join(jobs_dir, f"{base_name}.sub")
                    job_file_out = f"{base_name}.out"
                    job_file_err = f"{base_name}.err"
                    job_file_log = os.path.join(jobs_dir, f"{base_name}.log")
                    if ("/eos/user" in jobs_dir):
                        job_file_dir = "root://eosuser.cern.ch/" + jobs_dir
                    elif ("/eos/cms" in jobs_dir):
                        job_file_dir = "root://eoscms.cern.ch/" + jobs_dir
                    else:
                        job_file_dir = jobs_dir
                    with open(job_file_name, "w") as submit_file:
                        arguments = self.args_string.replace(
                            original_analysis_path, json_file
                        ).replace(" vanilla_lxplus", " iterative")
                        submit_file.write("executable = /usr/bin/env\n")
                        submit_file.write(
                            f"arguments = {sys.prefix}/bin/run_analysis.py {arguments} || exit 107\n"
                        )
                        submit_file.write(f"output = {job_file_out}\n")
                        submit_file.write(f"error = {job_file_err}\n")
                        submit_file.write(f"log = {job_file_log}\n")
                        submit_file.write(f"output_destination = {job_file_dir}\n")
                        submit_file.write(f"request_memory = {self.memory}\n")
                        submit_file.write("getenv = True\n")
                        submit_file.write(f'+JobFlavour = "{self.queue}"\n')
                        submit_file.write('on_exit_remove = (ExitBySignal == False) && (ExitCode == 0)\n')
                        submit_file.write('on_exit_hold = (ExitBySignal == True) && (ExitCode != 0)\n')
                        submit_file.write('periodic_release = (NumJobStarts < 3) && ((CurrentTime - EnteredCurrentStatus) > 600)\n')
                        submit_file.write('max_retries = 3\n')
                        submit_file.write('requirements = Machine =!= LastRemoteHost\n')
                        submit_file.write("queue 1\n")
                    self.job_files.append(job_file_name)

    def submit(self):
        """
        A method to submit all the jobs in the jobs_dir to the cluster
        """
        for jf in self.job_files:
            if self.current_dir.startswith("/eos"):
                # see https://batchdocs.web.cern.ch/troubleshooting/eos.html#no-eos-submission-allowed
                subprocess.run(["condor_submit", "-spool", jf])
            else:
                subprocess.run(["condor_submit", jf])
        return None
