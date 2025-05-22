import os
import subprocess
import glob

def submit_jobs(directory, suffix=""):
    if suffix != "":
        sub_files = glob.glob(f"{directory}/*{suffix}.sub")
    else:
        sub_files = glob.glob(f"{directory}/*.sub")
    for current_file in sub_files:
        subprocess.run(["condor_submit", "-spool", current_file])


def MKDIRP(dirpath, verbose=False, dry_run=False):
    if verbose:
        print("\033[1m" + ">" + "\033[0m" + ' os.mkdirs("' + dirpath + '")')
    if dry_run:
        return
    try:
        os.makedirs(dirpath)
    except OSError:
        if not os.path.isdir(dirpath):
            raise
    return


def htcondor_postprocessing(_opt, OUT_PATH, IN_PATH, CONDOR_PATH, SCRIPT_DIR, dirlist_path, var_dict, cat_dict_loc, var_dict_loc, genBinning_str, skip_normalisation_str, merge_data_str, do_syst_str, tbasket_str, job_flavor, memory, decompose_string, logger):

    job_flavor = job_flavor or "microcentury"

    if _opt.root_only:
        with open(dirlist_path) as fl:
            files = fl.readlines()
            if _opt.merge_data and (_opt.type.lower() == "data"):
                source_folder_path = f"{IN_PATH}"
                target_file_path = f"{OUT_PATH}/root/Data/merged.root"
                target_folder_path = f"{OUT_PATH}/root/Data"
                if os.path.exists(target_folder_path):
                    raise Exception(
                        f"The selected target path: {target_folder_path} already exists"
                    )
                MKDIRP(target_folder_path)
            for j, file in enumerate(files):
                if _opt.merge_data and (_opt.type.lower() == "data") and j > 0: continue

                file = file.split("\n")[0]
                # parent_id = 0
                # MC dataset are identified as everythingthat does not contain "data" or "Data" in the name.
                if _opt.logs != "":
                    jobs_dir = CONDOR_PATH
                else: 
                    jobs_dir = OUTPATH
                if ("/eos/home-" in os.path.realpath(jobs_dir)) or ("/eos/user" in os.path.realpath(jobs_dir)):
                    job_file_dir = "root://eosuser.cern.ch/" + os.path.realpath(jobs_dir)
                elif ("/eos/cms" in os.path.realpath(jobs_dir)):
                    job_file_dir = "root://eoscms.cern.ch/" + os.path.realpath(jobs_dir)
                else:
                    job_file_dir = os.path.realpath(jobs_dir)
                    
                if _opt.condor_logs != "":
                    job_file_executable = os.path.join(CONDOR_PATH, f"{file}.sh")
                    job_file_submit = os.path.join(CONDOR_PATH, f"{file}.sub")
                else:
                    job_file_executable = os.path.join(OUT_PATH, f"{file}.sh")
                    job_file_submit = os.path.join(OUT_PATH, f"{file}.sub")
                if not _opt.make_condor_logs:
                    job_file_out = "/dev/null"
                    job_file_err = "/dev/null"
                    job_file_log = "/dev/null"
                elif _opt.condor_logs != "":
                    job_file_out = f"{file}.$(ClusterId).$(ProcId).out"
                    job_file_err = f"{file}.$(ClusterId).$(ProcId).err"
                    job_file_log = os.path.join(CONDOR_PATH, f"{file}.$(ClusterId).log")
                else:
                    job_file_out = f"{file}.$(ClusterId).$(ProcId).out"
                    job_file_err = f"{file}.$(ClusterId).$(ProcId).err"
                    job_file_log = os.path.join(OUT_PATH, f"{file}.$(ClusterId).log")
                
                with open(job_file_executable, "w") as executable_file:
                    executable_file.write("#!/bin/sh\n")
                    if (not _opt.merge_data) or (_opt.type.lower() == "mc"):
                        source_folder_path = f"{IN_PATH}/{file}"
                        target_file_path = f"{OUT_PATH}/root/{file}/merged.root"
                        target_folder_path = f"{OUT_PATH}/root/{file}"
                        if os.path.exists(target_folder_path):
                            raise Exception(
                                f"The selected target path: {target_folder_path} already exists"
                            )

                        MKDIRP(target_folder_path)

                    os.chdir(SCRIPT_DIR)

                    print(f"python3 merge_root.py --source {source_folder_path} --target {target_file_path} --cats {cat_dict_loc} --abs {genBinning_str} --vars {var_dict_loc} --type {_opt.type} --process {decompose_string(file)} {skip_normalisation_str} {merge_data_str} {do_syst_str}")
                    executable_file.write(f"if [ $1 -eq 0 ]; then\n")
                    executable_file.write(f"    python3 merge_root.py --source {source_folder_path} --target {target_file_path} --cats {cat_dict_loc} --abs {genBinning_str} --vars {var_dict_loc} --type {_opt.type} --process {decompose_string(file)} {skip_normalisation_str} {merge_data_str} {do_syst_str} || exit 107\n")
                    executable_file.write("exit 0\n")
                    executable_file.write("fi\n")
                        
                os.system(f"chmod 775 {job_file_executable}")
                with open(job_file_submit, "w") as submit_file:
                    if _opt.condor_logs != "": submit_file.write(f"initialdir = {CONDOR_PATH}\n")
                    submit_file.write(f"executable = {job_file_executable}\n")
                    submit_file.write("arguments = $(ProcId)\n")
                    submit_file.write(f"output = {job_file_out}\n")
                    submit_file.write(f"error = {job_file_err}\n")
                    submit_file.write(f"log = {job_file_log}\n")
                    submit_file.write(f"output_destination = {job_file_dir}\n")
                    submit_file.write("on_exit_hold = (ExitBySignal == True) || (ExitCode != 0)\n")
                    submit_file.write("periodic_release =  (NumJobStarts < 3) && ((CurrentTime - EnteredCurrentStatus) > 600)\n")
                    if _opt.apptainer:
                        submit_file.write("MY.XRDCP_CREATE_DIR     = True\n")
                        submit_file.write("""MY.SingularityImage     = "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/higgsdna-project/higgsdna:latest"\n""")
                        submit_file.write("""MY.SINGULARITY_EXTRA_ARGUMENTS = "-B /afs -B /cvmfs/cms.cern.ch -B /tmp -B /etc/sysconfig/ngbauth-submit -B ${XDG_RUNTIME_DIR} -B /eos --env KRB5CCNAME='FILE:${XDG_RUNTIME_DIR}/krb5cc'"\n""")
                    submit_file.write("max_retries = 3\n")
                    submit_file.write("requirements = Machine =!= LastRemoteHost\n")
                    if memory != None:
                        submit_file.write(f"request_memory = {memory}\n")
                    submit_file.write(f'+JobFlavour = "{job_flavor}"\n')
                    submit_file.write(f"queue\n")
            if _opt.condor_logs != "":
                submit_jobs(CONDOR_PATH)
            else:
                submit_jobs(OUT_PATH)
    if _opt.merge:
        with open(dirlist_path) as fl:
            files = fl.readlines()
            if not _opt.merge_data:
                for file in files:
                    file = file.split("\n")[0]
                    # parent_id = 0
                    # MC dataset are identified as everythingthat does not contain "data" or "Data" in the name.
                    if _opt.logs != "":
                        jobs_dir = CONDOR_PATH
                    else: 
                        jobs_dir = OUTPATH
                    if ("/eos/home-" in os.path.realpath(jobs_dir)) or ("/eos/user" in os.path.realpath(jobs_dir)):
                        job_file_dir = "root://eosuser.cern.ch/" + os.path.realpath(jobs_dir)
                    elif ("/eos/cms" in os.path.realpath(jobs_dir)):
                        job_file_dir = "root://eoscms.cern.ch/" + os.path.realpath(jobs_dir)
                    else:
                        job_file_dir = os.path.realpath(jobs_dir)

                    if "data" not in file.lower():
                        if _opt.logs != "":
                            job_file_executable = os.path.join(CONDOR_PATH, f"{file}.sh")
                            job_file_submit = os.path.join(CONDOR_PATH, f"{file}.sub")
                        else:
                            job_file_executable = os.path.join(OUT_PATH, f"{file}.sh")
                            job_file_submit = os.path.join(OUT_PATH, f"{file}.sub")

                        if not _opt.make_condor_logs:
                            job_file_out = "/dev/null"
                            job_file_err = "/dev/null"
                            job_file_log = "/dev/null"
                        elif _opt.logs != "":
                            job_file_out = f"{file}.$(ClusterId).$(ProcId).out"
                            job_file_err = f"{file}.$(ClusterId).$(ProcId).err"
                            job_file_log = os.path.join(CONDOR_PATH, f"{file}.$(ClusterId).log")
                        else:
                            job_file_out = f"{file}.$(ClusterId).$(ProcId).out"
                            job_file_err = f"{file}.$(ClusterId).$(ProcId).err"
                            job_file_log = os.path.join(OUT_PATH, f"{file}.$(ClusterId).log")

                        with open(job_file_executable, "w") as executable_file:
                            executable_file.write("#!/bin/sh\n")
                            if os.path.exists(f"{OUT_PATH}/merged/{file}"):
                                raise Exception(
                                    f"The selected target path: {OUT_PATH}/merged/{file} already exists"
                                )

                            MKDIRP(f"{OUT_PATH}/merged/{file}")
                            if _opt.syst:
                                # if we have systematic variations in different files we have to split them in different directories
                                # otherwise they will be all merged at once in the same output file
                                i = 0
                                for var in var_dict:
                                    os.chdir(OUT_PATH)

                                    MKDIRP(f"{OUT_PATH}/merged/{file}/{var_dict[var]}")

                                    os.chdir(SCRIPT_DIR)
                                    logger.info(f"merge_parquet.py --source {IN_PATH}/{file}/{var_dict[var]} --target {OUT_PATH}/merged/{file}/{var_dict[var]}/ --cats {cat_dict_loc} {skip_normalisation_str} --abs {genBinning_str}")
                                    executable_file.write(f"if [ $1 -eq {i} ]; then\n")
                                    executable_file.write(f"    merge_parquet.py --source {IN_PATH}/{file}/{var_dict[var]} --target {OUT_PATH}/merged/{file}/{var_dict[var]}/ --cats {cat_dict_loc} {skip_normalisation_str} --abs {genBinning_str} || exit 107\n")
                                    executable_file.write("exit 0\n")
                                    executable_file.write("fi\n")
                                    i += 1

                            else:
                                i = 1
                                os.chdir(SCRIPT_DIR)
                                print(f"merge_parquet.py --source {IN_PATH}/{file}/nominal --target {OUT_PATH}/merged/{file}/ --cats {cat_dict_loc} {skip_normalisation_str} --abs {genBinning_str}")
                                executable_file.write(f"if [ $1 -eq 0 ]; then\n")
                                executable_file.write(f"    merge_parquet.py --source {IN_PATH}/{file}/nominal --target {OUT_PATH}/merged/{file}/ --cats {cat_dict_loc} {skip_normalisation_str} --abs {genBinning_str} || exit 107\n")
                                executable_file.write("exit 0\n")
                                executable_file.write("fi\n")

                        os.system(f"chmod 775 {job_file_executable}")
                        with open(job_file_submit, "w") as submit_file:
                            if _opt.logs != "": submit_file.write(f"initialdir = {CONDOR_PATH}\n")
                            submit_file.write(f"executable = {job_file_executable}\n")
                            submit_file.write("arguments = $(ProcId)\n")
                            submit_file.write(f"output = {job_file_out}\n")
                            submit_file.write(f"error = {job_file_err}\n")
                            submit_file.write(f"log = {job_file_log}\n")
                            submit_file.write(f"output_destination = {job_file_dir}\n")
                            submit_file.write("on_exit_hold = (ExitBySignal == True) || (ExitCode != 0)\n")
                            submit_file.write("periodic_release =  (NumJobStarts < 3) && ((CurrentTime - EnteredCurrentStatus) > 600)\n")
                            # if _opt.max_materialize != "": submit_file.write(f"max_materialize = {_opt.max_materialize}\n")
                            if (_opt.batch == "condor/apptainer"):
                                submit_file.write("MY.XRDCP_CREATE_DIR     = True\n")
                                submit_file.write("""MY.SingularityImage     = "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/higgsdna:lxplus-el9-latest"\n""")
                                submit_file.write("""MY.SINGULARITY_EXTRA_ARGUMENTS = "-B /afs -B /cvmfs/cms.cern.ch -B /tmp -B /etc/sysconfig/ngbauth-submit -B ${XDG_RUNTIME_DIR} -B /eos --env KRB5CCNAME='FILE:${XDG_RUNTIME_DIR}/krb5cc'"\n""")
                            submit_file.write("max_retries = 3\n")
                            submit_file.write("requirements = Machine =!= LastRemoteHost\n")
                            if memory != None:
                                submit_file.write(f"request_memory = {memory}\n")
                            submit_file.write(f'+JobFlavour = "{job_flavor}"\n')
                            submit_file.write(f"queue {i}\n")

                    else:
                        if _opt.logs != "":
                            jobs_dir = CONDOR_PATH
                        else: 
                            jobs_dir = OUTPATH
                        if ("/eos/home-" in os.path.realpath(jobs_dir)) or ("/eos/user" in os.path.realpath(jobs_dir)):
                            job_file_dir = "root://eosuser.cern.ch/" + os.path.realpath(jobs_dir)
                        elif ("/eos/cms" in os.path.realpath(jobs_dir)):
                            job_file_dir = "root://eoscms.cern.ch/" + os.path.realpath(jobs_dir)
                        else:
                            job_file_dir = os.path.realpath(jobs_dir)

                        if _opt.logs != "":
                            job_file_executable = os.path.join(CONDOR_PATH, f"{file}.sh")
                            job_file_submit = os.path.join(CONDOR_PATH, f"{file}.sub")
                        else:
                            job_file_executable = os.path.join(OUT_PATH, f"{file}.sh")
                            job_file_submit = os.path.join(OUT_PATH, f"{file}.sub")

                        if not _opt.make_condor_logs:
                            job_file_out = "/dev/null"
                            job_file_err = "/dev/null"
                            job_file_log = "/dev/null"
                        elif _opt.logs != "":
                            job_file_out = f"{file}.$(ClusterId).$(ProcId).out"
                            job_file_err = f"{file}.$(ClusterId).$(ProcId).err"
                            job_file_log = os.path.join(CONDOR_PATH, f"{file}.$(ClusterId).log")
                        else:
                            job_file_out = f"{file}.$(ClusterId).$(ProcId).out"
                            job_file_err = f"{file}.$(ClusterId).$(ProcId).err"
                            job_file_log = os.path.join(OUT_PATH, f"{file}.$(ClusterId).log")

                        with open(job_file_executable, "w") as executable_file:
                            executable_file.write("#!/bin/sh\n")
                            if os.path.exists(f"{OUT_PATH}/merged/{file}/{file}_merged.parquet"):
                                raise Exception(
                                    f"The selected target path: {OUT_PATH}/merged/{file}/{file}_merged.parquet already exists"
                                )
                            if not os.path.exists(f'{OUT_PATH}/merged/Data_{file.split("_")[-1]}'):
                                MKDIRP(f'{OUT_PATH}/merged/Data_{file.split("_")[-1]}')
                            os.chdir(SCRIPT_DIR)
                            print(f'merge_parquet.py --source {IN_PATH}/{file}/nominal --target {OUT_PATH}/merged/Data_{file.split("_")[-1]}/{file}_ --cats {cat_dict_loc} --is-data --abs {genBinning_str}')
                            executable_file.write(f"if [ $1 -eq 0 ]; then\n")
                            executable_file.write(f"    merge_parquet.py --source {IN_PATH}/{file}/nominal --target {OUT_PATH}/merged/Data_{file.split('_')[-1]}/{file}_ --cats {cat_dict_loc} --is-data --abs {genBinning_str} || exit 107\n")
                            executable_file.write("exit 0\n")
                            executable_file.write("fi\n")

                        os.system(f"chmod 775 {job_file_executable}")
                        with open(job_file_submit, "w") as submit_file:
                            if _opt.logs != "": submit_file.write(f"initialdir = {CONDOR_PATH}\n")
                            submit_file.write(f"executable = {job_file_executable}\n")
                            submit_file.write("arguments = $(ProcId)\n")
                            submit_file.write(f"output = {job_file_out}\n")
                            submit_file.write(f"error = {job_file_err}\n")
                            submit_file.write(f"log = {job_file_log}\n")
                            submit_file.write(f"output_destination = {job_file_dir}\n")
                            submit_file.write("on_exit_hold = (ExitBySignal == True) || (ExitCode != 0)\n")
                            submit_file.write("periodic_release =  (NumJobStarts < 3) && ((CurrentTime - EnteredCurrentStatus) > 600)\n")
                            # if _opt.max_materialize != "": submit_file.write(f"max_materialize = {_opt.max_materialize}\n")
                            if (_opt.batch == "condor/apptainer"):
                                submit_file.write("MY.XRDCP_CREATE_DIR     = True\n")
                                submit_file.write("""MY.SingularityImage     = "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/higgsdna:lxplus-el9-latest"\n""")
                                submit_file.write("""MY.SINGULARITY_EXTRA_ARGUMENTS = "-B /afs -B /cvmfs/cms.cern.ch -B /tmp -B /etc/sysconfig/ngbauth-submit -B ${XDG_RUNTIME_DIR} -B /eos --env KRB5CCNAME='FILE:${XDG_RUNTIME_DIR}/krb5cc'"\n""")
                            submit_file.write("max_retries = 3\n")
                            submit_file.write("requirements = Machine =!= LastRemoteHost\n")
                            if memory != None:
                                submit_file.write(f"request_memory = {memory}\n")
                            submit_file.write(f'+JobFlavour = "{job_flavor}"\n')
                            submit_file.write(f"queue\n")
                if _opt.logs != "":
                    submit_jobs(CONDOR_PATH)
                else:
                    submit_jobs(OUT_PATH)

            # at this point Data will be split in eras if any Data dataset is present, here we merge them again in one allData file to rule them all
            # we also skip this step if there is no Data
            if _opt.merge_data:
                j = 0
                for file in files:
                    if j != 0: continue
                    file = file.split("\n")[0]  # otherwise it contains an end of line and messes up the os.walk() call
                    if _opt.logs != "":
                        jobs_dir = CONDOR_PATH
                    else: 
                        jobs_dir = OUTPATH
                    if ("/eos/home-" in os.path.realpath(jobs_dir)) or ("/eos/user" in os.path.realpath(jobs_dir)):
                        job_file_dir = "root://eosuser.cern.ch/" + os.path.realpath(jobs_dir)
                    elif ("/eos/cms" in os.path.realpath(jobs_dir)):
                        job_file_dir = "root://eoscms.cern.ch/" + os.path.realpath(jobs_dir)
                    else:
                        job_file_dir = os.path.realpath(jobs_dir)

                    if _opt.logs != "":
                        job_file_executable = os.path.join(CONDOR_PATH, f"{file}_merge_data.sh")
                        job_file_submit = os.path.join(CONDOR_PATH, f"{file}_merge_data.sub")
                    else:
                        job_file_executable = os.path.join(OUT_PATH, f"{file}_merge_data.sh")
                        job_file_submit = os.path.join(OUT_PATH, f"{file}_merge_data.sub")

                    if not _opt.make_condor_logs:
                        job_file_out = "/dev/null"
                        job_file_err = "/dev/null"
                        job_file_log = "/dev/null"
                    elif _opt.logs != "":
                        job_file_out = f"{file}_merge_data.$(ClusterId).$(ProcId).out"
                        job_file_err = f"{file}_merge_data.$(ClusterId).$(ProcId).err"
                        job_file_log = os.path.join(CONDOR_PATH, f"{file}_merge_data.$(ClusterId).log")
                    else:
                        job_file_out = f"{file}_merge_data.$(ClusterId).$(ProcId).out"
                        job_file_err = f"{file}_merge_data.$(ClusterId).$(ProcId).err"
                        job_file_log = os.path.join(OUT_PATH, f"{file}_merge_data.$(ClusterId).log")
                    if "data" in file.lower() or "DoubleEG" in file:
                        with open(job_file_executable, "w") as executable_file:
                            executable_file.write("#!/bin/sh\n")
                            dirpath, dirnames, filenames = next(os.walk(f'{OUT_PATH}/merged/Data_{file.split("_")[-1]}'))
                            if len(filenames) > 0:
                                print(f'merge_parquet.py --source {OUT_PATH}/merged/Data_{file.split("_")[-1]} --target {OUT_PATH}/merged/Data_{file.split("_")[-1]}/allData_ --cats {cat_dict_loc} --is-data --abs {genBinning_str}')
                                executable_file.write(f"if [ $1 -eq 0 ]; then\n")
                                executable_file.write(f"    merge_parquet.py --source {OUT_PATH}/merged/Data_{file.split('_')[-1]} --target {OUT_PATH}/merged/Data_{file.split('_')[-1]}/allData_ --cats {cat_dict_loc} --is-data --abs {genBinning_str} || exit 107\n")
                                executable_file.write("exit 0\n")
                                executable_file.write("fi\n")
                                #break
                            else:
                                logger.info(f'No merged parquet found for {file} in the directory: {OUT_PATH}/merged/Data_{file.split("_")[-1]}')
                        with open(job_file_submit, "w") as submit_file:
                            if _opt.logs != "": submit_file.write(f"initialdir = {CONDOR_PATH}\n")
                            submit_file.write(f"executable = {job_file_executable}\n")
                            submit_file.write("arguments = $(ProcId)\n")
                            submit_file.write(f"output = {job_file_out}\n")
                            submit_file.write(f"error = {job_file_err}\n")
                            submit_file.write(f"log = {job_file_log}\n")
                            submit_file.write(f"output_destination = {job_file_dir}\n")
                            submit_file.write("on_exit_hold = (ExitBySignal == True) || (ExitCode != 0)\n")
                            submit_file.write("periodic_release =  (NumJobStarts < 3) && ((CurrentTime - EnteredCurrentStatus) > 600)\n")
                            # if _opt.max_materialize != "": submit_file.write(f"max_materialize = {_opt.max_materialize}\n")
                            if (_opt.batch == "condor/apptainer"):
                                submit_file.write("MY.XRDCP_CREATE_DIR     = True\n")
                                submit_file.write("""MY.SingularityImage     = "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/higgsdna:lxplus-el9-latest"\n""")
                                submit_file.write("""MY.SINGULARITY_EXTRA_ARGUMENTS = "-B /afs -B /cvmfs/cms.cern.ch -B /tmp -B /etc/sysconfig/ngbauth-submit -B ${XDG_RUNTIME_DIR} -B /eos --env KRB5CCNAME='FILE:${XDG_RUNTIME_DIR}/krb5cc'"\n""")
                            submit_file.write("max_retries = 3\n")
                            submit_file.write("requirements = Machine =!= LastRemoteHost\n")
                            if memory != None:
                                submit_file.write(f"request_memory = {memory}\n")
                            submit_file.write(f'+JobFlavour = "{job_flavor}"\n')
                            submit_file.write(f"queue\n")
                    os.system(f"chmod 775 {job_file_executable}")
                    j += 1
                if _opt.logs != "":
                    submit_jobs(CONDOR_PATH, "merge_data")
                else:
                    submit_jobs(OUT_PATH, "merge_data")

    if _opt.root:
        logger.info("Starting root step")
        if _opt.syst:
            logger.info("you've selected the run with systematics")
            args = "--do-syst"
        else:
            logger.info("you've selected the run without systematics")
            args = ""

        if _opt.merge:
            IN_PATH = OUT_PATH
        # Note, in my version of HiggsDNA I run the analysis splitting data per Era in different datasets
        # the treatment of data here is tested just with that structure
        with open(dirlist_path) as fl:
            files = fl.readlines()
            for file in files:
                file = file.split("\n")[0]
                if _opt.logs != "":
                    jobs_dir = CONDOR_PATH
                else: 
                    jobs_dir = OUTPATH
                if ("/eos/home-" in os.path.realpath(jobs_dir)) or ("/eos/user" in os.path.realpath(jobs_dir)):
                    job_file_dir = "root://eosuser.cern.ch/" + os.path.realpath(jobs_dir)
                elif ("/eos/cms" in os.path.realpath(jobs_dir)):
                    job_file_dir = "root://eoscms.cern.ch/" + os.path.realpath(jobs_dir)
                else:
                    job_file_dir = os.path.realpath(jobs_dir)
                if "data" not in file.lower() and (not "unknown" in decompose_string(file, era_flag=_opt.eraFlag)):
                    if _opt.logs != "":
                        job_file_executable = os.path.join(CONDOR_PATH, f"{file}_root.sh")
                    else:
                        job_file_executable = os.path.join(OUT_PATH, f"{file}_root.sh")

                    if not _opt.merge:
                        if _opt.logs != "":
                            job_file_submit = os.path.join(CONDOR_PATH, f"{file}_root.sub")
                        else:
                            job_file_submit = os.path.join(OUT_PATH, f"{file}_root.sub")
                        if not _opt.make_condor_logs:
                            job_file_out = "/dev/null"
                            job_file_err = "/dev/null"
                            job_file_log = "/dev/null"
                        elif _opt.logs != "":
                            job_file_out = f"{file}_root.$(ClusterId).$(ProcId).out"
                            job_file_err = f"{file}_root.$(ClusterId).$(ProcId).err"
                            job_file_log = os.path.join(CONDOR_PATH, f"{file}_root.$(ClusterId).log")
                        else:
                            job_file_out = f"{file}_root.$(ClusterId).$(ProcId).out"
                            job_file_err = f"{file}_root.$(ClusterId).$(ProcId).err"
                            job_file_log = os.path.join(OUT_PATH, f"{file}_root.$(ClusterId).log")
                    with open(job_file_executable, "w") as executable_file:
                        executable_file.write("#!/bin/sh\n")
                        if os.path.exists(f"{OUT_PATH}/root/{file}"):
                            raise Exception(
                                f"The selected target path: {OUT_PATH}/root/{file} already exists"
                            )

                        if os.listdir(f"{IN_PATH}/merged/{file}/"):
                            logger.info(f"Found merged files {IN_PATH}/merged/{file}/")
                        else:
                            raise Exception(f"Merged parquet not found at {IN_PATH}/merged/")
                        MKDIRP(f"{OUT_PATH}/root/{file}")
                        os.chdir(SCRIPT_DIR)
                        executable_file.write(f"if [ $1 -eq 0 ]; then\n")
                        executable_file.write(f"    convert_parquet_to_root.py {IN_PATH}/merged/{file}/merged.parquet {OUT_PATH}/root/{file}/merged.root mc --process {decompose_string(file)} {args} --cats {cat_dict_loc} --vars {var_dict_loc} --abs {genBinning_str} {tbasket_str} || exit 107\n")
                        executable_file.write("exit 0\n")
                        executable_file.write("fi\n")
                    os.system(f"chmod 775 {job_file_executable}")
                    with open(job_file_submit, "w") as submit_file:
                        if _opt.logs != "": submit_file.write(f"initialdir = {CONDOR_PATH}\n")
                        submit_file.write(f"executable = {job_file_executable}\n")
                        submit_file.write("arguments = $(ProcId)\n")
                        submit_file.write(f"output = {job_file_out}\n")
                        submit_file.write(f"error = {job_file_err}\n")
                        submit_file.write(f"log = {job_file_log}\n")
                        submit_file.write(f"output_destination = {job_file_dir}\n")
                        submit_file.write("on_exit_hold = (ExitBySignal == True) || (ExitCode != 0)\n")
                        submit_file.write("periodic_release =  (NumJobStarts < 3) && ((CurrentTime - EnteredCurrentStatus) > 600)\n")
                        # if _opt.max_materialize != "": submit_file.write(f"max_materialize = {_opt.max_materialize}\n")
                        if (_opt.batch == "condor/apptainer"):
                            submit_file.write("MY.XRDCP_CREATE_DIR     = True\n")
                            submit_file.write("""MY.SingularityImage     = "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/higgsdna:lxplus-el9-latest"\n""")
                            submit_file.write("""MY.SINGULARITY_EXTRA_ARGUMENTS = "-B /afs -B /cvmfs/cms.cern.ch -B /tmp -B /etc/sysconfig/ngbauth-submit -B ${XDG_RUNTIME_DIR} -B /eos --env KRB5CCNAME='FILE:${XDG_RUNTIME_DIR}/krb5cc'"\n""")
                        submit_file.write("max_retries = 3\n")
                        submit_file.write("requirements = Machine =!= LastRemoteHost\n")
                        if memory != None:
                            submit_file.write(f"request_memory = {memory}\n")
                        submit_file.write(f'+JobFlavour = "{job_flavor}"\n')
                        submit_file.write(f"queue\n")
                elif "data" in file.lower():
                    if _opt.logs != "":
                        job_file_executable = os.path.join(CONDOR_PATH, f"{file}_root.sh")
                    else:
                        job_file_executable = os.path.join(OUT_PATH, f"{file}_root.sh")

                    if not _opt.merge:
                        if _opt.logs != "":
                            job_file_submit = os.path.join(CONDOR_PATH, f"{file}_root.sub")
                        else:
                            job_file_submit = os.path.join(OUT_PATH, f"{file}_root.sub")

                        if not _opt.make_condor_logs:
                            job_file_out = "/dev/null"
                            job_file_err = "/dev/null"
                            job_file_log = "/dev/null"
                        elif _opt.logs != "":
                            job_file_out = f"{file}_root.$(ClusterId).$(ProcId).out"
                            job_file_err = f"{file}_root.$(ClusterId).$(ProcId).err"
                            job_file_log = os.path.join(CONDOR_PATH, f"{file}_root.$(ClusterId).log")
                        else:
                            job_file_out = f"{file}_root.$(ClusterId).$(ProcId).out"
                            job_file_err = f"{file}_root.$(ClusterId).$(ProcId).err"
                            job_file_log = os.path.join(OUT_PATH, f"{file}_root.$(ClusterId).log")

                    with open(job_file_executable, "w") as executable_file:
                        executable_file.write("#!/bin/sh\n")
                        if os.listdir(f'{IN_PATH}/merged/Data_{file.split("_")[-1]}/'):
                            logger.info(
                                f'Found merged data files in: {IN_PATH}/merged/Data_{file.split("_")[-1]}/'
                            )
                        else:
                            raise Exception(
                                f'Merged parquet not found at: {IN_PATH}/merged/Data_{file.split("_")[-1]}/'
                            )

                        if os.path.exists(
                            f'{OUT_PATH}/root/Data/allData_{file.split("_")[-1]}.root'
                        ):
                            logger.info(
                                f'Data already converted: {OUT_PATH}/root/Data/allData_{file.split("_")[-1]}.root'
                            )
                            continue
                        elif not os.path.exists(f"{OUT_PATH}/root/Data/"):
                            MKDIRP(f"{OUT_PATH}/root/Data")
                            os.chdir(SCRIPT_DIR)
                            executable_file.write(f"if [ $1 -eq 0 ]; then\n")
                            executable_file.write(f"    convert_parquet_to_root.py {IN_PATH}/merged/Data_{file.split('_')[-1]}/allData_merged.parquet {OUT_PATH}/root/Data/allData_{file.split('_')[-1]}.root data --cats {cat_dict_loc} --vars {var_dict_loc} --abs {genBinning_str} {tbasket_str} || exit 107\n")
                            executable_file.write("exit 0\n")
                            executable_file.write("fi\n")
                        else:
                            os.chdir(SCRIPT_DIR)
                            executable_file.write(f"if [ $1 -eq 0 ]; then\n")
                            executable_file.write(f"    convert_parquet_to_root.py {IN_PATH}/merged/Data_{file.split('_')[-1]}/allData_merged.parquet {OUT_PATH}/root/Data/allData_{file.split('_')[-1]}.root data --cats {cat_dict_loc} --vars {var_dict_loc} --abs {genBinning_str} {tbasket_str} || exit 107\n")
                            executable_file.write("exit 0\n")
                            executable_file.write("fi\n")
                os.system(f"chmod 775 {job_file_executable}")
                with open(job_file_submit, "w") as submit_file:
                    if _opt.logs != "": submit_file.write(f"initialdir = {CONDOR_PATH}\n")
                    submit_file.write(f"executable = {job_file_executable}\n")
                    submit_file.write("arguments = $(ProcId)\n")
                    submit_file.write(f"output = {job_file_out}\n")
                    submit_file.write(f"error = {job_file_err}\n")
                    submit_file.write(f"log = {job_file_log}\n")
                    submit_file.write(f"output_destination = {job_file_dir}\n")
                    submit_file.write("on_exit_hold = (ExitBySignal == True) || (ExitCode != 0)\n")
                    submit_file.write("periodic_release =  (NumJobStarts < 3) && ((CurrentTime - EnteredCurrentStatus) > 600)\n")
                    # if _opt.max_materialize != "": submit_file.write(f"max_materialize = {_opt.max_materialize}\n")
                    if (_opt.batch == "condor/apptainer"):
                        submit_file.write("MY.XRDCP_CREATE_DIR     = True\n")
                        submit_file.write("""MY.SingularityImage     = "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/higgsdna:lxplus-el9-latest"\n""")
                        submit_file.write("""MY.SINGULARITY_EXTRA_ARGUMENTS = "-B /afs -B /cvmfs/cms.cern.ch -B /tmp -B /etc/sysconfig/ngbauth-submit -B ${XDG_RUNTIME_DIR} -B /eos --env KRB5CCNAME='FILE:${XDG_RUNTIME_DIR}/krb5cc'"\n""")
                    submit_file.write("max_retries = 3\n")
                    submit_file.write("requirements = Machine =!= LastRemoteHost\n")
                    if memory != None:
                        submit_file.write(f"request_memory = {memory}\n")
                    submit_file.write(f'+JobFlavour = "{job_flavor}"\n')
                    submit_file.write(f"queue\n")
        if _opt.logs != "":
            submit_jobs(CONDOR_PATH, "root")
        else:
            submit_jobs(OUT_PATH, "root")
