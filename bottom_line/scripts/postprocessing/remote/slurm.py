#!/usr/bin/env python
import os
import sys
from time import sleep

def create_slurm_script(_opt, job_name, script_path, output_path, error_path, commands, time="01:00:00", partition="short", OUT_PATH="", memory="4G", mode="merged", random_delay=False):
    with open(script_path, "w") as script_file:
        script_file.write("#!/bin/bash\n")
        if random_delay:
            import random
            # Random delay between 1 and 20 seconds
            rand_int = random.randint(1, 20)
            script_file.write(f"#SBATCH --begin=now+{rand_int}seconds\n")
        script_file.write(f"#SBATCH --requeue\n")
        script_file.write(f"#SBATCH --job-name={job_name}\n")
        script_file.write(f"#SBATCH --output={output_path}\n")
        script_file.write(f"#SBATCH --error={error_path}\n")
        script_file.write(f"#SBATCH --time={time}\n")
        script_file.write(f"#SBATCH --partition={partition}\n")
        script_file.write(f"#SBATCH --mem={memory}\n")
        script_file.write("\n")

        if _opt.batch == "slurm/psi":
            script_file.write("export TARGET_PATH=/scratch/$USER/${SLURM_JOB_ID}\n")
            script_file.write("mkdir -p $TARGET_PATH\n")
            script_file.write("\n")
            script_file.write("\n".join(commands))
            script_file.write("\n")
            script_file.write(f"xrdcp -fr $TARGET_PATH/{mode} root://t3dcachedb.psi.ch:1094//{OUT_PATH}\n")
            script_file.write("rm -rf /scratch/$USER/${SLURM_JOB_ID}\n")


def submit_slurm_jobs(directory, suffix=""):
    for file in os.listdir(directory):
        if file.endswith(f"{suffix}.sh"):
            os.system(f"sbatch {os.path.join(directory, file)}")

def slurm_postprocessing(_opt, OUT_PATH, IN_PATH, dirlist_path, var_dict, cat_dict_loc, var_dict_loc, genBinning_str, skip_normalisation_str, merge_data_str, do_syst_str, tbasket_str, time, partition, memory, decompose_string, logger):

    time = time or "01:00:00"
    partition = partition or "short"
    memory = memory or "22GB"

    if _opt.root_only:
        with open(dirlist_path) as fl:
            files = fl.readlines()
            log_path = os.getcwd()
            if _opt.logs is not None:
                log_path = _opt.logs

            for j, file in enumerate(files):
                if _opt.merge_data and (_opt.type.lower() == "data") and j > 0: continue
                file = file.split("\n")[0]
                
                os.makedirs(os.path.join(log_path, file), exist_ok=True)
                job_script = os.path.join(log_path, file, f"{file}_root.sh")
                job_out = os.path.join(log_path, file, f"{file}_root.out")
                job_err = os.path.join(log_path, file, f"{file}_root.err")
                
                commands = []
                
                # Handle merge data for data type if applicable
                if _opt.merge_data and (_opt.type.lower() == "data"):
                    source_folder_path = f"{IN_PATH}"
                    # Create allData.root
                    target_file_path = f"{OUT_PATH}/root/Data/merged.root"
                    target_folder_path = f"{OUT_PATH}/root/Data"
                    if (_opt.batch == "slurm/psi"):
                        target_file_path = f"$TARGET_PATH/root/Data/merged.root"
                        target_folder_path = f"$TARGET_PATH/root/Data"
                    else:
                        os.makedirs(target_folder_path, exist_ok=True)

                    current_process = "data"
                elif (not _opt.merge_data) or (_opt.type.lower() == "mc"):
                    source_folder_path = f"{IN_PATH}/{file}"
                    target_file_path = f"{OUT_PATH}/root/{file}/merged.root"
                    target_folder_path = f"{OUT_PATH}/root/{file}"
                    if (_opt.batch == "slurm/psi"):
                        target_file_path = f"$TARGET_PATH/root/{file}/merged.root"
                        target_folder_path = f"$TARGET_PATH/root/{file}"
                    else:
                        os.makedirs(target_folder_path, exist_ok=True)

                    current_process = decompose_string(file)
                
                print(f"merge_root.py --source {source_folder_path} --target {target_file_path} --cats {cat_dict_loc} --abs {genBinning_str} --vars {var_dict_loc} --type {_opt.type} --process {current_process} {skip_normalisation_str} {merge_data_str} {do_syst_str}")
                commands.append(f"merge_root.py --source {source_folder_path} --target {target_file_path} --cats {cat_dict_loc} --abs {genBinning_str} --vars {var_dict_loc} --type {_opt.type} --process {current_process} {skip_normalisation_str} {merge_data_str} {do_syst_str}")
                
                if (_opt.type.lower() != "data"):
                    random_delay = True
                    sleeping = True
                else:
                    random_delay = False
                    sleeping = False
                
                create_slurm_script(_opt, file, job_script, job_out, job_err, commands, OUT_PATH=OUT_PATH, mode="root", random_delay=random_delay, memory=memory, time=time, partition=partition)
        
                submit_slurm_jobs(os.path.join(log_path, file), suffix="root")
                if sleeping:
                    sleep(2)
        return 0

    if _opt.merge and not _opt.merge_data:
        with open(dirlist_path) as fl:
            files = fl.readlines()
            log_path = os.getcwd()
            if _opt.logs is not None:
                log_path = _opt.logs
            if _opt.syst:
                for file in files:
                    for var in var_dict:
                        file = file.strip()
                        os.makedirs(os.path.join(log_path, file, var_dict[var]), exist_ok=True)
                        job_script = os.path.join(log_path, file, var_dict[var], f"{file}.sh")
                        job_out = os.path.join(log_path, file, var_dict[var], f"{file}.out")
                        job_err = os.path.join(log_path, file, var_dict[var], f"{file}.err")

                        commands = []
                        
                        if "data" not in file.lower():
                            target_path = f"{OUT_PATH}/merged/{file}/{var_dict[var]}/"
                            if (_opt.batch == "slurm/psi"):
                                target_path = f"$TARGET_PATH/merged/{file}/{var_dict[var]}/"
                            else:
                                os.makedirs(target_path, exist_ok=True)
                            print(f"merge_parquet.py --source {IN_PATH}/{file}/{var_dict[var]} --target {target_path} --cats {cat_dict_loc} --abs {genBinning_str}")
                            commands.append(f"merge_parquet.py --source {IN_PATH}/{file}/{var_dict[var]} --target {target_path} --cats {cat_dict_loc} --abs {genBinning_str}")
                        else:
                            target_path = f"{OUT_PATH}/merged/Data_{file.split('_')[-1]}"
                            if (_opt.batch == "slurm/psi"):
                                target_path = f"$TARGET_PATH/merged/Data_{file.split('_')[-1]}"
                            else:
                                os.makedirs(target_path, exist_ok=True)
                            print(f"merge_parquet.py --source {IN_PATH}/{file}/nominal --target {target_path}/{file}_ --cats {cat_dict_loc} --is-data --abs {genBinning_str}")
                            commands.append(f"merge_parquet.py --source {IN_PATH}/{file}/nominal --target {target_path}/{file}_ --cats {cat_dict_loc} --is-data --abs {genBinning_str}")
                        
                        create_slurm_script(_opt, file, job_script, job_out, job_err, commands, OUT_PATH=OUT_PATH, mode="merged", random_delay=True, memory=memory, time=time, partition=partition)
        
                        submit_slurm_jobs(os.path.join(log_path, file, var_dict[var]))
                        sleep(2)  # Wait 2 seconds before submitting the next job
            else:
                with open(dirlist_path) as fl:
                    files = fl.readlines()
                    log_path = os.getcwd()
                    for file in files:
                        file = file.strip()
                        if _opt.logs is not None:
                            log_path = _opt.logs
                        job_script = os.path.join(log_path, f"{file}.sh")
                        job_out = os.path.join(log_path, f"{file}.out")
                        job_err = os.path.join(log_path, f"{file}.err")

                        commands = []
                        
                        if "data" not in file.lower():
                            target_path = f"{OUT_PATH}/merged/{file}/nominal/"
                            if (_opt.batch == "slurm/psi"):
                                target_path = f"$TARGET_PATH/merged/{file}/nominal/"
                            else:
                                os.makedirs(target_path, exist_ok=True)
                            print(f"merge_parquet.py --source {IN_PATH}/{file}/nominal --target {target_path} --cats {cat_dict_loc} --abs {genBinning_str}")
                            commands.append(f"merge_parquet.py --source {IN_PATH}/{file}/nominal/ --target {target_path} --cats {cat_dict_loc} --abs {genBinning_str}")
                        else:
                            target_path = f"{OUT_PATH}/merged/Data_{file.split('_')[-1]}"
                            if (_opt.batch == "slurm/psi"):
                                target_path = f"$TARGET_PATH/merged/Data_{file.split('_')[-1]}"
                            else:
                                os.makedirs(target_path, exist_ok=True)
                            print(f"merge_parquet.py --source {IN_PATH}/{file}/nominal --target {target_path}/{file}_ --cats {cat_dict_loc} --is-data --abs {genBinning_str}")
                            commands.append(f"merge_parquet.py --source {IN_PATH}/{file}/nominal --target {target_path}/{file}_ --cats {cat_dict_loc} --is-data --abs {genBinning_str}")
                        
                        create_slurm_script(_opt, file, job_script, job_out, job_err, commands, OUT_PATH=OUT_PATH, mode="merged", memory=memory, time=time, partition=partition)
                
                submit_slurm_jobs(log_path)

    elif _opt.merge_data:
        j = 0.
        with open(dirlist_path) as fl:
            files = fl.readlines()
            log_path = os.getcwd()
            for file in files:
                if j != 0: continue
                file = file.split("\n")[0]  # otherwise it contains an end of line and messes up the os.walk() call
                file = file.strip()
                if _opt.logs is not None:
                    log_path = _opt.logs
                job_script = os.path.join(log_path, f"{file}_mergeData.sh")
                job_out = os.path.join(log_path, f"{file}_mergeData.out")
                job_err = os.path.join(log_path, f"{file}_mergeData.err")

                commands = []
            
                if "data" in file.lower() or "DoubleEG" in file:
                    target_path = f"{OUT_PATH}/merged/Data_{file.split('_')[-1]}"
                    print(target_path)
                    _, _, filenames = next(os.walk(target_path))
                    if len(filenames) > 0:
                        if (_opt.batch == "slurm/psi"):
                            target_path = f"$TARGET_PATH/merged/Data_{file.split('_')[-1]}"
                        else:
                            os.makedirs(target_path, exist_ok=True)
                        print(f"merge_parquet.py --source {IN_PATH}/{file} --target {target_path}/allData_ --cats {cat_dict_loc} --is-data --abs {genBinning_str}")
                        commands.append(f"merge_parquet.py --source {IN_PATH}/{file} --target {target_path}/allData_ --cats {cat_dict_loc} --is-data --abs {genBinning_str}")
                    else:
                        logger.info(f'No merged parquet found for {file} in the directory: {target_path}')
                else:
                    logger.error(f'Please choose data as input. {file} is not a data file.')
                    sys.exit(1)
                j += 1

                create_slurm_script(_opt, file, job_script, job_out, job_err, commands, OUT_PATH=OUT_PATH, mode="merged", memory=memory, time=time, partition=partition)

            submit_slurm_jobs(log_path, "mergeData")

    elif _opt.root:
        logger.info("Starting root step")
        if _opt.syst:
            logger.info("you've selected the run with systematics")
            args = "--do-syst"
        else:
            logger.info("you've selected the run without systematics")
            args = ""
        with open(dirlist_path) as fl:
            files = fl.readlines()
            log_path = os.getcwd()
            for file in files:
                file = file.strip()
                if _opt.logs is not None:
                    log_path = _opt.logs
                job_script = os.path.join(log_path, f"{file}_root.sh")
                job_out = os.path.join(log_path, f"{file}_root.out")
                job_err = os.path.join(log_path, f"{file}_root.err")

                commands = []

                if "data" not in file.lower() and (not "unknown" in decompose_string(file, era_flag=_opt.eraFlag)):
                    if os.listdir(f"{IN_PATH}/merged/{file}/"):
                        logger.info(f"Found merged files {IN_PATH}/merged/{file}/")
                    else:
                        raise Exception(f"Merged parquet not found at {IN_PATH}/merged/")

                    target_path = f"{OUT_PATH}/root/{file}/"
                    if (_opt.batch == "slurm/psi"):
                        target_path = f"$TARGET_PATH/root/{file}/"

                    print(f"convert_parquet_to_root.py {IN_PATH}/merged/{file}/merged.parquet {target_path}/merged.root mc --process {decompose_string(file)} {args} --cats {cat_dict_loc} --vars {var_dict_loc} --abs {genBinning_str} {tbasket_str}")
                    commands.append(f"convert_parquet_to_root.py {IN_PATH}/merged/{file}/merged.parquet {target_path}/merged.root mc --process {decompose_string(file)} {args} --cats {cat_dict_loc} --vars {var_dict_loc} --abs {genBinning_str} {tbasket_str}")
                
                elif "data" in file.lower():
                    
                    if os.listdir(f'{IN_PATH}/merged/Data_{file.split("_")[-1]}/'):
                        logger.info(
                            f'Found merged data files in: {IN_PATH}/merged/Data_{file.split("_")[-1]}/'
                        )
                    else:
                        raise Exception(
                            f'Merged parquet not found at: {IN_PATH}/merged/Data_{file.split("_")[-1]}/'
                        )
                                            
                    target_path = f"{OUT_PATH}/root/Data"
                    if (_opt.batch == "slurm/psi"):
                        target_path = f"$TARGET_PATH/root/Data"
                        
                    print(f"convert_parquet_to_root.py {IN_PATH}/merged/Data_{file.split('_')[-1]}/allData_merged.parquet {target_path}/allData_{file.split('_')[-1]}.root data --cats {cat_dict_loc} --vars {var_dict_loc} --abs {genBinning_str} {tbasket_str}")
                    commands.append(f"convert_parquet_to_root.py {IN_PATH}/merged/Data_{file.split('_')[-1]}/allData_merged.parquet {target_path}/allData_{file.split('_')[-1]}.root data --cats {cat_dict_loc} --vars {var_dict_loc} --abs {genBinning_str} {tbasket_str}")
                
                create_slurm_script(_opt, file, job_script, job_out, job_err, commands, OUT_PATH=OUT_PATH, mode="root", memory=memory, time=time, partition=partition)
        
            submit_slurm_jobs(log_path, "root")