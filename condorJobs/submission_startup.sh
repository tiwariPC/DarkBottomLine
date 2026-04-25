#!/bin/bash
set -e

# Interactive submission helper for condor jobs

# Change this to your DarkBottomLine directory
DBL=/eos/user/x/xdu/DarkBottomLine
# Change this to your condorJobs directory
# Keep files in folers named by year, e.g., condorJobs/2022, condorJobs/2023, ...
BASE=/afs/cern.ch/user/x/xdu/condorJobs
JOBDIR=$BASE/jobs

mkdir -p "$JOBDIR"

YEARS=(2022 2022EE 2023 2024)

# Auto submit flag
AUTO_SUBMIT=0
if [ "$1" = "-y" ] || [ "$1" = "--yes" ]; then
    AUTO_SUBMIT=1
fi

# Make globs expand to empty when there is no match (Bash only)
if [ -n "$BASH_VERSION" ]; then
    shopt -s nullglob
fi

declare -a SELECTED

create_jobscript() {
    local INPUTTXT="$1"
    local YEAR="$2"
    local SAMPLE
    SAMPLE=$(basename "$INPUTTXT" .txt)
    local JOBSCRIPT="$JOBDIR/run_${SAMPLE}_${YEAR}.sh"

    cat > "$JOBSCRIPT" <<EOF
#!/bin/bash
#ulimit -s unlimited
#set -e

INPUTPATH="$INPUTTXT"

cd /eos/user/x/xdu/DarkBottomLine
source /cvmfs/sft.cern.ch/lcg/views/LCG_109/x86_64-el9-gcc15-opt/setup.sh
export PYTHONPATH="/eos/user/x/xdu/DarkBottomLine/.local/lib/python3.9/site-packages:\$PYTHONPATH"

WORKDIR=\$PWD
echo "Running in \$WORKDIR"

cat \$INPUTPATH | while read path; do
    INPUT_XROOTD="\$path"
    BASENAME=\$(basename "\$path" .root)

    OUT_LOCAL="outputs/${SAMPLE}/${YEAR}/\${BASENAME}.pkl"
    OUT_EVENT="outputs/${SAMPLE}_EVENTSELECTION/${YEAR}/\${BASENAME}.pkl"

    echo "Input  : \$INPUT_XROOTD"
    echo "Output : \$OUT_LOCAL"

    python3 -m darkbottomline.cli analyze \\
      --config configs/${YEAR}.yaml \\
      --regions-config configs/regions.yaml \\
      --input "\$INPUT_XROOTD" \\
      --output "\$OUT_LOCAL" \\
      --chunk-size 100000 \\
      --event-selection-output "\$OUT_EVENT" \\
      --executor futures
done
EOF

    chmod +x "$JOBSCRIPT"
    echo "$JOBSCRIPT"
}

echo "Interactive job submission. Use Ctrl-C to quit anytime."

while :; do
    echo
    echo "Currently selected jobs:"
    if [ ${#SELECTED[@]} -eq 0 ]; then
        echo "  (none)"
    else
        for i in "${!SELECTED[@]}"; do
            txt=${SELECTED[$i]}
            sample=$(basename "$txt" .txt)
            # extract year from path: .../condorJobs/<year>/<file>
            year=$(basename "$(dirname "$txt")")
            echo "  [$i] $sample ($year)"
        done
    fi

    PS3="Choose a year to add a job, or select action: "
    options=("All years" "${YEARS[@]}" "Submit selected jobs" "Quit")
    select opt in "${options[@]}"; do
        if [[ -z "$opt" ]]; then
            echo "Invalid selection."; break
        fi

        if [[ "$opt" = "Submit selected jobs" ]]; then
            if [ ${#SELECTED[@]} -eq 0 ]; then
                echo "No jobs selected; nothing to submit."; break
            fi
            echo "Selected to submit:"; for t in "${SELECTED[@]}"; do echo "  - $(basename "$t" .txt) ($(basename $(dirname "$t")))"; done
            if [ "$AUTO_SUBMIT" -eq 0 ]; then
                read -p "Proceed to submit these ${#SELECTED[@]} jobs? [y/N] " conf
            else
                conf="y"
            fi
            if [[ "$conf" =~ ^[Yy] ]]; then
                for txt in "${SELECTED[@]}"; do
                    yr=$(basename "$(dirname "$txt")")
                    SAMPLE=$(basename "$txt" .txt)
                    JOBSCRIPT=$(create_jobscript "$txt" "$yr")
                    condor_submit <<SUBMIT
executable = $JOBSCRIPT
output     = $JOBDIR/${SAMPLE}_${yr}.out
error      = $JOBDIR/${SAMPLE}_${yr}.err
log        = $JOBDIR/${SAMPLE}_${yr}.log
request_cpus   = 4
request_memory = 4GB
+JobFlavour = "testmatch"
JobBatchName = "${SAMPLE}_${yr}"

queue
SUBMIT
                    echo "Submitted ${SAMPLE} (${yr})"
                done
            else
                echo "Submission cancelled."
            fi
            exit 0
        elif [[ "$opt" = "Quit" ]]; then
            echo "Exiting without submitting."; exit 0
        else
            # opted a year or All years
            if [[ "$opt" = "All years" ]]; then
                YEAR="ALL"
                files=()
                for y in "${YEARS[@]}"; do
                    for f in "$BASE/$y"/*.txt; do
                        files+=("$f")
                    done
                done
            else
                YEAR="$opt"
                FILE_DIR="$BASE/${YEAR}"
                files=( "$FILE_DIR"/*.txt )
            fi

            if [ ${#files[@]} -eq 0 ]; then
                echo "No .txt files found."; break
            fi

            PS3="Choose a file to add (or select action): "
            select f in "${files[@]}" "Add all files" "Back"; do
                if [[ "$REPLY" -ge 1 && "$REPLY" -le ${#files[@]} ]]; then
                    # single file chosen
                    chosen="${files[$REPLY-1]}"
                    SELECTED+=("$chosen")
                    yr=$(basename "$(dirname "$chosen")")
                    echo "Added $(basename "$chosen" .txt) ($yr)"
                    break
                elif [[ "$REPLY" -eq $((${#files[@]}+1)) ]]; then
                    # Add all files
                    for ff in "${files[@]}"; do
                        SELECTED+=("$ff")
                    done
                    echo "Added all ${#files[@]} files."
                    break
                elif [[ "$REPLY" -eq $((${#files[@]}+2)) ]]; then
                    # Back
                    break
                else
                    echo "Invalid selection"
                fi
            done
            break
        fi
    done
done

