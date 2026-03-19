#!/bin/bash -l
#$ -cwd
#$ -V
#$ -N ember
#$ -o logs/$JOB_NAME.$JOB_ID.$TASK_ID.out
#$ -e logs/$JOB_NAME.$JOB_ID.$TASK_ID.err
#$ -t 1-128

set -euo pipefail
module load miniconda


echo "Host: $(hostname)"
echo "Job:  $JOB_NAME  ID: $JOB_ID  Task: $SGE_TASK_ID"
echo "Cmd:  ember fit-sed widebinary_fluxes.pqt chains/thin/ --synthetic --xpphoto --fixedhe=30 --numtasks=128"

# Run the command
ember fit-sed widebinary_fluxes.pqt chains/thin/ --synthetic --xpphoto --fixedhe=30 --numtasks=128
