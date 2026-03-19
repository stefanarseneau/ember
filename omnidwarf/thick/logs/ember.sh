#!/bin/bash -l
#$ -cwd
#$ -V
#$ -N ember
#$ -o logs/$JOB_NAME.$JOB_ID.$TASK_ID.out
#$ -e logs/$JOB_NAME.$JOB_ID.$TASK_ID.err
#$ -t 1-1024

set -euo pipefail
module load miniconda


echo "Host: $(hostname)"
echo "Job:  $JOB_NAME  ID: $JOB_ID  Task: $SGE_TASK_ID"
echo "Cmd:  ember fit-sed /projectnb/mesaelm/ember/omnidwarf/omnidwarf_fluxes.pqt /projectnb/mesaelm/ember/omnidwarf/thick/ --xpphoto --synthetic --numtasks=1024"

# Run the command
ember fit-sed /projectnb/mesaelm/ember/omnidwarf/omnidwarf_fluxes.pqt /projectnb/mesaelm/ember/omnidwarf/thick/ --xpphoto --synthetic --numtasks=1024
