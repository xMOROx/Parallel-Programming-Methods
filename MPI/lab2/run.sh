#!/bin/bash -l
#SBATCH --nodes 1
#SBATCH --ntasks 12
#SBATCH --time=02:10:00
#SBATCH --partition=plgrid
#SBATCH --account=plgmpr25-cpu

module add .plgrid plgrid/tools/openmpi
module load python

set -x

python -u runner.py
