#!/bin/bash -l
#SBATCH --nodes 1
#SBATCH --ntasks 48
#SBATCH --time=10:00:00
#SBATCH --partition=plgrid
#SBATCH --account=plgmpr25-cpu

module load python

set -x

python -u runner.py
