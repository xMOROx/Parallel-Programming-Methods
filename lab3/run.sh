#!/bin/bash -l
#SBATCH --nodes 1
#SBATCH --ntasks 48
#SBATCH --mem 50G
#SBATCH --time=01:30:00
#SBATCH --partition=plgrid
#SBATCH --account=plgmpr25-cpu

module load python

set -x

python -u runner.py
