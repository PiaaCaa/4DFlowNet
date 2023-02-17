#!/bin/bash
#SBATCH --gpus 2
#SBATCH -N 1
#SBATCH -t 2:00:00
#SBATCH --mail-type ALL
#SBATCH --output /path-to-save-output/output.out
#SBATCH --error patch-to-save-error/output.err

cd patch-to-run-file
singularity run –nv tensorflow_2.6.1-gpu.sif python myfile.py













