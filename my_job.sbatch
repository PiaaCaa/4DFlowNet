#!/bin/bash
#SBATCH --gpus 1
#SBATCH -N 1
#SBATCH -t 01:15:00
#SBATCH --mail-type ALL
#SBATCH --mail-user pcallmer@kth.se
#SBATCH --output /proj/multipress/users/x_piaca/Temporal4DFlowNet/outputfiles/output.out
#SBATCH --error /proj/multipress/users/x_piaca/Temporal4DFlowNet/outputfiles/output.err

cd  /proj/multipress/users/x_piaca/
singularity run --nv tensorflow_2.6.1-gpu.sif python Temporal4DFlowNet/src/predictor_temporal_all_axis.py