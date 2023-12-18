#!/bin/bash
#SBATCH --gpus 1
#SBATCH -N 1
#SBATCH -t 00:01:00
#SBATCH --mail-type ALL
#SBATCH --mail-user pia.callmer@ki.se
#SBATCH --output /proj/multipress/users/x_piaca/Temporal4DFlowNet/outputfiles/output_predict.out
#SBATCH --error /proj/multipress/users/x_piaca/Temporal4DFlowNet/outputfiles/output_predict.err

cd  /proj/multipress/users/x_piaca/
singularity run --nv tensorflow_2.6.1-gpu.sif python Temporal4DFlowNet/src/test.py