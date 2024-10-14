#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=5:00:00

cd /gpfs/home5/pgort/


apptainer pull docker://pgort/mscthesis:v2