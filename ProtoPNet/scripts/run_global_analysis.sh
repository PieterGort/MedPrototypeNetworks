#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=1:00:00

cd $HOME/vscodethesis/ProtoPNet
mkdir wandb/$SLURM_JOBID

srun apptainer exec --nv /gpfs/home5/pgort/mscthesis_v2.sif /bin/bash /gpfs/home5/pgort/run_container_global_analysis.sh

#HOME PROJECT DIR:
#/gpfs/home5/pgort

# PROJECT SPACE AVAILABLE AT:
# cd /projects/0/prjs0976
