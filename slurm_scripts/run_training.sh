#!/bin/bash

#SBATCH --account=ucb722_asc1
#SBATCH --partition=aa100
#SBATCH --job-name=model_prototyping
#SBATCH --output=/home/rost5691/slurm_job_logs/model_prototyping%j.out
#SBATCH --time=24:00:00
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rost5691@colorado.edu
#SBATCH --gres=gpu:1


module purge
module load anaconda/2023.09
conda activate transportation-models-env
cd ../src/transportation_models/scripts
python model_prototyping.py
