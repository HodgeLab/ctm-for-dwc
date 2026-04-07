#!/bin/bash

#SBATCH --account=ucb722_asc1
#SBATCH --partition=amilan
#SBATCH --job-name=model_prototyping
#SBATCH --output=/home/rost5691/slurm_job_logs/model_prototyping%j.out
#SBATCH --time=12:00:00
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rost5691@colorado.edu


module purge
module load anaconda/2023.09
conda activate transportation-models-env
cd ../src/transportation_models/scripts
python model_prototyping.py --PhysWeight 0.0
