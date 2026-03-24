#!/bin/bash

#SBATCH --account=ucb722_asc1
#SBATCH --partition=amilan
#SBATCH --job-name=dataset_preprocessing
#SBATCH --output=/home/rost5691/slurm_job_logs/dataset_preprocessing%j.out
#SBATCH --time=16:00:00
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rost5691@colorado.edu

module purge
module load anaconda/2023.09
conda activate transportation-models-env
cd ../src/transportation_models/utils
python data_processing.py
