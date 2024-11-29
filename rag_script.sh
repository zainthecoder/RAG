#!/bin/bash
#SBATCH --partition=A40short
#SBATCH --time=0:10:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --output=/home/s28zabed/RAG/output_rag.out

# Activate the environment
source myenv/bin/activate

# Run your script
python script/Experiment_script_new_pipeline.py