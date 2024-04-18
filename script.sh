#!/bin/bash
#SBATCH --partition=A40medium
#SBATCH --gpus=1
#SBATCH --ntasks=1

python main.py