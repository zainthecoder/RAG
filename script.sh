#!/bin/bash
#SBATCH --partition=A100medium
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH -o "/home/s28zabed/RAG/output.out"

python main.py