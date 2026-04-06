#!/bin/bash
#SBATCH --job-name=generate
#SBATCH --output=generate.log
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=2
#SBATCH --time=10:00:00
#SBATCH --mem=10gb
#SBATCH --gres=gpu:3090:1

cd /mnt/data0/shared/colin/MNIST-SBM
source venv/bin/activate
python generate.py

