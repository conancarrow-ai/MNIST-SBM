#!/bin/bash
#SBATCH --job-name=train_SBM
#SBATCH --output=train_SBM.log
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --mem=10gb
#SBATCH --gres=gpu:3090:1

cd /mnt/data0/shared/colin/MNIST-SBM
source venv/bin/activate
python main.py

