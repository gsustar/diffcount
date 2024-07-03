#!/bin/sh

#SBATCH --job-name=run_train
#SBATCH --output=/d/hpc/projects/FRI/DL/gs1121/logs/R-%x.%j.out
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=32G
#SBATCH --partition=gpu
#SBATCH --constraint=h100
#SBATCH --signal=SIGTERM@300

export WANDB__SERVICE_WAIT=300
export HF_DATASETS_CACHE=/d/hpc/projects/FRI/DL/gs1121/.cache

srun --kill-on-bad-exit=1 \
	python train.py --config $1