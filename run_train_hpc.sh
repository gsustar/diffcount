#!/bin/sh

#SBATCH --job-name=fsc_train
#SBATCH --output=/d/hpc/projects/FRI/DL/gs1121/logs/$(basename "$1" .yaml).out
#SBATCH --error=/d/hpc/projects/FRI/DL/gs1121/logs/$(basename "$1" .yaml).err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=32G
#SBATCH --partition=gpu

export WANDB__SERVICE_WAIT=300

srun --kill-on-bad-exit=1 \
	python train.py --config $1