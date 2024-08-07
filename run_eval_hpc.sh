#!/bin/sh

#SBATCH --job-name=run_eval
#SBATCH --output=/d/hpc/projects/FRI/DL/gs1121/logs/R-%x.%j.out
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=32G
#SBATCH --partition=gpu

export HF_HOME=/d/hpc/projects/FRI/DL/gs1121/.cache

srun --kill-on-bad-exit=1 \
	python eval.py --expdir $1 --checkpoint $2 --batch_size 1 --use_fp16