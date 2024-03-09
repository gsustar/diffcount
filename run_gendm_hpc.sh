#!/bin/sh

#SBATCH --job-name=gendm
#SBATCH --output=/d/hpc/projects/FRI/DL/gs1121/logs/gendm.out
#SBATCH --error=/d/hpc/projects/FRI/DL/gs1121/logs/gendm.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16

ROOT="/d/hpc/projects/FRI/DL/gs1121/FSC147_384_V2"
SIGMA=0.5
SIZE=256

srun --kill-on-bad-exit=1 \
	python gendm.py --root $ROOT --sigma $SIGMA --size $SIZE