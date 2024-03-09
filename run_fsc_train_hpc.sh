#!/bin/sh

#SBATCH --job-name=fsc_train
#SBATCH --output=/d/hpc/projects/FRI/DL/gs1121/logs/fsc_train.out
#SBATCH --error=/d/hpc/projects/FRI/DL/gs1121/logs/fsc_train.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=32G
#SBATCH --partition=gpu

export WANDB__SERVICE_WAIT=300

DATA_DIR=" \
	--data_dir /d/hpc/projects/FRI/DL/gs1121/FSC147_384_V2/ \
	--target_dirname gt_density_maps_ksize=3_sig=0.25_size=256x256
"
LOG_DIR=" \
	--log_dir /d/hpc/projects/FRI/DL/gs1121/logs \
"
DIFFUSION_FLAGS=" \
	--deblur_diffusion True \
	--diffusion_steps 100 \
	--learn_sigma False \
	--noise_schedule linear \
	--blur_schedule log \
	--min_sigma 0.5 \
	--max_sigma 20.0 \
	--use_dct True \
	--loss_type l1 \
	--delta 0.01 \
	--image_size 256 \
	--deblur_diffusion True \
"

MODEL_FLAGS=" \
	--in_channels 4 \
	--out_channels 1 \
	--model_channels 192 \
	--num_res_blocks 2 \
	--channel_mult 1,1,2,2,4,4 \
	--attention_resolutions 32,16,8 \
	--num_head_channels 64 \
	--spatial_transformer_attn_type softmax-xformers \
	--use_checkpoint True \
	--use_fp16 True \
"
TRAIN_FLAGS=" \
	--batch_size 4 \
	--lr 1e-4 \
	--save_interval 10000 \
	--overfit_single_batch True \
	--validation_interval 200 \
	--num_epochs 10000 \
"
WANDB_FLAGS=" \
	--wandb_mode online \
"
srun --kill-on-bad-exit=1 \
	python fsc_train.py $DATA_DIR $LOG_DIR $DIFFUSION_FLAGS $MODEL_FLAGS $TRAIN_FLAGS $WANDB_FLAGS
