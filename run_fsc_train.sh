DATA_DIR=" \
	--data_dir /mnt/c/users/grega/faks/mag/FSC147_384_V2/ \
	--target_dirname gt_density_maps_ksize=5_sig=0.5_size=256x256
"
LOG_DIR=" \
	--log_dir experiments/dummy/ \
"
DIFFUSION_FLAGS=" \
	--diffusion_steps 200 \
	--learn_sigma False \
	--noise_schedule linear
"
MODEL_FLAGS=" \
	--in_channels 1 \
	--out_channels 1 \
	--model_channels 64 \
	--num_res_blocks 2 \
	--channel_mult 1,2,2 \
	--attention_resolutions 2 \
	--num_head_channels 32 \
	--transformer_depth 1 \
	--spatial_transformer_attn_type softmax-xformers \
	--use_checkpoint True \
	--use_fp16 True \
"
TRAIN_FLAGS=" \
	--batch_size 1 \
	--lr 2e-4 \
	--save_interval 1000 \
	--validation_interval 200 \
	--num_epochs 10000 \
	--overfit_single_batch True \
"
WANDB_FLAGS=" \
	--wandb_mode disabled \
"
python fsc_train.py $DATA_DIR $LOG_DIR $DIFFUSION_FLAGS $MODEL_FLAGS $TRAIN_FLAGS $WANDB_FLAGS
