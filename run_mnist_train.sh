DATA_DIR=" \
	--data_dir .data/mnist \
"
LOG_DIR=" \
	--log_dir experiments/dummy/
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
	--image_size 28 \
	--deblur_diffusion True \
"
MODEL_FLAGS=" \
	--in_channels 1 \
	--out_channels 1 \
	--model_channels 32 \
	--num_res_blocks 2 \
	--channel_mult 1,2,2 \
	--attention_resolutions 2 \
	--num_head_channels 1 \
	--transformer_depth 1 \
	--spatial_transformer_attn_type softmax-xformers \
	--use_checkpoint True \
	--use_fp16 False \
	--class_cond True \
	--context_dim 256 \
	--dropout 0.0 \
"
TRAIN_FLAGS=" \
	--batch_size 4 \
	--lr 2e-4 \
	--grad_clip 1.0 \
	--save_interval 10000 \
	--overfit_single_batch True \
	--validation_interval 500 \
	--num_epochs 10000 \
	--warmup 0
"
WANDB_FLAGS=" \
	--wandb_mode online \
"
python mnist_train.py $DATA_DIR $LOG_DIR $DIFFUSION_FLAGS $MODEL_FLAGS $TRAIN_FLAGS $WANDB_FLAGS
