DATA_DIR=" \
	--datadir ../FSC147_384_V2/ \
	--targetdir densitymaps/sig_0.5/size_256/
"
LOG_DIR=" \
	--logdir ../experiments/dummy/ \
"
DIFFUSION_FLAGS=" \
	--deblur_diffusion False \
	--diffusion_steps 100 \
	--learn_sigma False \
	--noise_schedule linear \
	--blur_schedule log \
	--min_sigma 0.5 \
	--max_sigma 20.0 \
	--use_dct True \
	--loss_type l1 \
	--delta 0.0 \
	--image_size 64 \
"
MODEL_FLAGS=" \
	--in_channels 4 \
	--out_channels 1 \
	--model_channels 64 \
	--num_res_blocks 2 \
	--channel_mult 1,2,3,4 \
	--attention_resolutions 32,16,8 \
	--num_head_channels 64 \
	--transformer_depth 1 \
	--spatial_transformer_attn_type softmax-xformers \
	--use_checkpoint True \
	--use_fp16 True \
"
TRAIN_FLAGS=" \
	--batch_size 4 \
	--lr 3e-4 \
	--save_interval 10000 \
	--validation_interval 200 \
	--num_epochs 10000 \
	--overfit_single_batch True \
	--dropout 0.1 \
"
WANDB_FLAGS=" \
	--wandb_mode disabled \
"
python fsc_train.py $DATA_DIR $LOG_DIR $DIFFUSION_FLAGS $MODEL_FLAGS $TRAIN_FLAGS $WANDB_FLAGS
