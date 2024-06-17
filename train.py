import argparse
import datetime
import pprint
import shutil

import os.path as osp
import torch as th

from diffcount import logger
from diffcount.resample import create_named_schedule_sampler
from diffcount.script_util import (
	create_model_and_diffusion,
	create_data_and_conditioner,
	namespace_to_dict,
	parse_config,
)
from diffcount.train_util import TrainLoop
from diffcount.deblur_diffusion import DeblurDiffusion

def main():

	args = parse_args()
	config = parse_config(args.config)
	now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

	config.log.logdir = osp.join(
		config.log.logdir,
		config.data.dataset.name,
		config.name,
		now
	) if config.log.logdir else None

	logger.configure(
		dir=config.log.logdir, 
		format_strs=['stdout', 'log', 'wandb'],
		wandb_kwargs=dict(
			project="diffcount",
			name=config.name,
			group=config.data.dataset.name,
			config=namespace_to_dict(config),
			mode=config.log.wandb_mode,
		),
		log_suffix='_train'
	)
	shutil.copy(args.config, osp.join(logger.get_dir(), "config.yaml"))

	logger.log(pprint.pformat(config))
	logger.log("creating model and diffusion...")
	model, diffusion = create_model_and_diffusion(
		config.model, 
		config.diffusion
	)

	dev = "cuda" if th.cuda.is_available() else "cpu"
	logger.log(f"moving model to '{dev}'...")
	model.to(dev)
	schedule_sampler = create_named_schedule_sampler(config.diffusion.schedule_sampler, diffusion)

	logger.log("creating data loader and conditioner...")
	config_conditioner = None
	if hasattr(config, "conditioner"):
		assert config.conditioner.type == config.data.dataset.name, \
				"conditioner type must match dataset"
		config_conditioner = config.conditioner
	train_data, val_data, _, conditioner = create_data_and_conditioner(
		config.data, 
		config_conditioner,
		train=True
	)
	conditioner.to(dev)

	if isinstance(diffusion, DeblurDiffusion):	# todo remove this
		diffusion.set_init_sample_set(train_data)

	logger.log("training...")
	TrainLoop(
		model=model,
		diffusion=diffusion,
		data=train_data,
		val_data=val_data,
		conditioner=conditioner,
		batch_size=config.data.dataloader.params.batch_size,
		lr=config.train.lr,
		ema_rate=config.train.ema_rate,
		log_interval=config.log.log_interval,
		save_interval=config.log.save_interval,
		validation_interval=config.train.validation_interval,
		resume_checkpoint=config.train.resume_checkpoint,
		use_fp16=config.train.use_fp16,
		schedule_sampler=schedule_sampler,
		weight_decay=config.train.weight_decay,
		num_epochs=config.train.num_epochs,
		device=dev,
		grad_clip=config.train.grad_clip,
		lr_scheduler=config.train.lr_scheduler
	).run_loop()


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", type=str)
	return parser.parse_args()


if __name__ == "__main__":
	main()
