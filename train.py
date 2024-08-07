import argparse
import datetime
import pprint
import shutil

import os.path as osp
import torch as th

from diffcount import logger
from diffcount.resample import create_named_schedule_sampler
from diffcount.script_util import (
	create_model,
	create_diffusion,
	create_data,
	create_conditioner,
	create_vae,
	create_cachedir,
	namespace_to_dict,
	parse_config,
	assert_config,
	seed_everything,
)
from diffcount.train_util import TrainLoop
from diffcount.deblur_diffusion import DeblurDiffusion

def main():

	args = parse_args()
	config = parse_config(args.config)
	assert_config(config)
	now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

	if config.train.seed is not None:
		seed_everything(config.train.seed)

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

	logger.log("creating model...")
	model = create_model(config.model)

	dev = "cuda" if th.cuda.is_available() else "cpu"
	logger.log(f"moving model to '{dev}'...")
	model.to(dev)

	logger.log("creating diffusion...")
	diffusion = create_diffusion(config.diffusion)
	schedule_sampler = create_named_schedule_sampler(config.diffusion.schedule_sampler, diffusion)

	logger.log("creating VAE...")
	vae = create_vae(
		getattr(config, "vae", None), device=dev
	)

	logger.log("creating data...")
	cachedir = create_cachedir(config.data.dataset.params.datadir, now)
	train_data, val_data, _ = create_data(config.data, train=True, cachedir=cachedir)

	logger.log("creating conditioner...")
	conditioner = create_conditioner(
		getattr(config, "conditioner", []),
		cachedir=cachedir,
		train=True,
	)
	conditioner.to(dev)

	if isinstance(diffusion, DeblurDiffusion):	# todo remove this
		diffusion.set_init_sample_set(train_data)

	logger.log("training...")
	TrainLoop(
		input_size=config.model.params.input_size,
		model=model,
		diffusion=diffusion,
		data=train_data,
		val_data=val_data,
		conditioner=conditioner,
		vae=vae,
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
		lr_scheduler=config.train.lr_scheduler,
		cachedir=cachedir,
	).run_loop()


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", type=str)
	return parser.parse_args()


if __name__ == "__main__":
	main()
