import argparse
import datetime
import numpy as np
import pprint

import os.path as osp
import torch as th
import torch.nn.functional as F

# from diffcount import dist_util, logger
from diffcount import logger
from diffcount.datasets import MNIST, load_data
from diffcount.resample import create_named_schedule_sampler
from diffcount.script_util import (
	model_and_diffusion_defaults,
	create_model_and_diffusion,
	create_mnist_conditioner,
	args_to_dict,
	add_dict_to_argparser,
)
from diffcount.train_util import TrainLoop
from diffcount.deblur_diffusion import DeblurDiffusion

def main():
	try:
		args = create_argparser().parse_args()
		now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
		args.logdir = osp.join(
			args.logdir,
			"mnist",
			now
		) if args.logdir else None
		
		pprint.pprint(vars(args))
		logger.configure(
			dir=args.logdir, 
			format_strs=['stdout', 'log', 'wandb'],
			wandb_kwargs=dict(
				project="diffcount",
				name=now,
				group="mnist",
				config=vars(args),
				mode=args.wandb_mode,
			)
		)

		logger.log("creating model...")
		model, diffusion = create_model_and_diffusion(
			**args_to_dict(args, model_and_diffusion_defaults().keys())
		)

		dev = "cuda" if th.cuda.is_available() else "cpu"
		logger.log(f"moving model to '{dev}'...")
		model.to(dev)
		schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

		logger.log("creating data loader...")
		train_data = load_data(
			dataset=MNIST(root=args.datadir, split='train'),
			batch_size=args.batch_size,
			shuffle=True,
			overfit_single_batch=args.overfit_single_batch,
		)
		val_data = load_data(
			dataset=MNIST(root=args.datadir, split='val'),
			batch_size=args.batch_size,
			shuffle=False,
		) if not args.overfit_single_batch else train_data

		logger.log("creating conditioner...")
		conditioner = create_mnist_conditioner(
			embed_dim=256, 
			is_trainable=True,
		).to(dev) if args.class_cond else None

		if isinstance(diffusion, DeblurDiffusion):
			diffusion.set_init_sample_set(train_data)

		logger.log("training...")
		TrainLoop(
			model=model,
			diffusion=diffusion,
			data=train_data,
			val_data=val_data,
			conditioner=conditioner,
			batch_size=args.batch_size,
			lr=args.lr,
			ema_rate=args.ema_rate,
			log_interval=args.log_interval,
			save_interval=args.save_interval,
			validation_interval=args.validation_interval,
			resume_checkpoint=args.resume_checkpoint,
			use_fp16=args.use_fp16,
			schedule_sampler=schedule_sampler,
			weight_decay=args.weight_decay,
			num_epochs=args.num_epochs,
			device=dev,
			grad_clip=args.grad_clip,
		).run_loop()
	except Exception as e:
		raise e
	finally:
		logger.log("closing...")
		logger.close()


def create_argparser():
	defaults = dict(
		datadir="",
		logdir="",
		schedule_sampler="uniform",
		lr=1e-4,
		weight_decay=0.0,
		batch_size=4,
		ema_rate=0.9999,
		log_interval=10,
		save_interval=50,
		validation_interval=10,
		resume_checkpoint="",
		use_fp16=False,
		overfit_single_batch=False,
		wandb_mode="online",
		num_epochs=100,
		class_cond=False,
		grad_clip=0.0,
	)
	defaults.update(model_and_diffusion_defaults())
	parser = argparse.ArgumentParser()
	add_dict_to_argparser(parser, defaults)
	return parser


if __name__ == "__main__":
	main()
