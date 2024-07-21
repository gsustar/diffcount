import os
import yaml
import random
import numpy as np
import torch as th
import torch.nn as nn
import os.path as osp

from types import SimpleNamespace
from diffusers import AutoencoderKL

from . import denoise_diffusion as dd
from . import deblur_diffusion as bd
from . import conditioning as cond

from .datasets import FSC147, MNIST, load_data
from .respace import SpacedDiffusion, space_timesteps
from .unet import UNetModel
from .dit import DiT_models
from .nn import disabled_train


def assert_config(config):
	for att in ["model", "diffusion", "data", "log"]:
		assert hasattr(config, att), f"config missing attribute: {att}"

	assert config.model.params.learn_sigma == config.diffusion.params.learn_sigma
	assert not hasattr(config.data.dataset.params, "split")

	for embconf in config.conditioner.embedders:
		assert hasattr(embconf, "input_keys"), (
			"input_keys must be specified for each conditioner."
		)

	if config.diffusion.type == "Deblur":
		assert not hasattr(config.diffusion, "vae"), (
			"Deblur diffusion does not support latent encoding."
		)
		assert not hasattr(config.diffusion.params, "learn_sigma"), (
			"learn_sigma is not supported for Deblur diffusion."
		)


def create_model(model_config):
	if model_config.type == "UNet":
		model = create_unet_model(**vars(model_config.params))
	elif model_config.type == "DiT":
		model = create_dit_model(**vars(model_config.params))
	else:
		raise ValueError(f"Unsupported model type: {model_config.type}")
	return model


def create_diffusion(diffusion_config):
	if diffusion_config.type == "Deblur":
		diffusion = create_deblur_diffusion(**vars(diffusion_config.params))
	elif diffusion_config.type == "Denoise":
		diffusion = create_denoise_diffusion(**vars(diffusion_config.params),)
	else:
		raise ValueError(f"Unsupported diffusion type: {diffusion_config.type}")
	return diffusion


def create_data(data_config, train=True):
	#todo if keep original aspect ratio or image size, make sure the batch size is one
	splits = ['train', 'train_val', None] if train else [None, 'test_val', 'test']
	if data_config.dataset.name == "FSC147":
		train_dataset, val_dataset, test_dataset = (
			FSC147(
				**vars(data_config.dataset.params),
				split=split
			) if split else None for split in splits
		)
	elif data_config.dataset.name == "MNIST":
		train_dataset, val_dataset, test_dataset = (
			MNIST(
				**vars(data_config.dataset.params),
				split=split,
			) if split else None for split in splits
		)
	else:
		raise ValueError(f"Unknown dataset: {data_config.dataset}")
	
	if train:
		train_data = load_data(
			dataset=train_dataset,
			batch_size=data_config.dataloader.params.batch_size,
			shuffle=True,
			overfit_single_batch=data_config.dataloader.params.overfit_single_batch,
		)
		test_data = None
	else:
		test_data = load_data(
			dataset=test_dataset,
			batch_size=data_config.dataloader.params.batch_size,
			shuffle=False,
		)
		train_data = None
	
	val_data = load_data(
		dataset=val_dataset,
		batch_size=data_config.dataloader.params.batch_size,
		shuffle=False,
	) if not data_config.dataloader.params.overfit_single_batch else train_data

	return train_data, val_data, test_data


def create_conditioner(conditioner_config, train=True):
	embedders = []
	for embconf in conditioner_config.embedders:
		params = vars(embconf.params) if hasattr(embconf, "params") else {}
		emb = getattr(cond, embconf.type)(**params)
		emb.is_trainable = getattr(embconf, "is_trainable", False) if train else False
		emb.ucg_rate = getattr(embconf, "ucg_rate", 0.0) if train else 0.0
		emb.input_keys = embconf.input_keys
		embedders.append(emb)
	return cond.Conditioner(embedders)


def create_vae(vae_config, device):
	vae = None
	if vae_config is not None and vae_config.enabled:
		vae = AutoencoderKL.from_pretrained(vae_config.path)
		vae.train = disabled_train
		vae.eval()
		vae.to(device)
	return vae


def create_unet_model(
	input_size,
	in_channels,
	model_channels,
	out_channels,
	num_res_blocks,
	attention_resolutions,
	dropout,
	channel_mult,
	conv_resample,
	dims,
	context_dim,
	y_dim,
	use_checkpoint,
	num_heads,
	num_head_channels,
	num_heads_upsample,
	use_scale_shift_norm,
	resblock_updown,
	learn_sigma,
	learn_count,
	adalnzero,
	transformer_depth,
):
	if channel_mult is None:
		if input_size == 512:
			channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
		elif input_size == 256:
			channel_mult = (1, 1, 2, 2, 4, 4)
		elif input_size == 128:
			channel_mult = (1, 1, 2, 3, 4)
		elif input_size == 64:
			channel_mult = (1, 2, 3, 4)
		else:
			raise ValueError(f"unsupported input size: {input_size}")
	else:
		channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult)

	attention_ds = []
	for res in attention_resolutions:
		attention_ds.append(input_size // int(res))

	return UNetModel(
		# image_size=image_size,
		in_channels=in_channels,
		model_channels=model_channels,
		out_channels=(out_channels if not learn_sigma else 2 * out_channels),
		num_res_blocks=num_res_blocks,
		attention_resolutions=tuple(attention_ds),
		dropout=dropout,
		channel_mult=channel_mult,
		conv_resample=conv_resample,
		dims=dims,
		context_dim=context_dim,
		y_dim=y_dim,
		use_checkpoint=use_checkpoint,
		num_heads=num_heads,
		num_head_channels=num_head_channels,
		num_heads_upsample=num_heads_upsample,
		use_scale_shift_norm=use_scale_shift_norm,
		resblock_updown=resblock_updown,
		adalnzero=adalnzero,
		learn_count=learn_count,
		transformer_depth=transformer_depth,
	)


def create_dit_model(
	dit_size,
	input_size,
	in_channels,
	out_channels,
	context_dim,
	adm_in_channels,
	learn_sigma,
):
	model = DiT_models[dit_size](
		input_size=input_size,
		in_channels=in_channels,
		out_channels=out_channels,
		context_dim=context_dim,
		adm_in_channels=adm_in_channels,
		learn_sigma=learn_sigma,
	)
	return model


def create_deblur_diffusion(
	diffusion_steps,
	blur_schedule,
	min_sigma,
	max_sigma,
	image_size,
	loss_type,
	use_dct,
	delta,
):	
	blur_schedule = bd.get_named_blur_schedule(blur_schedule, diffusion_steps, min_sigma, max_sigma)
	return bd.DeblurDiffusion(
		blur_sigmas=blur_schedule,
		image_size=image_size,
		loss_type=loss_type,
		use_dct=use_dct,
		delta=delta,
	)


def create_denoise_diffusion(
	diffusion_steps,
	learn_sigma,
	sigma_small,
	noise_schedule,
	use_kl,
	predict_xstart,
	rescale_timesteps,
	rescale_learned_sigmas,
	timestep_respacing,
	lmbd_vlb,
	lmbd_xs_count,
	lmbd_cb_count,
	t_mse_weighting_scheme,
	t_xs_count_weighting_scheme,
	t_cb_count_weighting_scheme,
):
	betas = dd.get_named_beta_schedule(noise_schedule, diffusion_steps)
	if use_kl:
		loss_type = dd.LossType.RESCALED_KL
	elif rescale_learned_sigmas:
		loss_type = dd.LossType.RESCALED_MSE
	else:
		loss_type = dd.LossType.MSE
	if not timestep_respacing:
		timestep_respacing = [diffusion_steps]
	return SpacedDiffusion(
		use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
		betas=betas,
		model_mean_type=(
			dd.ModelMeanType.EPSILON if not predict_xstart else dd.ModelMeanType.START_X
		),
		model_var_type=(
			(
				dd.ModelVarType.FIXED_LARGE
				if not sigma_small
				else dd.ModelVarType.FIXED_SMALL
			)
			if not learn_sigma
			else dd.ModelVarType.LEARNED_RANGE
		),
		loss_type=loss_type,
		rescale_timesteps=rescale_timesteps,
		lmbd_vlb=lmbd_vlb,
		lmbd_xs_count=lmbd_xs_count,
		lmbd_cb_count=lmbd_cb_count,
		t_mse_weighting_scheme=t_mse_weighting_scheme,
		t_xs_count_weighting_scheme=t_xs_count_weighting_scheme,
		t_cb_count_weighting_scheme=t_cb_count_weighting_scheme,
	)


def seed_everything(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	th.manual_seed(seed)
	th.cuda.manual_seed(seed)
	th.backends.cudnn.deterministic = True
	th.backends.cudnn.benchmark = True


def dict_to_namespace(d):
	x = SimpleNamespace()
	_ = [setattr(x, k,
				 dict_to_namespace(v) if isinstance(v, dict)
				 else [dict_to_namespace(e) if isinstance(e, dict) else e for e in v] if isinstance(v, list)
				 else v) for k, v in d.items()]
	return x


def namespace_to_dict(namespace):
    return {
        k: namespace_to_dict(v) if isinstance(v, SimpleNamespace) 
		else [namespace_to_dict(e) if isinstance(e, SimpleNamespace) else e for e in v] if isinstance(v, list)
		else v
        for k, v in vars(namespace).items()
    }


def parse_config(configpath):
	with open(configpath, "r") as stream:
		try:
			return dict_to_namespace(
				yaml.safe_load(stream)
			)
		except yaml.YAMLError as e:
			print(e)
