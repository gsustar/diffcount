import argparse
import inspect

from . import denoise_diffusion as dd
from . import deblur_diffusion as bd
from .respace import SpacedDiffusion, space_timesteps
from .unet import UNetModel
from .conditioning import Conditioner, ClassEmbedder, ImageConcatEmbedder

def denoise_diffusion_defaults():
	return dict(
		learn_sigma=False,
		diffusion_steps=1000,
		sigma_small=False,
		noise_schedule="linear",
		timestep_respacing="",
		use_kl=False,
		predict_xstart=False,
		rescale_timesteps=False,
		rescale_learned_sigmas=False,
	)

def deblur_diffsuion_defaults():
	return dict(
		blur_schedule="log",
		min_sigma=0.5,
		max_sigma=20.0,
		image_size=256,
		loss_type="l1",
		use_dct=False,
		delta=0.01,
	)

def model_and_diffusion_defaults():
	"""
 	Defaults for image training.
 	"""
	res = dict(
		in_channels=1,
		model_channels=32,
		out_channels=1,
		num_res_blocks=2,
		attention_resolutions="",
		dropout=0.0,
		channel_mult="1,2,2",
		conv_resample=True,
		dims=2,
		num_classes=None,
		use_checkpoint=False,
		num_heads=1,
		num_head_channels=64,
		num_heads_upsample=-1,
		use_scale_shift_norm=True,
		resblock_updown=False,
		transformer_depth=1,
		context_dim=None,
		disable_self_attentions = None,
		num_attention_blocks = None,
		disable_middle_self_attn = False,
		disable_middle_transformer = False,
		use_linear_in_transformer=False,
		spatial_transformer_attn_type="softmax-xformers",
		adm_in_channels=None,
		deblur_diffusion=True,
	)
	res.update(denoise_diffusion_defaults())
	res.update(deblur_diffsuion_defaults())
	return res


def create_model_and_diffusion(
	in_channels,
	model_channels,
	out_channels,
	num_res_blocks,
	attention_resolutions,
	dropout,
	channel_mult,
	conv_resample,
	dims,
	num_classes,
	use_checkpoint,
	num_heads,
	num_head_channels,
	num_heads_upsample,
	use_scale_shift_norm,
	resblock_updown,
	transformer_depth,
	context_dim,
	disable_self_attentions,
	num_attention_blocks,
	disable_middle_self_attn,
	disable_middle_transformer,
	use_linear_in_transformer,
	spatial_transformer_attn_type,
	adm_in_channels,
	deblur_diffusion,
	blur_schedule,
	min_sigma,
	max_sigma,
	image_size,
	loss_type,
	use_dct,
	delta,
	learn_sigma,
	diffusion_steps,
	noise_schedule,
	sigma_small,
	use_kl,
	predict_xstart,
	rescale_timesteps,
	rescale_learned_sigmas,
	timestep_respacing,
):
	model = create_model(
		in_channels,
		model_channels,
		out_channels,
		num_res_blocks,
		attention_resolutions,
		dropout,
		channel_mult,
		conv_resample,
		dims,
		num_classes,
		use_checkpoint,
		num_heads,
		num_head_channels,
		num_heads_upsample,
		use_scale_shift_norm,
		resblock_updown,
		transformer_depth,
		context_dim,
		disable_self_attentions,
		num_attention_blocks,
		disable_middle_self_attn,
		disable_middle_transformer,
		use_linear_in_transformer,
		spatial_transformer_attn_type,
		adm_in_channels,
		learn_sigma=learn_sigma,
	)
	if deblur_diffusion:
		diffusion = create_deblur_diffusion(
			steps=diffusion_steps,
			blur_schedule=blur_schedule,
			min_sigma=min_sigma,
			max_sigma=max_sigma,
			image_size=image_size,
			loss_type=loss_type,
			use_dct=use_dct,
			delta=delta,
		)
		return model, diffusion
	
	diffusion = create_denoise_diffusion(
		steps=diffusion_steps,
		learn_sigma=learn_sigma,
		sigma_small=sigma_small,
		noise_schedule=noise_schedule,
		use_kl=use_kl,
		predict_xstart=predict_xstart,
		rescale_timesteps=rescale_timesteps,
		rescale_learned_sigmas=rescale_learned_sigmas,
		timestep_respacing=timestep_respacing,
	)
	return model, diffusion


def create_model(
	in_channels,
	model_channels,
	out_channels,
	num_res_blocks,
	attention_resolutions="",
	dropout=0.0,
	channel_mult="1,2,3,4",
	conv_resample=True,
	dims=2,
	num_classes=None,
	use_checkpoint=False,
	num_heads=-1,
	num_head_channels=-1,
	num_heads_upsample=-1,
	use_scale_shift_norm=False,
	resblock_updown=False,
	transformer_depth=1,
	context_dim=None,
	disable_self_attentions=None,
	num_attention_blocks=None,
	disable_middle_self_attn=False,
	disable_middle_transformer=False,
	use_linear_in_transformer=False,
	spatial_transformer_attn_type="softmax",
	adm_in_channels=None,
	learn_sigma=False,
):
	channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))
	if attention_resolutions == "":
		attention_res = tuple()
	else:
		attention_res = tuple(int(res) for res in attention_resolutions.split(","))
	
	context_dim = int(context_dim) if context_dim is not None else None
	adm_in_channels = int(adm_in_channels) if adm_in_channels is not None else None

	return UNetModel(
		in_channels=in_channels,
		model_channels=model_channels,
		out_channels=(out_channels if not learn_sigma else 2 * out_channels),
		num_res_blocks=num_res_blocks,
		attention_resolutions=attention_res,
		dropout=dropout,
		channel_mult=channel_mult,
		conv_resample=conv_resample,
		dims=dims,
		num_classes=num_classes,
		use_checkpoint=use_checkpoint,
		num_heads=num_heads,
		num_head_channels=num_head_channels,
		num_heads_upsample=num_heads_upsample,
		use_scale_shift_norm=use_scale_shift_norm,
		resblock_updown=resblock_updown,
		transformer_depth=transformer_depth,
		context_dim=context_dim,
		disable_self_attentions=disable_self_attentions,
		num_attention_blocks=num_attention_blocks,
		disable_middle_self_attn=disable_middle_self_attn,
		disable_middle_transformer=disable_middle_transformer,
		use_linear_in_transformer=use_linear_in_transformer,
		spatial_transformer_attn_type=spatial_transformer_attn_type,
		adm_in_channels=adm_in_channels,
	)


# def create_model(
# 	image_size,
# 	num_channels,
# 	num_res_blocks,
# 	channel_mult="",
# 	learn_sigma=False,
# 	class_cond=False,
# 	use_checkpoint=False,
# 	attention_resolutions="16",
# 	num_heads=1,
# 	num_head_channels=-1,
# 	num_heads_upsample=-1,
# 	use_scale_shift_norm=False,
# 	dropout=0,
# 	resblock_updown=False,
# 	use_fp16=False,
# 	use_new_attention_order=False,
# ):
# 	if channel_mult == "":
# 		if image_size == 512:
# 			channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
# 		elif image_size == 256:
# 			channel_mult = (1, 1, 2, 2, 4, 4)
# 		elif image_size == 128:
# 			channel_mult = (1, 1, 2, 3, 4)
# 		elif image_size == 64:
# 			channel_mult = (1, 2, 3, 4)
# 		else:
# 			raise ValueError(f"unsupported image size: {image_size}")
# 	else:
# 		channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

# 	attention_ds = []
# 	for res in attention_resolutions.split(","):
# 		attention_ds.append(image_size // int(res))

# 	return UNetModel(
# 		image_size=image_size,
# 		in_channels=3,
# 		model_channels=num_channels,
# 		out_channels=(3 if not learn_sigma else 6),
# 		num_res_blocks=num_res_blocks,
# 		attention_resolutions=tuple(attention_ds),
# 		dropout=dropout,
# 		channel_mult=channel_mult,
# 		num_classes=(NUM_CLASSES if class_cond else None),
# 		use_checkpoint=use_checkpoint,
# 		use_fp16=use_fp16,
# 		num_heads=num_heads,
# 		num_head_channels=num_head_channels,
# 		num_heads_upsample=num_heads_upsample,
# 		use_scale_shift_norm=use_scale_shift_norm,
# 		resblock_updown=resblock_updown,
# 		use_new_attention_order=use_new_attention_order,
# 	)

def create_mnist_conditioner(
	embed_dim,
	is_trainable=True,
	add_sequence_dim=True,
):
	return Conditioner([
		ClassEmbedder(
			embed_dim=embed_dim, 
			is_trainable=is_trainable, 
			n_classes=10, 
			add_sequence_dim=add_sequence_dim
		)
	])


def create_fsc_conditioner():
	return Conditioner([
		ImageConcatEmbedder()
	])


def create_deblur_diffusion(
	*,
	steps=1000,
	blur_schedule="log",
	min_sigma=0.5,
	max_sigma=20.0,
	image_size=256,
	loss_type="l1",
	use_dct=True,
	delta=0.01,
):	
	blur_schedule = bd.get_named_blur_schedule(blur_schedule, steps, min_sigma, max_sigma)

	#TODO Add support for spaced diffusion
	return bd.DeblurDiffusion(
		blur_sigmas=blur_schedule,
		image_size=image_size,
		loss_type=loss_type,
		use_dct=use_dct,
		delta=delta,
	)


def create_denoise_diffusion(
	*,
	steps=1000,
	learn_sigma=False,
	sigma_small=False,
	noise_schedule="linear",
	use_kl=False,
	predict_xstart=False,
	rescale_timesteps=False,
	rescale_learned_sigmas=False,
	timestep_respacing="",
):
	betas = dd.get_named_beta_schedule(noise_schedule, steps)
	if use_kl:
		loss_type = dd.LossType.RESCALED_KL
	elif rescale_learned_sigmas:
		loss_type = dd.LossType.RESCALED_MSE
	else:
		loss_type = dd.LossType.MSE
	if not timestep_respacing:
		timestep_respacing = [steps]
	return SpacedDiffusion(
		use_timesteps=space_timesteps(steps, timestep_respacing),
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
	)


def add_dict_to_argparser(parser, default_dict):
	for k, v in default_dict.items():
		v_type = type(v)
		if v is None:
			v_type = str
		elif isinstance(v, bool):
			v_type = str2bool
		parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
	return {k: getattr(args, k) for k in keys}


def str2bool(v):
	"""
	https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
	"""
	if isinstance(v, bool):
		return v
	if v.lower() in ("yes", "true", "t", "y", "1"):
		return True
	elif v.lower() in ("no", "false", "f", "n", "0"):
		return False
	else:
		raise argparse.ArgumentTypeError("boolean value expected")
