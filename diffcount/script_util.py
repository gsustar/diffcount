import yaml
from types import SimpleNamespace
from . import denoise_diffusion as dd
from . import deblur_diffusion as bd
from .datasets import FSC147, MNIST, load_data
from .respace import SpacedDiffusion, space_timesteps
from .unet import UNetModel
from .dit import DiT_models
from .conditioning import Conditioner, ClassEmbedder, ImageConcatEmbedder, ViTExemplarEmbedder


def create_model_and_diffusion(
	model_config, 
	diffusion_config
):	
	if diffusion_config.type == "Deblur":
		learn_sigma = False
		# assert not hasattr(diffusion_config.params, "learn_sigma"), "learn_sigma is not supported for Deblur diffusion."
		diffusion = create_deblur_diffusion(
			**vars(diffusion_config.params)
		)
	elif diffusion_config.type == "Denoise":
		learn_sigma = diffusion_config.params.learn_sigma
		diffusion = create_denoise_diffusion(
			**vars(diffusion_config.params),
		)
	else:
		raise ValueError(f"Unsupported diffusion type: {diffusion_config.type}")

	# if hasattr(model_config.params, "learn_sigma"):
	# 	delattr(model_config.params, "learn_sigma")

	if model_config.type == "UNet":
		model = create_unet_model(
			learn_sigma=learn_sigma,
			**vars(model_config.params)
		)
	elif model_config.type == "DiT":
		model = create_dit_model(
			learn_sigma=learn_sigma,
			**vars(model_config.params),
		)
	else:
		raise ValueError(f"Unsupported model type: {model_config.type}")
	
	return model, diffusion


def create_data_and_conditioner(
	data_config,
	conditioner_config,
	train,
):
	splits = ['train', 'val', None] if train else [None, 'val', 'test']
	if data_config.dataset.name == "FSC147":
		train_dataset, val_dataset, test_dataset = (
			FSC147(
				**vars(data_config.dataset.params),
				split=split
			) if split else None for split in splits
		)
		create_conditioner_fn = create_fsc147_conditioner
	elif data_config.dataset.name == "MNIST":
		train_dataset, val_dataset, test_dataset = (
			MNIST(
				**vars(data_config.dataset.params),
				split=split,
			) if split else None for split in splits
		)
		create_conditioner_fn = create_mnist_conditioner
	else:
		raise ValueError(f"Unknown dataset: {data_config.dataset}")
	
	if conditioner_config is not None:
		conditioner = create_conditioner_fn(
			**vars(conditioner_config.params)
		)
	else:
		conditioner = create_empty_conditioner()

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

	return train_data, val_data, test_data, conditioner

def create_unet_model(
	image_size,
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
	adalnzero,
):
	if channel_mult is None:
		if image_size == 512:
			channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
		elif image_size == 256:
			channel_mult = (1, 1, 2, 2, 4, 4)
		elif image_size == 128:
			channel_mult = (1, 1, 2, 3, 4)
		elif image_size == 64:
			channel_mult = (1, 2, 3, 4)
		else:
			raise ValueError(f"unsupported image size: {image_size}")
	else:
		channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

	attention_ds = []
	for res in attention_resolutions:
		attention_ds.append(image_size // int(res))

	return UNetModel(
		image_size=image_size,
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
	)

# def create_unet_model(
# 	image_size,
# 	in_channels,
# 	model_channels,
# 	out_channels,
# 	num_res_blocks,
# 	attention_resolutions,
# 	dropout,
# 	channel_mult,
# 	conv_resample,
# 	dims,
# 	num_classes,
# 	use_checkpoint,
# 	num_heads,
# 	num_head_channels,
# 	num_heads_upsample,
# 	use_scale_shift_norm,
# 	resblock_updown,
# 	transformer_depth,
# 	context_dim,
# 	disable_self_attentions,
# 	num_attention_blocks,
# 	disable_middle_self_attn,
# 	disable_middle_transformer,
# 	use_linear_in_transformer,
# 	spatial_transformer_attn_type,
# 	adm_in_channels,
# 	learn_sigma,
# ):
# 	if channel_mult is None:
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


# 	attention_ds = []
# 	for res in attention_resolutions:
# 		attention_ds.append(image_size // int(res))
	
# 	if context_dim is not None:
# 		context_dim = int(context_dim)

# 	if adm_in_channels is not None:
# 		adm_in_channels = int(adm_in_channels)

# 	return UNetModel(
# 		in_channels=in_channels,
# 		model_channels=model_channels,
# 		out_channels=(out_channels if not learn_sigma else 2 * out_channels),
# 		num_res_blocks=num_res_blocks,
# 		attention_resolutions=tuple(attention_ds),
# 		dropout=dropout,
# 		channel_mult=channel_mult,
# 		conv_resample=conv_resample,
# 		dims=dims,
# 		num_classes=num_classes,
# 		use_checkpoint=use_checkpoint,
# 		num_heads=num_heads,
# 		num_head_channels=num_head_channels,
# 		num_heads_upsample=num_heads_upsample,
# 		use_scale_shift_norm=use_scale_shift_norm,
# 		resblock_updown=resblock_updown,
# 		transformer_depth=transformer_depth,
# 		context_dim=context_dim,
# 		disable_self_attentions=disable_self_attentions,
# 		num_attention_blocks=num_attention_blocks,
# 		disable_middle_self_attn=disable_middle_self_attn,
# 		disable_middle_transformer=disable_middle_transformer,
# 		use_linear_in_transformer=use_linear_in_transformer,
# 		spatial_transformer_attn_type=spatial_transformer_attn_type,
# 		adm_in_channels=adm_in_channels,
# 	)


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


def create_empty_conditioner():
	return Conditioner([])


def create_mnist_conditioner(
	embed_dim,
	is_trainable,
	add_sequence_dim,
):
	return Conditioner([
		ClassEmbedder(
			embed_dim=embed_dim, 
			is_trainable=is_trainable, 
			n_classes=10, 
			add_sequence_dim=add_sequence_dim
		)
	])


def create_fsc147_conditioner(
	image_size,
	out_channels,
	vit_size,
	freeze_backbone,
	remove_sequence_dim,
	is_trainable,
):
	return Conditioner([
		ImageConcatEmbedder(),
		ViTExemplarEmbedder(
			image_size=image_size,
			out_channels=out_channels,
			vit_size=vit_size,
			is_trainable=is_trainable,
			remove_sequence_dim=remove_sequence_dim,
			freeze_backbone=freeze_backbone,
		)
	])


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

	#TODO Add support for spaced diffusion
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
	)


def dict_to_namespace(d):
	x = SimpleNamespace()
	_ = [setattr(x, k,
				 dict_to_namespace(v) if isinstance(v, dict)
				 else v) for k, v in d.items()]
	return x


def namespace_to_dict(namespace):
    return {
        k: namespace_to_dict(v) if isinstance(v, SimpleNamespace) else v
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
