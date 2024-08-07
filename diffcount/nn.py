"""
Various utilities for neural networks.
"""

import math

import torch as th
import torch.nn as nn
from inspect import isfunction


def disabled_train(self, mode=True):
	"""Overwrite model.train with this function to make sure train/eval mode
	does not change anymore."""
	return self


def count_params(model, verbose=False):
	total_params = sum(p.numel() for p in model.parameters())
	if verbose:
		print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
	return total_params


def expand_dims_like(x, y):
    while x.dim() != y.dim():
        x = x.unsqueeze(-1)
    return x


def exists(val):
	return val is not None


# def uniq(arr):
# 	return {el: True for el in arr}.keys()


def default(val, d):
	if exists(val):
		return val
	return d() if isfunction(d) else d


# def max_neg_value(t):
# 	return -th.finfo(t.dtype).max


# def init_(tensor):
# 	dim = tensor.shape[-1]
# 	std = 1 / math.sqrt(dim)
# 	tensor.uniform_(-std, std)
# 	return tensor


# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
# class SiLU(nn.Module):
# 	def forward(self, x):
# 		return x * th.sigmoid(x)


# class GroupNorm32(nn.GroupNorm):
# 	def forward(self, x):
# 		return super().forward(x.float()).type(x.dtype)


# class LayerNorm(nn.LayerNorm):
# 	def forward(self, x):
# 		return super().forward(x.float()).type(x.dtype)


def conv_nd(dims, *args, **kwargs):
	"""
	Create a 1D, 2D, or 3D convolution module.
	"""
	if dims == 1:
		return nn.Conv1d(*args, **kwargs)
	elif dims == 2:
		return nn.Conv2d(*args, **kwargs)
	elif dims == 3:
		return nn.Conv3d(*args, **kwargs)
	raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
	"""
	Create a linear module.
	"""
	return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
	"""
	Create a 1D, 2D, or 3D average pooling module.
	"""
	if dims == 1:
		return nn.AvgPool1d(*args, **kwargs)
	elif dims == 2:
		return nn.AvgPool2d(*args, **kwargs)
	elif dims == 3:
		return nn.AvgPool3d(*args, **kwargs)
	raise ValueError(f"unsupported dimensions: {dims}")


# def update_ema(target_params, source_params, rate=0.99):
# 	"""
# 	Update target parameters to be closer to those of source parameters using
# 	an exponential moving average.

# 	:param target_params: the target parameter sequence.
# 	:param source_params: the source parameter sequence.
# 	:param rate: the EMA rate (closer to 1 means slower).
# 	"""
# 	for targ, src in zip(target_params, source_params):
# 		targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def zero_module(module):
	"""
	Zero out the parameters of a module and return it.
	"""
	for p in module.parameters():
		p.detach().zero_()
	return module


def scale_module(module, scale):
	"""
	Scale the parameters of a module and return it.
	"""
	for p in module.parameters():
		p.detach().mul_(scale)
	return module


def mean_flat(tensor):
	"""
	Take the mean over all non-batch dimensions.
	"""
	return tensor.mean(dim=list(range(1, len(tensor.shape))))


# def group_normalization(channels):
# 	"""
# 	Make a standard normalization layer.

# 	:param channels: number of input channels.
# 	:return: an nn.Module for normalization.
# 	"""
# 	return GroupNorm32(32, channels)


# def layer_normalization(dim):
# 	return LayerNorm(dim)

@th.no_grad
def possibly_vae_encode(x, vae=None):
	if vae is not None:
		_, ch, _, _ = x.shape
		if ch == 1:
			x = x.expand(-1, 3, -1, -1)
		return vae.encode(x).latent_dist.sample() * vae.config.scaling_factor
	return x

@th.no_grad
def possibly_vae_decode(z, vae=None, clip_decoded=False):
	if vae is not None:
		z = vae.decode(z / vae.config.scaling_factor).sample
	if clip_decoded:
		z = z.clamp(-1, 1)
	return z


# @th.no_grad
# def encode(batch, cond, vae):
# 	ENCODE_KEYS = ["img"]
# 	batch = possibly_vae_encode(batch, vae)
# 	for k in ENCODE_KEYS:
# 		if k in cond:
# 			new_k = f"z_{k}"
# 			cond[new_k] = possibly_vae_encode(cond[k], vae)
# 	return batch, cond


def torch_to(x, *args, **kwargs):
	if isinstance(x, th.Tensor):
		return x.to(*args, **kwargs)
	if isinstance(x, dict):
		return {k: torch_to(v, *args, **kwargs) for k, v in x.items()}
	if isinstance(x, (list, tuple)):
		return [torch_to(v, *args, **kwargs) for v in x]
	return x


def timestep_embedding(timesteps, dim, max_period=10000):
	"""
	Create sinusoidal timestep embeddings.

	:param timesteps: a 1-D Tensor of N indices, one per batch element.
					  These may be fractional.
	:param dim: the dimension of the output.
	:param max_period: controls the minimum frequency of the embeddings.
	:return: an [N x dim] Tensor of positional embeddings.
	"""
	half = dim // 2
	freqs = th.exp(
		-math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
	).to(device=timesteps.device)
	args = timesteps[:, None].float() * freqs[None]
	embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
	if dim % 2:
		embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
	return embedding