from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange

from .conditioning import RoIAlignExemplarEmbedder
from .attention import BasicTransformerBlock
from .dit import DiTBlock
from .count_utils import CountingBranch
from .nn import (
	conv_nd,
	linear,
	avg_pool_nd,
	zero_module,
	timestep_embedding,
)


class TimestepBlock(nn.Module):
	"""
	Any module where forward() takes timestep embeddings as a second argument.
	"""

	@abstractmethod
	def forward(self, x, emb):
		"""
		Apply the module to `x` given `emb` timestep embeddings.
		"""


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
	"""
	A sequential module that passes timestep embeddings to the children that
	support it as an extra input.
	"""
	# def forward(self, x, emb, y=None, context=None):
	def forward(self, x, emb, context=None, bboxes=None):
		for layer in self:
			if isinstance(layer, TimestepBlock):
				x = layer(x, emb)
			elif isinstance(layer, SpatialTransformer):
				# x = layer(x, y=y, context=context)
				x = layer(x, y=emb, context=context, bboxes=bboxes)
			else:
				x = layer(x)
		return x


class Upsample(nn.Module):
	"""
	An upsampling layer with an optional convolution.

	:param channels: channels in the inputs and outputs.
	:param use_conv: a bool determining if a convolution is applied.
	:param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
				 upsampling occurs in the inner-two dimensions.
	"""

	def __init__(self, channels, use_conv, dims=2, out_channels=None):
		super().__init__()
		self.channels = channels
		self.out_channels = out_channels or channels
		self.use_conv = use_conv
		self.dims = dims
		if use_conv:
			self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

	def forward(self, x):
		assert x.shape[1] == self.channels
		if self.dims == 3:
			x = F.interpolate(
				x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
			)
		else:
			x = F.interpolate(x, scale_factor=2, mode="nearest")
		if self.use_conv:
			x = self.conv(x)
		return x


class Downsample(nn.Module):
	"""
	A downsampling layer with an optional convolution.

	:param channels: channels in the inputs and outputs.
	:param use_conv: a bool determining if a convolution is applied.
	:param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
				 downsampling occurs in the inner-two dimensions.
	"""

	def __init__(self, channels, use_conv, dims=2, out_channels=None):
		super().__init__()
		self.channels = channels
		self.out_channels = out_channels or channels
		self.use_conv = use_conv
		self.dims = dims
		stride = 2 if dims != 3 else (1, 2, 2)
		if use_conv:
			self.op = conv_nd(
				dims, self.channels, self.out_channels, 3, stride=stride, padding=1
			)
		else:
			assert self.channels == self.out_channels
			self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

	def forward(self, x):
		assert x.shape[1] == self.channels
		return self.op(x)


class ResBlock(TimestepBlock):
	"""
	A residual block that can optionally change the number of channels.

	:param channels: the number of input channels.
	:param emb_channels: the number of timestep embedding channels.
	:param dropout: the rate of dropout.
	:param out_channels: if specified, the number of out channels.
	:param use_conv: if True and out_channels is specified, use a spatial
		convolution instead of a smaller 1x1 convolution to change the
		channels in the skip connection.
	:param dims: determines if the signal is 1D, 2D, or 3D.
	:param use_checkpoint: if True, use gradient checkpointing on this module.
	:param up: if True, use this block for upsampling.
	:param down: if True, use this block for downsampling.
	"""

	def __init__(
		self,
		channels,
		emb_channels,
		dropout,
		out_channels=None,
		use_conv=False,
		use_scale_shift_norm=False,
		dims=2,
		use_checkpoint=False,
		up=False,
		down=False,
	):
		super().__init__()
		self.channels = channels
		self.emb_channels = emb_channels
		self.dropout = dropout
		self.out_channels = out_channels or channels
		self.use_conv = use_conv
		self.use_checkpoint = use_checkpoint
		self.use_scale_shift_norm = use_scale_shift_norm

		self.in_layers = nn.Sequential(
			nn.GroupNorm(32, channels),
			nn.SiLU(),
			conv_nd(dims, channels, self.out_channels, 3, padding=1),
		)

		self.updown = up or down

		if up:
			self.h_upd = Upsample(channels, False, dims)
			self.x_upd = Upsample(channels, False, dims)
		elif down:
			self.h_upd = Downsample(channels, False, dims)
			self.x_upd = Downsample(channels, False, dims)
		else:
			self.h_upd = self.x_upd = nn.Identity()

		self.emb_layers = nn.Sequential(
			nn.SiLU(),
			linear(
				emb_channels,
				2 * self.out_channels if use_scale_shift_norm else self.out_channels,
			),
		)
		self.out_layers = nn.Sequential(
			nn.GroupNorm(32, self.out_channels),
			nn.SiLU(),
			nn.Dropout(p=dropout),
			zero_module(
				conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
			),
		)

		if self.out_channels == channels:
			self.skip_connection = nn.Identity()
		elif use_conv:
			self.skip_connection = conv_nd(
				dims, channels, self.out_channels, 3, padding=1
			)
		else:
			self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

	def forward(self, x, emb):
		"""
		Apply the block to a Tensor, conditioned on a timestep embedding.

		:param x: an [N x C x ...] Tensor of features.
		:param emb: an [N x emb_channels] Tensor of timestep embeddings.
		:return: an [N x C x ...] Tensor of outputs.
		"""
		if self.use_checkpoint:
			return checkpoint(self._forward, x, emb, use_reentrant=False)
		else:
			return self._forward(x, emb)
		# return checkpoint(
		# 	self._forward, (x, emb), self.parameters(), self.use_checkpoint
		# )

	def _forward(self, x, emb):
		if self.updown:
			in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
			h = in_rest(x)
			h = self.h_upd(h)
			x = self.x_upd(x)
			h = in_conv(h)
		else:
			h = self.in_layers(x)
		emb_out = self.emb_layers(emb)
		while len(emb_out.shape) < len(h.shape):
			emb_out = emb_out[..., None]
		if self.use_scale_shift_norm:
			out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
			scale, shift = th.chunk(emb_out, 2, dim=1)
			h = out_norm(h) * (1 + scale) + shift
			h = out_rest(h)
		else:
			h = h + emb_out
			h = self.out_layers(h)
		return self.skip_connection(x) + h


class SpatialTransformer(nn.Module):
	"""
	Transformer block for image-like data.
	First, project the input (aka embedding)
	and reshape to b, t, d.
	Then apply standard transformer action.
	Finally, reshape to image
	"""
	def __init__(
		self,
		in_channels,
		n_heads,
		d_head,
		time_embed_dim,
		depth=1,
		dropout=0.,
		context_dim=None,
		adalnzero=False,
		bbox_embed=None,
		bbox_y_embed=None,
	):
		super().__init__()

		if d_head == -1:
			d_head = in_channels // n_heads
		else:
			assert in_channels % d_head == 0, (
				f"q,k,v channels {in_channels} is not divisible by num_head_channels {d_head}"
			)
			n_heads = in_channels // d_head
	
		self.adalnzero = adalnzero
		self.in_channels = in_channels
		inner_dim = n_heads * d_head
		self.norm = nn.GroupNorm(32, in_channels, eps=1e-6, affine=True)
		self.bbox_embed = bbox_embed
		self.bbox_y_embed = bbox_y_embed

		self.proj_in = nn.Conv2d(in_channels,
								 inner_dim,
								 kernel_size=1,
								 stride=1,
								 padding=0)

		self.transformer_blocks = nn.ModuleList(
			[
				BasicTransformerBlock(
					inner_dim, 
					n_heads, 
					d_head, 
					dropout=dropout, 
					context_dim=context_dim
				) if not adalnzero else 
				DiTBlock(
					inner_dim,
					n_heads,
					dim_head=d_head,
					dropout=dropout,
					adaln_input_size=time_embed_dim,
				)
				for _ in range(depth)
			]
		)

		self.proj_out = zero_module(nn.Conv2d(inner_dim,
											  in_channels,
											  kernel_size=1,
											  stride=1,
											  padding=0))


	def forward(self, x, y=None, context=None, bboxes=None):
		# note: if no context is given, cross-attention defaults to self-attention
		b, c, h, w = x.shape
		x_in = x
		x = self.norm(x)
		x = self.proj_in(x)
		if bboxes is not None:
			assert self.bbox_embed is not None
			ex_emb = self.bbox_embed(x, bboxes)
			if self.adalnzero:
				assert self.bbox_y_embed is not None
				y = y + self.bbox_y_embed(ex_emb)
			else:
				context = (
					ex_emb if context is None else 
					th.cat((context, ex_emb), dim=2)
			   )
		c = y if self.adalnzero else context
		x = rearrange(x, 'b c h w -> b (h w) c')
		for block in self.transformer_blocks:
			x = block(x, c)
		x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
		x = self.proj_out(x)
		return x + x_in


class UNetModel(nn.Module):
	"""
	The full UNet model with attention and timestep embedding.

	:param in_channels: channels in the input Tensor.
	:param model_channels: base channel count for the model.
	:param out_channels: channels in the output Tensor.
	:param num_res_blocks: number of residual blocks per downsample.
	:param attention_resolutions: a collection of downsample rates at which
		attention will take place. May be a set, list, or tuple.
		For example, if this contains 4, then at 4x downsampling, attention
		will be used.
	:param dropout: the dropout probability.
	:param channel_mult: channel multiplier for each level of the UNet.
	:param conv_resample: if True, use learned convolutions for upsampling and
		downsampling.
	:param dims: determines if the signal is 1D, 2D, or 3D.
	:param num_classes: if specified (as an int), then this model will be
		class-conditional with `num_classes` classes.
	:param use_checkpoint: use gradient checkpointing to reduce memory usage.
	:param num_heads: the number of attention heads in each attention layer.
	:param num_heads_channels: if specified, ignore num_heads and instead use
							   a fixed channel width per attention head.
	:param num_heads_upsample: works with num_heads to set a different number
							   of heads for upsampling. Deprecated.
	:param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
	:param resblock_updown: use residual blocks for up/downsampling.
	:param use_new_attention_order: use a different attention pattern for potentially
									increased efficiency.
	"""

	def __init__(
		self,
		input_size,
		in_channels,
		model_channels,
		out_channels,
		num_res_blocks,
		attention_resolutions,
		dropout=0,
		channel_mult=(1, 2, 4, 8),
		conv_resample=True,
		dims=2,
		y_dim=None,
		context_dim=None,
		use_checkpoint=False,
		num_heads=1,
		num_head_channels=-1,
		num_heads_upsample=-1,
		use_scale_shift_norm=False,
		resblock_updown=False,
		adalnzero=False,
		learn_count=False,
		transformer_depth=1,
		initial_ds=1.0,
		bbox_dim=None,
		num_bboxes=None,
	):
		super().__init__()

		if num_heads_upsample == -1:
			num_heads_upsample = num_heads

		self.input_size = input_size
		self.in_channels = in_channels
		self.model_channels = model_channels
		self.out_channels = out_channels
		self.num_res_blocks = num_res_blocks
		self.attention_resolutions = attention_resolutions
		self.dropout = dropout
		self.channel_mult = channel_mult
		self.conv_resample = conv_resample
		self.context_dim = context_dim
		self.y_dim = y_dim
		self.use_checkpoint = use_checkpoint
		self.num_heads = num_heads
		self.num_head_channels = num_head_channels
		self.num_heads_upsample = num_heads_upsample
		self.adalnzero = adalnzero
		self.learn_count = learn_count

		if isinstance(transformer_depth, int):
			transformer_depth = len(channel_mult) * [transformer_depth]
		transformer_depth_middle = transformer_depth[-1]

		time_embed_dim = model_channels * 4
		self.time_embed = nn.Sequential(
			linear(model_channels, time_embed_dim),
			nn.SiLU(),
			linear(time_embed_dim, time_embed_dim),
		)

		if y_dim is not None:
			self.y_embed = nn.Sequential(
				nn.Linear(y_dim, time_embed_dim),
				nn.SiLU(),
				nn.Linear(time_embed_dim, time_embed_dim),
			)

		self.bbox_y_embed = None
		self.ex_embedders = [None] * len(channel_mult)
		context_dims = [context_dim] * len(channel_mult)
		if bbox_dim is not None:
			assert context_dim is None or context_dim == bbox_dim
			ds_ = 1
			self.ex_embedders = nn.ModuleList([])
			for level, mult in enumerate(channel_mult):
				if ds_ in attention_resolutions:
					ch_ = int(mult * model_channels)
					if bbox_dim == "adaptive":
						assert not adalnzero and context_dim is None
						context_dims[level] = ch_

					self.ex_embedders.append(
						RoIAlignExemplarEmbedder(
							in_channels=ch_,
							roi_output_size=7,
							out_channels=bbox_dim,
							spatial_scale=(initial_ds / ds_),
							remove_sequence_dim=adalnzero,
						)
					)
				else:
					self.ex_embedders.append(None)
				ds_ *= 2

			if adalnzero:
				assert num_bboxes is not None
				self.bbox_y_embed = nn.Sequential(
					nn.Linear(bbox_dim * num_bboxes, time_embed_dim),
					nn.SiLU(),
					nn.Linear(time_embed_dim, time_embed_dim)
				)

		ch = input_ch = int(channel_mult[0] * model_channels)
		self.input_blocks = nn.ModuleList(
			[TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
		)
		self._feature_size = ch
		input_block_chans = [ch]
		ds = 1
		for level, mult in enumerate(channel_mult):
			for _ in range(num_res_blocks):
				layers = [
					ResBlock(
						ch,
						time_embed_dim,
						dropout,
						out_channels=int(mult * model_channels),
						dims=dims,
						use_checkpoint=use_checkpoint,
						use_scale_shift_norm=use_scale_shift_norm,
					)
				]
				ch = int(mult * model_channels)
				if ds in attention_resolutions:
					layers.append(
	  					SpatialTransformer(
							ch, 
							n_heads=num_heads, 
							d_head=num_head_channels,
							depth=transformer_depth[level],
							time_embed_dim=time_embed_dim,
							context_dim=context_dims[level],
							adalnzero=adalnzero,
							bbox_embed=self.ex_embedders[level],
							bbox_y_embed=self.bbox_y_embed
						)
					)
				self.input_blocks.append(TimestepEmbedSequential(*layers))
				self._feature_size += ch
				input_block_chans.append(ch)
			if level != len(channel_mult) - 1:
				out_ch = ch
				self.input_blocks.append(
					TimestepEmbedSequential(
						ResBlock(
							ch,
							time_embed_dim,
							dropout,
							out_channels=out_ch,
							dims=dims,
							use_checkpoint=use_checkpoint,
							use_scale_shift_norm=use_scale_shift_norm,
							down=True,
						)
						if resblock_updown
						else Downsample(
							ch, conv_resample, dims=dims, out_channels=out_ch
						)
					)
				)
				ch = out_ch
				input_block_chans.append(ch)
				ds *= 2
				self._feature_size += ch

		self.middle_block = TimestepEmbedSequential(
			ResBlock(
				ch,
				time_embed_dim,
				dropout,
				dims=dims,
				use_checkpoint=use_checkpoint,
				use_scale_shift_norm=use_scale_shift_norm,
			),
			SpatialTransformer(
				ch, 
				n_heads=num_heads, 
				d_head=num_head_channels,
				depth=transformer_depth_middle,
				time_embed_dim=time_embed_dim,
				context_dim=context_dims[level],
				adalnzero=adalnzero,
				bbox_embed=self.ex_embedders[-1],
				bbox_y_embed=self.bbox_y_embed
			),
			ResBlock(
				ch,
				time_embed_dim,
				dropout,
				dims=dims,
				use_checkpoint=use_checkpoint,
				use_scale_shift_norm=use_scale_shift_norm,
			),
		)
		self._feature_size += ch

		self.output_blocks = nn.ModuleList([])
		for level, mult in list(enumerate(channel_mult))[::-1]:
			for i in range(num_res_blocks + 1):
				ich = input_block_chans.pop()
				layers = [
					ResBlock(
						ch + ich,
						time_embed_dim,
						dropout,
						out_channels=int(model_channels * mult),
						dims=dims,
						use_checkpoint=use_checkpoint,
						use_scale_shift_norm=use_scale_shift_norm,
					)
				]
				ch = int(model_channels * mult)
				if ds in attention_resolutions:
					layers.append(
	  					SpatialTransformer(
							ch, 
							n_heads=num_heads, 
							d_head=num_head_channels,
							depth=transformer_depth[level],
							time_embed_dim=time_embed_dim,
							context_dim=context_dims[level],
							adalnzero=adalnzero,
							bbox_embed=self.ex_embedders[level],
							bbox_y_embed=self.bbox_y_embed
						)
					)
				if level and i == num_res_blocks:
					out_ch = ch
					layers.append(
						ResBlock(
							ch,
							time_embed_dim,
							dropout,
							out_channels=out_ch,
							dims=dims,
							use_checkpoint=use_checkpoint,
							use_scale_shift_norm=use_scale_shift_norm,
							up=True,
						)
						if resblock_updown
						else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
					)
					ds //= 2
				self.output_blocks.append(TimestepEmbedSequential(*layers))
				self._feature_size += ch


		if learn_count:
			self.feat_extract_list = [1, 4, 7, 10, 13, 17, 20][:len(channel_mult)]
			feat_dims = {
				f"p{layer}": model_channels * mult for layer, mult in zip(self.feat_extract_list, channel_mult)
			}
			self.counting_branch = CountingBranch(feat_dims, hidden_dim=64)

		self.out = nn.Sequential(
			nn.GroupNorm(32, ch),
			nn.SiLU(),
			zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
		)

		self.initialize_weights()

	def initialize_weights(self):
		def _basic_init(module):
			if isinstance(module, nn.Linear):
				th.nn.init.xavier_uniform_(module.weight)
				if module.bias is not None:
					nn.init.constant_(module.bias, 0)
		self.apply(_basic_init)

		def _adalnzero_init(module):
			if isinstance(module, DiTBlock):
				# Zero-out adaLN modulation layers in DiT blocks:
				nn.init.constant_(module.adaLN_modulation[-1].weight, 0)
				nn.init.constant_(module.adaLN_modulation[-1].bias, 0)
		self.apply(_adalnzero_init)
		
		# Initialize timestep embedding MLP:
		nn.init.normal_(self.time_embed[0].weight, std=0.02)
		nn.init.normal_(self.time_embed[2].weight, std=0.02)


	def _forward(self, x, timesteps, y=None, context=None, bboxes=None):
		"""
		Apply the model to an input batch.

		:param x: an [N x C x ...] Tensor of inputs.
		:param timesteps: a 1-D batch of timesteps.
		:param y: an [N] Tensor of labels, if class-conditional.
		:return: an [N x C x ...] Tensor of outputs.
		"""
		xs = []
		en_feats = {}
		de_feats = {}
		emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
		
		if y is not None:
			assert y.shape[0] == x.shape[0]
			emb = emb + self.y_embed(y)

		for module in self.input_blocks:
			x = module(x, emb, context=context, bboxes=bboxes)
			xs.append(x)

		x = self.middle_block(x, emb, context=context, bboxes=bboxes)

		for layer, module in enumerate(self.output_blocks):
			e = xs.pop()
			x = th.cat([x, e], dim=1)
			x = module(x, emb, context=context, bboxes=bboxes)

			if self.learn_count and layer in self.feat_extract_list:
				en_feats[f"p{layer}"] = e.clone()
				de_feats[f"p{layer}"] = x.clone()

		count = None
		if self.learn_count:
			count = self.counting_branch(de_feats)
			
		out = self.out(x)
		return dict(out=out, count=count)


	def forward(self, x, t, cond, **kwargs):
		x = th.cat((x, cond.get("concat", th.tensor([]).type_as(x))), dim=1)
		return self._forward(
			x,
			timesteps=t,
			y=cond.get("vector", None),
			context=cond.get("crossattn", None),
			bboxes=cond.get("bboxes", None),
		)