import math
import torch
import torch.nn.functional as F

from torch import nn
from inspect import isfunction
from einops import rearrange
from torch.utils.checkpoint import checkpoint
from torch.nn.attention import SDPBackend, sdpa_kernel

from . import logger

def get_sdpa_settings():
	if torch.cuda.is_available():
		old_gpu = torch.cuda.get_device_properties(0).major < 7
		# only use Flash Attention on Ampere (8.0) or newer GPUs
		use_flash_attn = torch.cuda.get_device_properties(0).major >= 8
		if not use_flash_attn:
			logger.warn(
				"Flash Attention is disabled as it requires a GPU with Ampere (8.0) CUDA capability.",
			)
		# keep math kernel for PyTorch versions before 2.2 (Flash Attention v2 is only
		# available on PyTorch 2.2+, while Flash Attention v1 cannot handle all cases)
		pytorch_version = tuple(int(v) for v in torch.__version__.split(".")[:2])
		if pytorch_version < (2, 2):
			logger.warn(
				f"You are using PyTorch {torch.__version__} without Flash Attention v2 support. "
				"Consider upgrading to PyTorch 2.2+ for Flash Attention v2 (which could be faster).",
			)
		math_kernel_on = pytorch_version < (2, 2) or not use_flash_attn
	else:
		old_gpu = True
		use_flash_attn = False
		math_kernel_on = True

	backend = []
	if old_gpu:
		backend.append(SDPBackend.EFFICIENT_ATTENTION)
	if use_flash_attn:
		backend.append(SDPBackend.FLASH_ATTENTION)
	if math_kernel_on:
		backend.append(SDPBackend.MATH)

	return backend


def exists(val):
	return val is not None


def uniq(arr):
	return{el: True for el in arr}.keys()


def default(val, d):
	if exists(val):
		return val
	return d() if isfunction(d) else d


def max_neg_value(t):
	return -torch.finfo(t.dtype).max


def init_(tensor):
	dim = tensor.shape[-1]
	std = 1 / math.sqrt(dim)
	tensor.uniform_(-std, std)
	return tensor


# feedforward
class GEGLU(nn.Module):
	def __init__(self, dim_in, dim_out):
		super().__init__()
		self.proj = nn.Linear(dim_in, dim_out * 2)

	def forward(self, x):
		x, gate = self.proj(x).chunk(2, dim=-1)
		return x * F.gelu(gate)


class FeedForward(nn.Module):
	def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
		super().__init__()
		inner_dim = int(dim * mult)
		dim_out = default(dim_out, dim)
		project_in = nn.Sequential(
			nn.Linear(dim, inner_dim),
			nn.GELU()
		) if not glu else GEGLU(dim, inner_dim)

		self.net = nn.Sequential(
			project_in,
			nn.Dropout(dropout),
			nn.Linear(inner_dim, dim_out)
		)

	def forward(self, x):
		return self.net(x)


def zero_module(module):
	"""
	Zero out the parameters of a module and return it.
	"""
	for p in module.parameters():
		p.detach().zero_()
	return module


class CrossAttention(nn.Module):
	BACKEND = get_sdpa_settings()

	def __init__(
		self,
		query_dim,
		context_dim=None,
		heads=8,
		dim_head=64,
		dropout=0.0,
	):
		super().__init__()
		inner_dim = dim_head * heads
		context_dim = default(context_dim, query_dim)

		self.scale = dim_head**-0.5
		self.heads = heads

		self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
		self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
		self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

		self.dropout = dropout
		self.to_out = nn.Sequential(
			nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
		)

	def forward(
		self,
		x,
		context=None,
		mask=None,
	):
		h = self.heads
		q = self.to_q(x)
		context = default(context, x)
		k = self.to_k(context)
		v = self.to_v(context)

		q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

		with torch.nn.attention.sdpa_kernel(self.BACKEND):
			# print("dispatching into backend", self.backend, "q/k/v shape: ", q.shape, k.shape, v.shape)
			out = F.scaled_dot_product_attention(
				q, k, v, attn_mask=mask
			)  # scale is dim_head ** -0.5 per default

		del q, k, v
		out = rearrange(out, "b h n d -> b n (h d)", h=h)

		return self.to_out(out)


# class CrossAttention(nn.Module):

# 	def __init__(
# 		self, 
# 		query_dim, 
# 		context_dim=None, 
# 		heads=8, 
# 		dim_head=64, 
# 		dropout=0.
# 	):
# 		super().__init__()
# 		inner_dim = dim_head * heads
# 		context_dim = default(context_dim, query_dim)

# 		self.scale = dim_head ** -0.5
# 		self.heads = heads

# 		self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
# 		self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
# 		self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

# 		self.to_out = nn.Sequential(
# 			nn.Linear(inner_dim, query_dim),
# 			nn.Dropout(dropout)
# 		)

# 	def forward(self, x, context=None, mask=None):
# 		h = self.heads

# 		q = self.to_q(x)
# 		context = default(context, x)
# 		k = self.to_k(context)
# 		v = self.to_v(context)

# 		q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

# 		sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

# 		if exists(mask):
# 			mask = rearrange(mask, 'b ... -> b (...)')
# 			max_neg_value = -torch.finfo(sim.dtype).max
# 			mask = repeat(mask, 'b j -> (b h) () j', h=h)
# 			sim.masked_fill_(~mask, max_neg_value)

# 		# attention, what we cannot get enough of
# 		attn = sim.softmax(dim=-1)

# 		out = einsum('b i j, b j d -> b i d', attn, v)
# 		out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
# 		return self.to_out(out)


class BasicTransformerBlock(nn.Module):

	def __init__(
		self,
		dim, 
		n_heads, 
		d_head, 
		dropout=0., 
		context_dim=None, 
		gated_ff=True, 
		use_checkpoint=True
	):
		super().__init__()
		self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
		self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
		self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
									heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none

		self.norm1 = nn.LayerNorm(dim)
		self.norm2 = nn.LayerNorm(dim)
		self.norm3 = nn.LayerNorm(dim)
		self.use_checkpoint = use_checkpoint

	def forward(self, x, context=None):
		if self.use_checkpoint:
			return checkpoint(self._forward, x, context, use_reentrant=False)
		else:
			return self._forward(x, context)

	def _forward(self, x, context=None):
		x = self.attn1(self.norm1(x)) + x
		x = self.attn2(self.norm2(x), context=context) + x
		x = self.ff(self.norm3(x)) + x
		return x