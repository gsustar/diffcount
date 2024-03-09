
# TODO classifier-free guidance -> figure out how to null image embedding looks like

import torch as th
from contextlib import nullcontext

import torch.nn as nn

from .nn import disabled_train, count_params


class AbstractEmbModel(nn.Module):
	def __init__(self, input_keys, is_trainable=False):
		super().__init__()
		self.input_keys = input_keys
		self.is_trainable = is_trainable


class Conditioner(nn.Module):
	OUTPUT_DIM2KEYS = {2: "vector", 3: "crossattn", 4: "concat", 5: "concat"}
	KEY2CATDIM = {"vector": 1, "crossattn": 2, "concat": 1}

	def __init__(self, emb_models):
		super().__init__()
		embedders = []
		for n, embedder in enumerate(emb_models):
			assert isinstance(
				embedder, AbstractEmbModel
			), f"embedder model {embedder.__class__.__name__} has to inherit from AbstractEmbModel"
			if not embedder.is_trainable:
				embedder.train = disabled_train
				for param in embedder.parameters():
					param.requires_grad = False
				embedder.eval()
			print(
				f"Initialized embedder #{n}: {embedder.__class__.__name__} "
				f"with {count_params(embedder, False)} params. Trainable: {embedder.is_trainable}"
			)
			embedders.append(embedder)
		self.embedders = nn.ModuleList(embedders)

	def forward(self, cond):
		output = dict()
		for embedder in self.embedders:
			embedding_context = nullcontext if embedder.is_trainable else th.no_grad
			with embedding_context():
				emb_out = embedder(*[cond[k] for k in embedder.input_keys])
			assert isinstance(
				emb_out, (th.Tensor, list, tuple)
			), f"encoder outputs must be tensors or a sequence, but got {type(emb_out)}"
			if not isinstance(emb_out, (list, tuple)):
				emb_out = [emb_out]
			for emb in emb_out:
				out_key = self.OUTPUT_DIM2KEYS[emb.dim()]
				if out_key in output:
					output[out_key] = th.cat(
						(output[out_key], emb), self.KEY2CATDIM[out_key]
					)
				else:
					output[out_key] = emb
		return output


class ClassEmbedder(AbstractEmbModel):
	def __init__(self, embed_dim, is_trainable=False, n_classes=10, add_sequence_dim=False):
		super().__init__(input_keys=["cls"], is_trainable=is_trainable)
		self.embedding = nn.Embedding(n_classes, embed_dim)
		self.n_classes = n_classes
		self.add_sequence_dim = add_sequence_dim

	def forward(self, c):
		c = self.embedding(c)
		if self.add_sequence_dim:
			c = c[:, None, :]
		return c


class ImageConcatEmbedder(AbstractEmbModel):
	def __init__(self):
		super().__init__(input_keys=["img"], is_trainable=False)

	def forward(self, img):
		return img


class ExemplarEmbedder(AbstractEmbModel):
	def __init__(self, is_trainable=False):
		super().__init__(input_keys=["img", "bboxes"], is_trainable=is_trainable)


	def forward(self, img, bbs):
		pass