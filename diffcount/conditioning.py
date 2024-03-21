
# TODO classifier-free guidance -> figure out how to null image embedding looks like

import torch as th
import torch.nn as nn

from contextlib import nullcontext
from functools import partial
from itertools import chain

from detectron2.modeling import ViT, SimpleFeaturePyramid
from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes
from detectron2.checkpoint import DetectionCheckpointer

from .nn import disabled_train, count_params


VITDET_PRETRAINED_MODELS = {
	"B": "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/mask_rcnn_vitdet_b/f325346929/model_final_61ccd1.pkl",
	"L": "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/mask_rcnn_vitdet_l/f325599698/model_final_6146ed.pkl",
	"H": "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/mask_rcnn_vitdet_h/f329145471/model_final_7224f1.pkl",
}


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
		super().__init__(
			input_keys=["cls"], 
			is_trainable=is_trainable
		)
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
		super().__init__(
			input_keys=["img"], 
			is_trainable=False
		)

	def forward(self, img):
		return img


class ViTExemplarEmbedder(AbstractEmbModel):

	def __init__(
		self, 
		image_size,
		vit_size="B",
		freeze_backbone=True,
		is_trainable=False
	):
		super().__init__(
			input_keys=["img", "bboxes"], 
			is_trainable=is_trainable
		)
		self.image_size = image_size
		if vit_size == "B":
			embed_dim, depth, num_heads, dp = 768, 12, 12, 0.1
			# 2, 5, 8 11 for global attention
			window_block_ixs = list(chain(
				range(0,2), 
				range(3,5), 
				range(6,8), 
				range(9,11)
			))
		elif vit_size == "L":
			embed_dim, depth, num_heads, dp = 1024, 24, 16, 0.4
			# 5, 11, 17, 23 for global attention
			window_block_ixs = list(chain(
				range(0, 5),
				range(6, 11),
				range(12, 17),
				range(18, 23)
			))
		elif vit_size == "H":
			embed_dim, depth, num_heads, dp = 1280, 32, 16, 0.5
			# 7, 15, 23, 31 for global attention
			window_block_ixs = list(chain(
				range(0, 7),
				range(8, 15),
				range(16, 23),
				range(24, 31)
			))
		else:
			raise ValueError(f"unsupported ViT size: {vit_size}")
		
		self.backbone = SimpleFeaturePyramid(
			net=ViT(  # Single-scale ViT backbone
				img_size=image_size,
				patch_size=16,
				embed_dim=embed_dim,
				depth=depth,
				num_heads=num_heads,
				drop_path_rate=dp,
				window_size=14,
				mlp_ratio=4,
				qkv_bias=True,
				norm_layer=partial(nn.LayerNorm, eps=1e-6),
				window_block_indexes=window_block_ixs,
				residual_block_indexes=[],
				use_rel_pos=False, 
				out_feature="last_feat",
			),
			in_feature="last_feat",
			out_channels=256,
			scale_factors=(4.0, 2.0, 1.0, 0.5),
			norm="LN",
			square_pad=0,
		)
		self.roi_pooler = ROIPooler(
			output_size=7,
			scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
			sampling_ratio=0,
			pooler_type="ROIAlignV2",
		)
		self.fc1 = nn.Linear(256 * 7**2, 1024) # vit_out_channels * roi_output_size^2

		checkpointer = DetectionCheckpointer(self)
		checkpointer.load(VITDET_PRETRAINED_MODELS[vit_size])

		if freeze_backbone:
			self.backbone.train = disabled_train
			for param in self.backbone.parameters():
				param.requires_grad = False
			self.backbone.eval()

	def forward(self, img, bboxes):
		# TODO when n_exemplars is 0 this will fail - fix
		batch_size, n_exemplars = bboxes.shape[0], bboxes.shape[1]
		bbs = [Boxes(bb).to(img.device) for bb in bboxes]
		fpn = self.backbone(img)
		fpn = list(fpn.values())

		x = self.roi_pooler(fpn, bbs)
		x = x.flatten(start_dim=1)
		x = x.reshape(batch_size, n_exemplars, x.shape[1])
		x = self.fc1(x)
		
		return x
