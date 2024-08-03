
# TODO classifier-free guidance -> figure out how to null image embedding looks like
import torch as th
import torch.nn as nn

from contextlib import nullcontext
from functools import partial
from itertools import chain
from torchvision.ops import roi_align

from .nn import disabled_train, count_params


class AbstractEmbModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.input_keys = None
		self.ucg_rate = None
		self.is_trainable = None


class Conditioner(nn.Module):
	OUTPUT_DIM2KEYS = {2: "vector", 3: "crossattn", 4: "concat", 5: "concat"}
	KEY2CATDIM = {"vector": 1, "crossattn": 2, "concat": 1, "bboxes": 1}

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
				if isinstance(embedder, BBoxAppendEmbedder):
					out_key = "bboxes"
				else:
					out_key = self.OUTPUT_DIM2KEYS[emb.dim()]
				if out_key in output:
					output[out_key] = th.cat(
						(output[out_key], emb), self.KEY2CATDIM[out_key]
					)
				else:
					output[out_key] = emb
		return output


class ClassEmbedder(AbstractEmbModel):
	def __init__(self, embed_dim, n_classes=10, add_sequence_dim=False):
		super().__init__()
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
		super().__init__()

	def forward(self, img):
		return img
	

class BBoxAppendEmbedder(AbstractEmbModel):
	def __init__(self):
		super().__init__()

	def forward(self, bboxes):
		return bboxes


class ViTExemplarEmbedder(AbstractEmbModel):

	VITDET_PRETRAINED_MODELS = {
		"B": "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/mask_rcnn_vitdet_b/f325346929/model_final_61ccd1.pkl",
		"L": "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/mask_rcnn_vitdet_l/f325599698/model_final_6146ed.pkl",
		"H": "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/mask_rcnn_vitdet_h/f329145471/model_final_7224f1.pkl",
	}

	def __init__(
		self, 
		input_size,
		in_channels,
		out_channels,
		roi_output_size=7,
		vit_size="B",
		freeze_backbone=True,
		remove_sequence_dim=False,
	):
		super().__init__()

		try:
			from detectron2.modeling import SimpleFeaturePyramid, ViT
			from detectron2.structures import Boxes
			from detectron2.checkpoint import DetectionCheckpointer
			from detectron2.modeling.poolers import ROIPooler
		except ImportError:
			raise ImportError("detectron2 is required for ViTExemplarEmbedder")
		self._Boxes = Boxes

		self.input_size = input_size
		self.in_channels = in_channels
		self.remove_sequence_dim = remove_sequence_dim
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
				img_size=input_size,
				in_chans=in_channels,
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
			output_size=roi_output_size,
			scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
			sampling_ratio=0,
			pooler_type="ROIAlignV2",
		)
		self.fc1 = nn.Linear(256 * roi_output_size**2, out_channels) # vit_out_channels * roi_output_size^2

		checkpointer = DetectionCheckpointer(self)
		# checkpointer.load(ViTExemplarEmbedder.VITDET_PRETRAINED_MODELS[vit_size])
		checkpointer.load(self.VITDET_PRETRAINED_MODELS[vit_size])

		if freeze_backbone:
			self.backbone.train = disabled_train
			for param in self.backbone.parameters():
				param.requires_grad = False
			self.backbone.eval()

	def forward(self, img, bboxes):
		# TODO when n_exemplars is 0 this will fail - fix
		batch_size, n_exemplars = bboxes.shape[0], bboxes.shape[1]
		bbs = [self._Boxes(bb).to(img.device) for bb in bboxes]
		fpn = self.backbone(img)
		fpn = list(fpn.values())

		x = self.roi_pooler(fpn, bbs)
		x = x.flatten(start_dim=1)
		x = x.reshape(batch_size, n_exemplars, x.shape[1])
		x = self.fc1(x)

		if self.remove_sequence_dim:
			x = x.reshape(batch_size, -1)
		
		return x


class RoIAlignExemplarEmbedder(AbstractEmbModel):

	def __init__(
		self,
		in_channels,
		inner_dim,
		out_channels,
		roi_output_size,
		n_convs=2,
		spatial_scale=0.125,
		remove_sequence_dim=False,
		mlp_ratio=4,
	):
		super().__init__()

		self.in_channels = in_channels
		self.out_channels = out_channels
		self.roi_output_size = roi_output_size
		self.spatial_scale = spatial_scale
		self.remove_sequence_dim = remove_sequence_dim

		self.norm_in = nn.LayerNorm([in_channels, roi_output_size, roi_output_size])
		self.in_layers = nn.ModuleList([])
		ch = in_channels
		for _ in range(n_convs):
			self.in_layers.append(
				nn.Sequential(
					nn.Conv2d(
						ch,
						inner_dim,
						kernel_size=1,
						stride=1,
						padding=0
					),
					nn.GroupNorm(32, inner_dim),
					nn.ReLU()
				)
			)
			ch = inner_dim

		self.avgpool = nn.AdaptiveAvgPool2d(1)
		hidden_channels = mlp_ratio * inner_dim
		self.out = nn.Sequential(
			nn.Linear(inner_dim, hidden_channels),
			nn.LayerNorm(hidden_channels),
			nn.ReLU(),
			nn.Linear(hidden_channels, out_channels)
		)


	def forward(self, z, bboxes, ds=1.0):
		bs, ch, _, _ = z.shape
		assert ch > 3, "z must be a feature map not an image"
		x = roi_align(
			z, 
			boxes=list(bboxes), 
			output_size=self.roi_output_size, 
			spatial_scale=(self.spatial_scale / ds),
			aligned=True
		)
		x = self.norm_in(x)
		for module in self.in_layers:
			x = module(x)
		x = self.avgpool(x)
		x = x.reshape(bs, -1, x.shape[1])
		x = self.out(x)

		if self.remove_sequence_dim:
			x = x.reshape(bs, -1)
		return x


class LightRoIAlignExemplarEmbedder(AbstractEmbModel):
	
	def __init__(
		self,
		in_channels,
		out_channels,
		roi_output_size,
		spatial_scale=0.125,
		remove_sequence_dim=False,
	):
		super().__init__()

		self.skip_out = True if out_channels == "adaptive" else False
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.roi_output_size = roi_output_size
		self.spatial_scale = spatial_scale
		self.remove_sequence_dim = remove_sequence_dim

		self.avgpool = nn.AdaptiveAvgPool2d(1)
		if not self.skip_out:
			self.out = nn.Sequential(
				nn.Linear(in_channels, out_channels),
				nn.LayerNorm(out_channels),
				nn.ReLU(),
				nn.Linear(out_channels, out_channels)
			)

	def forward(self, z, bboxes, ds=1.0):
		bs, ch, _, _ = z.shape
		assert ch > 3, "z must be a feature map not an image"
		x = roi_align(
			z, 
			boxes=list(bboxes), 
			output_size=self.roi_output_size, 
			spatial_scale=(self.spatial_scale / ds),
			aligned=True
		)
		x = self.avgpool(x)
		x = x.reshape(bs, -1, x.shape[1])
		if not self.skip_out:
			x = self.out(x)

		if self.remove_sequence_dim:
			x = x.reshape(bs, -1)
		return x
	

class OPEExemplarEmbedder(AbstractEmbModel):

	def __init__():
		super().__init__()


	def forward(img, bboxes):
		pass



class BBoxShapeEmbedder(AbstractEmbModel):

	def __init__():
		super().__init__()


	def forward(bboxes):
		pass



class SAM2ExemplarMaskEmbedder(AbstractEmbModel):

	def __init__(self):
		super().__init__()
		try:
			from sam2.build_sam import build_sam2
			from sam2.sam2_image_predictor import SAM2ImagePredictor
		except ImportError:
			raise ImportError("sam2 is required for SAM2ImageMaskEmbedder")
		
		
	def forward(self):
		pass
