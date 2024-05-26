import os
import json
import random
import numpy as np
import torch as th
import shutil

from PIL import Image
from typing import Literal

from scipy.ndimage import gaussian_filter

import torchvision.datasets as thdata
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from torch.utils.data import Dataset, DataLoader, Subset

from . import logger


class MNIST(Dataset):

	def __init__(
		self, 
		datadir, 
		split='train'
	):
		self.dataset = thdata.MNIST(
			root=datadir,
			train=True if split == 'train' else False,
			download=True,
			transform=transforms.Compose([
				transforms.ToTensor(), transforms.Lambda(lambda x: x * 2.0 - 1.0)
			])
		)
	
	def __len__(self):
		return len(self.dataset)
	
	def __getitem__(self, index):
		img, cls = self.dataset[index]
		cls = th.tensor(cls)
		return img, dict(cls=cls)


class FSC147(Dataset):
	"""
	:param datadir: root directory of the dataset.
	:param targetdir: sub-directory where generated density maps will be stored.
	:param split: 'train', 'val' or 'test'.
	:param n_exemplars: Number of examplars

	Make sure the datadir directory has the following structure:

   datadir
	├── images_384_VarV2
	│       ├─ 2.jpg
	│       ├─ 3.jpg
	│       ├─ ...
	│       └─ 7714.jpg
	├── annotation_FSC147_384.json
	├── Train_Test_Val_FSC_147.json                         
	├── targetdir
	│       ├─ 2.npy
	│       ├─ 3.npy
	│       ├─ ...
	│       └─ 7714.npy
	└── ImageClasses_FSC147.json	(optional)

	"""

	def __init__(
			self,
			datadir: str,
			targetdir: str,
			split: Literal['train', 'val', 'test'] = 'train',
			n_exemplars: int = 3,
			image_size: int = 256,
			tile_size: int = 512,
			hflip_p: float = 0.5,
			cj_p: float = 0.8,
			sigma: float = 0.5,
			clear_cache: bool = True
	):
		self.datadir = datadir
		self.targetdir = os.path.join(self.datadir, targetdir)
		self.split = split
		self.n_exemplars = n_exemplars

		self.image_size = image_size
		self.tile_size = tile_size
		self.hflip_p = hflip_p
		self.cj_p = cj_p
		self.sigma = sigma

		self.img_names = None
		with open(os.path.join(self.datadir, 'Train_Test_Val_FSC_147.json'), 'rb') as f:
			self.img_names = json.load(f)[self.split]

		self.annotations = None
		with open(os.path.join(self.datadir, 'annotation_FSC147_384.json'), 'rb') as f:
			self.annotations = {k: v for k, v in json.load(f).items() if k in self.img_names}

		if clear_cache and not split in ['val', 'test']:
			if os.path.exists(self.targetdir):
				shutil.rmtree(self.targetdir)
			os.makedirs(self.targetdir)

	# todo remove this here and deal with it later with resizer
	def pad(
		self,
		img,
		center=False,
		value=0
	):
		"""
		Zero-pad the image to be divisible by the tile size
		"""
		h, w = img.shape[-2:]
		pad_h = (self.tile_size - h % self.tile_size) % self.tile_size
		pad_w = (self.tile_size - w % self.tile_size) % self.tile_size
		pad = [0, 0, pad_w, pad_h]
		if center:
			pad_b = pad_h // 2 + 1 if pad_h % 2 != 0 else pad_h // 2
			pad_r = pad_w // 2 + 1 if pad_w % 2 != 0 else pad_w // 2
			pad = [pad_w // 2, pad_h // 2, pad_r, pad_b]
		return F.pad(img, pad, padding_mode='constant', fill=value)


	def transform(self, img, bboxes, split='train'):
		# ToTensor
		img = F.to_tensor(img) # (C, H, W)

		# Resize
		old_h, old_w = img.shape[-2:]
		img = F.resize(img, (self.image_size, self.image_size), antialias=True)

		new_h, new_w = img.shape[-2:]
		rw = new_w / old_w
		rh = new_h / old_h
		bboxes = bboxes * th.tensor([rw, rh, rw, rh])

		# Pad
		img = self.pad(img)

		hflip = False
		if split == 'train':
			# RandomHorizontalFlip
			if th.rand(1) < self.hflip_p:
				img = F.hflip(img)
				bboxes[:, [0, 2]] = self.image_size - bboxes[:, [2, 0]]
				hflip = True

			# RandomColorJitter
			if th.rand(1) < self.cj_p:
				img = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)(img)

		# Rescale to [-1, 1]
		img = img * 2.0 - 1.0
		return img, bboxes, hflip, (new_w, new_h)
	

	def target_transform(self, target, hflip=False):
		target = F.to_tensor(target)
		if hflip:
			target = F.hflip(target)
		target = self.pad(target)
		target = target * 2.0 - 1.0
		return target


	def generate_density_map(self, src_size, new_size, points):
		w, h = src_size
		new_w, new_h = new_size
		rw, rh = new_w / w, new_h / h
		bitmap = np.zeros((new_h, new_w), dtype=np.float32)
		for point in points:
			x, y = int(point[0] * rw)-1, int(point[1] * rh)-1
			bitmap[y, x] = 1.0

		density_map = gaussian_filter(
			bitmap,
			self.sigma,
			truncate=3.0,
			mode='constant'
		)
		return density_map


	def __len__(self):
		return len(self.img_names)


	def __getitem__(self, index):
		if th.is_tensor(index):
			index = index.tolist()

		img = Image.open(
			os.path.join(
				self.datadir,
				'images_384_VarV2',
				self.img_names[index]
			)
		).convert('RGB')
		src_size = img.size # (w, h)

		bboxes = th.as_tensor(self.annotations[self.img_names[index]]['box_examples_coordinates'])
		assert len(bboxes) >= self.n_exemplars, f'Not enough examplars for image {self.img_names[index]}'
		bboxes = bboxes[:, [0, 2], :].reshape(-1, 4)
		img, bboxes, hflip, new_size = self.transform(img, bboxes, split=self.split)

		bboxes = bboxes[th.randperm(bboxes.shape[0])]
		bboxes = bboxes[:self.n_exemplars, ...]	# (x_min, y_min, x_max, y_max)

		npypath = os.path.join(
			self.targetdir, 
			os.path.splitext(self.img_names[index])[0] + '.npy'
		)
		if os.path.exists(npypath):
			target = np.load(npypath)
		else:
			target = self.generate_density_map(
				src_size,
				new_size,
				self.annotations[self.img_names[index]]['points']
			)
			np.save(npypath, target)
		target = self.target_transform(target, hflip)

		return target, dict(bboxes=bboxes, img=img)


def load_data(
	*,
	dataset,
	batch_size,
	shuffle=False,
	overfit_single_batch=False,
):
	"""
	For a dataset, create a dataloader of (target, kwargs) pairs.

	Each images is an NCHW float tensor, and the kwargs dict contains zero or
	more keys, each of which map to a batched Tensor of their own.

	:param dataset: The dataset to iterate over.
	:param batch_size: the batch size of each returned pair.
	:param shuffle: if True, yield results in a shuffle order.
	:param overfit_single_batch: if True, only return a single batch of data.
	"""
	if overfit_single_batch:
		ixs = [random.randint(0, len(dataset) - 1) for _ in range(batch_size)]
		dataset = Subset(dataset, ixs)
	loader = DataLoader(
		dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1, drop_last=False
	)
	return loader