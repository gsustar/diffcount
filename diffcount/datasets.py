import os
import json
import random
import numpy as np
import torch as th

from PIL import Image
from typing import Literal

import torchvision.datasets as thdata
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from torch.utils.data import Dataset, DataLoader, Subset
from . import logger


class MNIST(Dataset):

	def __init__(
		self, 
		root, 
		split='train'
	):
		self.dataset = thdata.MNIST(
			root=root,
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
	:param root: root directory of dataset.
	:param targetdir: Name of the directory containing the GT maps.
	:param split: 'train', 'val' or 'test'.
	:param n_examplars: Number of examplars
	:param target_transform: Optional transforms to be applied to the density map.

	Make sure the root directory has the following structure:

   root
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
			root: str,
			targetdir: str,
			split: Literal['train', 'val', 'test'] = 'train',
			n_examplars: int = 3,
			transform_kwargs: dict = dict(),
	):
		self.root = root
		self.targetdir = targetdir
		self.split = split
		self.n_examplars = n_examplars

		self.img_size = transform_kwargs.pop("img_size", 256)
		self.hflip_p = transform_kwargs.pop("hflip_p", 0.5)
		self.cj_p = transform_kwargs.pop("cj_p", 0.8)

		self.img_names = None
		with open(os.path.join(self.root, 'Train_Test_Val_FSC_147.json'), 'rb') as f:
			self.img_names = json.load(f)[self.split]

		self.annotations = None
		with open(os.path.join(self.root, 'annotation_FSC147_384.json'), 'rb') as f:
			self.annotations = {k: v for k, v in json.load(f).items() if k in self.img_names}


	def transform(self, img, bboxes, target, split='train'):
		# ToTensor
		img = F.to_tensor(img) # (C, H, W)
		target = F.to_tensor(target)

		# Resize
		old_h, old_w = img.shape[-2:]
		img = F.resize(img, (self.img_size, self.img_size), antialias=True)
		# Preferably create density maps with the same size as the input images and avoid resizing at this stage
		if target.shape[-2:] != img.shape[-2:]:
			original_sum = target.sum()
			target = F.resize(target, (self.img_size, self.img_size), antialias=True)
			target = target / target.sum() * original_sum

		new_h, new_w = img.shape[-2:]
		rw = new_w / old_w
		rh = new_h / old_h
		bboxes = bboxes * th.tensor([rw, rh, rw, rh])

		if split == 'train':
			# RandomHorizontalFlip
			if th.rand(1) < self.hflip_p:
				img = F.hflip(img)
				target = F.hflip(target)
				bboxes[:, [0, 2]] = self.img_size - bboxes[:, [2, 0]]

			# RandomColorJitter
			if th.rand(1) < self.cj_p:
				img = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)(img)

		# Rescale to [-1, 1]
		img = img * 2.0 - 1.0
		target = target * 2.0 - 1.0
		return img, bboxes, target


	def __len__(self):
		return len(self.img_names)


	def __getitem__(self, index):
		if th.is_tensor(index):
			index = index.tolist()

		img = Image.open(
			os.path.join(
				self.root,
				'images_384_VarV2',
				self.img_names[index]
			)
		).convert('RGB')

		target = np.load(
			os.path.join(
				self.root,
				self.targetdir,
				os.path.splitext(self.img_names[index])[0] + '.npy'
			)
		)

		bboxes = th.as_tensor(self.annotations[self.img_names[index]]['box_examples_coordinates'])
		assert len(bboxes) >= self.n_examplars, f'Not enough examplars for image {self.img_names[index]}'
		bboxes = bboxes[:, [0, 2], :].reshape(-1, 4)
		bboxes = bboxes[th.randperm(bboxes.shape[0])]
		bboxes = bboxes[:self.n_examplars, ...]	# (x_min, y_min, x_max, y_max)
		img, bboxes, target = self.transform(img, bboxes, target, split=self.split)
		return target, dict(bboxes=bboxes, img=img)


def load_data(
	*,
	dataset,
	batch_size,
	shuffle=False,
	fraction_of_data=1.0,
	overfit_single_batch=False,
):
	"""
	For a dataset, create a dataloader of (target, kwargs) pairs.

	Each images is an NCHW float tensor, and the kwargs dict contains zero or
	more keys, each of which map to a batched Tensor of their own.

	:param dataset: The dataset to iterate over.
	:param batch_size: the batch size of each returned pair.
	:param shuffle: if True, yield results in a shuffle order.
	:param fraction_of_data: if < 1.0, only use a fraction of the dataset.
	:param overfit_single_batch: if True, only return a single batch of data.
	"""
	assert 0 < fraction_of_data <= 1.0
	if overfit_single_batch and fraction_of_data < 1.0:
		logger.log("overfit_single_batch and fraction_of_data are both set, ignoring fraction_of_data")
	n = None
	if overfit_single_batch:
		n = batch_size
	elif fraction_of_data < 1.0:
		n = int(fraction_of_data * len(dataset))
		n = max(batch_size, n)
	if n is not None:
		ixs = [random.randint(0, len(dataset) - 1) for _ in range(n)]
		dataset = Subset(dataset, ixs)
	loader = DataLoader(
		dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1, drop_last=False
	)
	return loader