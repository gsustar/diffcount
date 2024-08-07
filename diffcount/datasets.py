import os
import csv
import json
import random
import numpy as np
import torch as th

from PIL import Image

from scipy.ndimage import gaussian_filter

import torchvision.datasets as thdata
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from torch.utils.data import Dataset, DataLoader, Subset
from .data_util import (
	resize, 
	resize_dm, 
	pad, 
	adjust_bboxes, 
	scale_to_bbox
)


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
		return img, dict(cls=cls, count=0)


class FSC147(Dataset):

	def __init__(
			self,
			datadir,
			split="train",
			n_exemplars=3,
			image_size=512,
			hflip_p=0.5,
			cj_p=0.0,
			sigma=0.5,
			center_pad=False,
			bbox_max_size=None,
			bbox_min_size=None,
			target_minmax_norm=False,
			cachedir=None,
	):
		self.datadir = datadir
		self.split = split
		self.n_exemplars = n_exemplars
		self.image_size = image_size
		self.hflip_p = hflip_p
		self.cj_p = cj_p
		self.sigma = sigma
		self.center_pad = center_pad
		self.bbox_max_size = bbox_max_size
		self.bbox_min_size = bbox_min_size
		self.target_minmax_norm = target_minmax_norm
		self.cachedir = cachedir

		assert self.split in ["train", "train_val", "test_val", "test"]
		assert isinstance(self.sigma, float) or self.sigma == "adaptive"

		self.adaptive_sigma = self.sigma == "adaptive"

		self.img_names = None
		with open(os.path.join(self.datadir, "Train_Test_Val_FSC_147.json"), "rb") as f:
			_split = "val" if self.split in ["train_val", "test_val"] else self.split
			self.img_names = json.load(f)[_split]

		self.annotations = None
		with open(os.path.join(self.datadir, "annotation_FSC147_384.json"), "rb") as f:
			self.annotations = {k: v for k, v in json.load(f).items() if k in self.img_names}

		self.img_classes = None
		with open(os.path.join(self.datadir, "ImageClasses_FSC147.txt"), "r") as f:
			self.img_classes = {k: v for (k, v) in csv.reader(f, delimiter="\t")}

		if split in ["train", "train_val"]:
			assert self.cachedir is not None
			self.targetdir = os.path.join(self.cachedir, "targets")
			if self.adaptive_sigma:
				self.targetdir = os.path.join(self.datadir, "gt_densitymaps_adaptive_384_VarV2")
			os.makedirs(self.targetdir, exist_ok=True)


	def generate_density_map(self, src_size, new_size, points):
		h, w = src_size
		new_h, new_w = new_size
		rh, rw = new_h / h, new_w / w
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


	def __train_val_transform(self, img, bboxes):
		# ToTensor
		img = F.to_tensor(img) # (C, H, W)

		# Resize
		img, scale = resize(img, self.image_size)
		bboxes = adjust_bboxes(bboxes, scale_factor=scale)

		# Scale to bboxes
		img, scale = scale_to_bbox(img, bboxes, self.image_size, self.bbox_max_size, self.bbox_min_size)
		new_size = img.shape[-2:]

		# Pad
		img, padding = pad(img, self.image_size, center=self.center_pad)
		bboxes = adjust_bboxes(bboxes, scale_factor=scale, padding=padding)
		assert img.shape[-1] == img.shape[-2] == self.image_size
		
		img = (img * 2.0 - 1.0).clamp(-1.0, 1.0)
		return img, bboxes, False, new_size


	def __train_transform(self, img, bboxes):
		# ToTensor
		img = F.to_tensor(img) # (C, H, W)

		# Resize
		img, scale = resize(img, self.image_size)
		bboxes = adjust_bboxes(bboxes, scale_factor=scale)

		# Scale to bboxes
		img, scale = scale_to_bbox(img, bboxes, self.image_size, self.bbox_max_size, self.bbox_min_size)
		new_size = img.shape[-2:]

		# Pad
		img, padding = pad(img, self.image_size, center=self.center_pad)
		bboxes = adjust_bboxes(bboxes, scale_factor=scale, padding=padding)
		assert img.shape[-1] == img.shape[-2] == self.image_size

		hflip = False
		# RandomHorizontalFlip
		if th.rand(1) < self.hflip_p:
			img = F.hflip(img)
			bboxes[:, [0, 2]] = self.image_size - bboxes[:, [2, 0]]
			hflip = True

		# RandomColorJitter
		if th.rand(1) < self.cj_p:
			img = transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)(img)

		img = (img * 2.0 - 1.0).clamp(-1.0, 1.0)
		return img, bboxes, hflip, new_size


	def __test_transfrom(self, img, bboxes):
		return self.__train_val_transform(img, bboxes)


	def __test_val_transform(self, img, bboxes):
		return self.__train_val_transform(img, bboxes)


	def transform(self, img, bboxes, split):
		if split == "train":
			img, bboxes, hflip, new_size = self.__train_transform(img, bboxes)
		elif split == "train_val":
			img, bboxes, hflip, new_size = self.__train_val_transform(img, bboxes)
		elif split == "test_val":
			img, bboxes, hflip, new_size = self.__test_val_transform(img, bboxes)
		elif split == "test":
			img, bboxes, hflip, new_size = self.__test_transfrom(img, bboxes)
		else:
			raise ValueError(f"Unknown split '{split}'")
		return img, bboxes, hflip, new_size
	

	def target_transform(self, target, hflip=False, target_size=None):
		# ToTensor
		target = F.to_tensor(target)

		# Resize if necessery
		if target_size is not None:
			target = resize_dm(target, target_size)

		# Pad
		target, _ = pad(target, self.image_size, center=self.center_pad)

		# Horizontal flip
		if hflip:
			target = F.hflip(target)

		# MinMax Normalization
		if self.target_minmax_norm:
			_tmax = target.max()
			_tmin = target.min()
			target = (target - _tmin) / (_tmax - _tmin)

		target = (target * 2.0 - 1.0).clamp(-1.0, 1.0)
		return target


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
		src_size = img.size[::-1]

		bboxes = th.as_tensor(
			self.annotations[self.img_names[index]]['box_examples_coordinates'], 
			dtype=th.float32
		)
		assert len(bboxes) >= self.n_exemplars, f'Not enough examplars for image {self.img_names[index]}'

		bboxes = bboxes[:, [0, 2], :].reshape(-1, 4)
		bboxes = bboxes[th.randperm(bboxes.shape[0])]
		bboxes = bboxes[:self.n_exemplars, ...]	# (x_min, y_min, x_max, y_max)
		
		points = self.annotations[self.img_names[index]]['points']
		target_count = th.tensor(len(points), dtype=th.float32)

		img, bboxes, hflip, new_size = self.transform(img, bboxes, split=self.split)

		img_id = os.path.splitext(self.img_names[index])[0]
		npypath = os.path.join(
			self.targetdir, 
			img_id + '.npy'
		) if self.cachedir is not None else None

		if npypath is not None and os.path.exists(npypath):
			target = np.load(npypath)
		else:
			target = self.generate_density_map(
				src_size,
				new_size,
				points
			)
			if self.split in ["train", "train_val"]:
				np.save(npypath, target)

		target = self.target_transform(
			target,
			hflip, 
			target_size=new_size if self.adaptive_sigma else None
		)
		assert target.shape[-2:] == img.shape[-2:], "target shape does not match image shape."
		return target, dict(bboxes=bboxes, img=img, count=target_count, id=img_id)


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