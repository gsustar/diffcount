import torch as th
import torchvision.transforms.functional as F

from itertools import product
from math import ceil
from collections.abc import Sequence



def resize(image, target_size, **kwargs):
	image_h, image_w = image.shape[-2:]
	target_h, target_w = __validate_size_type(target_size)
	scale_h = target_h / image_h
	scale_w = target_w / image_w
	resize_image = F.resize(
		image,
		size=(target_h, target_w),
		antialias=True,
		**kwargs
	)
	return resize_image, (scale_h, scale_w)


def resize_dm(density_map, target_size, **kwargs):
	target_h, target_w = __validate_size_type(target_size)
	_sum = density_map.sum()
	resize_density_map = F.resize(
		density_map,
		size=(target_h, target_w),
		antialias=True,
		**kwargs
	)
	resize_density_map = resize_density_map / resize_density_map.sum() * _sum
	return resize_density_map


def scale(image, scale_factor, **kwargs):
	if isinstance(scale_factor, (int, float)):
		scale_factor = (scale_factor, scale_factor)
	image_h, image_w = image.shape[-2:]
	scale_h, scale_w = scale_factor
	target_h = int(image_h * scale_h)
	target_w = int(image_w * scale_w)
	resize_image = F.resize(
		image,
		size=(target_h, target_w),
		antialias=True,
		**kwargs
	)
	return resize_image, (target_h, target_w)


def pad(image, target_size, center=False):
	image_h, image_w = image.shape[-2:]
	resize_h, resize_w = __validate_size_type(target_size)
	pad_h = resize_h - image_h
	pad_w = resize_w - image_w
	pad_t = 0
	pad_l = 0
	if center:
		pad_t = pad_h // 2
		pad_l = pad_w // 2
	pad_b = pad_h - pad_t
	pad_r = pad_w - pad_l
	padding = [pad_l, pad_t, pad_r, pad_b]
	pad_image = F.pad(image, padding)
	return pad_image, padding


def unpad(image, pad):
	image_h, image_w = image.shape[-2:]
	pad_l, pad_t, pad_r, pad_b = pad
	return image[
		:, 
		:, 
		pad_t:(image_h - pad_b), 
		pad_l:(image_w - pad_r)
	]


def adjust_bboxes(bboxes, scale_factor=(1.0, 1.0), padding=[0, 0, 0, 0]):
	if isinstance(scale_factor, (int, float)):
		scale_factor = (scale_factor, scale_factor)
	scale_h, scale_w = scale_factor
	pad_l, pad_t, _, _ = padding
	bboxes = bboxes * th.tensor([scale_w, scale_h, scale_w, scale_h])
	bboxes = bboxes + th.tensor([pad_l, pad_t, pad_l, pad_t])
	return bboxes


def scale_to_bbox(image, bboxes, target_size, bbox_max_size=50, bbox_min_size=11):
	target_size = __validate_size_type(target_size)

	scale_w = 1.0
	scale_h = 1.0

	if bbox_max_size is not None:
		scale_w = min(1.0, bbox_max_size / (bboxes[:, 2] - bboxes[:, 0]).mean())
		scale_h = min(1.0, bbox_max_size / (bboxes[:, 3] - bboxes[:, 1]).mean())

	if bbox_min_size is not None and not (scale_h < 1.0 or scale_w < 1.0):
		scale_w = min(max(1.0, bbox_min_size / (bboxes[:, 2] - bboxes[:, 0]).mean()), 1.9)
		scale_h = min(max(1.0, bbox_min_size / (bboxes[:, 3] - bboxes[:, 1]).mean()), 1.9)

	scale_w = (int(target_size[1] * scale_w) // 8 * 8) / target_size[1]
	scale_h = (int(target_size[0] * scale_h) // 8 * 8) / target_size[0]

	if scale_w != 1.0 or scale_h != 1.0:
		image, _ = scale(image, (scale_h, scale_w))

	return image, (scale_h, scale_w)



def preprocess_fsc147():
	pass


# Below code is modified from: https://github.com/openvinotoolkit/anomalib/blob/main/src/anomalib/data/utils/tiler.py
# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

def __validate_size_type(parameter):
	if isinstance(parameter, int):
		output = (parameter, parameter)
	elif isinstance(parameter, Sequence):
		output = (parameter[0], parameter[1])
	else:
		msg = f"Unknown type {type(parameter)} for tile or stride size. Could be int or Sequence type."
		raise TypeError(msg)

	if len(output) != 2:
		msg = f"Length of the size type must be 2 for height and width. Got {len(output)} instead."
		raise ValueError(msg)

	return output


def compute_new_image_size(image_size, tile_size, stride):
	"""Check if image size is divisible by tile size and stride. 
	If not divisible, it resizes the image size to make it divisible."""

	def __compute_new_edge_size(edge_size: int, tile_size: int, stride: int):
		"""Resize within the edge level."""
		if (edge_size - tile_size) % stride != 0:
			edge_size = (ceil((edge_size - tile_size) / stride) * stride) + tile_size
		return edge_size

	resized_h = __compute_new_edge_size(image_size[0], tile_size[0], stride[0])
	resized_w = __compute_new_edge_size(image_size[1], tile_size[1], stride[1])

	return resized_h, resized_w


def unfold(tensor, tile_size, stride):
		"""Unfolds tensor into tiles."""
		tile_size_h, tile_size_w = __validate_size_type(tile_size)
		stride_h, stride_w = __validate_size_type(stride)
		# identify device type based on input tensor
		device = tensor.device

		# extract and calculate parameters
		batch, channels, image_h, image_w = tensor.shape

		num_patches_h = int((image_h - tile_size_h) / stride_h) + 1
		num_patches_w = int((image_w - tile_size_w) / stride_w) + 1

		# create an empty torch tensor for output
		tiles = th.zeros(
			(num_patches_h, num_patches_w, batch, channels, tile_size_h, tile_size_w),
			device=device,
		)

		# fill-in output tensor with spatial patches extracted from the image
		for (tile_i, tile_j), (loc_i, loc_j) in zip(
			product(range(num_patches_h), range(num_patches_w)),
			product(
				range(0, image_h - tile_size_h + 1, stride_h),
				range(0, image_w - tile_size_w + 1, stride_w),
			),
			strict=True,
		):
			tiles[tile_i, tile_j, :] = tensor[
				:,
				:,
				loc_i : (loc_i + tile_size_h),
				loc_j : (loc_j + tile_size_w),
			]

		return tiles.permute(2, 0, 1, 3, 4, 5).contiguous()


def fold(tiles, stride):
		"""Fold the tiles back into the original tensor."""
		stride_h, stride_w = __validate_size_type(stride)
		batch_size, num_patches_h, num_patches_w, num_channels, tile_size_h, tile_size_w = tiles.shape

		# identify device type based on input tensor
		device = tiles.device

		# reconstructed image dimension
		resized_h = tile_size_h + (num_patches_h - 1) * stride_h
		resized_w = tile_size_w + (num_patches_w - 1) * stride_w
		image_size = (batch_size, num_channels, resized_h, resized_w)

		# rearrange input tiles in format [tile_count, batch, channel, tile_h, tile_w]
		tiles = tiles.contiguous().view(
			batch_size,
			num_patches_h,
			num_patches_w,
			num_channels,
			tile_size_h,
			tile_size_w,
		)
		tiles = tiles.permute(0, 3, 1, 2, 4, 5)
		tiles = tiles.contiguous().view(batch_size, num_channels, -1, tile_size_h, tile_size_w)
		tiles = tiles.permute(2, 0, 1, 3, 4)

		# create tensors to store intermediate results and outputs
		img = th.zeros(image_size, device=device)
		lookup = th.zeros(image_size, device=device)
		ones = th.ones(tile_size_h, tile_size_w, device=device)

		# reconstruct image by adding patches to their respective location and
		# create a lookup for patch count in every location
		for patch, (loc_i, loc_j) in zip(
			tiles,
			product(
				range(
					0,
					resized_h - tile_size_h + 1,
					stride_h,
				),
				range(
					0,
					resized_w - tile_size_w + 1,
					stride_w,
				),
			),
			strict=True,
		):
			img[:, :, loc_i : (loc_i + tile_size_h), loc_j : (loc_j + tile_size_w)] += patch
			lookup[:, :, loc_i : (loc_i + tile_size_h), loc_j : (loc_j + tile_size_w)] += ones

		# divide the reconstucted image by the lookup to average out the values
		img = th.divide(img, lookup)
		# alternative way of removing nan values
		img[img != img] = 0

		return img



if __name__ == "__main__":
	import matplotlib.pyplot as plt

	img = th.rand(2, 3, 100, 100)
	img, padding = pad(img, (150, 150), center=True)
	# print(img.shape)
	# plt.imshow(img[0].permute(1, 2, 0))
	# plt.show()
	# exit(0)
	print(img.shape)
	tiles = unfold(img, tile_size=(50, 50), stride=(25, 25))
	recon = fold(tiles, stride=(25, 25))
	# tile_size = 50
	# stride = int(0.9 * tile_size)
	# tiler = Tiler(tile_size=tile_size, stride=stride, mode="padding_center")
	# tiles = tiler.tile(img)
	print(tiles.shape)
	print(recon.shape)

	print(th.equal(img, recon))
	plt.imshow(img[0].permute(1, 2, 0))
	plt.show()
	plt.imshow(recon[0].permute(1, 2, 0))
	plt.show()