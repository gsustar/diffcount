import torch as th
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import make_grid
import io

def _ready_for_plotting(t):
	return (
		t.dtype == th.uint8
		and t.min() >= 0
		and t.max() <= 255
	)

def _maybe_to_plotting_range(x):
	if _ready_for_plotting(x):
		return x
	return (
		x
		.add(1)
		.div_(2)
		.mul_(255)
		.add_(0.5)
		.clamp_(0, 255)
		.to("cpu", th.uint8)
		.detach()
	)

def to_pil_image(tensor, cmap='gray', **grid_kwargs):
	# tensor must be in range [-1, 1] or already in [0, 255]
	if isinstance(tensor, Image.Image):
		return tensor
	grid_defaults = dict(
		nrow=int(
			np.sqrt(tensor.shape[0])
		),
		padding=2,
		pad_value=0.0
	)
	_, C, _, _ = tensor.shape
	grid_defaults.update(grid_kwargs)
	tensor = _maybe_to_plotting_range(tensor)
	grid = make_grid(tensor, **grid_defaults)
	grid = grid.permute(1, 2, 0)
	grid = grid.numpy()
	if C == 1:
		cm = plt.get_cmap(cmap)
		grid = cm(grid[:, :, 0], bytes=True)
		# grid = (grid * 255).astype(np.uint8)
	return Image.fromarray(grid)


def draw_denoising_process(imgs):
	assert imgs[0].dim() == 4
	imgs = th.stack(imgs)
	imgs = imgs[:, 0, :, :, :]
	imgs = to_pil_image(imgs, nrow=len(imgs), padding=1, pad_value=127)
	return imgs


def draw_result(img, density, pred_count, target_count, pred_coords=None):
	img = to_pil_image(img)
	density = to_pil_image(density, cmap="viridis")

	density.putalpha(192)
	img.paste(density, (0, 0), density)

	draw = ImageDraw.Draw(img)
	if pred_coords is not None:
		for coord in pred_coords:
			draw.circle(coord, radius=1, fill=(255, 0, 0))
	font = ImageFont.load_default(size=10)
	draw.text((0, 0), f"PR: {pred_count:>.1f}", fill="white", font=font)
	draw.text((0, 12), f"GT: {target_count:>.1f}", fill="chartreuse", font=font)
	return img


def draw_bboxes(img, bboxes):
	imgs = [
		torchvision.utils.draw_bounding_boxes(
			img, boxes=bboxes[i], colors="red"
		)
		for i, img in enumerate(_maybe_to_plotting_range(img))
	]
	return th.stack(imgs)


def draw_cls(xc):
	txts = list()
	size = 28, 28
	for c in xc.tolist():
		txt = Image.new("RGB", size, color="white")
		draw = ImageDraw.Draw(txt)
		font = ImageFont.load_default(size=24)
		draw.text((0, 0), str(c), fill="black", font=font)
		txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
		txts.append(txt)
	txts = np.stack(txts)
	return th.tensor(txts)