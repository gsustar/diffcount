import torch as th
import numpy as np
import torchvision
from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import make_grid

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

def to_pil_image(tensor, **grid_kwargs):
	# tensor must be in range [-1, 1] or already in [0, 255]
	grid_defaults = dict(
		nrow=int(
			np.sqrt(tensor.shape[0])
		),
		padding=2,
		pad_value=0.0
	)
	grid_defaults.update(grid_kwargs)
	tensor = _maybe_to_plotting_range(tensor)
	grid = make_grid(tensor, **grid_defaults)
	grid = grid.permute(1, 2, 0)
	grid = grid.numpy()
	return Image.fromarray(grid)

def draw_bboxes(t, bboxes):
	imgs = [
		torchvision.utils.draw_bounding_boxes(
			img, boxes=bboxes[i], colors="red"
		)
		for i, img in enumerate(_maybe_to_plotting_range(t))
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