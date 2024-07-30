import torch as th
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt

from skimage.feature import peak_local_max


def remove_background(density, eps=0.1):
	avg = density.mean(dim=(1,2,3), keepdim=True)
	return th.where(density > avg+eps, density, -1.0)


def sum_count(density):
	r = remove_background(density)
	r = (r+1)/2
	return r.sum(dim=(1,2,3))


def threshold_count(density, threshold=0.0):
	return (density > threshold).sum(dim=(1,2,3))


def pmax_threshold_count(density, p=0.5):
	_max, _ = th.max(density.view(density.size(0), -1), dim=1)
	_max = _max[:, None, None, None]
	threshold = _max * p
	return threshold_count(density, threshold)


def nms_count(density, eps=0.1):
	density = density.clamp_(-1.0, 1.0).detach().cpu()
	density = (density + 1.0) / 2.0
	density = density.mean(dim=0).squeeze()
	density[density < eps] = 0.0
	coords = peak_local_max(density.numpy(), exclude_border=0)
	return len(coords), coords


def contour_count(density, threshold=0.0):
	r = remove_background(density)
	t = (r > threshold).cpu().detach().numpy().astype(np.uint8) * 255
	conts = []
	for e in t:
		cont, _ = cv2.findContours(e[0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		conts.append(len(cont))
	return th.tensor(conts)



class XSCountPredictor(nn.Module):

	def __init__(self, input_dim, hidden_dim=64):
		super().__init__()

		self.input_dim = input_dim
		self.hidden_dim = hidden_dim

		self.norm = nn.LayerNorm(self.input_dim)
		self.avgpool = nn.AdaptiveAvgPool2d(1)
		self.mlp = nn.Sequential(
			nn.Linear(self.input_dim, self.hidden_dim),
			nn.ReLU(),
			nn.Linear(self.hidden_dim, 1)
		)

	# todo check if work
	def forward(self, x):
		x = self.avgpool(x)
		x = x.flatten(start_dim=1)
		x = self.norm(x)
		x = self.mlp(x)
		return x



class CountingBranch(nn.Module):
	
	def __init__(self, feat_dims, hidden_dim=64):
		super().__init__()
		self.num_feats = len(feat_dims)
		self.input_dim = int(sum(feat_dims.values()))

		self.avgpool = nn.AdaptiveAvgPool2d(1)
		self.norm = nn.LayerNorm(self.input_dim)
		self.mlp = nn.Sequential(
			nn.Linear(self.input_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, 1)
		)

	def forward(self, feats):
		x = [feats[key] for key in feats]
		x = th.cat([self.avgpool(feats[key]) for key in feats], dim=1)
		x = x.flatten(start_dim=1)
		x = self.norm(x)
		x = self.mlp(x)
		return x


if __name__ == "__main__":
	path = "/mnt/c/users/grega/downloads/final.npy"
	x = th.tensor(np.load(path)).float()
	# x[0] += 0.9
	# x[3, 0, 25, 25] = 0.11
	# print(sum_count(x))
	# print(threshold_count(x, threshold=0.2))
	# print(pmax_threshold_count(x, p=0.4))
	# print(contour_count(x))
	# plt.imshow(((x[0]+1) / 2).squeeze())
	# plt.show()
	print(nms_count(x[0]))
