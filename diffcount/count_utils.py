import torch as th
import torch.nn as nn
import numpy as np
import cv2

from skimage.feature import peak_local_max


def counting(d, mode="nms", threshold="auto", combine_mode="mean"):

	assert d.dim() == 3
	d = d.clamp_(-1.0, 1.0)
	d = (d + 1.0) / 2.0
	d = d.detach().cpu().numpy()

	if combine_mode == "mean":
		d = d.mean(axis=0).squeeze()
	elif combine_mode == "max":
		d = d.max(axis=0).squeeze()
	else:
		raise ValueError(f"Unsupported combine mode {combine_mode}")

	if threshold == "auto":
		threshold = d.mean() + 0.1
	assert isinstance(threshold, float)
	d[d < threshold] = 0.0

	coords = None
	if mode == "nms":
		coords = peak_local_max(d, exclude_border=0)
		cnt = len(coords)

	elif mode == "sum":
		cnt = d.sum(axis=(1,2,3))

	elif mode == "contour":
		d = (d > 0.0).astype(np.uint8) * 255
		conts, _ = cv2.findContours(d[0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnt = len(conts)

	else:
		raise ValueError(f"Unsupported counting mode {mode}")

	return cnt, coords


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


# if __name__ == "__main__":
# 	import matplotlib.pyplot as plt
# 	path = "/mnt/c/users/grega/downloads/final.npy"
# 	x = th.tensor(np.load(path)).float()
# 	# x[0] += 0.9
# 	# x[3, 0, 25, 25] = 0.11
# 	ix = 3
# 	i = (x[ix] + 1) / 2
# 	plt.imshow(i.squeeze())
# 	plt.show()
# 	cnt, coords = count(x[ix])

# 	print(cnt)
# 	plt.imshow(i.squeeze())
# 	plt.scatter(coords.T[1], coords.T[0], c="red")
# 	plt.show()
