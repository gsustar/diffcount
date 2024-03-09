""" Adjusted from: https://github.com/pytorch/vision/blob/main/torchvision/transforms/_functional_tensor.py """
import torch as th
import torch.nn.functional as F


def _gaussian_kernel1d(kernel_size, sigma):
	ksize_half = (kernel_size - 1) * 0.5
	x = th.arange(-ksize_half, ksize_half + 1, device=sigma.device)
	pdf = th.exp(-0.5 * (x / sigma).pow(2))
	kernel1d = pdf / pdf.sum()
	return kernel1d


def _gaussian_kernel2d(kernel_size, sigma, dtype, device):
	kernel1d_x = _gaussian_kernel1d(kernel_size, sigma).to(device, dtype=dtype)
	kernel1d_y = _gaussian_kernel1d(kernel_size, sigma).to(device, dtype=dtype)
	kernel2d = th.mm(kernel1d_y[:, None], kernel1d_x[None, :])
	return kernel2d


def _gaussian_blur(x, kernel_size, sigma, mode, value):
	kernel = _gaussian_kernel2d(kernel_size, sigma, dtype=x.dtype, device=x.device)
	kernel = kernel.expand(x.shape[-3], 1, kernel.shape[0], kernel.shape[1])
	# padding = (left, right, top, bottom)
	padding = [kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2]
	x = F.pad(x, padding, mode=mode, value=value)
	x = F.conv2d(x, kernel, groups=x.shape[-3])
	return x


def gaussian_blur(
		x,
		kernel_size,
		sigma,
		mode="reflect",
		value=-1.0
	):
	if isinstance(sigma, (float, int)):
		sigma = th.ones(x.shape[0], dtype=x.dtype, device=x.device) * float(sigma)
	if isinstance(sigma, (list, tuple)):
		sigma = th.tensor(sigma, dtype=x.dtype, device=x.device)
	sigma = th.flatten(sigma)
	assert sigma.shape[0] == x.shape[0]
	cval = value if mode == "constant" else None
	b = []
	for h, sig in zip(x, sigma):
		if sig > 0.0:
			h = _gaussian_blur(h, kernel_size, sig, mode, cval)
		b.append(h)
	return th.stack(b)