"""
Taken from:
https://github.com/AaltoML/generative-inverse-heat-dissipation/blob/main/model_code/torch_dct.py
"""

import torch as th
import numpy as np

def dct(x, norm=None):
	"""
	Discrete Cosine Transform, Type II (a.k.a. the DCT)
	For the meaning of the parameter `norm`, see:
	https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
	:param x: the input signal
	:param norm: the normalization, None or 'ortho'
	:return: the DCT-II of the signal over the last dimension
	"""
	x_shape = x.shape
	N = x_shape[-1]
	x = x.contiguous().view(-1, N)

	v = th.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

	#Vc = th.fft.rfft(v, 1)
	Vc = th.view_as_real(th.fft.fft(v, dim=1))

	k = - th.arange(N, dtype=x.dtype,
					   device=x.device)[None, :] * np.pi / (2 * N)
	W_r = th.cos(k)
	W_i = th.sin(k)

	V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

	if norm == 'ortho':
		V[:, 0] /= np.sqrt(N) * 2
		V[:, 1:] /= np.sqrt(N / 2) * 2

	V = 2 * V.view(*x_shape)

	return V


def idct(X, norm=None):
	"""
	The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
	Our definition of idct is that idct(dct(x)) == x
	For the meaning of the parameter `norm`, see:
	https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
	:param X: the input signal
	:param norm: the normalization, None or 'ortho'
	:return: the inverse DCT-II of the signal over the last dimension
	"""

	x_shape = X.shape
	N = x_shape[-1]

	X_v = X.contiguous().view(-1, x_shape[-1]) / 2

	if norm == 'ortho':
		X_v[:, 0] *= np.sqrt(N) * 2
		X_v[:, 1:] *= np.sqrt(N / 2) * 2

	k = th.arange(x_shape[-1], dtype=X.dtype,
					 device=X.device)[None, :] * np.pi / (2 * N)
	W_r = th.cos(k)
	W_i = th.sin(k)

	V_t_r = X_v
	V_t_i = th.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

	V_r = V_t_r * W_r - V_t_i * W_i
	V_i = V_t_r * W_i + V_t_i * W_r

	V = th.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

	#v = th.fft.irfft(V, 1)
	v = th.fft.irfft(th.view_as_complex(V), n=V.shape[1], dim=1)
	x = v.new_zeros(v.shape)
	x[:, ::2] += v[:, :N - (N // 2)]
	x[:, 1::2] += v.flip([1])[:, :N // 2]

	return x.view(*x_shape)


def dct_2d(x, norm=None):
	"""
	2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)
	For the meaning of the parameter `norm`, see:
	https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
	:param x: the input signal
	:param norm: the normalization, None or 'ortho'
	:return: the DCT-II of the signal over the last 2 dimensions
	"""
	X1 = dct(x, norm=norm)
	X2 = dct(X1.transpose(-1, -2), norm=norm)
	return X2.transpose(-1, -2)


def idct_2d(X, norm=None):
	"""
	The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III
	Our definition of idct is that idct_2d(dct_2d(x)) == x
	For the meaning of the parameter `norm`, see:
	https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
	:param X: the input signal
	:param norm: the normalization, None or 'ortho'
	:return: the DCT-II of the signal over the last 2 dimensions
	"""
	x1 = idct(X, norm=norm)
	x2 = idct(x1.transpose(-1, -2), norm=norm)
	return x2.transpose(-1, -2)


def dct_3d(x, norm=None):
	"""
	3-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)
	For the meaning of the parameter `norm`, see:
	https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
	:param x: the input signal
	:param norm: the normalization, None or 'ortho'
	:return: the DCT-II of the signal over the last 3 dimensions
	"""
	X1 = dct(x, norm=norm)
	X2 = dct(X1.transpose(-1, -2), norm=norm)
	X3 = dct(X2.transpose(-1, -3), norm=norm)
	return X3.transpose(-1, -3).transpose(-1, -2)


def idct_3d(X, norm=None):
	"""
	The inverse to 3D DCT-II, which is a scaled Discrete Cosine Transform, Type III
	Our definition of idct is that idct_3d(dct_3d(x)) == x
	For the meaning of the parameter `norm`, see:
	https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
	:param X: the input signal
	:param norm: the normalization, None or 'ortho'
	:return: the DCT-II of the signal over the last 3 dimensions
	"""
	x1 = idct(X, norm=norm)
	x2 = idct(x1.transpose(-1, -2), norm=norm)
	x3 = idct(x2.transpose(-1, -3), norm=norm)
	return x3.transpose(-1, -3).transpose(-1, -2)