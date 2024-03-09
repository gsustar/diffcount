import torch as th
import numpy as np

from .base_diffusion import BaseDiffusion
from .torch_dct import dct_2d, idct_2d
from .torch_gb import gaussian_blur
from .nn import mean_flat


def get_named_blur_schedule(schedule_name, num_timesteps, min_sigma=0.5, max_sigma=20.0):
	if schedule_name == "log":
		blur_schedule = np.exp(np.linspace(np.log(min_sigma),
										   np.log(max_sigma), 
										   num_timesteps))
	else :
		raise ValueError(f"Invalid blur schedule name: {schedule_name}")
	return blur_schedule


class DeblurDiffusion(BaseDiffusion):

	def __init__(
		self,
		blur_sigmas,
		image_size,
		loss_type="l1",
		use_dct=False,
		delta=0.01,
	):
		self.image_size = image_size
		self.loss_type = loss_type
		self.use_dct = use_dct
		self.init_sample_set = None

		blur_sigmas = np.array(blur_sigmas, dtype=np.float64)
		self.blur_sigmas = np.append(0.0, blur_sigmas)
		assert len(self.blur_sigmas.shape) == 1, "blur_sigmas must be 1-D"

		self.num_timesteps = blur_sigmas.shape[0]

		self.delta_train = delta
		self.delta_sample = self.delta_train * 1.25
		self.sampling_range = (-0.9, -0.6)

		dissipation_time = self.blur_sigmas ** 2 / 2
		freqs = np.pi * np.linspace(0, self.image_size-1, self.image_size) / self.image_size
		lmbda = freqs[:, None]**2 + freqs[None, :]**2
		self.freq_scaling = np.exp(-lmbda * dissipation_time[:, None, None, None])

	# def get_init_sample(self, shape, device):
	# 	r1, r2 = self.sampling_range
	# 	val = (r1 - r2) * th.rand(shape[0], 1, 1, 1) + r2
	# 	return (th.ones(shape) * val).to(device)
		
	def set_init_sample_set(self, init_sample_set):
		self.init_sample_set = init_sample_set

	def get_init_sample(self, shape, device):
		assert self.init_sample_set is not None
		init_sample = next(iter(self.init_sample_set))[0].to(device)
		init_sample = self.q_sample(init_sample,
									(self.num_timesteps-1) * th.ones(init_sample.shape[0], dtype=th.long).to(device))
		return init_sample

	def q_sample(self, x_start, t, noise=None):
		if self.use_dct:
			return self._dct_q_sample(x_start, t, noise=noise)
		return self._pix_q_sample(x_start, t, noise=noise)

	def _pix_q_sample(self, x_start, t, noise=None):
		return gaussian_blur(
			x_start,
			2 * self.image_size - 1,
			_extract_into_tensor(self.blur_sigmas, t)
		)
	
	def _dct_q_sample(self, x_start, t, noise=None):
		return idct_2d(
			_extract_into_tensor(self.freq_scaling, t) * dct_2d(x_start, norm="ortho"), 
			norm="ortho"
		)

	def p_sample(self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None):
		u_mean = model(x, t, **model_kwargs) + x
		noise = th.randn_like(x)
		return u_mean + noise*self.delta_sample

	def p_sample_loop(
		self,
		model,
		shape,
		noise=None,
		clip_denoised=True,
		denoised_fn=None,
		model_kwargs=None,
		device=None,
		progress=False,
	):
		final = None
		for sample in self.p_sample_loop_progressive(
			model,
			shape,
			noise=noise,
			clip_denoised=clip_denoised,
			denoised_fn=denoised_fn,
			model_kwargs=model_kwargs,
			device=device,
			progress=progress,
		):
			final = sample
		return final

	def p_sample_loop_progressive(
		self,
		model,
		shape,
		noise=None,
		clip_denoised=True,
		denoised_fn=None,
		model_kwargs=None,
		device=None,
		progress=False,
	):
		if device is None:
			device = next(model.parameters()).device
		assert isinstance(shape, (tuple, list))

		img = self.get_init_sample(shape, device)
		indices = list(range(self.num_timesteps))[::-1]

		if progress:
			# Lazy import so that we don't depend on tqdm.
			from tqdm.auto import tqdm

			indices = tqdm(indices)

		for i in indices:
			t = th.tensor([i] * shape[0], device=device)
			with th.no_grad():
				out = self.p_sample(
					model,
					img,
					t,
					clip_denoised=clip_denoised,
					denoised_fn=denoised_fn,
					model_kwargs=model_kwargs,
				)
				yield out
				img = out
	
	def training_losses(self, model, x_start, t, model_kwargs=None):
		x_t = self.q_sample(x_start, t)
		x_next = self.q_sample(x_start, t+1)

		noise = th.randn_like(x_next) * self.delta_train
		perturbed_data = noise + x_next
		model_output = model(perturbed_data, t, **model_kwargs)
		prediction = perturbed_data + model_output

		if self.loss_type == "l1":
			losses = (x_t - prediction).abs()
		elif self.loss_type == "l2":
			losses = (x_t - prediction)**2
		else:
			raise ValueError(f"Invalid loss type: {self.loss_type}")
		loss = mean_flat(losses)
		return {"loss": loss}


def _extract_into_tensor(arr, timesteps):
	"""
	Extract values from a 1-D numpy array for a batch of indices.

	:param arr: the 1-D numpy array.
	:param timesteps: a tensor of indices into the array to extract.
	"""
	res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
	return res