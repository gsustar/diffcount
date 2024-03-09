
class BaseDiffusion:

	def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
		raise NotImplementedError
	
	def q_sample(self, x_start, t, noise=None):
		raise NotImplementedError
	
	def p_sample(self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None):
		raise NotImplementedError
	
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
		raise NotImplementedError
	
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
		raise NotImplementedError
	
	def ddim_sample(
		self,
		model,
		x,
		t,
		clip_denoised=True,
		denoised_fn=None,
		model_kwargs=None,
		eta=0.0,
	):
		raise NotImplementedError
	
	def ddim_sample_loop_progressive(
		self,
		model,
		shape,
		noise=None,
		clip_denoised=True,
		denoised_fn=None,
		model_kwargs=None,
		device=None,
		progress=False,
		eta=0.0,
	):
		raise NotImplementedError
	
	def ddim_sample_loop(
		self,
		model,
		shape,
		noise=None,
		clip_denoised=True,
		denoised_fn=None,
		model_kwargs=None,
		device=None,
		progress=False,
		eta=0.0,
	):
		raise NotImplementedError