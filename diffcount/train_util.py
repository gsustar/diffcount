import blobfile as bf
import torch as th
import numpy as np
import pprint
import signal
import sys
import random
import os

from itertools import islice
from torch.optim import AdamW

from . import logger
from .resample import LossAwareSampler, UniformSampler
from .plot_utils import draw_bboxes, draw_cls, draw_denoising_process, draw_result
from .ema import ExponentialMovingAverage


class TrainLoop:
	def __init__(
		self,
		*,
		model,
		diffusion,
		data,
		val_data,
		conditioner,
		batch_size,
		lr,
		log_interval,
		save_interval,
		validation_interval,
		resume_checkpoint,
		device,
		ema_rate,
		use_fp16=False,
		schedule_sampler=None,
		weight_decay=0.0,
		num_epochs=0,
		grad_clip=0.0,
		lr_scheduler=None,
		seed=None
	):
		self.model = model
		self.diffusion = diffusion
		self.data = data
		self.val_data = val_data
		self.batch_size = batch_size
		self.lr = lr
		self.device = device
		self.log_interval = log_interval
		self.save_interval = save_interval
		self.validation_interval = validation_interval
		self.resume_checkpoint = resume_checkpoint
		self.use_fp16 = use_fp16
		self.ema_rate = ema_rate
		self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
		self.weight_decay = weight_decay
		self.num_epochs = num_epochs
		self.grad_clip = grad_clip
		self.lr_scheduler = lr_scheduler
		self.seed = seed

		self.step = 0
		self.epoch = 0
		# self.resume_step = 0
		# self.resume_epoch = 0

		self.conditioner = conditioner
		self.opt = self.configure_optimizer()
		self.scaler = th.cuda.amp.GradScaler(enabled=self.use_fp16)
		self.sch = self.configure_scheduler(self.opt)
		self.ema = ExponentialMovingAverage(
			self.model.parameters(),
			decay=ema_rate,
		)
		if self.resume_checkpoint:
			self.load()
		
		if self.seed is not None:
			seed_everything(self.seed)

		signal.signal(signal.SIGTERM, self.cleanup)
		signal.signal(signal.SIGINT, self.cleanup)

	def load(self):
		logger.log(f"loading model from checkpoint: {self.resume_checkpoint}...")
		checkpoint = th.load(self.resume_checkpoint, map_location=self.device)

		model_state_dict = checkpoint.get("model", checkpoint)
		optimizer_state_dict = checkpoint.get("optimizer", None)
		conditioner_state_dict = checkpoint.get("conditioner", None)
		scheduler_state_dict = checkpoint.get("scheduler", None)
		scaler_state_dict = checkpoint.get("scaler", None)
		ema_state_dict = checkpoint.get("ema", None)
		self.step = checkpoint.get("step", 0)
		self.epoch = checkpoint.get("epoch", 0)

		# self_model_state_dict = self.model.state_dict()
		# model_state_dict = {
		# 	k:v for k,v in model_state_dict.items() 
		# 	if k in self_model_state_dict and v.shape==self_model_state_dict[k].shape
		# }
		msd = dict()
		self_msd = self.model.state_dict()
		for k, v in model_state_dict.items():
			if k in self_msd and v.shape != self_msd[k].shape:
				logger.log(
					f"size mismatch for {k}: copying a param with shape {v.shape} from checkpoint, the shape in current model is {self_msd[k].shape}"
				)
			else:
				msd[k] = v
		m, u = self.model.load_state_dict(msd, strict=False)
		if m: logger.log(f"missing keys: {pprint.pformat(m)}\n")
		if u: logger.log(f"unexpected keys: {pprint.pformat(u)}\n")

		if optimizer_state_dict:
			self.opt.load_state_dict(optimizer_state_dict)
		if conditioner_state_dict:
			self.conditioner.load_state_dict(conditioner_state_dict)
		if scheduler_state_dict:
			self.sch.load_state_dict(scheduler_state_dict)
		if scaler_state_dict:
			self.scaler.load_state_dict(scaler_state_dict)
		if ema_state_dict:
			self.ema.load_state_dict(ema_state_dict)

	def run_loop(self):
		while (
			# (not self.num_epochs or self.epoch + self.resume_epoch < self.num_epochs)
			(not self.num_epochs or self.epoch < self.num_epochs)
		):
			self.run_epoch()
		# Save the last checkpoint if it wasn't already saved.
		if (self.epoch - 1) % self.save_interval != 0:
			self.save()

	def run_epoch(self):
		self.model.train()
		for batch, cond in self.data:
			self.run_step(batch, cond)
		if self.epoch % self.save_interval == 0 and self.epoch != 0:
			self.save()
		if self.epoch % self.validation_interval == 0 and self.epoch != 0:
			log_batch_with_cond(batch, cond, prefix="train", step=self.step)
			self.validate()
		self.epoch += 1

	def run_step(self, batch, cond):
		self.opt.zero_grad()
		batch = torch_to(batch, self.device)
		cond = torch_to(cond, self.device)
		count = cond.pop("count")
		t, weights = self.schedule_sampler.sample(batch.shape[0], self.device)
		with th.autocast(device_type=self.device, dtype=th.float16, enabled=self.use_fp16):
			losses = self.diffusion.training_losses(
				self.model,
				batch,
				t,
				model_kwargs=dict(
					cond=self.conditioner(cond),
					count=count,
				),
			)
		if isinstance(self.schedule_sampler, LossAwareSampler):
			self.schedule_sampler.update_with_all_losses(
				t, losses["loss"].detach()
			)
		loss = (losses["loss"] * weights).mean()

		self.scaler.scale(loss).backward()
		self.scaler.unscale_(self.opt)

		grad_norm, param_norm = self.compute_norms()
		if self.grad_clip > 0:
			th.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)

		self.scaler.step(self.opt)
		self.scaler.update()
		if self.sch is not None:
			self.sch.step()

		log_loss_dict(
			self.diffusion, t, {k: v * weights for k, v in losses.items()}
		)
		# logger.logkv("step", self.step + self.resume_step)
		logger.logkv("step", self.step)
		logger.logkv("epoch", self.epoch)
		logger.logkv("lr", self.opt.param_groups[0]["lr"])
		logger.logkv_mean("grad_norm", grad_norm)
		logger.logkv_mean("param_norm", param_norm)

		if self.step % self.log_interval == 0:
			logger.dumpkvs()
		self.step += 1

	def validate(self):
		logger.log("creating samples...")
		self.model.eval()
		batch, cond = next(iter(self.val_data))
		batch = torch_to(batch, self.device)
		cond = torch_to(cond, self.device)
		with th.autocast(device_type=self.device, dtype=th.float16, enabled=self.use_fp16):
			with self.ema.average_parameters(self.model.parameters()):
				samples = self.diffusion.p_sample_loop_progressive(
					self.model,
					batch.shape,
					model_kwargs=dict(
						cond=self.conditioner(cond)
					)
				)
				log_denoising_process(samples, self.diffusion, t_step=125, step=self.step)
				log_batch_with_cond(batch, cond, prefix="val", step=self.step)

	def save(self):
		logger.log(f"saving model...")
		filename = f"model{(self.epoch):06d}.pt"
		with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
			checkpoint = {
				"epoch": self.epoch,
				"step": self.step,
				"model": self.model.state_dict(),
				"optimizer": self.opt.state_dict(),
				"scheduler": self.sch.state_dict() if self.sch is not None else None,
				"scaler": self.scaler.state_dict(),
				"ema": self.ema.state_dict(),
				"conditioner": self.conditioner.state_dict()
			}
			th.save(checkpoint, f)
	
	def configure_optimizer(self):
		params = list(self.model.parameters())
		for embedder in self.conditioner.embedders:
			if embedder.is_trainable:
				params = params + list(embedder.parameters())
		opt = self.opt = AdamW(
			params, lr=self.lr, weight_decay=self.weight_decay
		)
		opt.register_step_post_hook(
			lambda optimizer, args, kwargs: self.ema.update(self.model.parameters()) 
		)
		return opt

	def configure_scheduler(self, opt):
		if self.lr_scheduler == "linear_warmup":
			sch = th.optim.lr_scheduler.LinearLR(
				optimizer=opt, 
				start_factor=0.01,
				total_iters=5000
			)
		elif self.lr_scheduler is None:
			sch = None
		else:
			raise ValueError(f"Unsupported lr_scheduler: {self.lr_scheduler}")
		return sch

	def compute_norms(self):
		grad_norm = 0.0
		param_norm = 0.0
		for p in self.model.parameters():
			with th.no_grad():
				param_norm += th.norm(p, p=2, dtype=th.float32).item() ** 2
				if p.grad is not None:
					grad_norm += th.norm(p.grad, p=2, dtype=th.float32).item() ** 2
		return np.sqrt(grad_norm), np.sqrt(param_norm)
	
	def cleanup(self, signum, frame):
		self.save()
		logger.log("closing...")
		logger.close()
		sys.exit(0)


def seed_everything(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	th.manual_seed(seed)
	th.cuda.manual_seed(seed)
	th.backends.cudnn.deterministic = True
	th.backends.cudnn.benchmark = True


def torch_to(x, *args, **kwargs):
	if isinstance(x, th.Tensor):
		return x.to(*args, **kwargs)
	elif isinstance(x, dict):
		return {k: torch_to(v, *args, **kwargs) for k, v in x.items()}
	elif isinstance(x, (list, tuple)):
		return [torch_to(v, *args, **kwargs) for v in x]
	else:
		raise ValueError(f"unsupported type: {type(x)}")


def get_blob_logdir():
	# You can change this to be a separate path to save checkpoints to
	# a blobstore or some external drive.
	return logger.get_dir()


def log_loss_dict(diffusion, ts, losses):
	for key, values in losses.items():
		logger.logkv_mean(key, values.mean().item())
		# Log the quantiles (four quartiles, in particular).
		for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
			quartile = int(4 * sub_t / diffusion.num_timesteps)
			logger.logkv_mean(f"{key}_q{quartile}", sub_loss)


def log_batch_with_cond(batch, cond, prefix="train", step=None):
	logger.logimg(batch, f"{prefix}_targets", step)
	if "img" in cond:
		img = cond["img"]
		if "bboxes" in cond:
			img = draw_bboxes(img, cond["bboxes"])
		logger.logimg(img, f"{prefix}_cond", step=step)
	if "cls" in cond:
		img = draw_cls(cond["cls"])
		logger.logimg(img, f"{prefix}_cond", step=step)


def log_denoising_process(samples, diffusion, t_step=125, step=None):
	assert diffusion.num_timesteps % t_step == 0
	out = []
	pred_xstart = []
	for i, s in enumerate(samples):
		if i % t_step == 0 or i == diffusion.num_timesteps - 1:
			out.append(s["sample"])
			pred_xstart.append(s["pred_xstart"])
	final = out[-1]
	imgs = draw_denoising_process(pred_xstart)
	logger.logimg(imgs, "pred_xstarts", step=step)
	logger.logimg(final, "final", step=step)
	logger.savetensor(final, "final", step)