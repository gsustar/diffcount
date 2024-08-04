import argparse
import torch as th
import os.path as osp
import warnings

from diffcount import logger
from diffcount.ema import ExponentialMovingAverage
from diffcount.plot_utils import draw_result
from diffcount.count_utils import counting
from diffcount.nn import (
	torch_to, 
	possibly_vae_decode, 
	encode
)
from diffcount.script_util import (
	create_model,
	create_diffusion,
	create_data,
	create_conditioner,
	create_vae,
	parse_config,
)

warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
	args = parse_args()
	expdir = args.expdir
	config = parse_config(osp.join(expdir, "config.yaml"))
	dev = "cuda" if th.cuda.is_available() else "cpu"
	ckpt = th.load(osp.join(expdir, args.checkpoint), map_location=dev)

	logger.configure(
		dir=expdir, 
		format_strs=['stdout', 'log'],
		log_suffix=f'_eval'
	)

	use_ddim = False
	if args.ddim_steps is not None:
		use_ddim = True
		config.diffusion.params.timestep_respacing = f"ddim{args.ddim_steps}"
	
	logger.log("creating model...")
	model = create_model(config.model)
	model.to(dev)
	model.load_state_dict(ckpt["model"])
	model.eval()

	logger.log("creating diffusion...")
	diffusion = create_diffusion(config.diffusion)

	logger.log("creating VAE...")
	vae = create_vae(
		getattr(config, "vae", None), device=dev
	)

	logger.log("creating data...")
	config.data.dataloader.params.overfit_single_batch = False
	config.data.dataloader.params.batch_size = args.batch_size
	_, val_data, test_data = create_data(config.data, train=False)

	logger.log("creating conditioner...")
	conditioner = create_conditioner(
		getattr(config, "conditioner", []),
		train=False
	)
	conditioner.to(dev)
	conditioner.load_state_dict(ckpt["conditioner"])
	conditioner.eval()

	logger.log("creating EMA...")
	ema = ExponentialMovingAverage(
		model.parameters(),
		decay=config.train.ema_rate
	)
	ema.load_state_dict(ckpt["ema"])

	splits = [args.split] if args.split is not None else ["val", "test"]
	for split in splits:
		logger.log(f"Evaluating on {split.upper()} set...")
		eval_data = val_data if split == "val" else test_data
		N = len(eval_data)
		MAE = 0.0
		RMSE = 0.0
		for i, (batch, cond) in enumerate(eval_data):
			batch = torch_to(batch, dev)
			cond = torch_to(cond, dev)
			en_batch, en_cond = encode(batch.clone(), cond.copy(), vae)
			sample_fn = (
				diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
			)
			with th.autocast(device_type=dev, dtype=th.float16, enabled=args.use_fp16):
				with ema.average_parameters(model.parameters()):
					samples = sample_fn(
						model,
						en_batch.shape,
						model_kwargs=dict(
							cond=conditioner(en_cond)
						),
						clip_denoised=True,
					)
		
			target_count = cond["count"].float().cpu()
			samples = possibly_vae_decode(samples, vae, clip_decoded=True)
			pred_count = th.zeros_like(target_count)
			for j, s in enumerate(samples):
				count, coords = counting(s)
				pred_count[j] = count

				if not args.skip_plotting:
					img = cond["img"][j].unsqueeze(0)
					density = s.unsqueeze(0)
					res = draw_result(img, density, float(count), target_count[j], coords)
					logger.logimg(res, name=f"{i*args.batch_size + j}", step=f"{split}")

			logger.log(f"{i+1}/{N}")
			MAE += th.sum(th.abs(target_count - pred_count))
			RMSE += th.sum((target_count - pred_count) ** 2)

		RMSE = th.sqrt(RMSE / N)
		MAE = MAE / N

		logger.log(f"{split.upper()}")
		logger.log(f"RMSE: {RMSE.item():.4f}")
		logger.log(f"MAE: {MAE.item():.4f}")


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--expdir", type=str)
	parser.add_argument("--checkpoint", type=str)
	parser.add_argument("--batch_size", type=int, default=16)
	parser.add_argument("--ddim_steps", type=int, default=None)
	parser.add_argument("--use_fp16", action="store_true")
	parser.add_argument("--skip_plotting", action="store_true")
	parser.add_argument("--split", type=str, default=None)
	return parser.parse_args()


if __name__ == "__main__":
	main()