import argparse
import torch as th
import os.path as osp

from PIL import Image
import matplotlib.pyplot as plt

from diffcount import logger
from diffcount.plot_utils import to_pil_image, draw_result
from diffcount.count import pmax_threshold_count
from diffcount.train_util import torch_to
from diffcount.script_util import (
	create_model_and_diffusion,
	create_data_and_conditioner,
	parse_config,
)


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
	
	logger.log("creating model and diffusion...")
	model, diffusion = create_model_and_diffusion(
		config.model, 
		config.diffusion
	)
	model.to(dev)
	model.load_state_dict(ckpt["model"])

	config_conditioner = None
	if hasattr(config, "conditioner"):
		config_conditioner = config.conditioner
	config_conditioner.params.is_trainable = False
	config.data.dataloader.params.overfit_single_batch = False
	config.data.dataloader.params.batch_size = args.batch_size

	logger.log("creating data loader and conditioner...")
	_, val_data, test_data, conditioner = create_data_and_conditioner(
		config.data, 
		config_conditioner,
		train=False
	)
	conditioner.to(dev)
	conditioner.load_state_dict(ckpt["conditioner"])

	model.eval()
	conditioner.eval()

	RMSE = {"val": 0.0, "test": 0.0}
	MAE = {"val": 0.0, "test": 0.0}
	for split in ["val", "test"]:
		logger.log(f"evaluating on {split} set...")
		eval_data = val_data if split == "val" else test_data
		N = len(eval_data)
		for i, (batch, cond) in enumerate(eval_data):
			batch = torch_to(batch, dev)
			cond = torch_to(cond, dev)
			sample_fn = (
				diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
			)
			with th.no_grad():
				with th.autocast(device_type=dev, dtype=th.float16, enabled=args.use_fp16):
					samples = sample_fn(
						model,
						(args.batch_size, 1, *batch.shape[2:]),
						model_kwargs=dict(
							cond=conditioner(cond)
						)
					)
				logger.log(f"{i+1}/{N}")

				
				target_count = cond["count"].float()
				pred_count = pmax_threshold_count(samples).float()

				if not args.skip_plotting:
					for j,s in enumerate(samples):
						img = cond["img"][j].unsqueeze(0)
						density = s.unsqueeze(0)
						res = draw_result(img, density, pred_count[j], target_count[j])
						logger.logimg(res, name=f"{i}_{j}", step="results")

				RMSE[split] += th.sqrt(th.mean((target_count - pred_count) ** 2)).item()
				MAE[split] += th.mean(th.abs(target_count - pred_count)).item()

		RMSE[split] /= N
		MAE[split] /= N

	logger.log(f"val RMSE: {RMSE['val']:.4f}")
	logger.log(f"val MAE: {MAE['val']:.4f}")
	logger.log(f"test RMSE: {RMSE['test']:.4f}")
	logger.log(f"test MAE: {MAE['test']:.4f}")	


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--expdir", type=str)
	parser.add_argument("--checkpoint", type=str)
	parser.add_argument("--batch_size", type=int, default=16)
	parser.add_argument("--ddim_steps", type=int, default=None)
	parser.add_argument("--use_fp16", action="store_true")
	parser.add_argument("--skip_plotting", action="store_false")
	return parser.parse_args()


if __name__ == "__main__":
	main()