import torch as th
import numpy as np
from diffcount.datasets import MNIST, load_data, FSC147
import matplotlib.pyplot as plt
import time
from diffcount.plot_utils import to_pil_image
from diffcount.denoise_diffusion import get_named_beta_schedule
from diffcount.deblur_diffusion import gaussian_blur


def load_single_mnist_batch(bsize=1):
	dataloader = load_data(
		dataset=MNIST(data_root=".data/mnist", split='train'),
		batch_size=bsize,
		shuffle=True,
		overfit_single_batch=False,
	)
	batch = next(iter(dataloader))
	images, labels = batch
	return images


def load_single_random_dotmap(size, num_dots):
	img = th.zeros((1, 1, size, size)) - 1.0
	ps = th.randint(0, size, (num_dots, 2))
	for p in ps:
		img[0, 0, 0, 0] = 1.0
		img[0, 0, size//2, size//2] = 1.0
		img[0, 0, size//2, 0] = 1.0
		img[0, 0, p[0], p[1]] = 1.0
	return img


def gaussian_blur_sequential_convolutions(x_0, num_conv, sigs):
	final_sig = 0.0
	for i in range(num_conv):
		x_0 = gaussian_blur(x_0, sigs[i])
		final_sig = np.sqrt(final_sig**2 + sigs[i]**2)
		# final_sig += sigs[i]**2
	# return x_0, np.sqrt(final_sig)
	return x_0, final_sig


def sig_schedule(sig_start, sig_end, num_steps, schedule_type="linear"):
	if schedule_type == "constant":
		return [sig_start]*num_steps
	elif schedule_type == "linear":
		return np.linspace(sig_start, sig_end, num_steps).tolist()
	elif schedule_type == "exponential":
		return np.exp(np.linspace(np.log(sig_start), np.log(sig_end), num_steps)).tolist()


def main():
	# k_size = 13
	sig1 = 0.5
	sig2 = 0.3
	sig3 = np.sqrt(sig1**2 + sig2**2)
	print(f"sig1: {sig1}, sig2: {sig2}, sig3: {sig3}")

	img = load_single_random_dotmap(256, 1000)
	img1 = gaussian_blur(img, sig3)
	img2, _ = gaussian_blur_sequential_convolutions(img, 2, [sig1, sig2])

	fig, ax = plt.subplots(1, 3, figsize=(10, 8))
	ax[0].imshow((img1.squeeze() + 1.0) / 2.0)
	ax[1].imshow((img2.squeeze() + 1.0) / 2.0)
	ax[2].imshow((img1 - img2).squeeze())
	print(img1.min().item(), img1.max().item())
	print(img2.min().item(), img2.max().item())
	print((img1 - img2).min().item(), (img1 - img2).max().item())
	plt.show()

	n = 200
	start = time.time()
	img = (img + 1.0) / 2.0
	test, fs = gaussian_blur_sequential_convolutions(img, n, [sig1]*n)
	img = img * 2.0 - 1.0
	end = time.time()
	print(f"{n} convolutions time: {end - start}")
	print(f"final sig: {fs}")
	fks = int(6*fs)
	if fks % 2 == 0:
		fks += 1
	print(f"final k_size: {fks}")

	start = time.time()
	img = (img + 1.0) / 2.0
	fimg = gaussian_blur(img, fs)
	img = img * 2.0 - 1.0
	end = time.time()
	print(f"single convolution time: {end - start}")
	fig, ax = plt.subplots(1, 3, figsize=(10, 8))
	ax[0].imshow((test.squeeze() + 1.0) / 2.0)
	ax[1].imshow((fimg.squeeze() + 1.0) / 2.0)
	ax[2].imshow((test - fimg).squeeze())
	plt.show()
	print(test.min().item(), test.max().item())
	print(fimg.min().item(), fimg.max().item())

	imgs = []
	sig = 0.83
	for i in range(10):
		print(img.min().item(), img.max().item())
		pil_img = to_pil_image(img)
		imgs.append(pil_img)
		img = gaussian_blur(img, sig)
	# 	images = th.clamp(images, -1.0, 1.0)
		sig = sig

	imgs[0].save("gauss.gif", save_all=True, optimize=False, append_images=imgs[1:], loop=0, duration=40)


def main2():
	from diffcount.deblur_diffusion import DeblurDiffusion
	num_timesteps = 100
	blur_sigma_max = 20
	blur_sigma_min = 0.83
	blur_schedule = np.exp(np.linspace(np.log(blur_sigma_min),
									   np.log(blur_sigma_max), num_timesteps))
	blur_schedule = np.array(
		# [0] + list(blur_schedule)
		list(blur_schedule)
	)
	diffusion1 = DeblurDiffusion(
		blur_sigmas=blur_schedule,
		image_size=28,
		dataloader=None,
		use_dct=False,
	)
	diffusion2 = DeblurDiffusion(
		blur_sigmas=blur_schedule,
		image_size=28,
		dataloader=None,
		use_dct=True,
	)
	x_t = load_single_mnist_batch(2).to("cuda")
	# x_t = load_single_random_dotmap(28, 3).to("cuda")

	t = th.tensor([0,20], device="cuda")
	y1_t = diffusion1.q_sample(x_t, t)
	y2_t = diffusion2.q_sample(x_t, t)
	plt.imshow(to_pil_image(th.cat((y1_t, y2_t))))
	plt.show()

	imgs = []
	for t in range(num_timesteps):
		x1_t = diffusion1.q_sample(x_t, th.tensor([t, t], device="cuda"))
		x2_t = diffusion2.q_sample(x_t, th.tensor([t, t], device="cuda"))
		imgs.append(to_pil_image(th.cat((x1_t, x2_t))))
		print(x1_t.min().item(), x1_t.max().item())
		print(x2_t.min().item(), x2_t.max().item())
		print()

	imgs[0].save("gauss.gif", save_all=True, optimize=False, append_images=imgs[1:], loop=0, duration=40)


def main4():
	# from diffusers import AutoencoderKL
	from transformers import VitDetConfig, ViTModel, ViTConfig, VitDetBackbone, ViTMAEConfig, ViTMAEModel
	# import open_clip


	device = 'cuda' if th.cuda.is_available() else 'cpu'
	image_size = 224

	data = load_data(
		dataset=FSC147(
			root="../FSC147_384_V2/",
			targetdir="densitymaps/sig_0.5/size_256/",
			split='train',
			transform_kwargs=dict(
				img_size=image_size,
			)
		),
		batch_size=1,
		shuffle=True,
		overfit_single_batch=True,
	)
	targets, cond = next(iter(data))

	# config = ViTMAEConfig(image_size=image_size, hidden_size=768)
	# model = ViTMAEModel(config).to(device)
	# pixel_values = cond['img'].to(device)
	# with th.no_grad():
	# 	outputs = model(pixel_values)
	# print(outputs.last_hidden_state.shape)
	# print(list(feature_maps[-1].shape))


	config = VitDetConfig(image_size=image_size, hidden_size=768)
	model = VitDetBackbone(config).from_pretrained("facebook/vit-mae-base")
	pixel_values = th.randn(1, 3, image_size, image_size)
	with th.no_grad():
		outputs = model(pixel_values)
	feature_maps = outputs.feature_maps
	print(list(feature_maps[-1].shape))

	# config = ViTConfig(image_size=image_size, hidden_size=768)
	# model = ViTModel(config).from_pretrained("google/vit-base-patch16-224")
	# pixel_values = th.randn(1, 3, image_size, image_size)
	# with th.no_grad():
	# 	outputs = model(pixel_values, output_hidden_states=True)
	# print(outputs.hidden_states[0].shape)
	# # feature_maps = outputs.last_hidden_state
	# # print(list(feature_maps.shape))


	# vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse").to(device)
	# z = vae.encode(cond['img'].to(device)).latent_dist.sample().mul_(0.18215)
	# print(z.shape)

	# model, _, preprocess = open_clip.create_model_and_transforms('RN50', pretrained='openai')
	# model = model.to(device)
	# # image = preprocess(cond['img'].to(device))
	# image = cond['img'].to(device)
	# image_features = model.encode_image(image)
	# print(image_features.shape)
	# image_features /= image_features.norm(dim=-1, keepdim=True)


	# fig, ax = plt.subplots(1, 2)
	# ax[0].imshow(to_pil_image(cond['img'][0]))
	# ax[0].imshow(to_pil_image(targets[0]), alpha=0.5)
	# x = vae.decode(z / 0.18215).sample
	# ax[1].imshow(to_pil_image(x[0]))
	# ax[1].imshow(to_pil_image(targets[0]), alpha=0.5)
	# plt.show()


def main5():
	import pickle
	from detectron2.modeling import ViT, SimpleFeaturePyramid
	from detectron2.modeling.poolers import ROIPooler
	from detectron2.structures import Boxes
	from functools import partial
	import torch.nn as nn

	# image_size = 224
	# config = VitDetConfig(image_size=image_size, hidden_size=768)
	# model = VitDetBackbone(config)

	# print(model._modules['encoder'].state_dict().keys())


	# with (open("/mnt/c/users/grega/downloads/model_final_435fa9.pkl", "rb")) as openfile:
	# 	obj = pickle.load(openfile)
	# print(type(obj['model']))
	# print(obj['model'])

	device = 'cuda' if th.cuda.is_available() else 'cpu'
	image_size = 512

	data = load_data(
		dataset=FSC147(
			root="../FSC147_384_V2/",
			targetdir="densitymaps/sig_0.5/size_256/",
			split='train',
			transform_kwargs=dict(
				img_size=image_size,
			)
		),
		batch_size=4,
		shuffle=True,
		overfit_single_batch=True,
	)
	targets, cond = next(iter(data))

	embed_dim, depth, num_heads, dp = 768, 12, 12, 0.1
	backbone = SimpleFeaturePyramid(
		net=ViT(  # Single-scale ViT backbone
			img_size=image_size,
			patch_size=16,
			embed_dim=embed_dim,
			depth=depth,
			num_heads=num_heads,
			drop_path_rate=dp,
			window_size=14,
			mlp_ratio=4,
			qkv_bias=True,
			norm_layer=partial(nn.LayerNorm, eps=1e-6),
			window_block_indexes=[
				# 2, 5, 8 11 for global attention
				0,
				1,
				3,
				4,
				6,
				7,
				9,
				10,
			],
			residual_block_indexes=[],
			use_rel_pos=False,
			out_feature="last_feat",
		),
		in_feature="last_feat",
		out_channels=256,
		scale_factors=(4.0, 2.0, 1.0, 0.5),
		# top_block=LastLevelMaxPool(),
		norm="LN",
		square_pad=0,
	)
	class Wrapper(nn.Module):
		def __init__(self, backbone):
			super().__init__()
			self.backbone = backbone

			from detectron2.checkpoint import DetectionCheckpointer
			checkpointer = DetectionCheckpointer(self)
			checkpointer.load("/mnt/c/users/grega/downloads/model_final_61ccd1.pkl")
			# checkpointer.load("https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/mask_rcnn_vitdet_b/f325346929/model_final_61ccd1.pkl")

		def forward(self, x):
			return self.backbone(x)
	model = Wrapper(backbone=backbone)
	model = model.eval()
	model = model.to(device)

	# bboxes = []
	# for t in cond['bboxes']:
	# 	bboxes.append(Boxes(t).to(device))
	# z = model(cond['img'].to(device))

	# pooler = ROIPooler(
	# 	output_size=(7, 7),
	# 	scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
	# 	sampling_ratio=0,
	# 	pooler_type="ROIAlignV2",
	# )
	# pooler = pooler.to(device)
	# out = pooler(list(z.values()), bboxes)
	# out = out.reshape(-1, 3, 256, 7, 7)
	# print(out.shape)
	# fl = nn.Flatten(start_dim=2)

	# print(fl(out).shape)
	# print(out.flatten(start_dim=2).shape)
	# for k in z.keys():
	# 	print(k, z[k].shape)

	# print(model.state_dict().keys())
	# with (open("/mnt/c/users/grega/downloads/model_final_61ccd1.pkl", "rb")) as openfile:
	# 	obj = pickle.load(openfile)
	# print(obj.keys())


if __name__ == "__main__":
	main5()