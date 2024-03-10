import torch as th
import numpy as np
from diffcount.datasets import MNIST, load_data
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


# def gaussian_blur(x, sig):
# 	k_size = int(6*sig)
# 	if k_size % 2 == 0:
# 		k_size += 1
# 	k_size = 3 if k_size < 3 else k_size
# 	return gaussian_blur2d(x, (k_size, k_size), (sig, sig), border_type="constant")


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

if __name__ == "__main__":
	main2()
	# x = th.zeros(4, 1, 10, 10)
	# x[0, 0, 5, 5] = 1.0
	# sigma = 1.0
	# kernel_size = 5
	# y = gaussian_blur(x, kernel_size, sigma)