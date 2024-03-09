""" Generate density maps from dot annotations and saves them to data_rootdir. """

import os
import json
import argparse
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

def generate_density_maps(root, sigma, size=None, dtype=np.float32):
	size_str = size or 'original'
	savedir = os.path.join(root, "densitymaps", f'sig{sigma}', f'size{size_str}')
	# if not os.path.isdir(savedir):
	os.makedirs(savedir, exist_ok=True)
	with open(os.path.join(root, 'annotation_FSC147_384.json'), 'rb') as f:
		annotations = json.load(f)
		for img_name, ann in annotations.items():
			w, h = Image.open(
				os.path.join(
					root,
					'images_384_VarV2',
					img_name
				)
			).size
			new_w, new_h = size if size is not None else (w, h)
			rw, rh = new_w / w, new_h / h
			bitmap = np.zeros((new_h, new_w), dtype=dtype)
			for point in ann['points']:
				x, y = int(point[0] * rw)-1, int(point[1] * rh)-1
				bitmap[y, x] = 1.0

			density_map = gaussian_filter(
				bitmap,
				sigma,
				truncate=4.0,
				mode='constant'
			)
			np.save(
				os.path.join(savedir, os.path.splitext(img_name)[0] + '.npy'), 
				density_map
			)
			print(f'{img_name}.npy saved')


def main():
	parser = argparse.ArgumentParser(description="Data Downloader")
	parser.add_argument("--root", type=str, default=".", help="Path to the dataset root directory")
	parser.add_argument("--sigma", type=float, default=1.0, help="Gaussian kernel standard deviation")
	parser.add_argument("--size", type=int, default=None, help="Size of the density maps")
	args = parser.parse_args()

	generate_density_maps(
		root=args.root, 
		sigma=args.sigma, 
	)


if __name__ == "__main__":
	main()