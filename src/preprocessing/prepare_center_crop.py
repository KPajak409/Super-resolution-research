import argparse
import glob
import h5py
import numpy as np
import PIL.Image as pil_image
from src.utils import calc_patch_size, convert_rgb_to_y


@calc_patch_size
def train(args):
    h5_file = h5py.File(args.output_path, "w")

    lr_patches = []
    hr_patches = []

    for image_path in sorted(glob.glob("{}/*".format(args.images_dir))):
        hr = pil_image.open(image_path).convert("RGB")
        hr_images = []

        if args.with_aug:
            for r in [0, 90, 180, 270]:
                hr = hr.rotate(r, expand=True)
                hr_images.append(hr)
        else:
            hr_images.append(hr)

        for hr in hr_images:
            hr_width = (hr.width // args.scale) * args.scale
            hr_height = (hr.height // args.scale) * args.scale

            hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
            lr = hr.resize(
                (hr.width // args.scale, hr_height // args.scale),
                resample=pil_image.BICUBIC,
            )
            hr = np.array(hr).astype(np.float32)
            lr = np.array(lr).astype(np.float32)
            hr = convert_rgb_to_y(hr)
            lr = convert_rgb_to_y(lr)

            lr_patches.append(lr)
            hr_patches.append(hr)

    lr_patches = np.array(lr_patches)
    hr_patches = np.array(hr_patches)

    h5_file.create_dataset("lr", data=lr_patches)
    h5_file.create_dataset("hr", data=hr_patches)

    h5_file.close()


def eval(args):
    h5_file = h5py.File(args.output_path, "w")

    lr_group = h5_file.create_group("lr")
    hr_group = h5_file.create_group("hr")

    for i, image_path in enumerate(sorted(glob.glob("{}/*".format(args.images_dir)))):
        hr = pil_image.open(image_path).convert("RGB")
        hr_width = (hr.width // args.scale) * args.scale
        hr_height = (hr.height // args.scale) * args.scale
        hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
        lr = hr.resize(
            (hr.width // args.scale, hr_height // args.scale),
            resample=pil_image.BICUBIC,
        )
        hr = np.array(hr).astype(np.float32)
        lr = np.array(lr).astype(np.float32)
        hr = convert_rgb_to_y(hr)
        lr = convert_rgb_to_y(lr)

        lr_group.create_dataset(str(i), data=lr)
        hr_group.create_dataset(str(i), data=hr)

    h5_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--with-aug", action="store_true")
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()
    if not args.eval:
        train(args)
    else:
        eval(args)
