import os
import cv2
import argparse


def resize_image(image, target_size):
    height, width, _ = image.shape
    new_height = height * target_size // min(image.shape[:2])
    new_width = width * target_size // min(image.shape[:2])
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)


def preprocessing_images(img_path, output, target_size):
    assert os.path.isdir(img_path)
    assert os.path.isdir(output)

    fns = os.listdir(img_path)
    fns = [f"{img_path}\\{fns[i]}" for i in range(len(fns))]

    for i in range(len(fns)):
        if i % 10000 == 0:
            print("%d/%d" % (i, len(fns)))
        # Load (as BGR)
        img = cv2.imread(fns[i])
        img = resize_image(img, target_size)
        assert img.shape[0] == target_size and img.shape[1] == target_size
        # Save (as RGB)
        # If it is in NP array, please revert last dim like [:,:,::-1]
        name, ext = os.path.splitext(os.path.basename(fns[i]))
        file_path = os.path.join(output, name + ".JPEG")
        cv2.imwrite(file_path, img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", "-i", required=True, type=str, help="ImageNet image path"
    )
    parser.add_argument(
        "--output", "-o", required=True, type=str, default="", help="Output image path"
    )
    parser.add_argument(
        "--target_size",
        "-df",
        required=True,
        type=int,
        default=240,
        help="target size of images",
    )
    args = parser.parse_args()
    if len(args.output) == 0:
        args.output = args.input
    print("Input dir:", args.input)
    print("Output dir:", args.output)
    print("Target size:", args.target_size)
    preprocessing_images(args.input, args.output, args.target_size)
