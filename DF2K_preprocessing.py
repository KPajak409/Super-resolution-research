# %%
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import os

CROP_SIZE = 480


def summary_images(img_folder_path):
    file_paths = glob.glob(f"{img_folder_path}/*")

    summary = {
        "max_width": 0,
        "max_height": 0,
        "min_width": 10000,
        "min_height": 10000,
    }

    widths = np.zeros(len(file_paths))
    heights = np.zeros(len(file_paths))
    channels = np.zeros(len(file_paths))

    for i in range(len(file_paths)):
        if i % (len(file_paths) * 0.1) == 0:
            print("%d/%d" % (i, len(file_paths)))

        # Load (as BGR)
        img = cv2.imread(file_paths[i])
        w, h, c = img.shape

        widths[i] = w
        heights[i] = h
        channels[i] = c

    summary["max_width"] = np.max(widths)
    summary["min_width"] = np.min(widths)
    summary["max_height"] = np.max(heights)
    summary["min_height"] = np.min(heights)
    summary["median_width"] = np.median(widths)
    summary["median_height"] = np.median(heights)
    summary["25_p_width"] = np.percentile(widths, 25)
    summary["25_p_height"] = np.percentile(heights, 25)

    return summary


def central_crop(image, crop_height, crop_width):
    height, width, _ = image.shape
    startx = width // 2 - (crop_width // 2)
    starty = height // 2 - (crop_height // 2)
    return image[starty : starty + crop_height, startx : startx + crop_width]


def preprocessing_DF2K(img_folder_path, output_folder_path, ds_name):
    file_paths = glob.glob(f"{img_folder_path}/*")

    for i in range(len(file_paths)):
        if i % (len(file_paths) * 0.1) == 0:
            print("%d/%d" % (i, len(file_paths)))

        # Load (as BGR)
        img = cv2.imread(file_paths[i])
        h, w, c = img.shape
        # plt.imshow(img)
        # plt.show()
        # print(w, h)
        name, ext = os.path.splitext(os.path.basename(file_paths[i]))
        if w > CROP_SIZE * 2 and h > CROP_SIZE * 2:
            cv2.imwrite(
                f"{output_folder_path}/{ds_name}_{name}_r0_0{ext}",
                img[0:CROP_SIZE, 0:CROP_SIZE],
            )  # left top
            cv2.imwrite(
                f"{output_folder_path}/{ds_name}_{name}_r0_1{ext}",
                img[0:CROP_SIZE, w - CROP_SIZE : w],
            )  # top right
            cv2.imwrite(
                f"{output_folder_path}/{ds_name}_{name}_r0_2{ext}",
                img[h - CROP_SIZE : h, 0:CROP_SIZE],
            )  # left bottom
            cv2.imwrite(
                f"{output_folder_path}/{ds_name}_{name}_r0_3{ext}",
                img[h - CROP_SIZE : h, w - CROP_SIZE : w],
            )  # right bottom
            cv2.imwrite(
                f"{output_folder_path}/{ds_name}_{name}_c0{ext}",
                central_crop(img, CROP_SIZE, CROP_SIZE),
            )  # center
            for degrees, cv2_rotation in zip(
                [90, 180, 270],
                [
                    cv2.ROTATE_90_CLOCKWISE,
                    cv2.ROTATE_180,
                    cv2.ROTATE_90_COUNTERCLOCKWISE,
                ],
            ):
                cv2.imwrite(
                    f"{output_folder_path}/{ds_name}_{name}_r{degrees}_0{ext}",
                    cv2.rotate(img[0:CROP_SIZE, 0:CROP_SIZE], cv2_rotation),
                )  # left top
                cv2.imwrite(
                    f"{output_folder_path}/{ds_name}_{name}_r{degrees}_1{ext}",
                    cv2.rotate(img[0:CROP_SIZE, w - CROP_SIZE : w], cv2_rotation),
                )  # top right
                cv2.imwrite(
                    f"{output_folder_path}/{ds_name}_{name}_r{degrees}_2{ext}",
                    cv2.rotate(img[h - CROP_SIZE : h, 0:CROP_SIZE], cv2_rotation),
                )  # left bottom
                cv2.imwrite(
                    f"{output_folder_path}/{ds_name}_{name}_r{degrees}_3{ext}",
                    cv2.rotate(img[h - CROP_SIZE : h, w - CROP_SIZE : w], cv2_rotation),
                )  # right bottom
                cv2.imwrite(
                    f"{output_folder_path}/{ds_name}_{name}_c{degrees}{ext}",
                    cv2.rotate(central_crop(img, CROP_SIZE, CROP_SIZE), cv2_rotation),
                )  # center
        else:
            cv2.imwrite(
                f"{output_folder_path}/{ds_name}_{name}_r0_0{ext}",
                central_crop(img, CROP_SIZE, CROP_SIZE),
            )  # left top
            cv2.imwrite(
                f"{output_folder_path}/{ds_name}_{name}_r90_1{ext}",
                cv2.rotate(
                    central_crop(img, CROP_SIZE, CROP_SIZE), cv2.ROTATE_90_CLOCKWISE
                ),
            )  # top right
            cv2.imwrite(
                f"{output_folder_path}/{ds_name}_{name}_r180_2{ext}",
                cv2.rotate(central_crop(img, CROP_SIZE, CROP_SIZE), cv2.ROTATE_180),
            )  # left bottom
            cv2.imwrite(
                f"{output_folder_path}/{ds_name}_{name}_r270_3{ext}",
                cv2.rotate(
                    central_crop(img, CROP_SIZE, CROP_SIZE),
                    cv2.ROTATE_90_COUNTERCLOCKWISE,
                ),
            )  # right bottom


preprocessing_DF2K("./data/DIV2K/valid_HR", "./data/DF2K_proc/val", ds_name="DIV2K")
