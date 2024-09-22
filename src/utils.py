# %%
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import torchvision.transforms.functional as f
from torchmetrics.functional.image import peak_signal_noise_ratio as psnr
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
import numpy as np


def calc_patch_size(func):
    def wrapper(args):
        if args.scale == 2:
            args.patch_size = 10
        elif args.scale == 3:
            args.patch_size = 7
        elif args.scale == 4:
            args.patch_size = 6
        else:
            raise Exception("Scale Error", args.scale)
        return func(args)

    return wrapper


def patchify(image, patch_size, stride, channels) -> torch.Tensor:
    """turn tensor of shape [1, kernels, height, width]
    to patches of shape [patches, kernels, height, width]"""
    overlap = patch_size - stride
    assert overlap % 2 == 0

    kc, kh, kw = channels, patch_size, patch_size  # kernel size
    dc, dh, dw = channels, stride, stride  # stride
    # Pad to multiples of kernel size
    pw, ph = 0, 0

    for i in range(1, kh, 1):
        if (image.shape[2] + i) % dh == 0:
            ph = i
        if (image.shape[3] + i) % dw == 0:
            pw = i

    image = torch.nn.functional.pad(
        image,
        (overlap // 2, pw + overlap // 2, overlap // 2, ph + overlap // 2),
    )
    patches = image.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
    unfold_shape = patches.size()
    patches = patches.contiguous().view(-1, kc, kh, kw)

    return patches, unfold_shape


def unpatchify(patches, unfold_shape, target_shape) -> torch.Tensor:
    """Return unpatchified tensor in shape of [batch, kernels, height, width]"""
    patches_orig = patches.view(unfold_shape)  # [1,1,c,row, height, column, width]
    output_c = unfold_shape[1] * unfold_shape[4]
    output_h = unfold_shape[2] * unfold_shape[5]
    output_w = unfold_shape[3] * unfold_shape[6]
    patches_orig = patches_orig.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
    patches_orig = patches_orig.view(1, output_c, output_h, output_w)

    # print((patches_orig == x[:, :output_c, :output_h, :output_w]).all())
    # print(x.shape, patches_orig.shape)

    p = f.crop(patches_orig, 0, 0, target_shape[2], target_shape[3])
    p = torch.flip(p, (2,))
    p = f.rotate(p, -90, expand=True)
    p = torch.permute(p, (0, 1, 3, 2))
    return p


def p2img_forward(image, target, scale, patch_size, stride, channels, model):
    # turn image into patches
    overlap = patch_size - stride
    p_lr, un_lr = patchify(
        image, patch_size=patch_size, stride=stride, channels=channels
    )

    # modify unfold dimensions to fit with upscaled image
    un_hr = [
        (un_lr[i] - overlap) * scale if i > 4 else un_lr[i] for i in range(len(un_lr))
    ]

    # forward batch of patches with overlap
    with torch.no_grad():
        output = model(p_lr)

    # discard of overlap
    output = output[
        :,
        :,
        (overlap // 2) * scale : -((overlap // 2) * scale),
        (overlap // 2) * scale : -((overlap // 2) * scale),
    ]

    # convert patches to single image with the same size as orginal
    return unpatchify(output, un_hr, target.shape)


def convert_rgb_to_y(img, dim_order="hwc"):
    if dim_order == "hwc":
        return (
            16.0
            + (65.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2])
            / 256.0
        )
    else:
        return 16.0 + (65.738 * img[0] + 129.057 * img[1] + 25.064 * img[2]) / 256.0


def convert_ycbcr_to_rgb(img, dim_order="hwc"):
    if dim_order == "hwc":
        r = 298.082 * img[..., 0] / 256.0 + 408.583 * img[..., 2] / 256.0 - 222.921
        g = (
            298.082 * img[..., 0] / 256.0
            - 100.291 * img[..., 1] / 256.0
            - 208.120 * img[..., 2] / 256.0
            + 135.576
        )
        b = 298.082 * img[..., 0] / 256.0 + 516.412 * img[..., 1] / 256.0 - 276.836
    else:
        r = 298.082 * img[0] / 256.0 + 408.583 * img[2] / 256.0 - 222.921
        g = (
            298.082 * img[0] / 256.0
            - 100.291 * img[1] / 256.0
            - 208.120 * img[2] / 256.0
            + 135.576
        )
        b = 298.082 * img[0] / 256.0 + 516.412 * img[1] / 256.0 - 276.836
    return np.array([r, g, b]).transpose([1, 2, 0])


def convert_rgb_to_ycbcr(img, dim_order="hwc"):
    if dim_order == "hwc":
        y = (
            16.0
            + (65.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2])
            / 256.0
        )
        cb = (
            128.0
            + (-37.945 * img[..., 0] - 74.494 * img[..., 1] + 112.439 * img[..., 2])
            / 256.0
        )
        cr = (
            128.0
            + (112.439 * img[..., 0] - 94.154 * img[..., 1] - 18.285 * img[..., 2])
            / 256.0
        )
    else:
        y = 16.0 + (65.738 * img[0] + 129.057 * img[1] + 25.064 * img[2]) / 256.0
        cb = 128.0 + (-37.945 * img[0] - 74.494 * img[1] + 112.439 * img[2]) / 256.0
        cr = 128.0 + (112.439 * img[0] - 94.154 * img[1] - 18.285 * img[2]) / 256.0
    return np.array([y, cb, cr]).transpose([1, 2, 0])


def preprocess(img, device):
    img = np.array(img).astype(np.float32)
    ycbcr = convert_rgb_to_ycbcr(img)
    x = ycbcr[..., 0]
    x /= 255.0
    x = torch.from_numpy(x).to(device)
    x = x.unsqueeze(0).unsqueeze(0)
    return x, ycbcr


def plot_patches(tensor, h_patches, w_patches):
    fig = plt.figure(figsize=(8, 8))
    grid = ImageGrid(fig, 111, nrows_ncols=(h_patches, w_patches), axes_pad=0.1)

    for i, ax in enumerate(grid):
        patch = tensor[i].permute(1, 2, 0).numpy()
        ax.imshow(patch)
        ax.axis("off")

    plt.show()


def plot_out_target(output, target):
    output_permute = torch.permute(output, (0, 2, 3, 1))
    target_permute = torch.permute(target, (0, 2, 3, 1))

    psnr_value = round(psnr(output, target).item(), 3)
    ssim_value = round(ssim(output, target).item(), 3)
    # pair = torch.hstack((output_permute[0], target_permute[0]))
    plt.subplot(1, 2, 1)
    plt.imshow(output_permute[0])
    plt.axis("off")
    plt.title(f"Psnr: {psnr_value}, SSIM: {ssim_value}")

    plt.subplot(1, 2, 2)
    plt.imshow(target_permute[0])
    plt.axis("off")
    plt.title("Ground Truth")
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def plot_src_out_target(src, output, target):
    bic = torch.nn.functional.interpolate(
        src, (target.shape[2], target.shape[3]), mode="bicubic"
    )

    src_permute = torch.permute(src, (0, 2, 3, 1))
    bic_permute = torch.permute(src, (0, 2, 3, 1))
    output_permute = torch.permute(output, (0, 2, 3, 1))
    target_permute = torch.permute(target, (0, 2, 3, 1))

    out_psnr = round(psnr(output, target).item(), 3)
    out_ssim = round(ssim(output, target).item(), 3)
    bic_psnr = round(psnr(bic, target).item(), 3)
    bic_ssim = round(ssim(bic, target).item(), 3)
    # pair = torch.hstack((output_permute[0], target_permute[0]))
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 4, 1)
    plt.imshow(src_permute[0])
    plt.axis("off")
    plt.title("LR", fontsize=9)

    plt.subplot(1, 4, 2)
    plt.imshow(bic_permute[0])
    plt.axis("off")
    plt.title(f"Bicubic Psnr: {bic_psnr},\nSSIM: {bic_ssim}", fontsize=9)

    plt.subplot(1, 4, 3)
    plt.imshow(output_permute[0])
    plt.axis("off")
    plt.title(f"Predict Psnr: {out_psnr},\nSSIM: {out_ssim}", fontsize=9)

    plt.subplot(1, 4, 4)
    plt.imshow(target_permute[0])
    plt.axis("off")
    plt.title("Ground Truth", fontsize=9)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def plot_lr_bic_out_hr_pil(src, bicubic, output, target, titles):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 4, 1)
    plt.imshow(src)
    plt.axis("off")
    plt.title(titles[0], fontsize=9)

    plt.subplot(1, 4, 2)
    plt.imshow(bicubic)
    plt.axis("off")
    plt.title(titles[1], fontsize=9)

    plt.subplot(1, 4, 3)
    plt.imshow(output)
    plt.axis("off")
    plt.title(titles[2], fontsize=9)

    plt.subplot(1, 4, 4)
    plt.imshow(target)
    plt.axis("off")
    plt.title(titles[3], fontsize=9)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
