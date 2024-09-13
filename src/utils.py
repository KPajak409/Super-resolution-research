# %%
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import torchvision.transforms.functional as f
from torchmetrics.functional.image import peak_signal_noise_ratio as psnr
from torchmetrics.functional.image import structural_similarity_index_measure as ssim


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


def forward_ycbcr(image, target, patch_size, stride, scale, model):
    # prepare channels for processing
    y = image[..., 0, :, :].unsqueeze(0)
    # target_y = target[..., 0, :, :].unsqueeze(0)
    cbcr = image[..., 1:, :, :]

    output = p2img_forward(y, target, scale, patch_size, stride, y.shape[1], model)
    cbcr_hr = torch.nn.functional.interpolate(
        cbcr, (target.shape[2], target.shape[3]), mode="bicubic"
    )
    output = torch.concat((output, cbcr_hr), 1)

    return output


def ycbcr_to_rgb(image):
    image *= 255.0

    y = image[..., 0, :, :]
    cb = image[..., 1, :, :]
    cr = image[..., 2, :, :]

    # r = (y * 1.16438355 + cb * 1.16438355 + cr * 1.16438355) - 222.921
    # g = (y + cb * (-0.3917616) + cr * 2.01723105) + 135.576
    # b = (y * 1.59602715 + cb * (-0.81296805) + cr) - 276.836

    r = (y * 1.16438355 + cb * 1.16438355 + cr * 1.16438355) - 222.921
    g = (y + cb * (-0.3917616) + cr * 2.01723105) + 135.576
    b = (y * 1.59602715 + cb * (-0.81296805) + cr) - 276.836

    image = torch.stack((r, g, b), 2)
    image /= 255.0
    return image


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
