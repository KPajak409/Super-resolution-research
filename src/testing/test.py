# %%
import torch, os, glob
import matplotlib.pyplot as plt
from torchmetrics.functional.image import peak_signal_noise_ratio as psnr
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
from src.models.ESPCN import ESPCN
from src.models.FSRCNN import FSRCNN
import src.utils as utils
from PIL import Image
import numpy as np

model = torch.load("../../trained_models/FSRCNNx2_112_DF2K-aug_v2_YCbCr").to("cpu")
model.eval()

file_paths = glob.glob("../../data/Set14/original/*")
file_names = os.listdir("../../data/Set14/original")

scale = 2
device = "cpu"

# %%
print("File name \t\t PSNR \t   SSIM     target shape")
total_psnr = 0
total_ssim = 0
for i in range(len(file_paths)):
    image = Image.open(file_paths[i]).convert("RGB")

    image_width = (image.width // scale) * scale
    image_height = (image.height // scale) * scale

    hr = image.resize((image_width, image_height), resample=Image.Resampling.BICUBIC)
    lr = hr.resize(
        (hr.width // scale, hr.height // scale), resample=Image.Resampling.BICUBIC
    )
    bicubic = lr.resize(
        (lr.width * scale, lr.height * scale), resample=Image.Resampling.BICUBIC
    )
    lr, _ = utils.preprocess(lr, device)
    hr, _ = utils.preprocess(hr, device)
    bic, ycbcr = utils.preprocess(bicubic, device)

    output = model(lr).clamp(0.0, 1.0).detach()
    psnr_value = psnr(bic, hr, (0.0, 1.0)).item()
    ssim_value = ssim(bic, hr, (0.0, 1.0)).item()

    total_psnr += psnr_value
    total_ssim += ssim_value
    print(
        f"{file_names[i]:15}:\t{psnr_value:2f}, {ssim_value:2f} H:{hr.shape[2]} W:{hr.shape[3]}"
    )

print(
    f"Average psnr: {total_psnr/len(file_paths):1f} \nAverage ssim: {total_ssim/len(file_paths):1f}"
)
# %%
image = Image.open(file_paths[4]).convert("RGB")

image_width = (image.width // scale) * scale
image_height = (image.height // scale) * scale

hr = image.resize((image_width, image_height), resample=Image.Resampling.BICUBIC)
lr = hr.resize(
    (hr.width // scale, hr.height // scale), resample=Image.Resampling.BICUBIC
)
bicubic = lr.resize(
    (lr.width * scale, lr.height * scale), resample=Image.Resampling.BICUBIC
)
lr_y, ycbcr_lr = utils.preprocess(lr, device)
hr_y, ycbcr__hr = utils.preprocess(hr, device)
bic, ycbcr = utils.preprocess(bicubic, device)


output = model(lr_y).clamp(0.0, 1.0).detach()
bic_psnr = round(psnr(bic, hr_y, (0.0, 1.0)).item(), 4)
out_psnr = round(psnr(output, hr_y, (0.0, 1.0)).item(), 4)


output = output.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)


output = np.array([output, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
output = np.clip(utils.convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
output = Image.fromarray(output)
print(output.size, lr.size)

titles = [
    "LR",
    f"Bicubic Psnr: {bic_psnr}",
    f"Predict Psnr: {out_psnr}",
    "Ground Truth",
]
utils.plot_lr_bic_out_hr_pil(lr, bicubic, output, hr, titles)
# %%
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes

fig, ax = plt.subplots(1, 3)
# left, right, down, up
x1, x2, y1, y2 = 50, 100, 150, 100
for i, img in enumerate([bicubic, output, hr]):
    ax[i].imshow(img)
    axins = ax[i].inset_axes(
        [0.53, 0.63, 0.47, 0.47],
        xlim=(x1, x2),
        ylim=(y1, y2),
        xticklabels=[],
        yticklabels=[],
    )
    axins.imshow(img)
    # manually invert axis to match corners
    _, p1, p2 = mark_inset(ax[i], axins, loc1=1, loc2=1, linewidth=0.5)
    p1.loc1 = 2
    p1.loc2 = 3
    p2.loc1 = 3
    p2.loc2 = 2
    ax[i].set_title(titles[i + 1], fontsize=9)
    ax[i].axis("off")

plt.show()
