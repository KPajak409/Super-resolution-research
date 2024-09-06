# %%
import torch
import matplotlib.pyplot as plt
from ImageNet_dataset import ImageNet
from torch.utils.data import DataLoader
import torchvision.transforms.functional as f
from torchmetrics.functional.image import peak_signal_noise_ratio as psnr
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
import utils

ds_lrx2 = ImageNet(
    input_folder_path="./data/Set14/LRbicx2",
    target_folder_path="./data/Set14/GTmod12",
)

ds_lrx4 = ImageNet(
    input_folder_path="./data/Set14/LRbicx4",
    target_folder_path="./data/Set14/GTmod12",
)

upsamplex2_lower = torch.load("./models/ESPCNx2_lower").to("cpu")
upsamplex2_upper = torch.load("./models/ESPCNx2_upper").to("cpu")
upsamplex4 = torch.load("./models/ESPCNx4").to("cpu")

upsamplex2_lower.eval()
upsamplex2_upper.eval()
upsamplex4.eval()


test_dl = DataLoader(
    dataset=ds_lrx2,
    batch_size=1,
    shuffle=False,
)

file_names = [fns.split("/")[-1] for fns in ds_lrx2.input_file_names]
print(file_names)
patch_size, stride, scale = 112, 102, 2
total_psnr, total_ssim = 0, 0
# %%
# patch_size - stride = overlap | stride < patch_size
print("File name \t\t PSNR \t   SSIM     target shape")
for i, (image, target) in enumerate(test_dl):
    output = utils.p2img_forward(
        image, target, scale, patch_size, stride, upsamplex2_upper
    )

    psnr_value = psnr(output, target).item()
    ssim_value = ssim(output, target).item()
    total_psnr += psnr_value
    total_ssim += ssim_value
    print(
        f"{file_names[i]}:\t\t{psnr_value:2f}, {ssim_value:2f} H:{target.shape[2]} W:{target.shape[3]}"
    )

print(
    f"Average psnr: {total_psnr/len(file_names):1f} \nAverage ssim: {total_ssim/len(file_names):1f}"
)

# %% Display desired pair output and target
image, target = ds_lrx2[0]
image = image.unsqueeze(0)
target = target.unsqueeze(0)
output = utils.p2img_forward(image, target, scale, patch_size, stride, upsamplex2_upper)
utils.plot_out_target(output, target)

# %%
