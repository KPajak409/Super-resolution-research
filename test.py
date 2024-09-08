# %%
import torch
import matplotlib.pyplot as plt
from ImageNet_dataset import ImageNet
from torch.utils.data import DataLoader
import torchvision.transforms.functional as f
from torchmetrics.functional.image import peak_signal_noise_ratio as psnr
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
import utils
import yaml

with open("./options/upsamplex4.yml", "r") as file:
    config_yml = yaml.safe_load(file)

model = torch.load(".\\models\\ESPCNx4").to("cpu")
model.eval()

imageNet_dataset = ImageNet(
    input_folder_path=config_yml["datasets"]["test1"]["dataroot_lq"],
    target_folder_path=config_yml["datasets"]["test1"]["dataroot_gt"],
)

test_dl = DataLoader(
    dataset=imageNet_dataset,
    batch_size=1,
    shuffle=False,
)

# patch_size - stride = overlap => stride < patch_size; overlap must be even
patch_size = config_yml["patches"]["patch_size"]
stride = config_yml["patches"]["stride"]
scale = config_yml["scale"]
total_psnr, total_ssim = 0, 0
file_names = [fns.split("/")[-1] for fns in imageNet_dataset.input_file_names]
# %%
print("File name \t\t PSNR \t   SSIM     target shape")
for i, (image, target) in enumerate(test_dl):
    output = utils.p2img_forward(image, target, scale, patch_size, stride, model)
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
image, target = imageNet_dataset[0]
image = image.unsqueeze(0)
target = target.unsqueeze(0)
output = utils.p2img_forward(image, target, scale, patch_size, stride, model)
print(torch.min(output), torch.max(output))
utils.plot_src_out_target(image, output, target)

# %%
