# %%
import torch
import matplotlib.pyplot as plt
from src.datasets.ImageNet_dataset import ImageNet
from torch.utils.data import DataLoader
import torchvision.transforms.functional as f
from torchmetrics.functional.image import peak_signal_noise_ratio as psnr
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
import src.utils as utils
import yaml

with open("./options/upsamplex2_upper.yml", "r") as file:
    config_yml = yaml.safe_load(file)

lower = torch.load(".\\models\\ESPCNx2_lower").to("cpu")
upper = torch.load(".\\models\\ESPCNx2_upper").to("cpu")
lower.eval()
upper.eval()


imageNet_dataset_x4 = ImageNet(
    input_folder_path="./data/Set14/LRbicx4",
    target_folder_path="./data/Set14/GTmod12",
)

imageNet_dataset_x2 = ImageNet(
    input_folder_path="./data/Set14/LRbicx2",
    target_folder_path="./data/Set14/GTmod12",
)

test_dl_x4 = DataLoader(
    dataset=imageNet_dataset_x4,
    batch_size=1,
    shuffle=False,
)
test_dl_x2 = DataLoader(
    dataset=imageNet_dataset_x2,
    batch_size=1,
    shuffle=False,
)


# patch_size - stride = overlap => stride < patch_size; overlap must be even
patch_size1, patch_size2 = 56, 112
stride1, stride2 = 46, 102
scale = 2
total_psnr, total_ssim = 0, 0
file_names = [fns.split("/")[-1] for fns in imageNet_dataset_x2.input_file_names]
print(len(file_names))
# %%
# print("File name \t\t PSNR \t   SSIM     target shape")
for i, ((image1, target), (image2, _)) in enumerate(zip(test_dl_x4, test_dl_x2)):

    output = utils.p2img_forward(image1, image2, scale, patch_size1, stride1, lower)
    output = utils.p2img_forward(output, target, scale, patch_size2, stride2, upper)
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
image1, target = imageNet_dataset_x4[0]
image2, _ = imageNet_dataset_x2[0]
image1 = image1.unsqueeze(0)
image2 = image2.unsqueeze(0)
target = target.unsqueeze(0)
print(image1.shape, image2.shape, target.shape)
output = utils.p2img_forward(image1, image2, scale, patch_size1, stride1, lower)
output = utils.p2img_forward(output, target, scale, patch_size2, stride2, upper)
print(torch.min(output), torch.max(output))
utils.plot_src_out_target(image1, output, target)

# image1_permute = torch.permute(image1, (0,2,3,1))
# plt.imshow(image1_permute[0])
# %%
