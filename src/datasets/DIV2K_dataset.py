# %%
import cv2
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as v2
import torchvision.transforms.functional as f
import torch
import matplotlib.pyplot as plt
import numpy as np


class DIV2KDataset(Dataset):
    def __init__(self, root_dir, transform, scale, mode="YCbCr"):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.scale = scale
        self.mode = mode
        self.samples = list(self.root_dir.glob("*"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        img = cv2.imread(str(img_path))
        if self.mode == "YCbCr":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
            img = np.expand_dims(img[:, :, 0], -1)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            target_img = self.transform(img)
        input_img = target_img.clone().detach()

        input_img = f.resize(
            input_img,
            [target_img.shape[1] // self.scale, target_img.shape[2] // self.scale],
            f.InterpolationMode.BICUBIC,
            antialias=False,
        ).clamp(0.0, 1.0)

        return input_img, target_img

    def dim(self):
        return len(self.samples)


if __name__ == "__main__":
    scale = 2
    img_size = 120
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.RandomCrop((img_size, img_size)),
            v2.RandomHorizontalFlip(0.5),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    ds = DIV2KDataset(
        "../../data/DF2k_aug_v2/train_HR", transform=transform, scale=2, mode="YCbCr"
    )
    train_dl = DataLoader(
        dataset=ds,
        batch_size=1,
        shuffle=False,
    )

    print(ds[14][0].shape, ds[14][1].shape)
    image, target = ds[4]
    plt.subplot(1, 2, 1)
    plt.imshow(torch.permute(image, (1, 2, 0)))
    plt.subplot(1, 2, 2)
    plt.imshow(torch.permute(target, (1, 2, 0)))
# %%
