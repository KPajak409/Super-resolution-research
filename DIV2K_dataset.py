# %%
import cv2
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as v2
import torch
from PIL import Image

import matplotlib.pyplot as plt


class DIV2KDataset(Dataset):
    def __init__(self, root_dir, scale_factor=2, transform=(None, None)):
        self.root_dir = Path(root_dir)
        self.transform_lr = transform[0]
        self.transform_hr = transform[1]
        self.samples = []
        self.scale_factor = scale_factor
        self.samples = list(self.root_dir.glob("*"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target = img.copy()
        if self.transform_lr:
            img = cv2.resize(
                img,
                (img.shape[1] // self.scale_factor, img.shape[0] // self.scale_factor),
            )
            img = self.transform_lr(img)
        if self.transform_hr:
            target = self.transform_hr(target)

        return img, target

    def dim(self):
        return len(self.samples)


if __name__ == "__main__":
    scale_factor = 2
    transform_lr = v2.Compose(
        [
            v2.ToImage(),
            v2.CenterCrop((480 // scale_factor, 480 // scale_factor)),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    transform_hr = v2.Compose(
        [
            v2.ToImage(),
            v2.CenterCrop((480, 480)),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    ds = DIV2KDataset(
        "./data/DIV2k_aug/train_HR", 2, transform=[transform_lr, transform_hr]
    )
    train_dl = DataLoader(
        dataset=ds,
        batch_size=1,
        shuffle=False,
    )

    print(ds[0][0].shape, ds[0][1].shape)
    plt.imshow(torch.permute(ds[4][1], (1, 2, 0)))
# %%
