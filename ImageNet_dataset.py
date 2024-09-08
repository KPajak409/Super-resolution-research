import os
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms
import torch
from PIL import Image


class ImageNet(Dataset):
    def __init__(self, input_folder_path, target_folder_path, transform=None):
        self.input_folder_path = input_folder_path
        self.target_folder_path = target_folder_path
        self.transform = transform

        input_fns = os.listdir(self.input_folder_path)
        target_fns = os.listdir(self.target_folder_path)

        self.img_to_tensor = transforms.Compose(
            [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]
        )

        self.input_file_names = [
            f"{input_folder_path}/{input_fns[i]}" for i in range(len(input_fns))
        ]
        self.target_file_names = [
            f"{target_folder_path}/{target_fns[i]}" for i in range(len(target_fns))
        ]

    def __len__(self):
        return len(self.input_file_names)

    def __getitem__(self, idx):
        input_image_pil = Image.open(self.input_file_names[idx])
        target_image_pil = Image.open(self.target_file_names[idx])

        if self.transform:
            input_image_pil, target_image_pil = self.transform(
                input_image_pil, target_image_pil
            )

        input_image, target_image = self.img_to_tensor(
            input_image_pil, target_image_pil
        )

        return input_image, target_image
