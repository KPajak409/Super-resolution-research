import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class ImageNet(Dataset):
    def __init__(self, input_folder_path, target_folder_path, transform=transforms.ToTensor()):
        self.input_folder_path = input_folder_path
        self.target_folder_path = target_folder_path
        self.transform = transform
        
        input_fns = os.listdir(self.input_folder_path)
        target_fns = os.listdir(self.target_folder_path)
        
        self.input_file_names = [f'{input_folder_path}\\{input_fns[i]}' for i in range(len(input_fns))]
        self.target_file_names = [f'{target_folder_path}\\{target_fns[i]}' for i in range(len(target_fns))]
        
        
    def __len__(self):
        return len(self.input_file_names)
    
    def __getitem__(self, idx):
        input_image_pil = Image.open(self.input_file_names[idx]) 
        target_image_pil = Image.open(self.target_file_names[idx]) 

        return self.transform(input_image_pil, target_image_pil)
    
        