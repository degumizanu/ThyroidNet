from torch.utils.data import Dataset
import os
import torchvision.transforms as T
import torch
from PIL import Image
import numpy as np
class NoduleDataset(Dataset):
    def __init__(self, data_dir, size=(256, 256)):
        self.data_dir = data_dir
        self.size = size
        self.image_files  = sorted(os.listdir(os.path.join(data_dir,"images")))
        self.mask_files = sorted(os.listdir(os.path.join(data_dir,"masks")))
        self.transforms = T.Compose([
            T.Resize(size),
            T.Lambda(lambda x: T.ToTensor()(x) + 0.05 * torch.randn_like(T.ToTensor()(x))),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):

        image_name = self.image_files[idx]
        mask_name = self.mask_files[idx]
        img = Image.open(os.path.join(os.path.join(self.data_dir, "images"),image_name)).convert('L')
        img  = self.transforms(img)
        mask = Image.open(os.path.join(os.path.join(self.data_dir,"masks"), mask_name)).convert('L')

        mask = np.array(mask).astype(np.float32) / 255.0
        mask = (mask > 0.5).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)
        return img, mask