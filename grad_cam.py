import torch
import torch.nn.functional as F
import numpy as np
import cv2
from model import CleanPhysicsUNet_v3
from torch.utils.data import Dataset
import torchvision.transforms as T
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

from torch.utils.data import RandomSampler
from dataset import NoduleDataset


def visualize_physics_maps(model, images,device):
    model.eval()

    maps_list = []

    with torch.no_grad():
        for image in images:
            # Ensure input has shape [B, C, H, W]
            if len(image.shape) == 3:
                image = image.unsqueeze(0).to(device)

            S, alpha, gain, logits, gland, aux = model(
                image,
                return_aux=True
            )

            maps = {
                "Input": image.squeeze().cpu().numpy(),
                "Scatter map": aux["attention"].abs().mean(
                dim=1
            ).cpu().squeeze().numpy(),
            "Attenuation map": aux["I_rec"].cpu().squeeze().numpy(),
            "Predicted Mask": (torch.sigmoid(
                logits
            ).cpu().squeeze().numpy()>0.5)
            }

            maps_list.append(maps)

    # Number of images and maps
    n_images = len(maps_list)
    n_maps = len(maps_list[0])

    fig, axes = plt.subplots(
        n_images,
        n_maps,
        figsize=(2 * n_maps, 4* n_images),
        squeeze=False
    )

    for i, maps in enumerate(maps_list):

        for j, (title, data) in enumerate(maps.items()):

            ax = axes[i, j]

            cmap = (
                "gray"
                if title in ["Input", "Predicted Mask"]
                else "viridis"
            )

            ax.imshow(
                data,
                cmap=cmap
            )

            if i == 0:
                ax.set_title(
                    title,
                    fontsize=12,
                    fontweight="bold"
                )

            if j == 0:
                ax.set_ylabel(
                    f"Image {i+1}",
                    fontsize=11
                )

            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.show()

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img = Image.open(self.img_files[idx]).convert('L').resize(self.size)
        img = np.array(img).astype(np.float32) / 255.0


        img = self.physics_transforms(img)
        
        mask = Image.open(self.mask_files[idx]).convert('L').resize(self.size)
        mask = np.array(mask).astype(np.float32) / 255.0
        mask = (mask > 0.5).astype(np.float32)

        
        # return img, torch.from_numpy(mask).unsqueeze(0)
        return img


# 1. Initialize your model and wrapper
# For U-Net, the bottleneck layer or the last encoder layer works best
import random
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CleanPhysicsUNet_v3(256, 256, pretrained=True).to(DEVICE)

weight = torch.load("best_dice_model_ddti_learnable_tresh.pth", map_location="cpu")
model.load_state_dict(weight)

# target_layer = model.encoder.layer4[-1] 
# cam_extractor = SegGradCAM(model, target_layer)

# 2. Run the image
image_dir = "sample_data_sir"
mask_dir  = "/home/mizanu/project/datasets/Seg/thyroid/val_set/mask"

dataset = NoduleDataset(image_dir, mask_dir)
indx = random.randint(0, len(dataset))
random_index = random.sample(range (len(dataset)),4)

# sample_image, mask= dataset[indx]
# sample_image = sample_image.unsqueeze(0).to(DEVICE)

random_samples = [dataset[i] for i in random_index]

visualize_physics_maps(model, random_samples,DEVICE)



