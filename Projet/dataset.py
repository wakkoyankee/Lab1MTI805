from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np

class CastleEldenDataset(Dataset):
    def __init__(self, root_elden, root_castle, transform=None):
        self.root_elden = root_elden
        self.root_castle = root_castle
        self.transform = transform

        self.elden_images = os.listdir(root_elden)
        self.castle_images = os.listdir(root_castle)
        self.length_dataset = max(len(self.elden_images), len(self.castle_images)) # 1000, 1500
        self.elden_len = len(self.elden_images)
        self.castle_len = len(self.castle_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        elden_img = self.elden_images[index % self.elden_len]
        castle_img = self.castle_images[index % self.castle_len]

        elden_path = os.path.join(self.root_elden, elden_img)
        castle_path = os.path.join(self.root_castle, castle_img)

        elden_img = np.array(Image.open(elden_path).convert("RGB"))
        castle_img = np.array(Image.open(castle_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=elden_img, image0=castle_img)
            elden_img = augmentations["image"]
            castle_img = augmentations["image0"]

        return elden_img, castle_img
