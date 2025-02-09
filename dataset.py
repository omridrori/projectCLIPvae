import os

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class CelebaDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['image_id']
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        # Extract attributes
        attrs = self.df.iloc[idx][['Hair_Color', 'Pale_Skin', 'Male', 'No_Beard']].values
        attrs = torch.tensor(attrs.astype('float32'))

        return image, attrs


# Define transforms
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),  # Automatically scales to [0,1] and converts to (C, H, W)
])



