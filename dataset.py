import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class CelebaDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, attribute_names=None):
        """
        Initialize dataset with flexible attributes

        Args:
            df: DataFrame containing image_id and attribute columns
            img_dir: Directory containing images
            transform: Image transforms
            attribute_names: List of attribute names to use
        """
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.attribute_names = attribute_names or [col for col in df.columns if col != 'image_id']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['image_id']
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Extract only specified attributes
        attrs = self.df.iloc[idx][self.attribute_names].values
        attrs = torch.tensor(attrs.astype('float32'))

        return image, attrs


# Define transforms
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])