import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pandas as pd
import numpy as np


class CelebaDataset(Dataset):
    """
    A flexible dataset class that can handle any number of attributes.
    """

    def __init__(self, df, img_dir, transform=None, attribute_names=None, img_size=(64, 64)):
        """
        Initialize dataset with flexible attributes.

        Args:
            df (pd.DataFrame): DataFrame containing 'image_id' and attribute columns
            img_dir (str): Directory containing images
            transform (callable, optional): Optional transform to be applied to images
            attribute_names (list, optional): List of attribute columns to use.
                                           If None, uses all columns except 'image_id'
            img_size (tuple): Target image size (height, width)
        """
        self.df = df
        self.img_dir = img_dir
        self.img_size = img_size

        # Set up transforms
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ])

        # Handle attribute names
        if attribute_names is None:
            self.attribute_names = [col for col in df.columns if col != 'image_id']
        else:
            # Validate that all specified attributes exist in the DataFrame
            missing_attrs = [attr for attr in attribute_names if attr not in df.columns]
            if missing_attrs:
                raise ValueError(f"Attributes {missing_attrs} not found in DataFrame")
            self.attribute_names = attribute_names

        # Create attribute information dictionary
        self.attribute_info = {}
        for attr in self.attribute_names:
            unique_values = sorted(df[attr].unique())
            self.attribute_info[attr] = {
                'num_values': len(unique_values),
                'values': unique_values
            }

        # Validate image files exist
        self._validate_images()

    def _validate_images(self):
        """Validate that all images in the DataFrame exist in the directory."""
        missing_images = []
        for img_name in self.df['image_id']:
            img_path = os.path.join(self.img_dir, img_name)
            if not os.path.exists(img_path):
                missing_images.append(img_name)

        if missing_images:
            raise FileNotFoundError(
                f"Could not find {len(missing_images)} images. "
                f"First few missing: {missing_images[:5]}"
            )

    def get_attribute_dims(self):
        """
        Get dictionary of attribute dimensions for VAE initialization.

        Returns:
            dict: Mapping of attribute names to number of possible values
        """
        return {attr: info['num_values'] for attr, info in self.attribute_info.items()}

    def get_attribute_info(self):
        """
        Get detailed information about attributes.

        Returns:
            dict: Dictionary containing attribute information including possible values
        """
        return self.attribute_info.copy()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.

        Args:
            idx (int): Index of the item

        Returns:
            tuple: (image, attributes) where attributes is a tensor of attribute values
        """
        # Get image name and path
        img_name = self.df.iloc[idx]['image_id']
        img_path = os.path.join(self.img_dir, img_name)

        try:
            # Load and transform image
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)

            # Extract attributes in order
            attrs = []
            for attr_name in self.attribute_names:
                attr_value = self.df.iloc[idx][attr_name]
                attrs.append(attr_value)

            # Convert attributes to tensor
            attrs = torch.tensor(attrs, dtype=torch.long)

            return image, attrs

        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            raise


def create_dataloader(df, img_dir, batch_size=32, num_workers=4, **dataset_kwargs):
    """
    Create a DataLoader with the specified parameters.

    Args:
        df (pd.DataFrame): DataFrame containing image IDs and attributes
        img_dir (str): Directory containing images
        batch_size (int): Batch size for training
        num_workers (int): Number of worker processes for data loading
        **dataset_kwargs: Additional arguments to pass to CelebaDataset

    Returns:
        DataLoader: Configured DataLoader instance
        dict: Attribute dimensions dictionary for VAE initialization
    """
    # Create dataset
    dataset = CelebaDataset(df=df, img_dir=img_dir, **dataset_kwargs)

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True  # Faster data transfer to GPU
    )

    # Get attribute dimensions for VAE initialization
    attribute_dims = dataset.get_attribute_dims()

    return dataloader, attribute_dims