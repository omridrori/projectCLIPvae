import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pandas as pd
import numpy as np


class CelebaDataset(Dataset):
    """
    A flexible dataset class with resampling strategies for imbalanced attributes.
    """

    def __init__(self, df, img_dir, transform=None, attribute_names=None,
                 resampling_strategy='none', sampling_attributes=None):
        """
        Initialize dataset with resampling options.

        Args:
            df (pd.DataFrame): DataFrame containing 'image_id' and attribute columns
            img_dir (str): Directory containing images
            transform (callable, optional): Optional transform for images
            attribute_names (list, optional): List of attribute columns to use
            resampling_strategy (str): One of ['none', 'oversample', 'undersample', 'smote']
            sampling_attributes (list, optional): Specific attributes to balance. If None, balance all
        """
        self.df = df.copy()
        self.img_dir = img_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

        # Set attribute names
        self.attribute_names = attribute_names or [col for col in df.columns if col != 'image_id']

        # Validate attributes exist
        missing_attrs = [attr for attr in self.attribute_names if attr not in df.columns]
        if missing_attrs:
            raise ValueError(f"Attributes {missing_attrs} not found in DataFrame")

        # Create attribute information dictionary
        self.attribute_info = {}
        for attr in self.attribute_names:
            unique_values = sorted(df[attr].unique())
            self.attribute_info[attr] = {
                'num_values': len(unique_values),
                'values': unique_values
            }

        # Apply resampling if requested
        self.sampling_attributes = sampling_attributes or self.attribute_names
        if resampling_strategy != 'none':
            self._apply_resampling(resampling_strategy)

        # Store synthetic features if SMOTE is used
        self.synthetic_features = None

        # Validate image files exist
        self._validate_images()

    def _validate_images(self):
        """Validate that all images in the DataFrame exist in the directory."""
        missing_images = []
        for img_name in self.df['image_id']:
            # Skip validation for synthetic images
            if str(img_name).startswith('synthetic_'):
                continue
            img_path = os.path.join(self.img_dir, img_name)
            if not os.path.exists(img_path):
                missing_images.append(img_name)

        if missing_images:
            raise FileNotFoundError(
                f"Could not find {len(missing_images)} images. "
                f"First few missing: {missing_images[:5]}"
            )

    def _apply_resampling(self, strategy):
        """
        Apply the specified resampling strategy.
        """
        print(f"\nApplying {strategy} resampling strategy...")

        if strategy == 'oversample':
            self._apply_oversampling()
        elif strategy == 'undersample':
            self._apply_undersampling()
        elif strategy == 'smote':
            self._apply_smote()
        else:
            raise ValueError(f"Unknown resampling strategy: {strategy}")

    def _apply_oversampling(self):
        """
        Oversample minority classes to match majority class.
        """
        resampled_dfs = []

        # Create combination groups for selected attributes
        self.df['_group'] = self.df[self.sampling_attributes].apply(tuple, axis=1)
        group_counts = self.df['_group'].value_counts()
        max_count = group_counts.max()

        print(f"Oversampling to {max_count} samples per group...")

        # Oversample each group
        for group in group_counts.index:
            group_df = self.df[self.df['_group'] == group]
            if len(group_df) < max_count:
                # Randomly sample with replacement to match max_count
                resampled = group_df.sample(n=max_count, replace=True)
                resampled_dfs.append(resampled)
            else:
                resampled_dfs.append(group_df)

        self.df = pd.concat(resampled_dfs, ignore_index=True)
        self.df = self.df.drop('_group', axis=1)

    def _apply_undersampling(self):
        """
        Undersample majority classes to match minority class.
        """
        resampled_dfs = []

        # Create combination groups for selected attributes
        self.df['_group'] = self.df[self.sampling_attributes].apply(tuple, axis=1)
        group_counts = self.df['_group'].value_counts()
        min_count = group_counts.min()

        print(f"Undersampling to {min_count} samples per group...")

        # Undersample each group
        for group in group_counts.index:
            group_df = self.df[self.df['_group'] == group]
            if len(group_df) > min_count:
                # Randomly sample without replacement to match min_count
                resampled = group_df.sample(n=min_count, replace=False)
                resampled_dfs.append(resampled)
            else:
                resampled_dfs.append(group_df)

        self.df = pd.concat(resampled_dfs, ignore_index=True)
        self.df = self.df.drop('_group', axis=1)

    def _apply_smote(self):
        """
        Apply SMOTE (Synthetic Minority Over-sampling Technique).
        """
        try:
            from imblearn.over_sampling import SMOTE
        except ImportError:
            raise ImportError("Please install imbalanced-learn: pip install imbalanced-learn")

        print("Applying SMOTE resampling...")

        # Extract features for SMOTE
        features = []
        for _, row in self.df.iterrows():
            img_path = os.path.join(self.img_dir, row['image_id'])
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            features.append(img.flatten().numpy())

        features = np.stack(features)

        # Create combined target for selected attributes
        self.df['_group'] = self.df[self.sampling_attributes].apply(tuple, axis=1)
        group_mapping = {group: idx for idx, group in enumerate(self.df['_group'].unique())}
        targets = self.df['_group'].map(group_mapping)

        # Apply SMOTE
        smote = SMOTE(random_state=42)
        features_resampled, targets_resampled = smote.fit_resample(features, targets)

        # Store synthetic features
        self.synthetic_features = features_resampled[len(self.df):]

        # Reconstruct DataFrame
        new_rows = []
        for idx, target in enumerate(targets_resampled):
            if idx < len(self.df):
                new_rows.append(self.df.iloc[idx])
            else:
                original_group = {v: k for k, v in group_mapping.items()}[target]
                new_row = {'image_id': f'synthetic_{idx}'}
                for attr_idx, attr in enumerate(self.sampling_attributes):
                    new_row[attr] = original_group[attr_idx]
                new_rows.append(new_row)

        self.df = pd.DataFrame(new_rows)
        if '_group' in self.df.columns:
            self.df = self.df.drop('_group', axis=1)

    def get_attribute_dims(self):
        """Get dictionary of attribute dimensions."""
        return {attr: info['num_values'] for attr, info in self.attribute_info.items()}

    def get_attribute_info(self):
        """Get detailed attribute information."""
        return self.attribute_info.copy()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """Get a single item from the dataset."""
        img_name = self.df.iloc[idx]['image_id']

        # Handle synthetic images from SMOTE
        if str(img_name).startswith('synthetic_'):
            synthetic_idx = int(img_name.split('_')[1]) - len(self.df) + len(self.synthetic_features)
            image = torch.from_numpy(self.synthetic_features[synthetic_idx]).view(3, 64, 64)
        else:
            # Load real image
            img_path = os.path.join(self.img_dir, img_name)
            try:
                image = Image.open(img_path).convert('RGB')
                image = self.transform(image)
            except Exception as e:
                print(f"Error loading image {img_path}: {str(e)}")
                raise

        # Extract attributes
        attrs = []
        for attr_name in self.attribute_names:
            attr_value = self.df.iloc[idx][attr_name]
            attrs.append(attr_value)

        return image, torch.tensor(attrs, dtype=torch.long)


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