import torch
import clip
from PIL import Image
import pandas as pd
from tqdm import tqdm
import os
from typing import Dict, List, Union
import torch.nn.functional as F


class CustomAttributeClassifier:
    def __init__(self, clip_model_name: str = "ViT-B/32", device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"Initializing CLIP model on {device}")
        self.model, self.preprocess = clip.load(clip_model_name, device=device)
        self.attribute_mappings = {}  # Will be populated when creating templates

    def create_text_templates(self, attribute_values: Dict[str, List[str]]) -> Dict[str, Dict[str, List[str]]]:
        templates = {}
        # Create attribute mappings while creating templates
        self.attribute_mappings = {
            attr: {idx: value for idx, value in enumerate(values)}
            for attr, values in attribute_values.items()
        }

        for attr, values in attribute_values.items():
            templates[attr] = {}
            for idx, value in enumerate(values):
                templates[attr][idx] = [
                    f"a photo of a person with {value} appearance",
                    f"a {value} person",
                    f"a face with {value} features"
                ]
        return templates

    def classify_images(self,
                        img_dir: str,
                        attribute_values: Dict[str, List[str]],
                        batch_size: int = 8) -> pd.DataFrame:
        """
        Classify images according to specified attributes using CLIP
        """
        # Verify directory exists
        if not os.path.exists(img_dir):
            raise ValueError(f"Directory not found: {img_dir}")

        # Create text templates
        templates = self.create_text_templates(attribute_values)

        # Get list of image files
        print(f"Scanning directory: {img_dir}")
        image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        if not image_files:
            raise ValueError(f"No image files found in {img_dir}")

        print(f"Found {len(image_files)} images")

        # Initialize results dictionary
        results = {
            'image_id': []
        }

        # Add columns for each attribute
        for attr in attribute_values.keys():
            results[attr] = []

        # Process images in batches
        for i in tqdm(range(0, len(image_files), batch_size),
                      desc="Classifying images",
                      position=0,
                      leave=True,
                      dynamic_ncols=True,
                      ascii=True):
            batch_files = image_files[i:i + batch_size]
            batch_images = []
            valid_files = []

            # Load and preprocess images
            for img_file in batch_files:
                try:
                    img_path = os.path.join(img_dir, img_file)
                    image = Image.open(img_path).convert('RGB')  # Ensure RGB format
                    processed_image = self.preprocess(image)
                    batch_images.append(processed_image)
                    valid_files.append(img_file)
                except Exception as e:
                    print(f"Error processing {img_file}: {str(e)}")
                    continue

            if not batch_images:
                continue

            # Stack images into a batch
            try:
                image_batch = torch.stack(batch_images).to(self.device)

                # Process each attribute
                with torch.no_grad():
                    image_features = self.model.encode_image(image_batch)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                    for attr, values in attribute_values.items():
                        # Prepare text features for all templates of this attribute
                        all_text_features = []
                        for idx in range(len(values)):
                            texts = templates[attr][idx]
                            text_tokens = clip.tokenize(texts).to(self.device)
                            text_features = self.model.encode_text(text_tokens)
                            text_features = text_features.mean(dim=0, keepdim=True)
                            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                            all_text_features.append(text_features)

                        text_features = torch.cat(all_text_features)
                        similarity = (100.0 * image_features @ text_features.T)
                        predictions = similarity.softmax(dim=-1).cpu().numpy()
                        class_indices = predictions.argmax(axis=1)

                        if attr not in results:
                            results[attr] = []
                        results[attr].extend(class_indices.tolist())

                results['image_id'].extend(valid_files)

            except Exception as e:
                print(f"Error processing batch: {str(e)}")
                continue

        # Create DataFrame
        df = pd.DataFrame(results)
        print(f"\nProcessed {len(df)} images successfully")
        print("\nAttribute distribution:")
        for attr, mapping in self.attribute_mappings.items():
            print(f"\n{attr.capitalize()} Distribution:")
            print("-" * 30)

            # Get value counts and calculate percentages
            counts = df[attr].value_counts().sort_index()
            total = len(df)

            # Print each category with count and percentage
            for idx in sorted(mapping.keys()):
                count = counts.get(idx, 0)
                percentage = (count / total) * 100
                print(f"{mapping[idx]:<25}: {count:>6,} ({percentage:>6.1f}%)")

        return df


def prepare_attributes_for_vae(df: pd.DataFrame,
                               attribute_mapping: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Prepare the classified attributes for use in the VAE

    Args:
        df: DataFrame with CLIP classifications
        attribute_mapping: Dictionary mapping attribute names to their possible values

    Returns:
        DataFrame with attributes encoded for VAE
    """
    vae_df = df.copy()

    # Convert categorical variables to numeric
    for attr, values in attribute_mapping.items():
        vae_df[attr] = vae_df[attr].astype(int)

    return vae_df