import torch
import torch.nn.functional as F
import clip
from typing import Dict, List, Union
import random


class CLIPAttributeConsistency:
    def __init__(self, clip_model_name: str = "ViT-B/32", device: str = "cuda"):
        """
        Initialize CLIP-based attribute consistency checker.

        Args:
            clip_model_name: Name of the CLIP model to use
            device: Device to run the model on
        """
        self.device = device
        self.model, self.preprocess = clip.load(clip_model_name, device=device)
        self.attribute_templates = {}

    def register_templates(self, templates: Dict[str, Dict[int, List[str]]]):
        """
        Register templates for attributes.

        Args:
            templates: Dictionary of attribute templates in the format:
                {
                    'attribute_name': {
                        value_index: ['template1', 'template2', ...],
                        ...
                    },
                    ...
                }
        """
        self.attribute_templates = templates

    def generate_default_templates(self, attribute_info: Dict[str, Dict]):
        """
        Generate default templates for attributes if none are provided.

        Args:
            attribute_info: Dictionary containing attribute information:
                {
                    'attribute_name': {
                        'num_values': int,
                        'values': List[str/int]
                    },
                    ...
                }
        """
        templates = {}

        for attr_name, info in attribute_info.items():
            templates[attr_name] = {}

            for idx, value in enumerate(info['values']):
                # Convert value to string if it's not already
                value_str = str(value).lower().replace('_', ' ')

                # Generate multiple templates for each value
                templates[attr_name][idx] = [
                    f"a photo of a person with {value_str} features",
                    f"a {value_str} person",
                    f"a face showing {value_str} characteristics",
                    f"a portrait of someone with {value_str} appearance",
                    f"a photograph depicting {value_str} traits"
                ]

        self.attribute_templates = templates

    @torch.no_grad()
    def get_image_features(self, images: torch.Tensor) -> torch.Tensor:
        """Get normalized CLIP image features."""
        # Resize images to CLIP input size
        images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        image_features = self.model.encode_image(images)
        return image_features / image_features.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def get_text_features(self, descriptions: List[str]) -> torch.Tensor:
        """Get normalized CLIP text features."""
        text_tokens = clip.tokenize(descriptions).to(self.device)
        text_features = self.model.encode_text(text_tokens)
        return text_features / text_features.norm(dim=-1, keepdim=True)

    def compute_attribute_loss(self, images: torch.Tensor, attrs: torch.Tensor,
                               attribute_names: List[str]) -> torch.Tensor:
        """
        Compute CLIP-based attribute consistency loss for a batch.

        Args:
            images: Batch of images
            attrs: Tensor of attribute values
            attribute_names: List of attribute names in order

        Returns:
            torch.Tensor: Computed loss
        """
        if not self.attribute_templates:
            raise ValueError("No templates registered. Call register_templates first.")

        batch_size = images.size(0)
        total_loss = torch.tensor(0.0, device=self.device)
        num_attributes = len(attribute_names)

        # Get image features once
        image_features = self.get_image_features(images)

        # For each image in the batch, randomly select an attribute to verify
        for i in range(batch_size):
            # Randomly select an attribute to verify
            attr_idx = random.randint(0, num_attributes - 1)
            attr_name = attribute_names[attr_idx]

            # Get the actual value for this attribute
            current_value = attrs[i, attr_idx].item()

            # Get templates for the current value
            if current_value not in self.attribute_templates[attr_name]:
                continue

            current_templates = self.attribute_templates[attr_name][current_value]

            # Select a different value for contrast
            possible_values = list(self.attribute_templates[attr_name].keys())
            possible_values.remove(current_value)
            contrast_value = random.choice(possible_values)
            contrast_templates = self.attribute_templates[attr_name][contrast_value]

            # Get text features for both current and contrast templates
            current_text_features = self.get_text_features(current_templates)
            contrast_text_features = self.get_text_features(contrast_templates)

            # Compute similarities
            current_sim = (image_features[i:i + 1] @ current_text_features.T).mean()
            contrast_sim = (image_features[i:i + 1] @ contrast_text_features.T).mean()

            # Compute contrastive loss with margin
            margin = 0.2
            loss = torch.relu(contrast_sim - current_sim + margin)
            total_loss += loss

        # Normalize by batch size
        return total_loss / batch_size

    def verify_single_attribute(self, image: torch.Tensor, attr_name: str,
                                attr_value: int) -> float:
        """
        Verify a single attribute value for an image.

        Args:
            image: Single image tensor
            attr_name: Name of the attribute to verify
            attr_value: Value of the attribute to verify

        Returns:
            float: Confidence score for the attribute (0 to 1)
        """
        with torch.no_grad():
            # Get image features
            image_features = self.get_image_features(image.unsqueeze(0))

            # Get templates for this attribute value
            templates = self.attribute_templates[attr_name][attr_value]

            # Get text features
            text_features = self.get_text_features(templates)

            # Compute similarity
            similarity = (image_features @ text_features.T).mean().item()

            # Convert to probability
            return torch.sigmoid(torch.tensor(similarity * 100.0)).item()

    def batch_verify_attributes(self, images: torch.Tensor,
                                attrs: torch.Tensor,
                                attribute_names: List[str]) -> torch.Tensor:
        """
        Verify all attributes for a batch of images.

        Args:
            images: Batch of images
            attrs: Tensor of attribute values
            attribute_names: List of attribute names in order

        Returns:
            torch.Tensor: Tensor of confidence scores for each attribute
        """
        batch_size = images.size(0)
        num_attributes = len(attribute_names)
        confidences = torch.zeros(batch_size, num_attributes, device=self.device)

        with torch.no_grad():
            image_features = self.get_image_features(images)

            for attr_idx, attr_name in enumerate(attribute_names):
                for i in range(batch_size):
                    attr_value = attrs[i, attr_idx].item()
                    templates = self.attribute_templates[attr_name][attr_value]
                    text_features = self.get_text_features(templates)

                    similarity = (image_features[i:i + 1] @ text_features.T).mean()
                    confidences[i, attr_idx] = torch.sigmoid(similarity * 100.0)

        return confidences