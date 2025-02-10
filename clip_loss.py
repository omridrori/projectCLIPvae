import torch
import torch.nn.functional as F
import clip
from random import choice


class CLIPAttributeConsistency:
    def __init__(self, clip_model_name="ViT-B/32", device="cuda"):
        self.device = device
        self.model, self.preprocess = clip.load(clip_model_name, device=device)

        self.attribute_templates = {
            'hair_color': {
                0: ["a photo of a person with blonde hair", "a face with blonde hair"],
                1: ["a photo of a person with brown hair", "a face with brown hair"],
                2: ["a photo of a person with black hair", "a face with black hair"],
                3: ["a photo of a person with red hair", "a face with red hair"]
            },
            'pale_skin': {
                0: ["a photo of a person with dark skin", "a dark-skinned person"],
                1: ["a photo of a person with pale skin", "a pale-skinned person"]
            },
            'gender': {
                0: ["a photo of a woman", "a female face"],
                1: ["a photo of a man", "a male face"]
            },
            'beard': {
                0: ["a photo of a person without a beard", "a face without facial hair"],
                1: ["a photo of a person with a beard", "a face with a beard"]
            }
        }

    @torch.no_grad()
    def get_image_features(self, images):
        """Get normalized CLIP image features"""
        images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        image_features = self.model.encode_image(images)
        return image_features / image_features.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def get_text_features(self, descriptions):
        """Get normalized CLIP text features for a list of descriptions"""
        text_tokens = clip.tokenize(descriptions).to(self.device)
        text_features = self.model.encode_text(text_tokens)
        return text_features / text_features.norm(dim=-1, keepdim=True)

    def compute_attribute_loss(self, vae, encoder_output, attrs):
        """
        Compute CLIP-based attribute consistency loss for entire batch in parallel
        """
        batch_size = attrs.size(0)

        # Get latent representations
        z = encoder_output[:, :64]  # [batch_size, 64]

        # Randomly select attributes to manipulate for each image in batch
        attr_indices = torch.randint(0, 4, (batch_size,), device=attrs.device)  # [batch_size]

        # Create mask for multiclass (hair_color) vs binary attributes
        is_hair_color = (attr_indices == 0)

        # Get original attribute values
        orig_attr_vals = torch.gather(attrs, 1, attr_indices.unsqueeze(1)).squeeze(1)  # [batch_size]

        # Create perturbed attributes tensor
        perturbed_attrs = attrs.clone().float()  # [batch_size, num_attrs]

        # Handle hair color (4 classes)
        hair_mask = is_hair_color
        if hair_mask.any():
            # For hair color, randomly select a different class
            new_hair_vals = torch.randint(0, 4, (hair_mask.sum(),), device=attrs.device).float()
            # Make sure new values are different from original
            same_hair = new_hair_vals == orig_attr_vals[hair_mask]
            if same_hair.any():
                new_hair_vals[same_hair] = (new_hair_vals[same_hair] + 1) % 4
            perturbed_attrs[hair_mask, 0] = new_hair_vals

        # Handle binary attributes
        binary_mask = ~is_hair_color
        if binary_mask.any():
            # For binary attributes, flip the value
            binary_indices = attr_indices[binary_mask]
            perturbed_attrs[binary_mask, binary_indices] = 1 - orig_attr_vals[binary_mask]

        # Get reconstructions for entire batch
        with torch.no_grad():
            # Original reconstructions
            orig_decoded = vae.decoder(torch.cat([z, attrs], dim=1))

            # Perturbed reconstructions
            perturbed_decoded = vae.decoder(torch.cat([z, perturbed_attrs], dim=1))

            # Get CLIP embeddings for all images
            orig_features = self.get_image_features(orig_decoded)
            perturbed_features = self.get_image_features(perturbed_decoded)

            # Initialize lists for text features
            all_orig_text_features = []
            all_perturbed_text_features = []

            # Get text features for each attribute type
            attr_names = ['hair_color', 'pale_skin', 'gender', 'beard']
            for attr_type in range(4):
                mask = (attr_indices == attr_type)
                if not mask.any():
                    continue

                # Get original descriptions
                orig_descriptions = []
                for idx in range(batch_size):
                    if mask[idx]:
                        orig_descriptions.extend(
                            self.attribute_templates[attr_names[attr_type]][int(orig_attr_vals[idx].item())]
                        )

                # Get perturbed descriptions
                perturbed_descriptions = []
                for idx in range(batch_size):
                    if mask[idx]:
                        perturbed_descriptions.extend(
                            self.attribute_templates[attr_names[attr_type]][int(perturbed_attrs[idx, attr_type].item())]
                        )

                if orig_descriptions:  # If we have any descriptions for this attribute
                    orig_text_features = self.get_text_features(orig_descriptions)
                    perturbed_text_features = self.get_text_features(perturbed_descriptions)

                    # Reshape to account for multiple descriptions per image
                    num_desc = len(self.attribute_templates[attr_names[attr_type]][0])
                    orig_text_features = orig_text_features.view(-1, num_desc, orig_text_features.size(-1))
                    perturbed_text_features = perturbed_text_features.view(-1, num_desc,
                                                                           perturbed_text_features.size(-1))

                    all_orig_text_features.append((mask, orig_text_features))
                    all_perturbed_text_features.append((mask, perturbed_text_features))

            # Compute losses for each attribute type and combine
            total_loss = 0
            margin = 0.2

            for (mask, orig_text), (_, perturbed_text) in zip(all_orig_text_features, all_perturbed_text_features):
                if not mask.any():
                    continue

                # Compute similarities for masked batch
                orig_features_masked = orig_features[mask]
                perturbed_features_masked = perturbed_features[mask]

                # Compute mean similarity across multiple descriptions
                orig_sim = (orig_features_masked @ orig_text.mean(dim=1).t()).mean(dim=1)
                perturbed_sim = (perturbed_features_masked @ perturbed_text.mean(dim=1).t()).mean(dim=1)

                # Cross similarities
                cross_orig_sim = (orig_features_masked @ perturbed_text.mean(dim=1).t()).mean(dim=1)
                cross_perturbed_sim = (perturbed_features_masked @ orig_text.mean(dim=1).t()).mean(dim=1)

                # Compute contrastive loss
                loss = (
                        torch.relu(cross_orig_sim - orig_sim + margin) +
                        torch.relu(cross_perturbed_sim - perturbed_sim + margin)
                ).mean()

                total_loss += loss

            return total_loss / len(all_orig_text_features)
