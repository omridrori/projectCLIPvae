import os
import tempfile

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from dotenv import load_dotenv

from models import CVAE, loss_function
from dataset import CelebaDataset
from clip_classifier import CustomAttributeClassifier
from clip_loss import CLIPAttributeConsistency
from utils import (validate_attribute_config,
                   convert_clip_to_vae_format,
                   create_attribute_noise)
from plots import (plot_reconstructions_with_perturbations,
                   plot_training_progress,
                   plot_attribute_distributions)

# Define transform for images
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])


def generate_clip_labels(user_attributes, img_dir, max_images=100):
    """
    Generate labels for images using CLIP classifier.

    Args:
        user_attributes: Dictionary of attribute names and their possible values
        img_dir: Directory containing the images
        max_images: Maximum number of images to process

    Returns:
        results_df: DataFrame containing image IDs and their attribute classifications

    Raises:
        ValueError: If no valid images are found or processing fails
    """
    print("Generating labels using CLIP...")

    # Load environment variables
    load_dotenv()

    # Get the API key
    api_key = os.getenv("OPENAI_API_KEY")

    # Create the classifier with explicit API key
    classifier = CustomAttributeClassifier(
        clip_model_name="ViT-B/16",
        openai_api_key=api_key
    )

    # Validate directory exists
    if not os.path.exists(img_dir):
        raise ValueError(f"Image directory not found: {img_dir}")

    # Get list of all image files
    image_files = [f for f in os.listdir(img_dir)
                   if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    if not image_files:
        raise ValueError(f"No valid image files found in {img_dir}")

    # Take only first max_images
    selected_files = image_files[:max_images]
    print(f"\nProcessing {len(selected_files)} images out of {len(image_files)} total images")

    # Create a temporary directory with symlinks to selected images
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create symlinks for selected images
        for img_file in selected_files:
            src = os.path.join(img_dir, img_file)
            dst = os.path.join(temp_dir, img_file)
            if os.path.exists(src):
                os.symlink(src, dst)

        # Classify only selected images
        results_df = classifier.classify_images(
            img_dir=temp_dir,
            attribute_values=user_attributes,
            batch_size=16
        )

    # Validate results
    if results_df is None or len(results_df) == 0:
        raise ValueError(
            "CLIP classification failed to produce any valid results. Please check your input data and paths.")

    return results_df


def train_vae_model(results_df, user_attributes, img_dir, device="cuda",
                    batch_size=32, num_epochs=100, vis_dir="training_outputs"):
    """
    Train the VAE model using CLIP-generated labels.

    Args:
        results_df: DataFrame with CLIP-generated labels
        user_attributes: Dictionary of attribute names and their possible values
        img_dir: Directory containing images
        device: Device to run training on
        batch_size: Training batch size
        num_epochs: Number of training epochs
        vis_dir: Directory to save visualizations
    """
    print("\nPreparing for training...")

    # Convert CLIP results to VAE format
    vae_df, attribute_dims = convert_clip_to_vae_format(
        results_df,
        list(user_attributes.keys())
    )

    # Initialize dataset and dataloader
    dataset = CelebaDataset(
        df=vae_df,
        img_dir=img_dir,
        transform=transform,
        attribute_names=list(user_attributes.keys())
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Initialize models
    vae = CVAE(attribute_dims=attribute_dims).to(device)
    clip_consistency = CLIPAttributeConsistency(device=device)

    # Register templates with CLIP consistency checker
    clip_consistency.generate_default_templates(dataset.get_attribute_info())

    # Initialize optimizer
    optimizer = optim.Adam(vae.parameters(), lr=0.0001)

    # Initialize loss history
    loss_history = {
        'total': [],
        'reconstruction': [],
        'kl': [],
        'clip': []
    }

    print("\nStarting training...")
    os.makedirs(vis_dir, exist_ok=True)

    # Training loop
    for epoch in range(num_epochs):
        vae.train()
        epoch_losses = {k: 0.0 for k in loss_history.keys()}

        pbar = tqdm(total=len(dataloader),
                    desc=f'Epoch {epoch}/{num_epochs}',
                    position=0,
                    leave=True,
                    dynamic_ncols=True)

        # Store first batch for visualization
        vis_batch = None

        for batch_idx, (images, attrs) in enumerate(dataloader):
            if batch_idx == 0:
                vis_batch = (images[:5].clone(), attrs[:5].clone())

            images = images.to(device)
            attrs = attrs.to(device)

            optimizer.zero_grad()

            # Forward pass
            recon_images, mu, logvar = vae(images, attrs)

            # Compute losses
            total_loss, recon_loss, kl_loss, clip_loss = loss_function(
                recon_images, images, mu, logvar,
                clip_consistency=clip_consistency,
                attrs=attrs,
                attribute_names=list(user_attributes.keys()),
                beta_vae=1.0,
                beta_clip=1
            )

            # Backward pass
            total_loss.backward()
            optimizer.step()

            # Update loss tracking
            epoch_losses['total'] += total_loss.item()
            epoch_losses['reconstruction'] += recon_loss.item()
            epoch_losses['kl'] += kl_loss.item()
            epoch_losses['clip'] += clip_loss.item()

            pbar.update(1)

        pbar.close()

        # Average losses
        for k in epoch_losses:
            avg_loss = epoch_losses[k] / len(dataloader)
            loss_history[k].append(avg_loss)

        # Print progress on same line
        loss_str = f"Epoch {epoch}/{num_epochs} | " + " | ".join(
            [f"{k.capitalize()}: {v / len(dataloader):.6f}" for k, v in epoch_losses.items()]
        )
        tqdm.write(loss_str)

        # Save visualizations and checkpoint every 10 epochs
        if epoch % 1 == 0:
            if vis_batch is not None:
                vae.eval()
                with torch.no_grad():
                    images, attrs = vis_batch
                    images = images.to(device)
                    attrs = attrs.to(device)

                    # Get reconstructions
                    recon_images, mu, logvar = vae(images, attrs)

                    # Create perturbed versions
                    perturbed_attrs = attrs.clone()
                    for i in range(len(perturbed_attrs)):
                        attr_idx = torch.randint(0, len(user_attributes), (1,)).item()
                        current_val = perturbed_attrs[i, attr_idx].item()
                        num_values = len(user_attributes[list(user_attributes.keys())[attr_idx]])
                        new_val = (current_val + torch.randint(1, num_values, (1,)).item()) % num_values
                        perturbed_attrs[i, attr_idx] = new_val

                    # Generate perturbed images
                    perturbed_z = torch.cat([mu, perturbed_attrs], dim=1)
                    perturbed_images = vae.decoder(perturbed_z)

                    # Plot results
                    plot_reconstructions_with_perturbations(
                        images, recon_images, perturbed_images,
                        attrs, perturbed_attrs,
                        list(user_attributes.keys()),
                        user_attributes,
                        epoch,
                        vis_dir
                    )

            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': vae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_history,
                'attribute_dims': attribute_dims
            }
            torch.save(checkpoint, os.path.join(vis_dir, f'checkpoint_epoch_{epoch}.pt'))

            # Save training progress plot
            plot_training_progress(loss_history, vis_dir)

    return vae, loss_history


