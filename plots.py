from matplotlib import pyplot as plt


def plot_reconstructions(original_images, reconstructed_images, attrs, epoch):
    """
    Plot 5 original and reconstructed images side by side with their attributes
    """
    fig, axes = plt.subplots(5, 2, figsize=(12, 15))
    fig.suptitle(f'Original vs Reconstructed Images - Epoch {epoch}')

    # Labels for attributes
    hair_labels = ['Blond', 'Brown', 'Black', 'Other']

    for idx in range(5):
        if idx < len(original_images):
            # Original image
            orig_img = original_images[idx].cpu().permute(1, 2, 0).detach().numpy()
            axes[idx, 0].imshow(orig_img)
            axes[idx, 0].set_title('Original')

            # Reconstructed image
            recon_img = reconstructed_images[idx].cpu().permute(1, 2, 0).detach().numpy()
            axes[idx, 1].imshow(recon_img)
            axes[idx, 1].set_title('Reconstructed')

            # Get attributes for this image
            attr = attrs[idx]
            attr_text = f'Hair: {hair_labels[int(attr[0])]} | '
            attr_text += f'Pale Skin: {bool(attr[1])} | '
            attr_text += f'Male: {bool(attr[2])} | '
            attr_text += f'No Beard: {bool(attr[3])}'

            # Add attributes as text below the images
            plt.figtext(0.5, 0.98 - (idx * 0.2), attr_text,
                        ha='center', va='top', bbox=dict(facecolor='white', alpha=0.8))

            # Remove axes
            axes[idx, 0].axis('off')
            axes[idx, 1].axis('off')

    plt.tight_layout()
    plt.savefig(f'/home/omrid/Desktop/jungo /projectCLIPvae/plot_reconstructions/reconstructions_epoch_{epoch}.png')
    plt.close()


def plot_training_progress(loss_history):
    """
    Plot training progress including losses and beta values
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot losses
    ax1.plot(loss_history['total'], label='Total Loss')
    ax1.plot(loss_history['reconstruction'], label='Reconstruction Loss')
    ax1.plot(loss_history['kl'], label='KL Loss')
    ax1.set_title('Training Losses Over Time')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.legend()

    # Plot beta values
    ax2.plot(loss_history['beta'], label='Beta Value', color='purple')
    ax2.set_title('Beta Schedule Over Time')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Beta')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
import torch


def plot_reconstructions_with_perturbations(original_images, reconstructed_images,
                                            perturbed_images, orig_attrs, perturbed_attrs,
                                            attr_indices, epoch):
    """
    Plot original, reconstructed, and perturbed images side by side

    Args:
        original_images: Original input images
        reconstructed_images: VAE reconstructed images
        perturbed_images: Images with perturbed attributes
        orig_attrs: Original attributes
        perturbed_attrs: Perturbed attributes
        attr_indices: Which attribute was modified for each image
        epoch: Current training epoch
    """
    num_samples = min(5, len(original_images))
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4 * num_samples))
    fig.suptitle(f'Original vs Reconstructed vs Perturbed Images - Epoch {epoch}')

    # Labels for attributes
    attr_names = ['Hair Color', 'Pale Skin', 'Gender', 'No Beard']
    hair_labels = ['Blonde', 'Brown', 'Black', 'Other']
    binary_labels = {
        'Pale Skin': ['Dark', 'Pale'],
        'Gender': ['Female', 'Male'],
        'No Beard': ['Has Beard', 'No Beard']
    }

    for idx in range(num_samples):
        if idx < len(original_images):
            # Original image
            orig_img = original_images[idx].cpu().permute(1, 2, 0).detach().numpy()
            axes[idx, 0].imshow(orig_img)
            axes[idx, 0].set_title('Original')

            # Reconstructed image
            recon_img = reconstructed_images[idx].cpu().permute(1, 2, 0).detach().numpy()
            axes[idx, 1].imshow(recon_img)
            axes[idx, 1].set_title('Reconstructed')

            # Perturbed image
            perturbed_img = perturbed_images[idx].cpu().permute(1, 2, 0).detach().numpy()
            axes[idx, 2].imshow(perturbed_img)

            # Get perturbed attribute info
            attr_idx = attr_indices[idx].item()
            attr_name = attr_names[attr_idx]

            if attr_idx == 0:  # Hair color
                orig_val = hair_labels[int(orig_attrs[idx, attr_idx].item())]
                new_val = hair_labels[int(perturbed_attrs[idx, attr_idx].item())]
            else:  # Binary attributes
                orig_val = binary_labels[attr_name][int(orig_attrs[idx, attr_idx].item())]
                new_val = binary_labels[attr_name][int(perturbed_attrs[idx, attr_idx].item())]

            axes[idx, 2].set_title(f'Perturbed: {attr_name}\n{orig_val} → {new_val}')

            # Remove axes
            for ax in axes[idx]:
                ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'plot_reconstructions/reconstructions_epoch_{epoch}.png')
    plt.close()


def generate_perturbation_samples(vae, clip_consistency, images, attrs, device):
    """
    Generate samples with perturbed attributes for visualization
    """
    vae.eval()
    with torch.no_grad():
        # Get encoder output
        encoder_output = vae.encoder(images)
        z = encoder_output[:, :64]

        # Get original reconstructions
        recon_images = vae.decoder(torch.cat([z, attrs], dim=1))

        # Create perturbed attributes
        batch_size = attrs.size(0)
        attr_indices = torch.randint(0, 4, (batch_size,), device=device)

        # Create perturbed attributes following same logic as in clip loss
        perturbed_attrs = attrs.clone().float()

        # Handle hair color (4 classes)
        hair_mask = (attr_indices == 0)
        if hair_mask.any():
            new_hair_vals = torch.randint(0, 4, (hair_mask.sum(),), device=device).float()
            same_hair = new_hair_vals == attrs[hair_mask, 0]
            if same_hair.any():
                new_hair_vals[same_hair] = (new_hair_vals[same_hair] + 1) % 4
            perturbed_attrs[hair_mask, 0] = new_hair_vals

        # Handle binary attributes
        binary_mask = ~hair_mask
        if binary_mask.any():
            binary_indices = attr_indices[binary_mask]
            orig_vals = torch.gather(attrs[binary_mask], 1, binary_indices.unsqueeze(1)).squeeze(1)
            perturbed_attrs[binary_mask, binary_indices] = 1 - orig_vals

        # Generate perturbed reconstructions
        perturbed_images = vae.decoder(torch.cat([z, perturbed_attrs], dim=1))

        return recon_images, perturbed_images, perturbed_attrs, attr_indices


def save_training_visualizations(vae, clip_consistency, vis_batch, epoch):
    """
    Generate and save training visualizations
    """
    images, attrs = vis_batch
    images = images.to(vae.encoder.conv1.weight.device)
    attrs = attrs.to(vae.encoder.conv1.weight.device)

    # Generate samples
    recon_images, perturbed_images, perturbed_attrs, attr_indices = generate_perturbation_samples(
        vae, clip_consistency, images, attrs, images.device
    )

    # Plot results
    plot_reconstructions_with_perturbations(
        images, recon_images, perturbed_images,
        attrs, perturbed_attrs, attr_indices, epoch
    )