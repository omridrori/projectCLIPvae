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