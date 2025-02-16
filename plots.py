import pandas as pd
import torch
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple, Optional
import seaborn as sns
from utils import attribute_to_text, get_attribute_changes


def plot_reconstructions(original_images: torch.Tensor,
                         reconstructed_images: torch.Tensor,
                         attrs: torch.Tensor,
                         attribute_names: List[str],
                         attribute_values: Dict[str, List[str]],
                         epoch: int,
                         save_dir: str,
                         max_samples: int = 5) -> None:
    """
    Plot original and reconstructed images with their attributes.

    Args:
        original_images: Tensor of original images
        reconstructed_images: Tensor of reconstructed images
        attrs: Tensor of attribute values
        attribute_names: List of attribute names
        attribute_values: Dictionary mapping attributes to possible values
        epoch: Current epoch number
        save_dir: Directory to save plots
        max_samples: Maximum number of samples to plot
    """
    num_samples = min(max_samples, original_images.size(0))
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 3 * num_samples))
    fig.suptitle(f'Original vs Reconstructed Images - Epoch {epoch}')

    # Get attribute descriptions
    descriptions = attribute_to_text(attrs[:num_samples], attribute_names, attribute_values)

    for idx in range(num_samples):
        # Original image
        orig_img = original_images[idx].cpu().permute(1, 2, 0).detach().numpy()
        axes[idx, 0].imshow(orig_img)
        axes[idx, 0].set_title('Original')
        axes[idx, 0].axis('off')

        # Reconstructed image
        recon_img = reconstructed_images[idx].cpu().permute(1, 2, 0).detach().numpy()
        axes[idx, 1].imshow(recon_img)
        axes[idx, 1].set_title('Reconstructed')
        axes[idx, 1].axis('off')

        # Add attributes as text
        plt.figtext(0.5, 0.98 - (idx * 0.2), descriptions[idx],
                    ha='center', va='top', bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'reconstructions_epoch_{epoch}.png'))
    plt.close()


def plot_training_progress(loss_history: Dict[str, List[float]], save_dir: str) -> None:
    """
    Plot training progress including all loss components.

    Args:
        loss_history: Dictionary containing loss values over time
        save_dir: Directory to save plots
    """
    plt.figure(figsize=(12, 8))

    # Plot all loss components
    for loss_name, values in loss_history.items():
        if loss_name != 'beta':  # Plot beta separately
            plt.plot(values, label=loss_name.capitalize())

    plt.title('Training Losses Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()

    # Save the plot
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'training_losses.png'))
    plt.close()

    # Plot beta schedule separately if it exists
    if 'beta' in loss_history:
        plt.figure(figsize=(12, 4))
        plt.plot(loss_history['beta'], color='purple')
        plt.title('Beta Schedule Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Beta')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'beta_schedule.png'))
        plt.close()


def plot_reconstructions_with_perturbations(original_images: torch.Tensor,
                                            reconstructed_images: torch.Tensor,
                                            perturbed_images: torch.Tensor,
                                            orig_attrs: torch.Tensor,
                                            perturbed_attrs: torch.Tensor,
                                            attribute_names: List[str],
                                            attribute_values: Dict[str, List[str]],
                                            epoch: int,
                                            save_dir: str,
                                            max_samples: int = 5) -> None:
    """
    Plot original, reconstructed, and attribute-perturbed images.

    Args:
        original_images: Original input images
        reconstructed_images: VAE reconstructed images
        perturbed_images: Images with perturbed attributes
        orig_attrs: Original attributes
        perturbed_attrs: Perturbed attributes
        attribute_names: List of attribute names
        attribute_values: Dictionary mapping attributes to possible values
        epoch: Current training epoch
        save_dir: Directory to save visualizations
        max_samples: Maximum number of samples to plot
    """
    num_samples = min(max_samples, original_images.size(0))
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4 * num_samples))
    fig.suptitle(f'Original vs Reconstructed vs Perturbed Images - Epoch {epoch}')

    # Get attribute changes
    changes = get_attribute_changes(orig_attrs[:num_samples],
                                    perturbed_attrs[:num_samples],
                                    attribute_names,
                                    attribute_values)

    for idx in range(num_samples):
        # Original image
        orig_img = original_images[idx].cpu().permute(1, 2, 0).detach().numpy()
        axes[idx, 0].imshow(orig_img)
        axes[idx, 0].set_title('Original')
        axes[idx, 0].axis('off')

        # Reconstructed image
        recon_img = reconstructed_images[idx].cpu().permute(1, 2, 0).detach().numpy()
        axes[idx, 1].imshow(recon_img)
        axes[idx, 1].set_title('Reconstructed')
        axes[idx, 1].axis('off')

        # Perturbed image
        pert_img = perturbed_images[idx].cpu().permute(1, 2, 0).detach().numpy()
        axes[idx, 2].imshow(pert_img)

        # Create change description
        change_text = []
        for attr_name, change in changes[idx].items():
            change_text.append(f"{attr_name}: {change}")
        change_desc = " | ".join(change_text)

        axes[idx, 2].set_title('Perturbed\n' + change_desc, fontsize=10)
        axes[idx, 2].axis('off')

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'perturbations_epoch_{epoch}.png'))
    plt.close()


def plot_attribute_distributions(df: pd.DataFrame,
                                 attribute_names: List[str],
                                 attribute_values: Dict[str, List[str]],
                                 save_dir: str) -> None:
    """
    Plot distribution of attributes in the dataset.

    Args:
        df: DataFrame containing attribute values
        attribute_names: List of attribute names
        attribute_values: Dictionary mapping attributes to possible values
        save_dir: Directory to save plots
    """
    num_attrs = len(attribute_names)
    fig = plt.figure(figsize=(15, 5 * ((num_attrs + 1) // 2)))

    for idx, attr_name in enumerate(attribute_names, 1):
        plt.subplot(((num_attrs + 1) // 2), 2, idx)

        # Get value counts
        value_counts = df[attr_name].value_counts().sort_index()
        values = [attribute_values[attr_name][i] for i in value_counts.index]

        # Create bar plot
        sns.barplot(x=values, y=value_counts.values)
        plt.title(f'Distribution of {attr_name}')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Count')

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'attribute_distributions.png'))
    plt.close()


def plot_latent_space(latent_vectors: torch.Tensor,
                      attrs: torch.Tensor,
                      attribute_names: List[str],
                      attribute_values: Dict[str, List[str]],
                      save_dir: str) -> None:
    """
    Plot 2D visualization of latent space colored by attributes.

    Args:
        latent_vectors: Encoded latent vectors
        attrs: Attribute tensors
        attribute_names: List of attribute names
        attribute_values: Dictionary mapping attributes to possible values
        save_dir: Directory to save plots
    """
    # Reduce dimensionality to 2D using PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_vectors.cpu().numpy())

    # Create a plot for each attribute
    for attr_idx, attr_name in enumerate(attribute_names):
        plt.figure(figsize=(10, 8))

        # Get attribute values for coloring
        attr_vals = attrs[:, attr_idx].cpu().numpy()
        values = attribute_values[attr_name]

        # Create scatter plot
        scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1],
                              c=attr_vals, cmap='tab10',
                              alpha=0.6)

        # Add legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor=scatter.cmap(scatter.norm(i)),
                                      label=val, markersize=10)
                           for i, val in enumerate(values)]
        plt.legend(handles=legend_elements, title=attr_name)

        plt.title(f'Latent Space Colored by {attr_name}')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')

        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'latent_space_{attr_name}.png'))
        plt.close()


def plot_attribute_correlations(df: pd.DataFrame,
                                attribute_names: List[str],
                                save_dir: str) -> None:
    """
    Plot correlation matrix between attributes.

    Args:
        df: DataFrame containing attribute values
        attribute_names: List of attribute names
        save_dir: Directory to save plots
    """
    # Calculate correlations
    corr_matrix = df[attribute_names].corr()

    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f', cbar_kws={'label': 'Correlation'})

    plt.title('Attribute Correlations')
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'attribute_correlations.png'))
    plt.close()


def plot_clip_confidence_distribution(confidence_scores: torch.Tensor,
                                      attribute_names: List[str],
                                      save_dir: str) -> None:
    """
    Plot distribution of CLIP confidence scores for each attribute.

    Args:
        confidence_scores: Tensor of CLIP confidence scores
        attribute_names: List of attribute names
        save_dir: Directory to save plots
    """
    num_attrs = len(attribute_names)
    fig = plt.figure(figsize=(15, 5 * ((num_attrs + 1) // 2)))

    for idx, attr_name in enumerate(attribute_names, 1):
        plt.subplot(((num_attrs + 1) // 2), 2, idx)

        # Get confidence scores for this attribute
        scores = confidence_scores[:, idx].cpu().numpy()

        # Create histogram
        sns.histplot(scores, bins=50)
        plt.title(f'CLIP Confidence Distribution for {attr_name}')
        plt.xlabel('Confidence Score')
        plt.ylabel('Count')

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'clip_confidence_distributions.png'))
    plt.close()