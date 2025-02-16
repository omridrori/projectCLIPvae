import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from clip_loss import CLIPAttributeConsistency


class Encoder(nn.Module):
    def __init__(self, latent_dim=64):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim

        # Convolutional layers for feature extraction
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 64, 4, 2, 1)
        self.conv4 = nn.Conv2d(64, 128, 4, 2, 1)

        # Pooling layers
        self.maxp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxp2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxp3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxp4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # FC layers for mu and logvar with dynamic latent dimension
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, x):
        # Feature extraction
        out = self.conv1(x)
        out = self.maxp1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.maxp2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.maxp3(out)
        out = F.relu(out)

        out = self.conv4(out)
        out = self.maxp4(out)
        out = F.relu(out)

        # Flatten for FC layers
        out = out.view(out.size(0), -1)

        # Generate latent distribution parameters
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)

        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, attribute_dims, latent_dim=64, embedding_dim=10):
        """
        Initialize decoder with dynamic attribute dimensions

        Args:
            attribute_dims: Dictionary mapping attribute names to number of possible values
                          e.g., {"ethnicity": 3, "facial_expression": 3}
            latent_dim: Dimension of the latent space
            embedding_dim: Dimension of each attribute embedding
        """
        super(Decoder, self).__init__()

        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.attribute_dims = attribute_dims

        # Create embeddings for each attribute
        self.embeddings = nn.ModuleDict({
            attr_name: nn.Embedding(num_values, embedding_dim)
            for attr_name, num_values in attribute_dims.items()
        })

        # Calculate total input dimension (latent + all attribute embeddings)
        total_embedding_dims = len(attribute_dims) * embedding_dim
        self.total_input_dim = latent_dim + total_embedding_dims

        # Transposed convolution layers
        self.transconv1 = nn.ConvTranspose2d(self.total_input_dim, 64, 8, 4, 2)
        self.transconv2 = nn.ConvTranspose2d(64, 64, 8, 4, 2)
        self.transconv3 = nn.ConvTranspose2d(64, 64, 4, 2, 1)
        self.transconv4 = nn.ConvTranspose2d(64, 3, 4, 2, 1)

        # Store attribute names in order
        self.attribute_names = list(attribute_dims.keys())

    def forward(self, x):
        """
        Forward pass with dynamic attribute handling

        Args:
            x: Concatenated tensor of [latent_vector, attr1, attr2, ...]
        """
        # Split latent vector and attributes
        z = x[:, :self.latent_dim]
        current_idx = self.latent_dim

        # Process each attribute through its embedding
        embeddings = []
        for attr_name in self.attribute_names:
            attr_val = x[:, current_idx].long()
            embedding = self.embeddings[attr_name](attr_val)
            embeddings.append(embedding)
            current_idx += 1

        # Concatenate latent vector with all embeddings
        z = torch.cat([z] + embeddings, dim=1)

        # Reshape for transposed convolutions
        z = z.view(z.shape[0], z.shape[1], 1, 1)

        # Decode
        out = F.relu(self.transconv1(z))
        out = F.relu(self.transconv2(out))
        out = F.relu(self.transconv3(out))
        out = torch.sigmoid(self.transconv4(out))

        return out


class CVAE(nn.Module):
    def __init__(self, attribute_dims, latent_dim=64, embedding_dim=10):
        """
        Initialize CVAE with dynamic attributes

        Args:
            attribute_dims: Dictionary mapping attribute names to number of possible values
            latent_dim: Dimension of the latent space
            embedding_dim: Dimension of each attribute embedding
        """
        super(CVAE, self).__init__()

        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(attribute_dims, latent_dim, embedding_dim)
        self.latent_dim = latent_dim

    def forward(self, x, attrs):
        # Get latent distribution parameters
        mu, logvar = self.encoder(x)

        # Reparameterization trick
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu

        # Concatenate with attributes
        z = torch.cat([z, attrs], dim=1)

        # Decode
        out = self.decoder(z)
        return out, mu, logvar


def loss_function(recon_x, x, mu, logvar, clip_consistency=None, attrs=None,
                  attribute_names=None, beta_vae=1.0, beta_clip=1.0):
    """
    Compute VAE loss with optional CLIP consistency

    Args:
        recon_x: Reconstructed image
        x: Original image
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        clip_consistency: Optional CLIPAttributeConsistency instance
        attrs: Attribute tensor (required if clip_consistency is provided)
        attribute_names: List of attribute names in order
        beta_vae: Weight for KL divergence term
        beta_clip: Weight for CLIP consistency loss
    """
    # Reconstruction loss
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    recon_loss = recon_loss / x.size(0)

    # KL divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kld = kld / x.size(0)

    # Initialize total loss
    total_loss = recon_loss + beta_vae * kld

    # Add CLIP consistency loss if provided
    clip_loss = 0
    if clip_consistency is not None and attrs is not None and attribute_names is not None:
        clip_loss = clip_consistency.compute_attribute_loss(recon_x, attrs, attribute_names)
        total_loss += beta_clip * clip_loss

    return total_loss, recon_loss, kld, clip_loss