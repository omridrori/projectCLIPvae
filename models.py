import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from clip_loss import  CLIPAttributeConsistency

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # Keep original encoder architecture
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 64, 4, 2, 1)
        self.conv4 = nn.Conv2d(64, 128, 4, 2, 1)
        self.maxp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxp2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxp3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxp4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Add FC layers for mu and logvar
        self.fc_mu = nn.Linear(128, 64)
        self.fc_logvar = nn.Linear(128, 64)

    def forward(self, x):
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

        # Flatten
        out = out.view(out.size(0), -1)

        # Split into mu and logvar
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)

        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, attribute_dims):
        """
        attribute_dims: Dictionary of attribute names and their number of possible values
        e.g., {"ethnicity": 3, "age": 3}
        """
        super(Decoder, self).__init__()

        # Core architecture remains the same
        # Calculate total embedding dimension
        total_embedding_dims = sum(10 for dims in attribute_dims.values())  # 10-dim embedding for each attribute
        self.total_input_dim = 64 + total_embedding_dims

        self.transconv1 = nn.ConvTranspose2d(self.total_input_dim, 64, 8, 4, 2)
        self.transconv2 = nn.ConvTranspose2d(64, 64, 8, 4, 2)
        self.transconv3 = nn.ConvTranspose2d(64, 64, 4, 2, 1)
        self.transconv4 = nn.ConvTranspose2d(64, 3, 4, 2, 1)

        # Dynamic embeddings based on attributes
        self.embeddings = nn.ModuleDict()
        for attr_name, num_values in attribute_dims.items():
            self.embeddings[attr_name] = nn.Embedding(num_values, 10)

        # Store attribute names in order
        self.attribute_names = list(attribute_dims.keys())

    def forward(self, x):

        # Split latent vector and attributes
        z = x[:, :64]
        current_idx = 64

        # Process each attribute through its embedding
        embeddings = []
        for attr_name in self.attribute_names:
            attr_val = x[:, current_idx].long()
            embedding = self.embeddings[attr_name](attr_val)
            embeddings.append(embedding)
            current_idx += 1

        # Concatenate latent vector with all embeddings
        z = torch.cat([z] + embeddings, dim=1)

        # Decode
        out = self.transconv1(z.view(z.shape[0], z.shape[1], 1, 1))
        out = F.relu(out)
        out = self.transconv2(out)
        out = F.relu(out)
        out = self.transconv3(out)
        out = F.relu(out)
        out = self.transconv4(out)
        out = F.sigmoid(out)

        return out


class CVAE(nn.Module):
    def __init__(self, encoder, decoder, attribute_dims):
        super(CVAE, self).__init__()
        self.encoder = encoder()
        self.decoder = decoder(attribute_dims)

    def forward(self, x, attrs):
        # Get mu and logvar from encoder
        mu, logvar = self.encoder(x)

        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        # Concatenate with attributes
        z = torch.cat([z, attrs], dim=1)

        # Decode
        out = self.decoder(z)
        return out, mu, logvar

def loss_function(recon_x, x, mu, logvar, vae, encoder_output, attrs,
                      beta_vae=1.0, beta_clip=1.0, clip_consistency=None):
    """
    Enhanced VAE loss function with CLIP-based attribute manipulation loss

    Args:
        recon_x: reconstructed input
        x: original input
        mu: mean of the latent distribution
        logvar: log variance of the latent distribution
        vae: VAE model
        encoder_output: full encoder output
        attrs: attribute tensor
        beta_vae: weight for the KL divergence term
        beta_clip: weight for the CLIP-based attribute loss
        clip_consistency: CLIPAttributeConsistency instance
    """
    # Original VAE losses
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    recon_loss = recon_loss / (x.size(0) * 3 * 64 * 64)

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD = KLD / (x.size(0) * 3 * 64 * 64)

    # Initialize CLIP consistency checker if not provided
    if clip_consistency is None:
        clip_consistency = CLIPAttributeConsistency(device=x.device)

    # Compute CLIP-based attribute consistency loss
    clip_loss = clip_consistency.compute_attribute_loss(vae, encoder_output, attrs)

    # Combine all losses
    total_loss = recon_loss + beta_vae * KLD + beta_clip * clip_loss

    return total_loss, recon_loss, KLD, clip_loss


class CosineScheduler:
    def __init__(self, start_value, end_value, num_cycles, num_epochs):
        """
        Cosine annealing scheduler for beta value

        Args:
            start_value: Initial beta value
            end_value: Final beta value
            num_cycles: Number of cycles for the cosine schedule
            num_epochs: Total number of epochs
        """
        self.start_value = start_value
        self.end_value = end_value
        self.num_cycles = num_cycles
        self.num_epochs = num_epochs

    def get_value(self, epoch):
        """
        Calculate the current beta value based on the epoch
        """
        # Calculate the progress within the current cycle
        cycle_length = self.num_epochs / self.num_cycles
        cycle_progress = (epoch % cycle_length) / cycle_length

        # Calculate cosine value (0 to 1)
        cosine_value = 0.5 * (1 + np.cos(np.pi * cycle_progress))

        # Interpolate between start and end values
        current_value = self.end_value + (self.start_value - self.end_value) * cosine_value
        return current_value