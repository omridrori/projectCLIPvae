import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from clip_loss import  CLIPAttributeConsistency


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # channels_in ,  channels_out, kernel_size, stride , padding,
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 64, 4, 2, 1)
        self.conv4 = nn.Conv2d(64, 128, 4, 2, 1)
        self.maxp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxp2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxp3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxp4 = nn.MaxPool2d(kernel_size=2, stride=2)

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
        return out.view(out.shape[0], -1)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # channels_in ,  channels_out, kernel_size, stride , padding,
        self.transconv1 = nn.ConvTranspose2d(64 + 40, 64, 8, 4, 2)
        self.transconv2 = nn.ConvTranspose2d(64, 64, 8, 4, 2)
        self.transconv3 = nn.ConvTranspose2d(64, 64, 4, 2, 1)
        self.transconv4 = nn.ConvTranspose2d(64, 3, 4, 2, 1)


        self.hairEmbedding = nn.Embedding(4, 10)
        self.beardEmbedding = nn.Embedding(2, 10)
        self.genderEmbedding = nn.Embedding(2, 10)
        self.paleSkinEmbedding = nn.Embedding(2, 10)

    def forward(self, x):
        z = x[:, :64]
        hair = self.hairEmbedding(x[:, 64].long())
        paleSkin = self.paleSkinEmbedding(x[:, 65].long())
        gender = self.genderEmbedding(x[:, 66].long())
        beard = self.beardEmbedding(x[:, 67].long())
        """
        Concating the embeddings and the encoded image
        """
        z = torch.cat([z, hair, beard, gender, paleSkin], dim=1)

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
    def __init__(self, encoder, decoder):
        super(CVAE, self).__init__()
        self.encoder = encoder()
        self.decoder = decoder()

    def forward(self, x, attrs):
        h = self.encoder(x)

        mu = h[:, :64]
        logvar = h[:, 64:]
        # this part is for the reparameterization trick
        s = torch.exp(logvar)
        eps = torch.randn_like(s)
        z = s * eps + mu

        z = torch.cat([z, attrs], dim=1)
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