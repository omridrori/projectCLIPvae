import torch
from torch import nn
import torch.nn.functional as F


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






def loss_function(recon_x, x, mu, logvar):
    """
    Compute VAE loss with BCE reconstruction loss + KL divergence
    """
    # BCE reconstruction loss
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    recon_loss = recon_loss / (x.size(0) * 3 * 64 * 64)  # Normalize by batch size and dimensions

    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD = KLD / (x.size(0) * 3 * 64 * 64)  # Normalize by batch size and dimensions

    total_loss = recon_loss + KLD
    return total_loss, recon_loss, KLD