from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
from PIL import Image

import torch.optim as  optim
from tqdm import tqdm

from dataset import CelebaDataset, transform
from models import Encoder, Decoder, CVAE, loss_function, CosineScheduler
from plots import plot_reconstructions

if torch.cuda.is_available():
  dev = "cuda:0"
  print("gpu up")
else:
  dev = "cpu"
device = torch.device(dev)

df = pd.read_csv("celeba_dataset/list_attr_celeba.csv")

def haircolor(x):
    if x["Blond_Hair"] == 1:
        return 0
    elif x["Brown_Hair"] == 1:
        return 1
    elif x["Black_Hair"] == 1:
        return 2
    else:
        return 3


df["Hair_Color"] = df.apply(haircolor, axis=1)

df = df[["image_id","Hair_Color",'Pale_Skin',"Male","No_Beard"]]
df.Pale_Skin = df.Pale_Skin.apply(lambda x: max(x,0))
df.Male = df.Male.apply(lambda x: max(x,0))
df.No_Beard = df.No_Beard.apply(lambda x: max(x,0))



# Initialize dataset and dataloader
dataset = CelebaDataset(
    df=df,
    img_dir = "/home/omrid/Desktop/jungo /projectCLIPvae/celeba_dataset/img_align_celeba/img_align_celeba/",
    transform=transform
)

dataloader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=True,
    num_workers=4,  # Parallel data loading
)



vae = CVAE(Encoder, Decoder)
vae.to(device)


def train_vae(vae, dataloader, optimizer, device, num_epochs=1201, save_interval=100):
    """
    Train the VAE model using BCE loss with beta scheduling
    """
    vae.train()
    loss_history = {
        'total': [],
        'reconstruction': [],
        'kl': [],
        'beta': []
    }

    # Initialize beta scheduler
    beta_scheduler = CosineScheduler(
        start_value=0.0,  # Start with low KL weight
        end_value=1.0,  # End with full KL weight
        num_cycles=4,  # Number of cycles during training
        num_epochs=num_epochs
    )

    for epoch in range(num_epochs):
        pbar = tqdm(total=len(dataloader), desc=f'Epoch {epoch}')
        epoch_total_loss = 0
        epoch_recon_loss = 0
        epoch_kl_loss = 0

        # Get current beta value
        current_beta = beta_scheduler.get_value(epoch)

        # Store first batch for visualization
        vis_batch = None

        for batch_idx, (images, attrs) in enumerate(dataloader):
            if batch_idx == 0:
                vis_batch = (images[:5].clone(), attrs[:5].clone())

            images = images.to(device)
            attrs = attrs.to(device)

            optimizer.zero_grad()
            recon_images, mu, logvar = vae(images, attrs)

            # Compute losses with current beta value
            total_loss, recon_loss, kl_loss = loss_function(recon_images, images, mu, logvar, current_beta)

            total_loss.backward()
            optimizer.step()

            # Accumulate losses
            epoch_total_loss += total_loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()

            pbar.update(1)

        pbar.close()

        # Calculate averages
        avg_total = epoch_total_loss / len(dataloader)
        avg_recon = epoch_recon_loss / len(dataloader)
        avg_kl = epoch_kl_loss / len(dataloader)

        # Store in history
        loss_history['total'].append(avg_total)
        loss_history['reconstruction'].append(avg_recon)
        loss_history['kl'].append(avg_kl)
        loss_history['beta'].append(current_beta)

        print(f"\nEpoch {epoch} Summary:")
        print(f"Total Loss: {avg_total:.6f}")
        print(f"BCE Loss: {avg_recon:.6f}")
        print(f"KL Loss: {avg_kl:.6f}")
        print(f"Current Beta: {current_beta:.6f}\n")

        # Generate and save reconstructions
        if vis_batch is not None:
            vae.eval()
            with torch.no_grad():
                vis_images, vis_attrs = vis_batch
                vis_images = vis_images.to(device)
                vis_attrs = vis_attrs.to(device)
                recon_images, _, _ = vae(vis_images, vis_attrs)
                plot_reconstructions(vis_images, recon_images, vis_attrs, epoch)
            vae.train()

        # Save checkpoint at intervals
        if epoch % save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': vae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_total,
                'beta': current_beta
            }, f"vae_checkpoint_epoch_{epoch}.pt")

    return loss_history


# Setup training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
vae = vae.to(device)
optimizer =  optim.Adam(vae.parameters(), lr=0.0001)  # Adjust learning rate if needed

# Train the model
loss_history = train_vae(
    vae=vae,
    dataloader=dataloader,
    optimizer=optimizer,
    device=device,
    num_epochs=1201,
    save_interval=100
)
# Plot training progress
plt.figure(figsize=(10, 5))
plt.plot(loss_history)
plt.title('Training Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()