import os

import pandas as pd
import torch

from celeba_project import train_vae_model
from utils import validate_attribute_config

# User-defined attributes
user_attributes = {
    "ethnicity": ["African", "Asian", "European"],
    "facial_expression": ["happy", "sad", "angry"]
}
results_df = pd.read_csv("results.csv")

validate_attribute_config({name: len(values) for name, values in user_attributes.items()})

# Setup directories
img_dir = r"/home/omrid/Desktop/jungo /projectCLIPvae/celeba_dataset/img_align_celeba/img_align_celeba"
vis_dir = os.path.join('training_outputs', 'visualizations')
os.makedirs(vis_dir, exist_ok=True)

# Train VAE
print("\nStep 2: Training VAE...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae, loss_history = train_vae_model(
    results_df=results_df,
    user_attributes=user_attributes,
    img_dir=img_dir,
    device=device,
    batch_size=32,
    num_epochs=100,
    vis_dir=vis_dir
)

print("\nTraining complete! Check the visualization directory for results.")