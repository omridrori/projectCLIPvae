import torch
import pandas as pd
from typing import Dict, List, Union, Tuple
import numpy as np


def validate_attribute_config(attribute_dims: Dict[str, int]) -> bool:
    """
    Validate attribute configuration dictionary.

    Args:
        attribute_dims: Dictionary mapping attribute names to number of possible values

    Returns:
        bool: True if configuration is valid

    Raises:
        ValueError: If configuration is invalid
    """
    if not attribute_dims:
        raise ValueError("attribute_dims cannot be empty")

    for attr_name, num_values in attribute_dims.items():
        if not isinstance(num_values, int):
            raise ValueError(f"Number of values for {attr_name} must be integer")
        if num_values < 2:
            raise ValueError(f"Attribute {attr_name} must have at least 2 values")

    return True


def convert_clip_to_vae_format(clip_df: pd.DataFrame,
                               attribute_names: List[str]) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Convert CLIP classifier output to VAE-compatible format.

    Args:
        clip_df: DataFrame from CLIP classifier
        attribute_names: List of attribute names to process

    Returns:
        Tuple containing:
            - DataFrame with processed attributes
            - Dictionary of attribute dimensions
    """
    vae_df = clip_df.copy()
    attribute_dims = {}

    for attr in attribute_names:
        if attr not in vae_df.columns:
            raise ValueError(f"Attribute {attr} not found in DataFrame")

        # Get number of unique values
        unique_values = sorted(vae_df[attr].unique())
        attribute_dims[attr] = len(unique_values)

        # Ensure values are zero-based indices
        value_map = {val: idx for idx, val in enumerate(unique_values)}
        vae_df[attr] = vae_df[attr].map(value_map)

    return vae_df, attribute_dims


def create_attribute_noise(batch_size: int,
                           attribute_dims: Dict[str, int],
                           device: str = "cuda") -> torch.Tensor:
    """
    Create random attribute tensor for generation.

    Args:
        batch_size: Number of samples to generate
        attribute_dims: Dictionary of attribute dimensions
        device: Device to create tensor on

    Returns:
        torch.Tensor: Random attribute tensor
    """
    attrs = []
    for num_values in attribute_dims.values():
        attr = torch.randint(0, num_values, (batch_size,), device=device)
        attrs.append(attr)
    return torch.stack(attrs, dim=1)


def interpolate_attributes(start_attrs: torch.Tensor,
                           end_attrs: torch.Tensor,
                           steps: int) -> torch.Tensor:
    """
    Create interpolation between two attribute configurations.

    Args:
        start_attrs: Starting attribute tensor
        end_attrs: Ending attribute tensor
        steps: Number of interpolation steps

    Returns:
        torch.Tensor: Interpolated attribute tensors
    """
    # For categorical attributes, we'll gradually change probability distribution
    alphas = torch.linspace(0, 1, steps)
    interp_attrs = []

    for alpha in alphas:
        # Use alpha to interpolate between one-hot encodings
        interp = torch.lerp(start_attrs.float(), end_attrs.float(), alpha)
        # Convert back to categorical by taking argmax
        categorical = torch.argmax(interp, dim=-1)
        interp_attrs.append(categorical)

    return torch.stack(interp_attrs)


def get_attribute_changes(orig_attrs: torch.Tensor,
                          perturbed_attrs: torch.Tensor,
                          attribute_names: List[str],
                          attribute_values: Dict[str, List[str]]) -> List[Dict[str, str]]:
    """
    Get human-readable description of attribute changes.

    Args:
        orig_attrs: Original attribute tensor
        perturbed_attrs: Modified attribute tensor
        attribute_names: List of attribute names
        attribute_values: Dictionary mapping attributes to their possible values

    Returns:
        List[Dict[str, str]]: List of changes in format {"attribute": "old -> new"}
    """
    changes = []

    for i in range(orig_attrs.size(0)):
        sample_changes = {}
        for j, attr_name in enumerate(attribute_names):
            orig_val = orig_attrs[i, j].item()
            pert_val = perturbed_attrs[i, j].item()

            if orig_val != pert_val:
                values = attribute_values[attr_name]
                change = f"{values[orig_val]} → {values[pert_val]}"
                sample_changes[attr_name] = change

        changes.append(sample_changes)

    return changes


def attribute_to_text(attrs: torch.Tensor,
                      attribute_names: List[str],
                      attribute_values: Dict[str, List[str]]) -> List[str]:
    """
    Convert attribute tensor to human-readable text descriptions.

    Args:
        attrs: Attribute tensor
        attribute_names: List of attribute names
        attribute_values: Dictionary mapping attributes to their possible values

    Returns:
        List[str]: Human-readable descriptions
    """
    descriptions = []

    for i in range(attrs.size(0)):
        attr_strs = []
        for j, attr_name in enumerate(attribute_names):
            val_idx = attrs[i, j].item()
            value = attribute_values[attr_name][val_idx]
            attr_strs.append(f"{attr_name}: {value}")
        descriptions.append(" | ".join(attr_strs))

    return descriptions


def validate_attributes_tensor(attrs: torch.Tensor,
                               attribute_dims: Dict[str, int]) -> bool:
    """
    Validate attribute tensor against expected dimensions.

    Args:
        attrs: Attribute tensor to validate
        attribute_dims: Dictionary of expected attribute dimensions

    Returns:
        bool: True if tensor is valid

    Raises:
        ValueError: If tensor is invalid
    """
    expected_dims = len(attribute_dims)
    if attrs.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got {attrs.dim()}D")

    if attrs.size(1) != expected_dims:
        raise ValueError(f"Expected {expected_dims} attributes, got {attrs.size(1)}")

    for i, (attr_name, num_values) in enumerate(attribute_dims.items()):
        if torch.any((attrs[:, i] < 0) | (attrs[:, i] >= num_values)):
            raise ValueError(f"Invalid values for attribute {attr_name}")

    return True