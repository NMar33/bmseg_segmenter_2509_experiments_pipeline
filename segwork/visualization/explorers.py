# src/visualization/explorers.py

"""
Functions for interactive exploration of datasets, designed to be called
from a Jupyter/Colab notebook environment.
"""
import random
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import tifffile as tiff
import cv2
import numpy as np


def explore_random_samples(
    dataset_path: str | Path,
    split: str = "train",
    n_samples: int = 9,
    figsize_per_sample: tuple = (5, 5)
) -> List[str]:
    """
    Displays a grid of random samples from a prepared dataset.

    For each sample, it shows the image overlaid with its ground truth mask
    and prints the sample's ID, which can be used for more detailed analysis.

    Args:
        dataset_path: Path to the root of the prepared (canonical) dataset.
        split: The data split to sample from (e.g., "train", "val", "test").
        n_samples: The number of random samples to display.
        figsize_per_sample: The (width, height) in inches for each sample's plot.

    Returns:
        A list of the selected sample IDs.
    """
    # DEV: Эта функция - прямой перенос логики из ячейки Colab в переиспользуемую
    # форму. Она зависит только от Matplotlib и библиотек для чтения изображений.
    
    print(f"Showing {n_samples} random samples from the '{split}' split...")
    
    # --- 1. Find data files ---
    dataset_path = Path(dataset_path)
    image_dir = dataset_path / "images" / split
    mask_dir = dataset_path / "masks" / split

    all_image_paths = sorted(list(image_dir.glob("*.tif")))
    if not all_image_paths:
        print(f"ERROR: No images found in {image_dir}. Please check the path and split name.")
        return []

    # --- 2. Select random samples ---
    num_to_sample = min(n_samples, len(all_image_paths))
    sampled_paths = random.sample(all_image_paths, num_to_sample)

    # --- 3. Plotting ---
    # Calculate grid layout
    cols = 3
    if num_to_sample <= 3:
        cols = num_to_sample
    rows = (num_to_sample + cols - 1) // cols
    
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * figsize_per_sample[0], rows * figsize_per_sample[1]),
        squeeze=False
    )
    axes = axes.flatten()
    
    selected_ids: List[str] = []
    
    for i, img_path in enumerate(sampled_paths):
        ax = axes[i]
        
        # Load image and corresponding mask
        image = tiff.imread(img_path)
        mask_path = mask_dir / img_path.with_suffix(".png").name
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else np.zeros_like(image)
        
        # Create a visually appealing overlay
        img_u8 = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        img_rgb = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)
        
        overlay = img_rgb.copy()
        overlay[mask > 0] = (0, 255, 0) # Bright green for the mask
        
        # Blend the two for a semi-transparent effect
        display_img = cv2.addWeighted(overlay, 0.4, img_rgb, 0.6, 0)
        
        ax.imshow(display_img)
        
        # Add the ID as the title
        image_id = img_path.stem
        ax.set_title(f"ID: {image_id}", fontsize=10)
        ax.axis('off')
        
        selected_ids.append(image_id)

    # Hide any unused subplots in the grid
    for j in range(num_to_sample, len(axes)):
        axes[j].axis('off')
        
    fig.tight_layout()
    plt.show()
    
    return selected_ids