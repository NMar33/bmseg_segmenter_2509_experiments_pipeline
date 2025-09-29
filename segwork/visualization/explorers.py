# segwork/visualization/explorers.py

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
import yaml
import itertools

from segwork.data import filters as filter_functions
from segwork.visualization.plotter import Plotter
from segwork.visualization.generators import (
    generate_raw_panel,
    generate_overlay_panel,
    generate_filter_panel
)

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

def visualize_filters_for_image(config: dict):
    """
    Loads data and generates a filter visualization plot based on a config dictionary.
    Designed to be called directly from a notebook.

    Args:
        config: A dictionary with the same structure as a visualize_filters.yaml file.
    """
    # DEV: Эта функция - это, по сути, тело скрипта `visualize_filters.py`,
    # но без парсинга аргументов. Она принимает готовый конфиг.

    # 1. Загрузка данных
    print("Loading data...")
    input_cfg = config['input_data']
    img_path = Path(input_cfg['image_source'])
    image = tiff.imread(img_path)
    
    gt_mask_path = input_cfg.get('gt_mask_source')
    gt_mask = cv2.imread(str(gt_mask_path), cv2.IMREAD_GRAYSCALE) if gt_mask_path else None
    
    pred_mask_path = input_cfg.get('pred_mask_source')
    pred_mask = cv2.imread(str(pred_mask_path), cv2.IMREAD_GRAYSCALE) if pred_mask_path else None

    # 2. Инициализация отрисовщика
    plotter = Plotter(config['output']['style'])

    # 3. Добавление базовых панелей
    plotter.add_panel(generate_raw_panel(image), "Original Image")
    if gt_mask is not None:
        plotter.add_panel(generate_overlay_panel(image, gt_mask, color=(0, 255, 0)), "Ground Truth")
    if pred_mask is not None:
        plotter.add_panel(generate_overlay_panel(image, pred_mask, color=(255, 0, 0)), "Prediction")

    # 4. Применение и добавление фильтров
    print("Applying filters...")
    FILTER_CATALOG = {
        "clahe": filter_functions.apply_clahe,
        "scharr": filter_functions.scharr_mag,
        "laplacian": filter_functions.laplacian,
        "log_sigma": filter_functions.log_filter,
        "frangi": filter_functions.frangi_filter,
        "gabor": filter_functions.gabor,
    }
    
    for filter_cfg in config.get('filters_to_visualize', []):
        name = filter_cfg['name']
        params = filter_cfg.get('params', {})
        param_names = list(params.keys())
        param_values = [v if isinstance(v, list) else [v] for v in params.values()]
        
        for param_combination in itertools.product(*param_values):
            current_params = dict(zip(param_names, param_combination))
            title = f"{name.capitalize()}\n" + ", ".join(f"{k}={v}" for k, v in current_params.items())
            
            filter_func = FILTER_CATALOG.get(name)
            if not filter_func: continue
            
            img_u8 = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            input_image_for_filter = image if name in ['frangi'] else img_u8
            
            filtered_img = filter_func(input_image_for_filter, **current_params)
            panel = generate_filter_panel(filtered_img, config['output']['style'].get('colormap', 'viridis'))
            plotter.add_panel(panel, title.strip())

    # 5. Рендеринг
    print("Rendering plot...")
    output_cfg = config['output']
    save_path = None
    if output_cfg.get('save_dir'):
        save_dir = Path(output_cfg['save_dir'])
        save_path = save_dir / f"{img_path.stem}_filter_visualization.png"

    # `show=True` теперь будет работать, так как мы в сессии ноутбука
    plotter.render(save_path=save_path, show=output_cfg.get('show_interactive', True))