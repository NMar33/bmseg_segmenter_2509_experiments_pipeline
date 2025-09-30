# segwork/visualization/explorers.py

"""
Functions for interactive exploration of datasets, designed to be called
from a Jupyter/Colab notebook environment.
"""
import random
from pathlib import Path
from typing import List, Dict, Any

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
    generate_filter_panel,
    generate_mask_panel,       
    generate_error_map_panel,  
    generate_info_panel,     
    _normalize_to_uint8        
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


def generate_evaluation_visuals(
    image: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    metrics: Dict[str, Any],
    feature_bank_config: Dict[str, Any],
    viz_config: Dict[str, Any],
    output_dir: Path,
    image_id: str,
    sample_type: str
):
    """
    Generates and saves a suite of visualizations for a single evaluation sample.
    """
    viz_style = viz_config.get('style', {})

    # --- 1. Generate and save the main 2x3 comparison panel ---
    try:
        plotter = Plotter(viz_style)
        
        # --- Row 1 ---
        # Panel 1: Info
        info_metrics = metrics.copy()
        info_metrics['type'] = sample_type.capitalize() # e.g., "Worst", "Best"
        plotter.add_panel(generate_info_panel(image.shape, info_metrics, viz_style), "") # No title for info panel
        
        # Panel 2: Input Image
        plotter.add_panel(generate_raw_panel(image), "Input Image")

        # Panel 3: Ground Truth (Mask only)
        plotter.add_panel(generate_mask_panel(gt_mask), "Ground Truth")
        
        # --- Row 2 ---
        # Panel 4: Prediction (Mask only)
        primary_metric_name = viz_config.get('primary_metric', 'dice').upper()
        metric_val = metrics.get(viz_config.get('primary_metric', 'dice'), 0.0)
        pred_title = f"Prediction\n{primary_metric_name}={metric_val:.4f}"
        plotter.add_panel(generate_mask_panel(pred_mask), pred_title)

        # Panel 5: Error Map
        plotter.add_panel(generate_error_map_panel(gt_mask, pred_mask, viz_style), "Error Map (FP: Red, FN: Blue)")

        # Panel 6: Overlay
        pred_overlay_color = viz_style.get('pred_overlay_color_rgb', [227, 27, 27])
        overlay_img = generate_overlay_panel(image, pred_mask, color=pred_overlay_color, alpha=viz_style.get('overlay_alpha', 0.4))
        plotter.add_panel(overlay_img, "Overlay")

        # Render the 2x3 grid
        plotter.render(
            save_path=output_dir / f"{image_id}_comparison.png",
            suptitle=f"Evaluation for Image: {image_id}",
            force_cols=viz_style.get('comparison_layout_cols', 3), # Force 3 columns
            show=False
        )
    except Exception as e:
        print(f"\n[Warning] Failed to generate comparison panel for '{image_id}'. Error: {e}")

    # --- 2. (Optional) Generate and save the feature bank panel ---
    if viz_config.get('visualize_feature_bank', False) and feature_bank_config.get('use', False):
        # Этот блок остается таким же, как я присылал ранее,
        # так как он уже был спроектирован правильно.
        try:
            fb_plotter = Plotter(viz_style)
            channels = feature_bank_config.get('channels', ['raw'])
            img_u8 = _normalize_to_uint8(image)
            
            for channel_name in channels:
                try:
                    title = channel_name.capitalize()
                    filtered_img = None
                    # ... (остальная логика применения фильтров без изменений)
                    if channel_name == 'raw':
                        filtered_img = image
                    elif channel_name == 'clahe':
                        filtered_img = filter_functions.apply_clahe(img_u8)
                    elif channel_name == 'scharr':
                        filtered_img = filter_functions.scharr_mag(img_u8)
                    elif channel_name == 'laplacian':
                        filtered_img = filter_functions.laplacian(img_u8)
                    elif channel_name.startswith("log_sigma"):
                        sigma = float(channel_name.replace("log_sigma", ""))
                        title = f"LoG (σ={sigma})"
                        filtered_img = filter_functions.log_filter(img_u8, sigma)
                    elif channel_name == 'frangi':
                        filtered_img = filter_functions.frangi_filter(image)
                    
                    if filtered_img is not None:
                        panel = generate_filter_panel(filtered_img, viz_style.get('colormap', 'viridis'))
                        fb_plotter.add_panel(panel, title)
                except Exception as e:
                    print(f"Warning: Could not apply filter '{channel_name}' for viz. Error: {e}")

            if fb_plotter.panels:
                fb_plotter.render(
                    save_path=output_dir / f"{image_id}_feature_bank.png",
                    suptitle=f"Feature Bank for Image: {image_id}",
                    show=False
                )
        except Exception as e:
            print(f"\n[Warning] Failed to generate feature bank panel for '{image_id}'. Error: {e}")