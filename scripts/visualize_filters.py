# scripts/visualize_filters.py

"""
Main script for the visualization tool.

This script reads a configuration file to:
1. Load a source image and optional masks (ground truth, prediction).
2. Apply a series of specified image filters with varying parameters.
3. Generate a grid plot of the results for visual analysis and debugging.
4. Save the plot to a file and/or display it interactively.

Example usage from the `ml_pipeline` directory:
`python scripts/visualize_filters.py --config configs/viz/explore_isbi_filters.yaml`
"""
import argparse
import yaml
from pathlib import Path
import itertools
import tifffile as tiff
import cv2

# Import components from our ML pipeline's source code
from segwork.data import filters as filter_functions
from segwork.visualization.plotter import Plotter
from segwork.visualization.generators import (
    generate_raw_panel,
    generate_overlay_panel,
    generate_filter_panel
)

# DEV: Каталог фильтров для удобного вызова по имени из конфига.
# Он связывает строковое имя с функцией из нашего модуля `filters`.
# Это делает скрипт легко расширяемым: добавил функцию в filters.py,
# добавил запись сюда, и оно работает.
FILTER_CATALOG = {
    "clahe": filter_functions.apply_clahe,
    "scharr": filter_functions.scharr_mag,
    "laplacian": filter_functions.laplacian,
    "log_sigma": filter_functions.log_filter,
    "frangi": filter_functions.frangi_filter,
    "gabor": filter_functions.gabor,
}


def main():
    """Main function to orchestrate the visualization process."""
    parser = argparse.ArgumentParser(description="Visualize image filters and segmentation masks.")
    parser.add_argument("--config", required=True, help="Path to the visualization config file.")
    args = parser.parse_args()

    # Load the YAML configuration
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # --- 1. Load Data ---
    print("Loading data...")
    input_cfg = cfg['input_data']
    
    img_path = Path(input_cfg['image_source'])
    image = tiff.imread(img_path)
    print(f"Loaded image: {img_path}")
    
    gt_mask_path = input_cfg.get('gt_mask_source')
    gt_mask = cv2.imread(str(gt_mask_path), cv2.IMREAD_GRAYSCALE) if gt_mask_path else None
    
    pred_mask_path = input_cfg.get('pred_mask_source')
    pred_mask = cv2.imread(str(pred_mask_path), cv2.IMREAD_GRAYSCALE) if pred_mask_path else None

    # --- 2. Initialize the Plotter ---
    plotter = Plotter(cfg['output']['style'])

    # --- 3. Add Base Panels (Image and Overlays) ---
    plotter.add_panel(generate_raw_panel(image), "Original Image")
    if gt_mask is not None:
        plotter.add_panel(generate_overlay_panel(image, gt_mask, color=(0, 255, 0)), "Ground Truth Overlay")
    if pred_mask is not None:
        plotter.add_panel(generate_overlay_panel(image, pred_mask, color=(255, 0, 0)), "Prediction Overlay")

    # --- 4. Apply and Add Filter Panels ---
    print("Applying filters...")
    for filter_cfg in cfg.get('filters_to_visualize', []):
        name = filter_cfg['name']
        params = filter_cfg.get('params', {})
        
        # Expand parameter lists into individual combinations for grid search
        # DEV: `itertools.product` - это мощный инструмент. Он создает декартово
        # произведение списков, что позволяет легко перебирать все комбинации
        # параметров для "гридсерча".
        param_names = list(params.keys())
        param_values = [v if isinstance(v, list) else [v] for v in params.values()]
        
        for param_combination in itertools.product(*param_values):
            current_params = dict(zip(param_names, param_combination))
            
            # Create a descriptive title for the panel
            title = f"{name.capitalize()}"
            if current_params:
                title += "\n" + ", ".join(f"{k}={v}" for k, v in current_params.items())
            
            # Get the filter function from our catalog
            filter_func = FILTER_CATALOG.get(name)
            if not filter_func:
                print(f"Warning: Filter '{name}' not found in catalog. Skipping.")
                continue
                
            # Prepare the correct input format for the filter
            # DEV: Некоторые фильтры ожидают uint8, другие - float. Готовим оба варианта.
            # В будущем можно добавить метаданные в каталог, чтобы автоматизировать этот выбор.
            img_u8 = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            input_image_for_filter = image if name in ['frangi'] else img_u8
            
            # Apply the filter
            filtered_img = filter_func(input_image_for_filter, **current_params)
            
            # Generate the final panel with a colormap and add it to the plotter
            panel = generate_filter_panel(filtered_img, cfg['output']['style'].get('colormap', 'viridis'))
            plotter.add_panel(panel, title)

    # --- 5. Render the Final Plot ---
    print("Rendering plot...")
    output_cfg = cfg['output']
    save_path = None
    if output_cfg.get('save_dir'):
        save_dir = Path(output_cfg['save_dir'])
        # The output filename is automatically generated from the input image's name
        save_path = save_dir / f"{img_path.stem}_filter_visualization.png"

    plotter.render(save_path=save_path, show=output_cfg.get('show_interactive', False))
    print("Visualization process complete.")


if __name__ == "__main__":
    # DEV: Убедитесь, что вы запускаете этот скрипт из папки `ml_pipeline/`,
    # чтобы относительные пути в конфиге и импорты `segwork` работали корректно.
    main()