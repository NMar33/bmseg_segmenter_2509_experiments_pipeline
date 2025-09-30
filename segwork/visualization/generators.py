# segwork/visualization/generators.py

"""
Functions to generate visual panels (NumPy arrays) for plotting.
These functions prepare raw data and filter outputs for display with Matplotlib.
"""
import numpy as np
import cv2
from matplotlib import cm
from typing import Dict, Any

def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """
    Safely normalizes any NumPy array to the [0, 255] range for visualization.

    Args:
        arr: The input NumPy array.

    Returns:
        A NumPy array of type uint8 scaled to the [0, 255] range.
    """
    # DEV: Это важный хелпер. Matplotlib и OpenCV ожидают данные в определенных
    # форматах (float [0,1] или uint8 [0,255]). Эта функция унифицирует все
    # входные данные, делая остальной код проще.
    arr_f32 = arr.astype(np.float32)
    min_val, max_val = arr_f32.min(), arr_f32.max()
    
    if max_val <= min_val:
        return np.zeros_like(arr, dtype=np.uint8)
    
    # Scale to [0, 1] range first
    scaled_arr = (arr_f32 - min_val) / (max_val - min_val)
    
    # Then scale to [0, 255] and convert to uint8
    return (scaled_arr * 255).astype(np.uint8)


def generate_raw_panel(image: np.ndarray) -> np.ndarray:
    """
    Prepares a raw grayscale image for display by converting it to a 3-channel BGR format.

    Args:
        image: The input grayscale image (H, W).

    Returns:
        A 3-channel BGR image (H, W, 3) ready for plotting.
    """
    img_u8 = _normalize_to_uint8(image)
    return cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)


# def generate_overlay_panel(image: np.ndarray, mask: np.ndarray, color: tuple = (0, 255, 0), alpha: float = 0.4) -> np.ndarray:
#     """
#     Creates a BGR panel with a colored mask overlaid on the image.

#     Args:
#         image: The base grayscale image.
#         mask: The binary mask to overlay.
#         color: The (R, G, B) color for the mask.
#         alpha: The transparency of the mask overlay.

#     Returns:
#         A 3-channel BGR image with the mask overlay.
#     """
#     img_u8 = _normalize_to_uint8(image)
#     mask_u8 = _normalize_to_uint8(mask)

#     img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    
#     # Convert RGB color to BGR for OpenCV
#     color_bgr = (color[2], color[1], color[0])

#     # Create a solid color layer of the same size as the image
#     color_layer = np.zeros_like(img_bgr)
#     color_layer[:] = color_bgr

#     # Create a 3-channel version of the mask to apply the color
#     mask_3ch = (mask_u8 > 0).astype(np.uint8)[:, :, np.newaxis]
    
#     # Blend the image and the colored mask
#     overlayed = cv2.addWeighted(color_layer, alpha, img_bgr, 1 - alpha, 0)
    
#     # Apply the colored overlay only where the mask is positive
#     final_image = np.where(mask_3ch.astype(bool), overlayed, img_bgr)
    
#     return final_image.astype(np.uint8)

def generate_overlay_panel(image: np.ndarray, mask: np.ndarray, color: tuple = (0, 255, 0), alpha: float = 0.4) -> np.ndarray:
    """
    Creates a BGR panel with a colored mask overlaid on the image.

    Args:
        image: The base grayscale image.
        mask: The binary mask to overlay.
        color: The (R, G, B) color for the mask.
        alpha: The transparency of the mask overlay.

    Returns:
        A 3-channel BGR image with the mask overlay.
    """
    img_u8 = _normalize_to_uint8(image)
    # Ensure mask is binary (0 or 255) before creating the overlay
    mask_u8 = (mask > 0).astype(np.uint8) * 255

    img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    
    # Convert RGB color to BGR for OpenCV
    color_bgr = (color[2], color[1], color[0])

    # Create a solid color layer of the same size as the image
    color_layer = np.zeros_like(img_bgr)
    color_layer[:] = color_bgr

    # --- УЛУЧШЕННАЯ ЛОГИКА ---
    # DEV: Старый метод смешивал всё изображение с цветным слоем,
    # что могло немного окрашивать фон. Новый метод более точен.
    # Он смешивает цветной слой с изображением, а затем применяет
    # результат только там, где маска положительна.
    
    # 1. Смешиваем цветной слой с BGR-изображением
    overlay = cv2.addWeighted(color_layer, alpha, img_bgr, 1 - alpha, 0)
    
    # 2. Создаем 3-канальную маску для применения результата
    mask_3ch = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR) > 0
    
    # 3. Копируем исходное изображение и заменяем пиксели под маской на смешанные
    final_image = img_bgr.copy()
    final_image[mask_3ch] = overlay[mask_3ch]
    
    return final_image.astype(np.uint8)

def generate_filter_panel(filtered_image: np.ndarray, colormap_name: str = "viridis") -> np.ndarray:
    """
    Applies a Matplotlib colormap to a single-channel filter output to make it
    visually informative.

    Args:
        filtered_image: The single-channel output from a filter.
        colormap_name: The name of the Matplotlib colormap to apply.

    Returns:
        A 3-channel BGR image representing the color-mapped filter output.
    """
    # DEV: Результаты фильтров (например, Лапласиан) могут быть отрицательными.
    # Нормализация и применение colormap - стандартный способ их красиво показать.
    img_u8 = _normalize_to_uint8(filtered_image)
    
    # Get the specified colormap from Matplotlib
    try:
        colormap = cm.get_cmap(colormap_name)
    except ValueError:
        print(f"Warning: Colormap '{colormap_name}' not found. Falling back to 'viridis'.")
        colormap = cm.get_cmap("viridis")
    
    # Apply the colormap. It returns an RGBA image.
    colored_rgba = colormap(img_u8)
    
    # Convert RGBA to BGR for consistency with OpenCV and our plotter
    colored_bgr = cv2.cvtColor((colored_rgba * 255).astype(np.uint8), cv2.COLOR_RGBA2BGR)
    
    return colored_bgr

# --- НОВЫЕ ФУНКЦИИ ---

def generate_mask_panel(mask: np.ndarray) -> np.ndarray:
    """
    Converts a single-channel binary mask to a 3-channel BGR image (black and white).
    
    Args:
        mask: The input binary or grayscale mask.
        
    Returns:
        A 3-channel BGR image with white foreground on a black background.
    """
    # Ensure the mask is binary (0 or 255) and of type uint8
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    # Convert grayscale to BGR so it can be concatenated with other color panels
    return cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)


def generate_error_map_panel(gt_mask: np.ndarray, pred_mask: np.ndarray, style: Dict[str, Any]) -> np.ndarray:
    """
    Generates a color-coded error map visualizing True Positives (TP),
    False Positives (FP), and False Negatives (FN).

    Args:
        gt_mask: The ground truth binary mask.
        pred_mask: The prediction binary mask.
        style: The visualization style configuration dictionary, used to get colors.

    Returns:
        A 3-channel BGR image representing the error map.
    """
    gt_bin = (gt_mask > 0)
    pred_bin = (pred_mask > 0)

    # Create a blank BGR image
    H, W = gt_bin.shape
    error_map = np.zeros((H, W, 3), dtype=np.uint8)

    # Get colors from the style config, converting from RGB to BGR for OpenCV
    tp_color = np.array(style.get('error_map_tp_color_rgb', [80, 80, 80]), dtype=np.uint8)[::-1]
    fp_color = np.array(style.get('error_map_fp_color_rgb', [255, 0, 0]), dtype=np.uint8)[::-1]
    fn_color = np.array(style.get('error_map_fn_color_rgb', [0, 0, 255]), dtype=np.uint8)[::-1]
    
    # Assign colors based on pixel-wise comparison
    error_map[gt_bin & pred_bin] = tp_color      # True Positive
    error_map[~gt_bin & pred_bin] = fp_color     # False Positive
    error_map[gt_bin & ~pred_bin] = fn_color     # False Negative
    
    return error_map


def generate_info_panel(shape: tuple, metrics: Dict[str, Any], style: Dict[str, Any]) -> np.ndarray:
    """
    Creates an image panel with textual information about the sample and its metrics.

    Args:
        shape: The (height, width) of the panel to generate.
        metrics: A dictionary containing the ID, type, and metric values for the sample.
        style: The visualization style configuration dictionary.

    Returns:
        A 3-channel BGR image with rendered text.
    """
    H, W = shape
    
    # Create a background image
    bg_color = style.get('info_panel_bg_color_rgb', [20, 20, 20])[::-1] # to BGR
    panel = np.full((H, W, 3), bg_color, dtype=np.uint8)

    # Get text styling from config
    font_color = tuple(style.get('info_panel_font_color_bgr', [255, 255, 255]))
    font_scale = style.get('info_panel_font_scale', 0.8)
    font_thickness = style.get('info_panel_font_thickness', 1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Position text line by line, starting from the top
    y_pos = 40
    line_height = 40

    # Main title
    cv2.putText(panel, "Analysis Info", (10, y_pos), font, font_scale * 1.2, font_color, font_thickness + 1, cv2.LINE_AA)
    y_pos += line_height + 10
    
    # Basic Info (ID, Type)
    for key in ['image_id', 'type']:
        if key in metrics:
            text = f"{key.replace('_', ' ').title()}: {metrics[key]}"
            cv2.putText(panel, text, (10, y_pos), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
            y_pos += line_height

    # Metrics section
    y_pos += 20 # Add some space
    cv2.putText(panel, "--- Metrics ---", (10, y_pos), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
    y_pos += line_height

    for key, value in metrics.items():
        if key not in ['image_id', 'type'] and isinstance(value, (float, int)):
            text = f"{key.upper()}: {value:.4f}"
            cv2.putText(panel, text, (10, y_pos), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
            y_pos += line_height
            
    return panel