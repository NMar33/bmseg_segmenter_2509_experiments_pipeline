# segwork/visualization/generators.py

"""
Functions to generate visual panels (NumPy arrays) for plotting.
These functions prepare raw data and filter outputs for display with Matplotlib.
"""
import numpy as np
import cv2
from matplotlib import cm

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
    mask_u8 = _normalize_to_uint8(mask)

    img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    
    # Convert RGB color to BGR for OpenCV
    color_bgr = (color[2], color[1], color[0])

    # Create a solid color layer of the same size as the image
    color_layer = np.zeros_like(img_bgr)
    color_layer[:] = color_bgr

    # Create a 3-channel version of the mask to apply the color
    mask_3ch = (mask_u8 > 0).astype(np.uint8)[:, :, np.newaxis]
    
    # Blend the image and the colored mask
    overlayed = cv2.addWeighted(color_layer, alpha, img_bgr, 1 - alpha, 0)
    
    # Apply the colored overlay only where the mask is positive
    final_image = np.where(mask_3ch.astype(bool), overlayed, img_bgr)
    
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