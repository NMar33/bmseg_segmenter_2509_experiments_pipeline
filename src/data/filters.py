# src/data/filters.py

"""
Functions for applying classical image filters to create a multi-channel feature bank.
"""
import cv2
import numpy as np

def norm_zscore(img: np.ndarray) -> np.ndarray:
    """Normalizes an image using z-score."""
    # DEV: Z-score нормализация (mean=0, std=1) - это стандартный и очень
    # надежный способ подготовки данных для нейронных сетей.
    x = img.astype(np.float32)
    return (x - x.mean()) / (x.std() + 1e-6)

def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    """Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)."""
    # DEV: CLAHE очень эффективен для изображений микроскопии, так как он
    # выравнивает локальный контраст, делая детали более заметными.
    # Важно: CLAHE в OpenCV работает с uint8, поэтому мы должны безопасно
    # преобразовать входное изображение.
    if img.dtype != np.uint8:
        # Безопасное масштабирование из любого диапазона в [0, 255]
        img_u8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    else:
        img_u8 = img
        
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    return clahe.apply(img_u8).astype(np.float32)

# DEV: Здесь мы позже добавим другие фильтры: scharr_mag, laplacian, log_filter и т.д.

def build_feature_stack(gray_image: np.ndarray, config: dict) -> np.ndarray:
    """
    Builds a multi-channel stack from a grayscale image based on config.

    Args:
        gray_image: A 2D NumPy array (H, W).
        config: The `feature_bank` section of the main configuration.

    Returns:
        A 3D NumPy array (C, H, W) of type float32.
    """
    # DEV: Эта функция - оркестратор. Она берет одно изображение и на его основе
    # создает "бутерброд" из разных каналов согласно конфигу.
    
    # Сначала применяем базовую нормализацию, если она указана.
    if config.get("base_norm", "zscore") == "zscore":
        base_normalized = norm_zscore(gray_image)
    else:
        base_normalized = gray_image.astype(np.float32)

    features = []
    for channel_name in config["channels"]:
        if channel_name == "raw":
            features.append(base_normalized)
        elif channel_name == "clahe":
            clahe_img = apply_clahe(gray_image)
            features.append(norm_zscore(clahe_img)) # Нормализуем результат каждого фильтра
        # elif channel_name == "scharr": ... (добавим на Шаге 4)
        else:
            raise ValueError(f"Unknown feature channel in config: {channel_name}")
            
    return np.stack(features, axis=0).astype(np.float32)
