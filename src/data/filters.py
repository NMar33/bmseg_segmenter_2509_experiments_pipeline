# src/data/filters.py

"""
Functions for applying classical image filters to create a multi-channel feature bank.
Each function takes a 2D NumPy array and returns a processed 2D array.
"""
import cv2
import numpy as np
from skimage.filters import frangi

def norm_zscore(img: np.ndarray) -> np.ndarray:
    """
    Normalizes an image using z-score (mean=0, std=1).

    Args:
        img: A 2D NumPy array.

    Returns:
        A normalized 2D NumPy array of type float32.
    """
    # DEV: Z-score нормализация - это стандартный и очень надежный способ
    # подготовки данных для нейронных сетей. Добавляем эпсилон для
    # предотвращения деления на ноль на пустых/черных тайлах.
    x = img.astype(np.float32)
    return (x - x.mean()) / (x.std() + 1e-6)

def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Args:
        img: A 2D NumPy array.
        clip_limit: Threshold for contrast limiting.
        tile_grid_size: Size of the grid for histogram equalization.

    Returns:
        A processed 2D NumPy array of type float32.
    """
    # DEV: CLAHE очень эффективен для изображений микроскопии, так как он
    # выравнивает локальный контраст, делая детали более заметными.
    # Важно: CLAHE в OpenCV работает с uint8, поэтому мы должны безопасно
    # преобразовать входное изображение из любого диапазона в [0, 255].
    if img.dtype != np.uint8:
        img_u8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    else:
        img_u8 = img
        
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    return clahe.apply(img_u8).astype(np.float32)

def scharr_mag(img: np.ndarray) -> np.ndarray:
    """
    Computes the magnitude of the Scharr gradient.

    Args:
        img: A 2D NumPy array, typically uint8.

    Returns:
        A 2D NumPy array of type float32 representing the gradient magnitude.
    """
    # DEV: Scharr - это как Sobel, но с более точными весами, что делает его
    # более изотропным (результат меньше зависит от ориентации градиента).
    gx = cv2.Scharr(img, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(img, cv2.CV_32F, 0, 1)
    return np.sqrt(gx**2 + gy**2)

def laplacian(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Applies the Laplacian filter to detect edges.

    Args:
        img: A 2D NumPy array, typically uint8.
        ksize: Aperture size for the Laplacian operator.

    Returns:
        A 2D NumPy array of type float32.
    """
    # DEV: Лапласиан - оператор второго порядка, выделяет области резкого изменения
    # интенсивности, полезен для поиска "острых" краев.
    return cv2.Laplacian(img, cv2.CV_32F, ksize=ksize)

def log_filter(img: np.ndarray, sigma: float) -> np.ndarray:
    """
    Applies a Laplacian of Gaussian (LoG) filter.

    Args:
        img: A 2D NumPy array, typically uint8.
        sigma: Standard deviation of the Gaussian kernel.

    Returns:
        A 2D NumPy array of type float32.
    """
    # DEV: LoG - это Лапласиан, примененный к Гауссову размытию.
    # Он менее чувствителен к шуму, чем чистый Лапласиан. Варьируя `sigma`,
    # мы можем находить "блобы" (каплевидные структуры) разного размера.
    img_blur = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return cv2.Laplacian(img_blur, cv2.CV_32F)

def frangi_filter(img: np.ndarray) -> np.ndarray:
    """
    Applies the Frangi vesselness filter to detect tube-like structures.

    Args:
        img: A 2D NumPy array (can be any dtype).

    Returns:
        A 2D NumPy array of type float32.
    """
    # DEV: Фильтр Франги специально разработан для выделения трубчатых структур
    # (сосуды, волокна). Он будет очень полезен для некоторых наших датасетов.
    # Skimage требует, чтобы вход был в диапазоне [0, 1].
    img_f32 = img.astype(np.float32)
    min_val, max_val = img_f32.min(), img_f32.max()
    img_norm = (img_f32 - min_val) / (max_val - min_val + 1e-6)
    return frangi(img_norm).astype(np.float32)


def build_feature_stack(gray_image: np.ndarray, config: dict) -> np.ndarray:
    """
    Builds a multi-channel stack from a grayscale image based on the config.
    This function acts as an orchestrator for all other filter functions.

    Args:
        gray_image: A 2D NumPy array (H, W).
        config: The `feature_bank` section of the main configuration dictionary.

    Returns:
        A 3D NumPy array (C, H, W) of type float32.
    """
    # DEV: Эта функция - сердце нашего препроцессинга. Она берет одно серое
    # изображение и на его основе создает "бутерброд" из разных каналов
    # согласно списку `channels` в конфиге.

    # Сначала применяем базовую нормализацию, если она указана.
    if config.get("base_norm", "zscore") == "zscore":
        base_normalized = norm_zscore(gray_image)
    else:
        # Если нормализация не z-score, просто приводим к float32
        base_normalized = gray_image.astype(np.float32)

    # Некоторые фильтры OpenCV работают с uint8, поэтому создадим 8-битную версию
    if gray_image.dtype != np.uint8:
        img_u8 = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    else:
        img_u8 = gray_image

    features = []
    for channel_name in config["channels"]:
        # DEV: Мы нормализуем выход каждого фильтра z-score'ом. Это важно
        # для того, чтобы каналы имели сопоставимые диапазоны значений и
        # нейронная сеть могла эффективно с ними работать.
        if channel_name == "raw":
            features.append(base_normalized)
        elif channel_name == "clahe":
            feat = apply_clahe(img_u8)
            features.append(norm_zscore(feat))
        elif channel_name == "scharr":
            feat = scharr_mag(img_u8)
            features.append(norm_zscore(feat))
        elif channel_name == "laplacian":
            feat = laplacian(img_u8)
            features.append(norm_zscore(feat))
        elif channel_name.startswith("log_sigma"):
            # Динамически парсим sigma из имени канала, например, "log_sigma2.5"
            try:
                sigma = float(channel_name.replace("log_sigma", ""))
            except ValueError:
                raise ValueError(f"Could not parse sigma from channel name: {channel_name}")
            feat = log_filter(img_u8, sigma)
            features.append(norm_zscore(feat))
        elif channel_name == "frangi":
            # Франги лучше работает на исходном диапазоне, а не на uint8
            feat = frangi_filter(gray_image)
            features.append(norm_zscore(feat))
        else:
            raise ValueError(f"Unknown feature channel in config: {channel_name}")
            
    return np.stack(features, axis=0).astype(np.float32)