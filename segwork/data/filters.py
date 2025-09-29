# segwork/data/filters.py

"""
Functions for applying classical image filters to create a multi-channel feature bank.
"""
import cv2
import numpy as np
from skimage.filters import frangi

# DEV: `norm_zscore` и `apply_clahe` остаются без изменений.
def norm_zscore(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float32)
    return (x - x.mean()) / (x.std() + 1e-6)

def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    if img.dtype != np.uint8:
        img_u8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    else:
        img_u8 = img
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    return clahe.apply(img_u8).astype(np.float32)

# --- НОВЫЕ ФИЛЬТРЫ ---

def scharr_mag(img: np.ndarray) -> np.ndarray:
    """Computes the magnitude of the Scharr gradient."""
    # DEV: Scharr - это как Sobel, но с более точными весами, что делает его
    # более изотропным (результат меньше зависит от ориентации градиента).
    gx = cv2.Scharr(img, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(img, cv2.CV_32F, 0, 1)
    return np.sqrt(gx**2 + gy**2)

def laplacian(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Applies the Laplacian filter."""
    # DEV: Лапласиан - оператор второго порядка, выделяет области резкого изменения
    # интенсивности, полезен для поиска "острых" краев.
    return cv2.Laplacian(img, cv2.CV_32F, ksize=ksize)

def log_filter(img: np.ndarray, sigma: float) -> np.ndarray:
    """Applies a Laplacian of Gaussian (LoG) filter."""
    # DEV: LoG - это Лапласиан, примененный к Гауссову размытию.
    # Он менее чувствителен к шуму, чем чистый Лапласиан. Варьируя `sigma`,
    # мы можем находить "блобы" разного размера.
    img_blur = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return cv2.Laplacian(img_blur, cv2.CV_32F)

def frangi_filter(img: np.ndarray) -> np.ndarray:
    """Applies the Frangi vesselness filter."""
    # DEV: Фильтр Франги специально разработан для выделения трубчатых структур
    # (сосуды, волокна). Он будет очень полезен для некоторых наших датасетов.
    # Skimage требует вход в диапазоне [0, 1].
    img_norm = (img.astype(np.float32) - img.min()) / (img.max() - img.min() + 1e-6)
    return frangi(img_norm).astype(np.float32)

# --- НОВАЯ ДОБАВЛЕННАЯ ФУНКЦИЯ ---
def gabor(
    img: np.ndarray,
    ksize: int = 21,
    theta_deg: float = 0,
    sigma: float = 4.0,
    lambd: float = 10.0,
    gamma: float = 0.5,
    psi: float = 0
) -> np.ndarray:
    """
    Applies a Gabor filter to the image.
    ... (docstring)
    """
    # DEV: Используем np.pi для консистентности с остальным кодом,
    # основанным на NumPy. Это лучшая практика.
    theta_rad = theta_deg * np.pi / 180.0
    kernel = cv2.getGaborKernel(
        (ksize, ksize), sigma, theta_rad, lambd, gamma, psi, ktype=cv2.CV_32F
    )
    return cv2.filter2D(img.astype(np.float32), cv2.CV_32F, kernel)

# --- ОБНОВЛЕННЫЙ ОРКЕСТРАТОР ---
# DEV: Теперь build_feature_stack знает о фильтре Габора.
def build_feature_stack(gray_image: np.ndarray, config: dict) -> np.ndarray:
    """
    Builds a multi-channel stack from a grayscale image based on config.
    """
    if config.get("base_norm", "zscore") == "zscore":
        base_normalized = norm_zscore(gray_image)
    else:
        base_normalized = gray_image.astype(np.float32)

    if gray_image.dtype != np.uint8:
        img_u8 = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    else:
        img_u8 = gray_image

    features = []
    for channel_name in config["channels"]:
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
            sigma = float(channel_name.replace("log_sigma", ""))
            feat = log_filter(img_u8, sigma)
            features.append(norm_zscore(feat))
        elif channel_name == "frangi":
            feat = frangi_filter(gray_image)
            features.append(norm_zscore(feat))
        elif channel_name.startswith("gabor"):
            # DEV: Парсим параметры прямо из имени канала, например "gabor_ksize31_th45"
            # Это позволяет нам не создавать отдельную секцию для каждого фильтра в конфиге.
            params = {}
            parts = channel_name.split('_')
            for part in parts[1:]:
                if part.startswith("ksize"): params["ksize"] = int(part[5:])
                elif part.startswith("th"): params["theta_deg"] = float(part[2:])
                elif part.startswith("sig"): params["sigma"] = float(part[3:])
                elif part.startswith("lam"): params["lambd"] = float(part[3:])
            feat = gabor(img_u8, **params)
            features.append(norm_zscore(feat))
        else:
            raise ValueError(f"Unknown feature channel in config: {channel_name}")
            
    return np.stack(features, axis=0).astype(np.float32)