# /segwork/data/dataset.py

"""
PyTorch Dataset classes for reading preprocessed data and building augmentation pipelines.
"""
from pathlib import Path
from typing import List, Callable, Dict, Any
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
import random

def binarize_mask(mask: np.ndarray, config: Dict[str, Any] | None) -> np.ndarray:
    """
    Binarizes a mask according to the strategy defined in the config.

    Args:
        mask: The input mask as a NumPy array (0-255).
        config: The `data.mask_processing` section of the main configuration.

    Returns:
        A binary mask with values {0.0, 1.0} of type float32.
    """
    cfg = config or {}
    mode = cfg.get("binarize_mode", "nonzero")

    if mode == "nonzero":
        binary_mask = (mask > 127).astype(np.float32)
    elif mode == "zero":
        binary_mask = (mask < 128).astype(np.float32)
    elif mode == "ignore_zero":
        binary_mask = (mask > 0).astype(np.float32)
    else:
        raise ValueError(f"Unknown binarize_mode in config: '{mode}'")
    
    return binary_mask

class TilesDataset(Dataset):
    """PyTorch Dataset for reading tiled images and masks."""

    def __init__(
        self,
        tiles_root: Path,
        tile_ids: List[str],
        is_train: bool, # <-- НОВЫЙ АРГУМЕНТ
        augmentations: Callable | None = None,
        mask_processing_cfg: Dict[str, Any] | None = None,
        feature_bank_cfg: Dict[str, Any] | None = None # <-- НОВЫЙ АРГУМЕНТ
    ):
        """
        Args:
            tiles_root: Path to the directory containing `images` and `masks` folders.
            tile_ids: A list of tile IDs to include in this dataset instance.
            augmentations: An albumentations composition.
            mask_processing_cfg: Configuration for how to binarize masks.
            is_train: Flag to indicate if this is a training dataset (to enable dropout).
            feature_bank_cfg: The `data.feature_bank` section from the config.
        """
        self.root = tiles_root
        self.ids = tile_ids
        self.aug = augmentations
        self.mask_cfg = mask_processing_cfg or {}
        
        # --- ИЗМЕНЕНИЕ: Сохраняем конфиги для dropout ---
        self.is_train = is_train
        self.feature_bank_cfg = feature_bank_cfg or {}
        
        self.img_path = self.root / "images"
        self.mask_path = self.root / "masks"
        
    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        tile_id = self.ids[index]

        image_stack = np.load(self.img_path / f"{tile_id}.npy").astype(np.float32)
        
        mask_raw = cv2.imread(str(self.mask_path / f"{tile_id}.png"), cv2.IMREAD_GRAYSCALE)
        
        # Use the centralized binarization function
        mask = binarize_mask(mask_raw, self.mask_cfg)

        # Add channel dimension
        mask = mask[np.newaxis, ...]

        # Apply augmentations
        if self.aug:
            # Albumentations expects (H, W, C), but we have (C, H, W). Transpose is needed.
            image_stack_t = np.transpose(image_stack, (1, 2, 0)) # CHW -> HWC
            mask_t = np.transpose(mask, (1, 2, 0))
            
            augmented = self.aug(image=image_stack_t, mask=mask_t)
            
            image_stack = np.transpose(augmented['image'], (2, 0, 1)) # HWC -> CHW
            mask = np.transpose(augmented['mask'], (2, 0, 1))
        
        # --- НОВЫЙ БЛОК: Channel Dropout ---
        # DEV: Применяется ПОСЛЕ геометрических аугментаций, только для train сета.
        dropout_cfg = self.feature_bank_cfg.get('channel_dropout', {})
        if self.is_train and dropout_cfg.get('p', 0) > 0 and image_stack.shape[0] > 1:
            
            channel_names = self.feature_bank_cfg.get('channels', [])
            always_keep = dropout_cfg.get('always_keep', [])
            
            for i, channel_name in enumerate(channel_names):
                if channel_name not in always_keep:
                    if random.random() < dropout_cfg['p']:
                        # Обнуляем канал
                        image_stack[i, :, :] = 0.0

        # Use .copy() to avoid a UserWarning from PyTorch about negative strides.
        return torch.from_numpy(image_stack.copy()), torch.from_numpy(mask.copy())
    
    
# def default_train_aug() -> A.Compose:
#     """Returns a default set of augmentations for training."""
#     # DEV: Этот набор аугментаций довольно стандартный и безопасный.
#     # Он включает геометрические преобразования и легкие изменения интенсивности.
#     return A.Compose([
#         A.HorizontalFlip(p=0.5),
#         A.VerticalFlip(p=0.5),
#         A.RandomRotate90(p=0.5),
#         A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.7),
#         A.RandomBrightnessContrast(p=0.2),
#     ])

def build_augmentations(aug_cfg: Dict[str, Any]) -> Callable | None:
    """
    Builds an Albumentations augmentation pipeline from a config dictionary.
    """
    if not aug_cfg:
        return None

    augs = []
    for aug_name, params in aug_cfg.items():
        # DEV: Мы проверяем, что у аугментации есть вероятность `p` и она больше нуля.
        # Это позволяет легко включать/выключать аугментации в конфиге.
        if params and params.get('p', 0) > 0:
            try:
                # Находим класс аугментации в библиотеке `albumentations` по имени
                aug_class = getattr(A, aug_name)
                augs.append(aug_class(**params))
            except AttributeError:
                print(f"Warning: Augmentation '{aug_name}' not found in Albumentations library. Skipping.")
    
    if not augs:
        return None
        
    return A.Compose(augs)