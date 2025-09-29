# /segwork/data/dataset.py

"""
PyTorch Dataset classes for reading preprocessed data.
"""
from pathlib import Path
from typing import List, Callable
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A

class TilesDataset(Dataset):
    """PyTorch Dataset for reading tiled images and masks."""

    def __init__(self, interim_root: Path, tile_ids: List[str], augmentations: Callable | None = None):
        """
        Args:
            interim_root: Path to the interim data directory.
            tile_ids: A list of tile IDs to include in this dataset instance.
            augmentations: An albumentations composition.
        """
        self.root = interim_root
        self.ids = tile_ids
        self.aug = augmentations
        
        self.img_path = self.root / "images"
        self.mask_path = self.root / "masks"

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        tile_id = self.ids[index]

        # Загружаем многоканальный .npy файл
        image_stack = np.load(self.img_path / f"{tile_id}.npy")
        
        # Загружаем маску
        mask = cv2.imread(str(self.mask_path / f"{tile_id}.png"), cv2.IMREAD_GRAYSCALE)
        
        # Преобразуем в float32, как ожидает PyTorch
        image_stack = image_stack.astype(np.float32)
        # Маску бинаризуем и добавляем "канальное" измерение
        mask = (mask > 127).astype(np.float32)[np.newaxis, ...]

        # Применяем аугментации
        if self.aug:
            # Albumentations ожидает (H, W, C), а у нас (C, H, W). Нужно транспонировать.
            image_stack_t = np.transpose(image_stack, (1, 2, 0)) # C,H,W -> H,W,C
            mask_t = np.transpose(mask, (1, 2, 0))
            
            augmented = self.aug(image=image_stack_t, mask=mask_t)
            
            image_stack = np.transpose(augmented['image'], (2, 0, 1)) # H,W,C -> C,H,W
            mask = np.transpose(augmented['mask'], (2, 0, 1))

        return torch.from_numpy(image_stack), torch.from_numpy(mask)

def default_train_aug() -> A.Compose:
    """Returns a default set of augmentations for training."""
    # DEV: Этот набор аугментаций довольно стандартный и безопасный.
    # Он включает геометрические преобразования и легкие изменения интенсивности.
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.7),
        A.RandomBrightnessContrast(p=0.2),
    ])