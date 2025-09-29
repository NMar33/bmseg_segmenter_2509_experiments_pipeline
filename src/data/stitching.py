# src/data/stitching.py

"""
Utility for stitching overlapping tile predictions back into a full image mask.
"""
import numpy as np
from typing import List, Tuple

def stitch_tiles(
    tiles: List[np.ndarray],
    coords: List[Tuple[int, int]],
    full_image_shape: Tuple[int, int],
    tile_size: int,
) -> np.ndarray:
    """
    Stitches overlapping tiles back into a single image by averaging overlaps.

    Args:
        tiles: A list of predicted tiles (as NumPy arrays, e.g., probability maps).
        coords: A list of (y, x) coordinates for the top-left corner of each tile.
        full_image_shape: The (height, width) of the final stitched image.
        tile_size: The size (height and width) of each square tile.

    Returns:
        A NumPy array representing the fully stitched image mask.
    """
    # DEV: Это классический и очень надежный способ склейки.
    # Мы создаем два "холста": один для суммы предсказаний, другой для
    # подсчета количества перекрытий в каждом пикселе. Финальный результат —
    # это просто их поэлементное деление.
    
    h, w = full_image_shape
    prediction_sum = np.zeros((h, w), dtype=np.float32)
    overlap_count = np.zeros((h, w), dtype=np.uint8)

    for tile, (y, x) in zip(tiles, coords):
        # Убираем "канальное" измерение, если оно есть
        if tile.ndim == 3:
            tile = tile.squeeze(0)
        
        # Добавляем предсказание тайла на "холст"
        prediction_sum[y : y + tile_size, x : x + tile_size] += tile
        # Увеличиваем счетчик перекрытий
        overlap_count[y : y + tile_size, x : x + tile_size] += 1

    # Избегаем деления на ноль в областях, куда не попал ни один тайл
    overlap_count[overlap_count == 0] = 1
    
    # Усредняем предсказания в областях перекрытия
    stitched_mask = prediction_sum / overlap_count
    
    return stitched_mask