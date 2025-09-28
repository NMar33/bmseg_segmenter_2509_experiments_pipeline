# scripts/preprocess.py

"""
Script to preprocess canonical data into a tiled, feature-rich format for training.
It reads full-size images and masks, cuts them into tiles, builds feature stacks
for each image tile, and saves them in an intermediate directory.
"""
import argparse
from pathlib import Path
import numpy as np
import tifffile as tiff
import cv2
from tqdm import tqdm

from src.utils import load_config
from src.data.filters import build_feature_stack

def main():
    parser = argparse.ArgumentParser(description="Preprocess data into tiles and feature banks.")
    parser.add_argument("--config", required=True, help="Path to the experiment config file.")
    args = parser.parse_args()

    print("--- Starting Preprocessing ---")
    cfg = load_config(args.config)
    data_cfg = cfg['data']

    # Определяем пути
    prepared_root = Path(data_cfg['prepared_data_root'])
    interim_root = Path(data_cfg['interim_data_root'])
    print(f"Reading from: {prepared_root}")
    print(f"Writing to:   {interim_root}")

    # Создаем выходные директории
    (interim_root / "images").mkdir(parents=True, exist_ok=True)
    (interim_root / "masks").mkdir(parents=True, exist_ok=True)

    # Находим все изображения в каноническом датасете
    image_paths = sorted((prepared_root / "images").rglob("*.tif"))
    if not image_paths:
        raise FileNotFoundError(f"No .tif images found in {prepared_root / 'images'}")

    # --- Основной цикл по изображениям ---
    for img_path in tqdm(image_paths, desc="Tiling and featurizing"):
        original_stem = img_path.stem
        
        # Читаем изображение и соответствующую маску
        image = tiff.imread(img_path)
        mask_path = prepared_root / "masks" / img_path.relative_to(prepared_root / "images").with_suffix(".png")
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        H, W = image.shape
        tile_size = data_cfg['tile_size']
        stride = tile_size - data_cfg['overlap']

        # Нарезаем на тайлы
        for y in range(0, H - tile_size + 1, stride):
            for x in range(0, W - tile_size + 1, stride):
                tile_img = image[y:y+tile_size, x:x+tile_size]
                tile_mask = mask[y:y+tile_size, x:x+tile_size]
                
                # Пропускаем тайлы с очень малым количеством информации в маске (опционально)
                if tile_mask.sum() < 10: # DEV: Простое правило, можно сделать настраиваемым
                    continue

                # --- Создание Feature Bank ---
                if data_cfg['feature_bank']['use']:
                    stack = build_feature_stack(tile_img, data_cfg['feature_bank'])
                else:
                    # Если feature bank не используется, создаем 1-канальный z-scored массив
                    stack = norm_zscore(tile_img)[np.newaxis, ...].astype(np.float32)

                if data_cfg['save_float16']:
                    stack = stack.astype(np.float16)

                # Сохраняем тайлы
                tile_id = f"{original_stem}_y{y:04d}_x{x:04d}"
                np.save(interim_root / "images" / f"{tile_id}.npy", stack)
                cv2.imwrite(str(interim_root / "masks" / f"{tile_id}.png"), tile_mask)

    print("\n--- Preprocessing finished successfully! ---")

if __name__ == "__main__":
    main()