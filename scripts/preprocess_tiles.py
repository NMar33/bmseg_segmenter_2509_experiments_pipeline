# scripts/preprocess_tiles.py

"""
Script for the first stage of preprocessing: Tiling and Feature Bank Creation.

This script reads full-size images from the canonical dataset, cuts them into
overlapping tiles, optionally builds multi-channel feature stacks for each tile,
and saves the results to an intermediate directory (`interim_data`).

It produces two main outputs:
1. A `tiles/` directory containing `images/` (.npy) and `masks/` (.png).
2. A `master_split.csv` file that maps every generated tile back to its
   original source image and source folder (e.g., 'train', 'test').

This is a computationally expensive, one-time operation per data configuration.
"""
import argparse
from pathlib import Path
import numpy as np
import tifffile as tiff
import cv2
from tqdm import tqdm
import pandas as pd

from segwork.utils import load_config
from segwork.data.filters import build_feature_stack, norm_zscore

def main():
    parser = argparse.ArgumentParser(description="Step 1: Preprocess canonical data into tiles and feature banks.")
    parser.add_argument("--config", required=True, help="Path to the experiment config file.")
    args = parser.parse_args()

    print("--- Starting Step 1: Tiling and Featurizing ---")
    cfg = load_config(args.config)
    data_cfg = cfg['data']

    prepared_root = Path(data_cfg['prepared_data_root'])
    interim_root = Path(data_cfg['interim_data_root'])
    print(f"Reading from: {prepared_root}")
    print(f"Writing to:   {interim_root}")

    # Create output directories for tiles
    (interim_root / "tiles" / "images").mkdir(parents=True, exist_ok=True)
    (interim_root / "tiles" / "masks").mkdir(parents=True, exist_ok=True)

    # --- Step 1: Collect all source folders to be processed ---
    source_folders = set()
    split_gen_cfg = data_cfg.get('split_generation', {})
    for split_key in ['train', 'val', 'test']:
        split_val = split_gen_cfg.get('source_splits', {}).get(split_key)
        if isinstance(split_val, list):
            source_folders.update(split_val)
    
    if not source_folders:
        raise ValueError("No source folders specified in `data.split_generation.source_splits`.")
        
    print(f"Found source folders to process: {sorted(list(source_folders))}")

    # --- Step 2: Iterate through folders, tile images, and create master_split records ---
    master_split_records = []
    
    for folder_name in sorted(list(source_folders)):
        image_dir = prepared_root / "images" / folder_name
        mask_dir = prepared_root / "masks" / folder_name
        
        if not image_dir.is_dir():
            print(f"Warning: Source directory not found, skipping: {image_dir}")
            continue

        for img_path in tqdm(sorted(image_dir.glob("*.tif")), desc=f"Tiling folder '{folder_name}'"):
            original_stem = img_path.stem
            
            image = tiff.imread(img_path)
            mask_path = mask_dir / f"{original_stem}.png"
            if not mask_path.exists():
                print(f"Warning: Mask not found for {img_path.name}, skipping.")
                continue
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            H, W = image.shape
            tile_size = data_cfg['tiling']['tile_size']
            stride = tile_size - data_cfg['tiling']['overlap']

            for y in range(0, H - tile_size + 1, stride):
                for x in range(0, W - tile_size + 1, stride):
                    tile_img = image[y:y+tile_size, x:x+tile_size]
                    tile_mask = mask[y:y+tile_size, x:x+tile_size]
                    
                    if tile_mask.sum() < 10: # Optional: skip almost empty tiles
                        continue

                    # Create Feature Bank if enabled
                    if data_cfg.get('feature_bank', {}).get('use', False):
                        stack = build_feature_stack(tile_img, data_cfg['feature_bank'])
                    else:
                        stack = norm_zscore(tile_img)[np.newaxis, ...].astype(np.float32)

                    if data_cfg.get('save_float16', True):
                        stack = stack.astype(np.float16)

                    # Save tiles
                    tile_id = f"{original_stem}_y{y:04d}_x{x:04d}"
                    np.save(interim_root / "tiles" / "images" / f"{tile_id}.npy", stack)
                    cv2.imwrite(str(interim_root / "tiles" / "masks" / f"{tile_id}.png"), tile_mask)
                    
                    master_split_records.append({
                        "tile_id": tile_id,
                        "source_image_id": original_stem,
                        "source_split": folder_name
                    })

    # --- Step 3: Save the master_split.csv file ---
    if not master_split_records:
        raise RuntimeError("Preprocessing finished, but no tiles were generated. Check data and config.")

    master_df = pd.DataFrame(master_split_records)
    master_df.to_csv(interim_root / "master_split.csv", index=False)
    
    print(f"\n--- Tiling and featurizing finished successfully! ---")
    print(f"Generated {len(master_df)} tiles.")
    print(f"Master split map saved to: {interim_root / 'master_split.csv'}")


if __name__ == "__main__":
    main()