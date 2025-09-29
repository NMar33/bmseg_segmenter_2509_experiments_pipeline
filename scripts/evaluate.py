# scripts/evaluate.py

"""
Main script for evaluating a trained model on full-size images.
This script performs tiling on-the-fly, runs inference, stitches the results,
and calculates metrics against the full ground truth masks.
"""
import argparse
from pathlib import Path
import json
import numpy as np
import torch
import tifffile as tiff
import cv2
from tqdm import tqdm

from src.utils import load_config, load_split_ids
from src.models.model_builder import build_model
from src.data.filters import build_feature_stack
from src.data.stitching import stitch_tiles
from src.metrics.core import dice_score

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained segmentation model.")
    parser.add_argument("--config", required=True, help="Path to the experiment config file.")
    parser.add_argument("--checkpoint", required=True, help="Path to the trained model checkpoint (.pt file).")
    parser.add_argument("--split", default="test", help="Which split to evaluate on (e.g., 'val', 'test').")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Evaluation on '{args.split}' split ---")
    print(f"Using device: {device}")

    # --- 1. Load Model ---
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print("Model loaded successfully from checkpoint.")

    # --- 2. Load Data IDs ---
    prepared_root = Path(cfg['data']['prepared_data_root'])
    split_dir = prepared_root / "splits"
    test_ids = load_split_ids(split_dir, cfg['eval'][f'{args.split}_split_file'])
    print(f"Found {len(test_ids)} images to evaluate in '{args.split}' split.")

    # --- 3. Main Evaluation Loop ---
    all_dices = []
    for image_id in tqdm(test_ids, desc=f"Evaluating on {args.split} set"):
        # Load full-size image and mask
        full_image = tiff.imread(prepared_root / "images" / args.split / f"{image_id}.tif")
        full_mask_gt = cv2.imread(str(prepared_root / "masks" / args.split / f"{image_id}.png"), cv2.IMREAD_GRAYSCALE)
        full_mask_gt = (full_mask_gt > 127).astype(np.float32)

        # --- On-the-fly Tiling and Featurizing ---
        H, W = full_image.shape
        tile_size = cfg['data']['tile_size']
        stride = tile_size - cfg['data']['overlap']
        
        tiles, coords = [], []
        for y in range(0, H - tile_size + 1, stride):
            for x in range(0, W - tile_size + 1, stride):
                tile_img = full_image[y:y+tile_size, x:x+tile_size]
                if cfg['data']['feature_bank']['use']:
                    stack = build_feature_stack(tile_img, cfg['data']['feature_bank'])
                else:
                    stack = (tile_img.astype(np.float32) - tile_img.mean()) / (tile_img.std() + 1e-6)
                    stack = stack[np.newaxis, ...]
                tiles.append(stack)
                coords.append((y, x))
        
        # --- Batch Inference ---
        predicted_tiles = []
        with torch.no_grad():
            for i in range(0, len(tiles), cfg['eval']['batch_size']):
                batch_tiles = tiles[i : i + cfg['eval']['batch_size']]
                batch_tensor = torch.from_numpy(np.array(batch_tiles)).to(device)
                
                with torch.cuda.amp.autocast(enabled=cfg['train']['amp']):
                    logits = model(batch_tensor)
                    probs = torch.sigmoid(logits)
                
                predicted_tiles.extend(probs.cpu().numpy())

        # --- Stitching ---
        stitched_prob_mask = stitch_tiles(predicted_tiles, coords, (H, W), tile_size)

        # --- Metric Calculation ---
        # Convert to tensors for metric calculation
        stitched_tensor = torch.from_numpy(stitched_prob_mask).unsqueeze(0).unsqueeze(0)
        gt_tensor = torch.from_numpy(full_mask_gt).unsqueeze(0).unsqueeze(0)
        
        dice = dice_score(stitched_tensor, gt_tensor).item()
        all_dices.append(dice)

    # --- 4. Reporting ---
    results = {
        "config_file": args.config,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "num_images": len(test_ids),
        "dice_mean": np.mean(all_dices),
        "dice_std": np.std(all_dices)
    }
    
    print("\n--- Evaluation Results ---")
    for key, value in results.items():
        print(f"{key}: {value}")

    # Save results to a JSON file
    report_dir = Path("./reports")
    report_dir.mkdir(exist_ok=True)
    exp_name = Path(args.config).stem
    seed = cfg['seed']
    report_path = report_dir / f"{exp_name}_seed{seed}_{args.split}_results.json"
    
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to: {report_path}")


if __name__ == "__main__":
    main()