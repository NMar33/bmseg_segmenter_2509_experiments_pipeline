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

from segwork.utils import load_config, load_split_ids
from segwork.models.model_builder import build_model
from segwork.data.filters import build_feature_stack
from segwork.data.stitching import stitch_tiles
from segwork.metrics.core import dice_score, iou_score, boundary_f1_score
from segwork.metrics.advanced import v_rand_score, warping_error_score

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
    print(f"Model loaded successfully from checkpoint: {args.checkpoint}")

    # --- 2. Load Data IDs ---
    prepared_root = Path(cfg['data']['prepared_data_root'])
    split_dir = prepared_root / "splits"
    # DEV: Имя файла со сплитом берется из конфига, чтобы обеспечить гибкость.
    split_filename = cfg['eval'][f'{args.split}_split_file']
    image_ids = load_split_ids(split_dir, split_filename)
    print(f"Found {len(image_ids)} images to evaluate in '{args.split}' split.")

    # --- 3. Main Evaluation Loop ---
    metrics_to_calc = cfg['eval']['metrics']
    metric_results = {metric: [] for metric in metrics_to_calc}
    
    for image_id in tqdm(image_ids, desc=f"Evaluating on {args.split} set"):
        # Load full-size image and mask
        image_path = prepared_root / "images" / args.split / f"{image_id}.tif"
        mask_path = prepared_root / "masks" / args.split / f"{image_id}.png"
        
        full_image = tiff.imread(image_path)
        full_mask_gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        # Binarize ground truth to {0.0, 1.0} for calculations
        full_mask_gt = (full_mask_gt > 127).astype(np.float32)

        # --- On-the-fly Tiling and Featurizing ---
        H, W = full_image.shape
        tile_size = cfg['data']['tile_size']
        stride = tile_size - cfg['data']['overlap']
        
        tiles, coords = [], []
        # DEV: Убедимся, что даже если изображение меньше тайла, мы его обработаем.
        # Это крайний случай, но он важен для робастности.
        if H < tile_size or W < tile_size:
            # Simple resize for tiny images
            tile_img = cv2.resize(full_image, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
            tiles.append(tile_img)
            coords.append((0,0)) # Dummy coords
        else:
            for y in range(0, H - tile_size + 1, stride):
                for x in range(0, W - tile_size + 1, stride):
                    tile_img = full_image[y:y+tile_size, x:x+tile_size]
                    tiles.append(tile_img)
                    coords.append((y, x))
        
        # --- Batch Inference ---
        predicted_tiles_probs = []
        with torch.no_grad():
            for i in range(0, len(tiles), cfg['eval']['batch_size']):
                batch_tiles_raw = tiles[i : i + cfg['eval']['batch_size']]
                
                # Apply featurization to the batch
                batch_stacks = []
                for tile_img_raw in batch_tiles_raw:
                    if cfg['data']['feature_bank']['use']:
                        stack = build_feature_stack(tile_img_raw, cfg['data']['feature_bank'])
                    else:
                        stack = norm_zscore(tile_img_raw)[np.newaxis, ...].astype(np.float32)
                    batch_stacks.append(stack)

                batch_tensor = torch.from_numpy(np.array(batch_stacks)).to(device)
                
                with torch.cuda.amp.autocast(enabled=cfg['train']['amp']):
                    logits = model(batch_tensor)
                    probs = torch.sigmoid(logits)
                
                predicted_tiles_probs.extend(probs.cpu().numpy())

        # --- Stitching ---
        if H < tile_size or W < tile_size:
            # Simple resize back for tiny images
            stitched_prob_mask = cv2.resize(predicted_tiles_probs[0].squeeze(), (W, H), interpolation=cv2.INTER_AREA)
        else:
            stitched_prob_mask = stitch_tiles(predicted_tiles_probs, coords, (H, W), tile_size)

        # --- Metric Calculation ---
        gt_tensor = torch.from_numpy(full_mask_gt).unsqueeze(0).unsqueeze(0)
        prob_tensor = torch.from_numpy(stitched_prob_mask).unsqueeze(0).unsqueeze(0)
        pred_bin_tensor = (prob_tensor > 0.5)

        if "dice" in metric_results:
            metric_results["dice"].append(dice_score(prob_tensor, gt_tensor).item())
        if "iou" in metric_results:
            metric_results["iou"].append(iou_score(prob_tensor, gt_tensor).item())
        if "bf1" in metric_results:
            bf1 = boundary_f1_score(pred_bin_tensor, gt_tensor.bool())
            metric_results["bf1"].extend(bf1)
        if "v_rand" in metric_results:
            score = v_rand_score(pred_bin_tensor.numpy(), gt_tensor.numpy())
            metric_results["v_rand"].append(score)
        if "warping" in metric_results:
            score = warping_error_score(pred_bin_tensor.numpy(), gt_tensor.numpy())
            metric_results["warping"].append(score)

    # --- 4. Reporting ---
    final_results = {
        "config_file": args.config,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "num_images": len(image_ids),
    }
    
    print("\n--- Evaluation Results ---")
    for metric, values in metric_results.items():
        if values:
            final_results[f"{metric}_mean"] = np.mean(values)
            final_results[f"{metric}_std"] = np.std(values)
            print(f"{metric.upper():<10}: {final_results[f'{metric}_mean']:.4f} ± {final_results[f'{metric}_std']:.4f}")

    # Save results to a JSON file for later aggregation
    report_dir = Path("./reports")
    report_dir.mkdir(exist_ok=True, parents=True)
    exp_name = Path(args.config).stem
    seed = cfg['seed']
    report_path = report_dir / f"{exp_name}_seed{seed}_{args.split}_results.json"
    
    with open(report_path, 'w') as f:
        json.dump(final_results, f, indent=4)
    print(f"\nResults saved to: {report_path}")


if __name__ == "__main__":
    main()