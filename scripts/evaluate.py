# scripts/evaluate.py

"""
Main script for evaluating a trained model on full-size images.

This script operates on a specific seed's test split. It works by:
1. Identifying the unique source images that correspond to the test tiles.
2. For each source image, it performs on-the-fly tiling, featurizing, and inference.
3. Stitching the tile predictions back into a full-resolution mask.
4. Calculating a suite of metrics by comparing the stitched mask against the
   full-resolution ground truth.
5. Saving the aggregated results to a JSON report file.
"""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import tifffile as tiff
import cv2
from tqdm import tqdm

from segwork.utils import set_seed, load_config, flatten_config
from segwork.models.model_builder import build_model
from segwork.data.filters import build_feature_stack, norm_zscore
from segwork.data.stitching import stitch_tiles
from segwork.metrics.core import dice_score, iou_score, boundary_f1_score
from segwork.metrics.advanced import v_rand_score, warping_error_score

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained segmentation model.")
    parser.add_argument("--config", required=True, help="Path to the experiment config file.")
    parser.add_argument("--seed", required=True, type=int, help="The random seed for the split to evaluate.")
    parser.add_argument("--checkpoint", required=True, help="Path to the trained model checkpoint (.pt file).")
    # DEV: Аргументы для переопределения данных. Критически важны для Эксперимента B (Cross-Domain).
    parser.add_argument("--override_prepared_data_root", default=None, help="Override prepared_data_root for cross-domain evaluation.")
    parser.add_argument("--override_interim_data_root", default=None, help="Override interim_data_root for cross-domain evaluation.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    # Сид здесь используется для поиска правильного файла сплита
    seed = args.seed
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Evaluation for Seed: {seed} ---")
    print(f"Using device: {device}")

    # --- 1. Load Model ---
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print(f"Model loaded successfully from checkpoint: {args.checkpoint}")

    # --- 2. Resolve Test Set Source Images ---
    # DEV: Пути могут быть переопределены из командной строки.
    prepared_root = Path(args.override_prepared_data_root or cfg['data']['prepared_data_root'])
    interim_root = Path(args.override_interim_data_root or cfg['data']['interim_data_root'])
    
    print(f"Evaluating on data from: {prepared_root}")

    # Читаем нужный файл сплита
    splits_dir = interim_root / "splits" / f"seed_{seed}"
    test_split_path = splits_dir / "test.txt"
    if not test_split_path.exists():
        raise FileNotFoundError(f"Test split file not found for seed {seed} at {test_split_path}")
    
    with open(test_split_path, 'r') as f:
        test_tile_ids = {line.strip() for line in f if line.strip()}

    # Находим уникальные исходные изображения, которые нужно оценить
    master_df = pd.read_csv(interim_root / "master_split.csv")
    test_df = master_df[master_df['tile_id'].isin(test_tile_ids)]
    images_to_evaluate = test_df.groupby('source_image_id')
    
    print(f"Found {len(images_to_evaluate)} full images to evaluate in 'test' split for seed {seed}.")

    # --- 3. Main Evaluation Loop ---
    metrics_to_calc = cfg['eval']['metrics']
    metric_results = {metric: [] for metric in metrics_to_calc}
    
    for source_id, group in tqdm(images_to_evaluate, desc="Evaluating full images"):
        source_split_folder = group['source_split'].iloc[0]
        
        full_image = tiff.imread(prepared_root / "images" / source_split_folder / f"{source_id}.tif")
        full_mask_gt = cv2.imread(str(prepared_root / "masks" / source_split_folder / f"{source_id}.png"), cv2.IMREAD_GRAYSCALE)
        full_mask_gt = (full_mask_gt > 127).astype(np.float32)

        H, W = full_image.shape
        tile_size = cfg['data']['tiling']['tile_size']
        stride = tile_size - cfg['data']['tiling']['overlap']
        
        tiles_raw, coords = [], []
        for y in range(0, W - tile_size + 1, stride):
            for x in range(0, H - tile_size + 1, stride):
                tiles_raw.append(full_image[x:x+tile_size, y:y+tile_size])
                coords.append((x, y))

        predicted_tiles_probs = []
        with torch.no_grad():
            for i in range(0, len(tiles_raw), cfg['eval']['batch_size']):
                batch_tiles_raw = tiles_raw[i : i + cfg['eval']['batch_size']]
                
                batch_stacks = []
                for tile_img_raw in batch_tiles_raw:
                    if cfg['data'].get('feature_bank', {}).get('use', False):
                        stack = build_feature_stack(tile_img_raw, cfg['data']['feature_bank'])
                    else:
                        stack = norm_zscore(tile_img_raw)[np.newaxis, ...].astype(np.float32)
                    batch_stacks.append(stack)

                batch_tensor = torch.from_numpy(np.array(batch_stacks)).to(device)
                
                with torch.cuda.amp.autocast(enabled=cfg['train']['amp']):
                    logits = model(batch_tensor)
                    probs = torch.sigmoid(logits)
                
                predicted_tiles_probs.extend(probs.cpu().numpy())

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
        # ... (placeholders for advanced metrics)

    # --- 4. Reporting ---
    final_results = {
        "config_file": args.config,
        "seed": seed,
        "checkpoint": args.checkpoint,
        "evaluated_on": str(prepared_root.name),
    }
    
    print("\n--- Evaluation Results ---")
    for metric, values in metric_results.items():
        if values:
            final_results[f"{metric}_mean"] = np.mean(values)
            final_results[f"{metric}_std"] = np.std(values)
            print(f"{metric.upper():<10}: {final_results[f'{metric}_mean']:.4f} ± {final_results[f'{metric}_std']:.4f}")

    report_dir = Path("./reports")
    report_dir.mkdir(exist_ok=True, parents=True)
    exp_name = Path(args.config).stem
    report_path = report_dir / f"{exp_name}_seed{seed}_on_{prepared_root.name}.json"
    
    with open(report_path, 'w') as f:
        json.dump(final_results, f, indent=4)
    print(f"\nResults saved to: {report_path}")

if __name__ == "__main__":
    main()