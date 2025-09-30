# scripts/evaluate.py

"""
Main script for evaluating a trained model on full-size images.

This script operates on a specific seed's test split. It works by:
1. Identifying the unique source images that correspond to the test split.
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

from segwork.utils import load_config
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
    parser.add_argument("--override_prepared_data_root", default=None, help="Override prepared_data_root for cross-domain evaluation.")
    parser.add_argument("--override_interim_data_root", default=None, help="Override interim_data_root for cross-domain evaluation.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed
    
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)
    print(f"--- Starting Evaluation for Seed: {seed} ---")
    print(f"Using device: {device_type}")

    # --- 1. Load Model ---
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print(f"Model loaded successfully from checkpoint: {args.checkpoint}")

    # --- 2. Resolve Test Set Source Images ---
    prepared_root = Path(args.override_prepared_data_root or cfg['data']['prepared_data_root'])
    interim_root = Path(args.override_interim_data_root or cfg['data']['interim_data_root'])
    
    print(f"Evaluating on data from: {prepared_root}")

    # --- ИЗМЕНЕНИЕ ЗДЕСЬ: Упрощенная логика загрузки ---
    # `evaluate` теперь читает `_images.txt` файл, который содержит
    # готовый список ID исходных изображений для теста.
    splits_dir = interim_root / "splits" / f"seed_{seed}"
    test_images_path = splits_dir / "test_images.txt"
    if not test_images_path.exists():
        raise FileNotFoundError(f"Test split file not found for seed {seed} at {test_images_path}")
    
    with open(test_images_path, 'r') as f:
        source_image_ids = [line.strip() for line in f if line.strip()]
        
    if not source_image_ids:
        print("Warning: Test set is empty. No evaluation will be performed.")
        return

    # Нам все еще нужен master_split, чтобы узнать, из какой исходной папки ('train'/'test')
    # пришло изображение, чтобы правильно построить путь.
    master_df = pd.read_csv(interim_root / "master_split.csv")
    
    print(f"Found {len(source_image_ids)} full images to evaluate in 'test' split for seed {seed}.")

    # --- 3. Main Evaluation Loop ---
    metrics_to_calc = cfg['eval']['metrics']
    metric_results = {metric: [] for metric in metrics_to_calc}
    
    for source_id in tqdm(source_image_ids, desc="Evaluating full images"):
        # Находим исходную папку для этого изображения
        source_split_folder_series = master_df[master_df['source_image_id'] == source_id]['source_split']
        if source_split_folder_series.empty:
            print(f"Warning: Could not find source split for image '{source_id}' in master_split.csv. Skipping.")
            continue
        source_split_folder = source_split_folder_series.iloc[0]
        
        full_image = tiff.imread(prepared_root / "images" / source_split_folder / f"{source_id}.tif")
        full_mask_gt = cv2.imread(str(prepared_root / "masks" / source_split_folder / f"{source_id}.png"), cv2.IMREAD_GRAYSCALE)
        full_mask_gt = (full_mask_gt > 127).astype(np.float32)

        H, W = full_image.shape
        tile_size = cfg['data']['tiling']['tile_size']
        stride = tile_size - cfg['data']['tiling']['overlap']
        
        tiles_raw, coords = [], []
        # Исправленный цикл тайлинга
        for y in range(0, H - tile_size + 1, stride):
            for x in range(0, W - tile_size + 1, stride):
                tiles_raw.append(full_image[y:y+tile_size, x:x+tile_size])
                coords.append((y, x))

        # Обработка случая, если ни один тайл не был создан (изображение меньше tile_size)
        if not tiles_raw:
            print(f"Warning: Image '{source_id}' is smaller than tile size. Skipping evaluation for this image.")
            continue

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
                
                with torch.amp.autocast(device_type=device_type, enabled=cfg['train']['amp']):
                    logits = model(batch_tensor)
                    probs = torch.sigmoid(logits)
                
                predicted_tiles_probs.extend(probs.cpu().numpy())

        stitched_prob_mask = stitch_tiles(predicted_tiles_probs, coords, (H, W), tile_size)

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
        "checkpoint": Path(args.checkpoint).name, # Сохраняем только имя файла, а не временный путь
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