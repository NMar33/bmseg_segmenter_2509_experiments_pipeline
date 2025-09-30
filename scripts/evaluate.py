# scripts/evaluate.py

"""
Main script for evaluating a trained model on full-size images.

This script connects to an existing MLflow run (specified by a run ID)
to log all evaluation results. It operates by:
1. Identifying the unique source images for the test split of a given seed.
2. For each source image, it performs on-the-fly tiling, featurizing, and inference.
3. Stitching the tile predictions back into a full-resolution mask.
4. Calculating a suite of metrics for each full image.
5. Logging the aggregated metrics (mean, std) and visualization artifacts
   back to the original MLflow run.
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
import random
import shutil
import mlflow

from segwork.utils import load_config
from segwork.models.model_builder import build_model
from segwork.data.filters import build_feature_stack, norm_zscore
from segwork.data.stitching import stitch_tiles
from segwork.metrics.core import dice_score, iou_score, boundary_f1_score
from segwork.metrics.advanced import v_rand_score, warping_error_score
from segwork.visualization.explorers import generate_evaluation_visuals

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained segmentation model and log to MLflow.")
    parser.add_argument("--config", required=True, help="Path to the experiment config file.")
    parser.add_argument("--seed", required=True, type=int, help="The random seed for the split to evaluate.")
    parser.add_argument("--checkpoint", required=True, help="Path to the trained model checkpoint (.pt file).")
    parser.add_argument("--mlflow_run_id", required=True, help="The MLflow Run ID to log results to.")
    parser.add_argument("--override_prepared_data_root", default=None)
    parser.add_argument("--override_interim_data_root", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed
    
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)
    print(f"--- Starting Evaluation for Seed: {seed} ---")
    print(f"Using device: {device_type}")

    # --- 1. Load Model & Connect to MLflow Run ---
    # DEV: Мы "подключаемся" к существующему MLflow run, созданному `train.py`,
    # чтобы добавить в него метрики и артефакты оценки.
    with mlflow.start_run(run_id=args.mlflow_run_id):
        print(f"Connected to existing MLflow Run ID: {args.mlflow_run_id}")
        
        model = build_model(cfg).to(device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        model.eval()
        print(f"Model loaded successfully from checkpoint: {args.checkpoint}")

        viz_cfg = cfg['eval'].get('visualization', {})
        
        # --- 2. Resolve Test Set Source Images ---
        prepared_root = Path(args.override_prepared_data_root or cfg['data']['prepared_data_root'])
        interim_root = Path(args.override_interim_data_root or cfg['data']['interim_data_root'])
        print(f"Evaluating on data from: {prepared_root}")

        splits_dir = interim_root / "splits" / f"seed_{seed}"
        test_images_path = splits_dir / "test_images.txt"
        if not test_images_path.exists():
            raise FileNotFoundError(f"Test split file (`test_images.txt`) not found for seed {seed} at {splits_dir}")
        
        with open(test_images_path, 'r') as f:
            source_image_ids = [line.strip() for line in f if line.strip()]
            
        if not source_image_ids:
            print("Warning: Test set is empty. No evaluation will be performed.")
            return

        master_df = pd.read_csv(interim_root / "master_split.csv")
        print(f"Found {len(source_image_ids)} full images to evaluate in 'test' split.")

        # --- 3. First Pass: Inference and Metric Calculation ---
        all_image_results = []
        temp_preds_dir = interim_root / "tmp_preds_eval" / f"seed_{seed}"
        if temp_preds_dir.exists(): shutil.rmtree(temp_preds_dir)
        temp_preds_dir.mkdir(parents=True)

        for source_id in tqdm(source_image_ids, desc="Pass 1/2: Running Inference"):
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
            for y in range(0, H - tile_size + 1, stride):
                for x in range(0, W - tile_size + 1, stride):
                    tiles_raw.append(full_image[y:y+tile_size, x:x+tile_size])
                    coords.append((y, x))

            if not tiles_raw:
                print(f"Warning: Image '{source_id}' is smaller than tile size. Skipping.")
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
            
            # Save prediction to temporary directory
            tiff.imwrite(temp_preds_dir / f"{source_id}_pred.tif", stitched_prob_mask)

            # Calculate individual metrics
            gt_tensor = torch.from_numpy(full_mask_gt).unsqueeze(0).unsqueeze(0)
            prob_tensor = torch.from_numpy(stitched_prob_mask).unsqueeze(0).unsqueeze(0)
            pred_bin_tensor = (prob_tensor > 0.5)
            
            image_metrics = {'image_id': source_id}
            if "dice" in cfg['eval']['metrics']:
                image_metrics['dice'] = dice_score(prob_tensor, gt_tensor).item()
            if "iou" in cfg['eval']['metrics']:
                image_metrics['iou'] = iou_score(prob_tensor, gt_tensor).item()
            if "bf1" in cfg['eval']['metrics']:
                image_metrics['bf1'] = np.mean(boundary_f1_score(pred_bin_tensor, gt_tensor.bool()))
            
            all_image_results.append(image_metrics)

        # --- 4. Second Pass: Analysis, Visualization, and Logging ---
        df_results = pd.DataFrame(all_image_results)
        
        # --- 4.1 Log Aggregated Metrics to MLflow ---
        print("\n--- Aggregated Evaluation Results ---")
        metrics_to_log = {}
        # Собираем метрики из DataFrame, чтобы избежать повторных вычислений
        for metric_col in df_results.columns:
            if metric_col == 'image_id': continue
            values = df_results[metric_col]
            if not values.empty:
                mean_val, std_val = values.mean(), values.std()
                # Логируем в MLflow с префиксом 'test/', как в train.py
                metrics_to_log[f"test/{metric_col}_mean"] = mean_val
                metrics_to_log[f"test/{metric_col}_std"] = std_val
                print(f"test_{metric_col.upper():<10}: {mean_val:.4f} ± {std_val:.4f}")
        
        # Записываем все метрики в MLflow одним вызовом
        if metrics_to_log:
            mlflow.log_metrics(metrics_to_log)
        
        # --- 4.2 Generate and Log Visualizations to MLflow ---
        if viz_cfg.get('enabled', False) and not df_results.empty:
            primary_metric = viz_cfg.get('primary_metric', 'dice')
            if primary_metric not in df_results.columns:
                print(f"Warning: Primary metric '{primary_metric}' for visualization not found in results. Skipping visualization.")
            else:
                df_results = df_results.sort_values(by=primary_metric, ascending=True)
                
                # --- Логика выбора ID для визуализации ---
                worst_ids = df_results.head(viz_cfg.get('num_worst', 0))
                best_ids = df_results.tail(viz_cfg.get('num_best', 0))
                
                remaining_df = df_results.drop(index=worst_ids.index).drop(index=best_ids.index)
                num_random = min(viz_cfg.get('num_random', 0), len(remaining_df))
                random_ids = remaining_df.sample(n=num_random, random_state=seed) if num_random > 0 else pd.DataFrame()

                # Собираем все в один DataFrame для итерации
                samples_to_visualize = pd.concat([
                    worst_ids.assign(type='Worst'),
                    best_ids.assign(type='Best'),
                    random_ids.assign(type='Random')
                ])

                print(f"\nSelected {len(samples_to_visualize)} images for visualization (best, worst, random)...")

                temp_viz_dir = Path("./temp_visuals_for_mlflow")
                if temp_viz_dir.exists(): shutil.rmtree(temp_viz_dir)
                temp_viz_dir.mkdir()
                
                for _, row in tqdm(samples_to_visualize.iterrows(), total=len(samples_to_visualize), desc="Pass 2/2: Generating Visuals"):
                    source_id = row['image_id']
                    
                    source_split_folder = master_df[master_df['source_image_id'] == source_id]['source_split'].iloc[0]
                    full_image = tiff.imread(prepared_root / "images" / source_split_folder / f"{source_id}.tif")
                    full_mask_gt = cv2.imread(str(prepared_root / "masks" / source_split_folder / f"{source_id}.png"), cv2.IMREAD_GRAYSCALE)
                    pred_prob_mask = tiff.imread(temp_preds_dir / f"{source_id}_pred.tif")
                    
                    # Собираем все метрики для этого изображения в один словарь
                    image_metrics = row.to_dict()

                    generate_evaluation_visuals(
                        image=full_image,
                        gt_mask=full_mask_gt,
                        pred_mask=(pred_prob_mask > 0.5),
                        metrics=image_metrics, # Передаем весь словарь
                        feature_bank_config=cfg['data'].get('feature_bank', {}),
                        viz_config=viz_cfg,
                        output_dir=temp_viz_dir,
                        image_id=source_id,
                        sample_type=row['type'] # Передаем тип сэмпла
                    )
                
                print(f"Logging {len(list(temp_viz_dir.glob('*.png')))} visualization artifacts to MLflow...")
                mlflow.log_artifacts(str(temp_viz_dir), artifact_path="evaluation_visuals")
                shutil.rmtree(temp_viz_dir)

        # --- 4.3 Log Per-Image Results CSV to MLflow ---
        # DEV: Сохраняем детальные результаты по каждому изображению как артефакт.
        # Это очень полезно для последующего глубокого анализа.
        report_path = Path("./temp_per_image_results.csv")
        df_results.to_csv(report_path, index=False)
        mlflow.log_artifact(str(report_path), artifact_path="reports")
        report_path.unlink()

        # Cleanup temporary predictions directory
        if temp_preds_dir.exists():
            shutil.rmtree(temp_preds_dir)

    print("\n✅ Evaluation finished. All results logged to MLflow.")