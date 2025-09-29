# scripts/train.py

"""
Main script for training a segmentation model.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from segmentation_models_pytorch.losses import DiceLoss, SoftBCEWithLogitsLoss
import mlflow
from tqdm import tqdm

from segwork.utils import set_seed, load_config, flatten_config
from segwork.data.dataset import TilesDataset, default_train_aug
from segwork.models.model_builder import build_model
from segwork.metrics.core import dice_score

def load_tile_ids(interim_root: Path, split_files: dict) -> dict:
    """Finds all tile IDs and splits them based on original image splits."""
    # DEV: Это очень важная логика. Мы не можем просто перемешать все тайлы.
    # Тайлы от одного и того же исходного изображения должны оставаться в одном сплите.
    # Этот helper гарантирует, что мы соблюдаем исходное разделение данных.
    canonical_splits_dir = Path(interim_root.parent.name) / "splits" # Предполагаем стандартную структуру
    
    # Загружаем ID исходных изображений для каждого сплита
    source_ids = {}
    for split, filename in split_files.items():
        with open(canonical_splits_dir / filename, 'r') as f:
            source_ids[split] = {line.strip() for line in f}

    # Находим все тайлы и распределяем их по сплитам
    all_tile_paths = (interim_root / "images").glob("*.npy")
    split_tile_ids = {split: [] for split in split_files}

    for tile_path in all_tile_paths:
        original_stem = "_".join(tile_path.stem.split("_")[:-2]) # e.g., "isbi_s001_y0000_x0000" -> "isbi_s001"
        for split, ids_set in source_ids.items():
            if original_stem in ids_set:
                split_tile_ids[split].append(tile_path.stem)
                break
    return split_tile_ids

def main():
    parser = argparse.ArgumentParser(description="Train a segmentation model.")
    parser.add_argument("--config", required=True, help="Path to the experiment config file.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- 1. Data Loading ---
    interim_root = Path(cfg['data']['interim_data_root'])
    # DEV: Мы должны откуда-то знать, как называются файлы со сплитами в каноническом датасете.
    # Пока захардкодим стандартные имена.
    split_files = {"train": "train.txt", "val": "val.txt"}
    split_tile_ids = load_tile_ids(interim_root, split_files)

    ds_train = TilesDataset(interim_root, split_tile_ids['train'], augmentations=default_train_aug())
    ds_val = TilesDataset(interim_root, split_tile_ids['val'], augmentations=None)
    
    dl_train = DataLoader(ds_train, batch_size=cfg['train']['batch_size'], shuffle=True, num_workers=cfg['train']['num_workers'], pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=cfg['eval']['batch_size'], shuffle=False, num_workers=cfg['train']['num_workers'], pin_memory=True)
    print(f"Data loaded: {len(ds_train)} train tiles, {len(ds_val)} validation tiles.")

    # --- 2. Model, Optimizer, Loss ---
    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['train']['optimizer']['lr'], weight_decay=cfg['train']['optimizer']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['train']['epochs'])
    
    dice_loss = DiceLoss(mode='binary', from_logits=True)
    bce_loss = SoftBCEWithLogitsLoss(pos_weight=torch.tensor([cfg['loss']['params']['pos_weight']]).to(device))
    loss_weights = {'dice': cfg['loss']['params']['dice_weight'], 'bce': cfg['loss']['params']['bce_weight']}

    scaler = torch.cuda.amp.GradScaler(enabled=cfg['train']['amp'])

    # --- 3. MLflow Logging ---
    mlflow.set_tracking_uri(f"file://{Path(cfg['logging']['artifact_uri']).resolve()}")
    mlflow.set_experiment(cfg['logging']['experiment_name'])
    run_name = f"{Path(args.config).stem}_seed{cfg['seed']}"
    
    with mlflow.start_run(run_name=run_name) as run:
        print(f"MLflow Run ID: {run.info.run_id}")
        mlflow.log_params(flatten_config(cfg))

        best_val_dice = -1.0
        for epoch in range(cfg['train']['epochs']):
            # --- Training Epoch ---
            model.train()
            train_loss = 0.0
            for images, masks in tqdm(dl_train, desc=f"Epoch {epoch+1}/{cfg['train']['epochs']} [Train]"):
                images, masks = images.to(device), masks.to(device)
                
                with torch.cuda.amp.autocast(enabled=cfg['train']['amp']):
                    logits = model(images)
                    loss = loss_weights['dice'] * dice_loss(logits, masks) + \
                           loss_weights['bce'] * bce_loss(logits, masks)
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
            
            scheduler.step()

            # --- Validation Epoch ---
            model.eval()
            val_dices = []
            with torch.no_grad():
                for images, masks in tqdm(dl_val, desc=f"Epoch {epoch+1}/{cfg['train']['epochs']} [Val]"):
                    images, masks = images.to(device), masks.to(device)
                    logits = model(images)
                    probs = torch.sigmoid(logits)
                    batch_dice = dice_score(probs, masks).cpu().numpy()
                    val_dices.extend(batch_dice)
            
            avg_val_dice = np.mean(val_dices)
            avg_train_loss = train_loss / len(dl_train)
            
            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Dice = {avg_val_dice:.4f}")
            mlflow.log_metrics({"train_loss": avg_train_loss, "val_dice": avg_val_dice, "lr": optimizer.param_groups[0]['lr']}, step=epoch)

            # --- Save Best Model ---
            if avg_val_dice > best_val_dice:
                best_val_dice = avg_val_dice
                mlflow.log_metric("best_val_dice", best_val_dice, step=epoch)
                # Сохраняем чекпоинт локально, MLflow сам его заберет
                checkpoint_path = Path("./best_model.pt")
                torch.save(model.state_dict(), checkpoint_path)
                mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoints")
                print(f"New best model saved with Dice: {best_val_dice:.4f}")

if __name__ == "__main__":
    # DEV: Перед запуском этого скрипта убедитесь, что вы запустили
    # `scripts/preprocess.py` с соответствующим конфигом, чтобы создать
    # `interim_data`. Также убедитесь, что в `prepared_data/.../splits/`
    # лежат файлы `train.txt` и `val.txt`.
    main()