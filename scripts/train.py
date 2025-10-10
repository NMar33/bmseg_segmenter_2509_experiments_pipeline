"""
Main script for training a segmentation model.

This script is a "dumb executor". It expects that all data preprocessing
(tiling, feature generation) and split generation has already been completed.

It takes a config and a seed as input, finds the corresponding `_tiles.txt`
split files, and runs the training loop, logging results to MLflow.
"""
import argparse
from pathlib import Path
import random
from typing import List, Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from segmentation_models_pytorch.losses import DiceLoss, SoftBCEWithLogitsLoss
import mlflow
from tqdm import tqdm

import torch.nn as nn # <-- Добавить этот импорт

from segwork.utils import set_seed, load_config, flatten_config
# --- ИЗМЕНЕНИЕ: Импортируем `build_augmentations` вместо `default_train_aug` ---
from segwork.data.dataset import TilesDataset, build_augmentations
from segwork.models.model_builder import build_model

from segwork.metrics.core import dice_score
# Импортируем обе наши новые функции логирования
from segwork.mlflow_loggers.training_loggers import log_validation_visuals, log_adapter_weights

def load_tile_ids_from_file(split_path: Path) -> list[str]:
    """Reads a list of tile IDs from a .txt file."""
    if not split_path.exists():
        print(f"Warning: Split file not found at {split_path}. Returning empty list.")
        return []
    with open(split_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def set_encoder_grad(model: nn.Module, requires_grad: bool):
    """Sets the `requires_grad` flag for the encoder of an SMP model."""
    encoder = None
    if isinstance(model, nn.Sequential) and hasattr(model[1], 'encoder'): # Модель с адаптером
        encoder = model[1].encoder
    elif hasattr(model, 'encoder'): # Стандартная SMP модель
        encoder = model.encoder
    
    if encoder:
        for param in encoder.parameters():
            param.requires_grad = requires_grad
        print(f"INFO: Encoder gradients have been {'ENABLED' if requires_grad else 'DISABLED'}.")

def main():
    parser = argparse.ArgumentParser(description="Train a segmentation model for a specific seed.")
    parser.add_argument("--config", required=True, help="Path to the experiment config file.")
    parser.add_argument("--seed", required=True, type=int, help="The random seed for this specific run.")
    parser.add_argument("--init_checkpoint", default=None, help="Optional path to a checkpoint for fine-tuning.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)
    
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)
    print(f"--- Starting Training for Seed: {args.seed} ---")
    print(f"Using device: {device_type}")
    
    # --- 1. Data Loading ---
    interim_root = Path(cfg['data']['interim_data_root'])
    splits_dir = interim_root / "splits" / f"seed_{args.seed}"
    tiles_root = interim_root / "tiles"

    train_tile_ids = load_tile_ids_from_file(splits_dir / "train_tiles.txt")
    val_tile_ids = load_tile_ids_from_file(splits_dir / "val_tiles.txt")

    if not train_tile_ids:
        raise ValueError(f"Training set is empty. Check if `train_tiles.txt` was generated correctly in {splits_dir}")

    # --- ИЗМЕНЕНИЕ: Динамически строим аугментации из конфига ---
    train_augs = build_augmentations(cfg['data'].get('augmentations', {}))

    ds_train = TilesDataset(
            tiles_root, 
            train_tile_ids, 
            augmentations=train_augs,
            mask_processing_cfg=cfg['data'].get('mask_processing')
        )
    ds_val = TilesDataset(
        tiles_root, 
        val_tile_ids, 
        augmentations=None, 
        mask_processing_cfg=cfg['data'].get('mask_processing')
    ) if val_tile_ids else None

    dl_train = DataLoader(ds_train, batch_size=cfg['train']['batch_size'], shuffle=True, num_workers=cfg['train']['num_workers'], pin_memory=(device_type == 'cuda'))
    dl_val = DataLoader(ds_val, batch_size=cfg['eval']['batch_size'], shuffle=False, num_workers=cfg['train']['num_workers'], pin_memory=(device_type == 'cuda')) if ds_val else None
    print(f"Data loaded: {len(ds_train)} train tiles, {len(ds_val) if ds_val else 0} validation tiles.")

    # --- 1.1 Select fixed samples for visualization (paths only) ---
    fixed_val_sample_paths = []
    viz_cfg = cfg['train'].get('validation_visualization', {})
    if viz_cfg.get('enabled', False) and val_tile_ids:
        num_samples = viz_cfg.get('num_samples', 4)
        num_to_sample = min(num_samples, len(val_tile_ids))
        
        # Use a separate random generator to not affect other random processes
        rng = random.Random(args.seed)
        sample_ids = rng.sample(val_tile_ids, num_to_sample)
        
        for tile_id in sample_ids:
            fixed_val_sample_paths.append({
                "image_path": tiles_root / "images" / f"{tile_id}.npy",
                "mask_path": tiles_root / "masks" / f"{tile_id}.png"
            })
        print(f"Selected {len(fixed_val_sample_paths)} fixed samples for validation visualization.")

    # --- 2.1 Model Initialization and Encoder Freezing ---
    model = build_model(cfg).to(device)
    
    if args.init_checkpoint:
        print(f"Initializing model weights from checkpoint: {args.init_checkpoint}")
        model.load_state_dict(torch.load(args.init_checkpoint, map_location=device), strict=False)

    freeze_epochs = cfg['model'].get('adapter', {}).get('freeze_encoder_epochs', 0)
    if freeze_epochs > 0:
        set_encoder_grad(model, requires_grad=False)

    # --- 2.2 Optimizer and Scheduler ---
    # DEV: Важно передавать в оптимизатор только те параметры, которые сейчас обучаемы.
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params, 
        lr=float(cfg['train']['optimizer']['lr']), 
        weight_decay=float(cfg['train']['optimizer']['weight_decay'])
    )

    # DEV: Планировщик создается один раз, но мы будем его обновлять, если оптимизатор изменится.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['train']['epochs'])


    # # --- 2. Model, Optimizer, Loss ---
    # model = build_model(cfg).to(device)
    
    # if args.init_checkpoint:
    #     print(f"Initializing model weights from checkpoint: {args.init_checkpoint}")
    #     model.load_state_dict(torch.load(args.init_checkpoint, map_location=device), strict=False)

    # # --- ИЗМЕНЕНИЕ: Добавляем логику заморозки энкодера ---
    # freeze_epochs = cfg['model'].get('adapter', {}).get('freeze_encoder_epochs', 0)
    # if freeze_epochs > 0:
    #     set_encoder_grad(model, requires_grad=False)

    # optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg['train']['optimizer']['lr']), weight_decay=float(cfg['train']['optimizer']['weight_decay']))

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['train']['epochs'])
    dice_loss = DiceLoss(mode='binary', from_logits=True)
    bce_loss = SoftBCEWithLogitsLoss(pos_weight=torch.tensor([cfg['loss']['params']['pos_weight']]).to(device))
    loss_weights = {'dice': cfg['loss']['params']['dice_weight'], 'bce': cfg['loss']['params']['bce_weight']}

    scaler = torch.amp.GradScaler(device_type, enabled=cfg['train']['amp'])
    
    # --- 3. MLflow Logging ---
    mlflow.set_tracking_uri(f"file://{Path(cfg['logging']['artifact_uri']).resolve()}")
    mlflow.set_experiment(cfg['logging']['experiment_name'])
    run_name = f"{Path(args.config).stem}_seed{args.seed}"
    
    with mlflow.start_run(run_name=run_name) as run:
        print(f"MLflow Run ID: {run.info.run_id}")
        mlflow.log_params(flatten_config(cfg))
        mlflow.log_param("cmd_line_seed", args.seed)
        mlflow.log_artifact(args.config, artifact_path="configs")

        best_val_dice = -1.0

        for epoch in range(cfg['train']['epochs']):

            # --- Unfreeze encoder logic ---
            if freeze_epochs > 0 and epoch == freeze_epochs:
                print(f"\n--- Unfreezing encoder at epoch {epoch} ---")
                set_encoder_grad(model, requires_grad=True)
                
                # DEV: Пересоздаем оптимизатор, чтобы он "увидел" новые,
                # размороженные параметры энкодера. Это самая надежная практика.
                print("Re-initializing optimizer to include all trainable parameters.")
                # trainable_params = filter(lambda p: p.requires_grad, model.parameters())
                optimizer = torch.optim.AdamW(
                    model.parameters(), 
                    lr=float(cfg['train']['optimizer']['lr']), 
                    weight_decay=float(cfg['train']['optimizer']['weight_decay'])
                )
                
                # DEV: Адаптируем планировщик к новому оптимизатору и оставшимся эпохам.
                # Мы "перематываем" его состояние на текущую эпоху.
                print("Re-initializing scheduler.")
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, 
                    T_max=cfg['train']['epochs'],
                    last_epoch=epoch - 1 # Устанавливаем текущее состояние
                )

            # # --- ИЗМЕНЕНИЕ: Логика разморозки энкодера в начале эпохи ---
            # if freeze_epochs > 0 and epoch == freeze_epochs:
            #     print(f"\n--- Unfreezing encoder at epoch {epoch} ---")
            #     set_encoder_grad(model, requires_grad=True)
            #     # DEV: Пересоздавать оптимизатор здесь не обязательно для AdamW,
            #     # так как он адаптирует моменты для новых параметров.
            
            model.train()
            train_loss = 0.0

            for images, masks in tqdm(dl_train, desc=f"Epoch {epoch+1}/{cfg['train']['epochs']} [Train]"):
                images, masks = images.to(device), masks.to(device)
                with torch.amp.autocast(device_type=device_type, enabled=cfg['train']['amp']):
                    logits = model(images)
                    loss = loss_weights['dice'] * dice_loss(logits, masks) + loss_weights['bce'] * bce_loss(logits, masks)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()
            
            scheduler.step()
            avg_train_loss = train_loss / len(dl_train)

            avg_val_dice = 0.0
            if dl_val:
                model.eval()
                val_dices = []
                with torch.no_grad():
                    for images, masks in tqdm(dl_val, desc=f"Epoch {epoch+1}/{cfg['train']['epochs']} [Val]"):
                        images, masks = images.to(device), masks.to(device)
                        with torch.amp.autocast(device_type=device_type, enabled=cfg['train']['amp']):
                            logits = model(images)
                        probs = torch.sigmoid(logits)
                        batch_dice = dice_score(probs, masks).cpu().numpy()
                        val_dices.extend(batch_dice)
                avg_val_dice = np.mean(val_dices) if val_dices else 0.0
            
            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Dice = {avg_val_dice:.4f}")
            mlflow.log_metrics({"train_loss": avg_train_loss, "val_dice": avg_val_dice, "lr": optimizer.param_groups[0]['lr']}, step=epoch)

            # --- Log Visuals and Adapter Weights ---
            if fixed_val_sample_paths:
                log_validation_visuals(
                    model=model,
                    val_sample_paths=fixed_val_sample_paths,
                    epoch=epoch,
                    viz_config=viz_cfg,
                    device=device,
                    amp_enabled=cfg['train']['amp']
                )
            
            # Log adapter weights if the model uses one
            if cfg['model'].get('adapter', {}).get('use', False):
                log_adapter_weights(
                    model=model,
                    epoch=epoch,
                    feature_bank_channels=cfg['data']['feature_bank']['channels']
                )

            # --- Save Best Model ---
            if avg_val_dice > best_val_dice:
                best_val_dice = avg_val_dice
                mlflow.log_metric("best_val_dice", best_val_dice, step=epoch)
                checkpoint_path = Path("./best_model.pt")
                torch.save(model.state_dict(), checkpoint_path)
                mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoints")
                print(f"New best model saved with Dice: {best_val_dice:.4f}")

if __name__ == "__main__":
    main()