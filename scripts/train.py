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
from segmentation_models_pytorch.losses import DiceLoss, SoftBCEWithLogitsLoss, FocalLoss
import mlflow
from tqdm import tqdm

import torch.nn as nn # <-- Добавить этот импорт

from segwork.utils import set_seed, load_config, flatten_config
# --- ИЗМЕНЕНИЕ: Импортируем `build_augmentations` вместо `default_train_aug` ---
from segwork.data.dataset import TilesDataset, build_augmentations
from segwork.models.model_builder import build_model

from segwork.metrics.core import dice_score, iou_score, boundary_f1_score, pixel_error, main_metrics

from segwork.mlflow_loggers.training_loggers import (
    log_validation_visuals,
    log_adapter_weights,
    log_gradient_norms
)

from segwork.models.adapter import ChannelAdapter

def load_tile_ids_from_file(split_path: Path) -> list[str]:
    """Reads a list of tile IDs from a .txt file."""
    if not split_path.exists():
        print(f"Warning: Split file not found at {split_path}. Returning empty list.")
        return []
    with open(split_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]
    
def get_encoder(model: nn.Module) -> nn.Module | None:
    """
    Safely retrieves the encoder module from a standard SMP model or an
    SMP model wrapped in an nn.Sequential (e.g., with a ChannelAdapter).
    """
    if isinstance(model, nn.Sequential) and hasattr(model[1], 'encoder'):
        # Case M2/M3: Model is nn.Sequential(ChannelAdapter, Unet)
        return model[1].encoder
    elif hasattr(model, 'encoder'):
        # Case M1: Model is a standard Unet
        return model.encoder
    return None

def get_decoder(model: nn.Module) -> nn.Module | None:
    """Safely retrieves the decoder module from an SMP model."""
    if isinstance(model, nn.Sequential) and hasattr(model[1], 'decoder'):
        return model[1].decoder
    elif hasattr(model, 'decoder'):
        return model.decoder
    return None

def get_adapter(model: nn.Module) -> nn.Module | None:
    """Safely retrieves the ChannelAdapter module."""
    if isinstance(model, nn.Sequential) and isinstance(model[0], ChannelAdapter):
        return model[0]
    return None

def set_encoder_grad(model: nn.Module, requires_grad: bool):
    """Sets the `requires_grad` flag for the encoder of an SMP model."""
    encoder = get_encoder(model)
    if encoder:
        for param in encoder.parameters():
            param.requires_grad = requires_grad
        print(f"INFO: Encoder gradients have been {'ENABLED' if requires_grad else 'DISABLED'}.")

def create_optimizer(model: nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """
    Creates an optimizer with parameter groups for differential learning rates.
    This implementation robustly separates model parameters into 'encoder' and
    'decoder_and_adapter' groups.
    """
    optimizer_cfg = config['train']['optimizer']
    optimizer_name = optimizer_cfg.get('name', 'adamw').lower()
    
    encoder_params = []
    decoder_and_adapter_params = []
    
    # --- 1. Find the encoder module ---
    encoder_module = get_encoder(model)
    
    # --- 2. Split parameters based on module membership ---
    if encoder_module:
        # Get the IDs of all parameters belonging to the encoder
        encoder_param_ids = {id(p) for p in encoder_module.parameters()}
        
        # Iterate through all model parameters and assign them to a group
        for param in model.parameters():
            if id(param) in encoder_param_ids:
                encoder_params.append(param)
            else:
                decoder_and_adapter_params.append(param)
        
        print(f"Successfully split parameters into {len(encoder_params)} encoder params "
              f"and {len(decoder_and_adapter_params)} decoder/adapter params.")
    else:
        # Fallback if no encoder is found: treat all parameters as one group.
        print("Warning: Encoder module not found. Treating all parameters as a single group.")
        decoder_and_adapter_params = list(model.parameters())

    # --- 3. Create parameter groups for the optimizer ---
    base_lr = float(optimizer_cfg['lr'])
    encoder_lr = float(optimizer_cfg.get('encoder_lr', base_lr))
    
    param_groups = [
        {'params': decoder_and_adapter_params, 'lr': base_lr},
    ]
    # Only add the encoder group if it has parameters
    if encoder_params:
        param_groups.append({'params': encoder_params, 'lr': encoder_lr})
    
    print(f"Optimizer: Building {optimizer_name.upper()} with {len(param_groups)} param groups.")
    print(f"  - Decoder/Adapter LR: {base_lr}")
    if encoder_params:
        print(f"  - Encoder LR: {encoder_lr}")

    # --- 4. Create the optimizer instance ---
    if optimizer_name == 'adamw':
        return torch.optim.AdamW(
            param_groups,
            weight_decay=float(optimizer_cfg.get('weight_decay', 1e-4))
        )
    else:
        raise NotImplementedError(f"Optimizer '{optimizer_name}' is not implemented in `create_optimizer`.")

def create_scheduler(optimizer: torch.optim.Optimizer, config: Dict[str, Any]) -> torch.optim.lr_scheduler._LRScheduler:
    """Creates a scheduler with an optional linear warmup phase."""
    scheduler_cfg = config['train']['scheduler']
    total_epochs = config['train']['epochs']
    warmup_epochs = scheduler_cfg.get('warmup_epochs', 0)
    
    if warmup_epochs > 0:
        print(f"Using a scheduler with {warmup_epochs} warmup epochs.")
        # Scheduler for the warmup phase
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_epochs
        )
        # Main scheduler for the post-warmup phase
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs - warmup_epochs
        )
        # Chain them together
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs]
        )
    else:
        print("Using a standard CosineAnnealingLR scheduler without warmup.")
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

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
            is_train=True, 
            augmentations=train_augs,
            mask_processing_cfg=cfg['data'].get('mask_processing'),
            feature_bank_cfg=cfg['data'].get('feature_bank')
        )
    ds_val = TilesDataset(
        tiles_root, 
        val_tile_ids,
        is_train=False, 
        augmentations=None, 
        mask_processing_cfg=cfg['data'].get('mask_processing'),
        feature_bank_cfg=cfg['data'].get('feature_bank')
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

    # --- 2. Model, Optimizer, Scheduler ---
    model = build_model(cfg).to(device)
    
    if args.init_checkpoint:
        print(f"Initializing model weights from checkpoint: {args.init_checkpoint}")
        model.load_state_dict(torch.load(args.init_checkpoint, map_location=device), strict=False)

    # --- Создаем оптимизатор и планировщик ---
    # DEV: Сначала вызываем фабрики, и только потом замораживаем.
    # Это гарантирует, что оптимизатор "знает" обо всех параметрах с самого начала.
    optimizer = create_optimizer(model, cfg)
    scheduler = create_scheduler(optimizer, cfg)

    # Замораживаем энкодер, если это требуется по стратегии
    freeze_epochs = cfg['train'].get('fine_tuning', {}).get('freeze_encoder_epochs', 0)
    if freeze_epochs > 0:
        set_encoder_grad(model, requires_grad=False)
    
    dice_loss = DiceLoss(mode='binary', from_logits=True)
    focal_loss = FocalLoss(mode='binary', gamma=2)
    # bce_loss = SoftBCEWithLogitsLoss(pos_weight=torch.tensor([cfg['loss']['params']['pos_weight']]).to(device))
    loss_weights = {'dice': cfg['loss']['params']['dice_weight'], 'focal': cfg['loss']['params']['focal_weight']}

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
        global_step = 0

        for epoch in range(cfg['train']['epochs']):

            # --- Unfreeze encoder logic ---
            if freeze_epochs > 0 and epoch == freeze_epochs:
                print(f"\n--- Unfreezing encoder at epoch {epoch} ---")
                set_encoder_grad(model, requires_grad=True)
                # DEV: Больше ничего делать не нужно! Оптимизатор уже знает об этих
                # параметрах, а планировщик продолжит свою работу без изменений.

            model.train()
            train_loss = 0.0
            train_dice_score_lv = 0.0
            # train_dice_scores = []
            train_metrics = {
                    'acc': [],
                    'rec': [],
                    'prc': [],
                    'dice': [],
                    'iou': [],
                    'perr': [],                   
                }

            for images, masks in tqdm(dl_train, desc=f"Epoch {epoch+1}/{cfg['train']['epochs']} [Train]"):
                images, masks = images.to(device), masks.to(device)
                with torch.amp.autocast(device_type=device_type, enabled=cfg['train']['amp']):
                    logits = model(images)
                    dice_loss_value = dice_loss(logits, masks)
                    loss = loss_weights['dice'] * dice_loss_value + loss_weights['focal'] * focal_loss(logits, masks)
                
                with torch.no_grad():
                    probs = torch.sigmoid(logits)
                    # batch_dice = dice_score(probs, masks) # Вызываем нашу новую обертку
                    # train_dice_scores.extend(batch_dice.cpu().numpy()) # Собираем результаты
                    train_main_metrics = main_metrics(probs, masks)
                    for k, v in train_main_metrics.items():
                        if k in train_metrics:
                            train_metrics[k].extend(v.cpu().numpy())
                optimizer.zero_grad()
                scaler.scale(loss).backward()

                # --- Log gradient norms for each component ---
                modules_to_log = {
                    "adapter": get_adapter(model),
                    "encoder": get_encoder(model),
                    "decoder": get_decoder(model)
                }
                log_gradient_norms(modules=modules_to_log, global_step=global_step)

                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()
                train_dice_score_lv += 1 - dice_loss_value.item()
                global_step += 1
            
            scheduler.step()
            
            avg_train_metrics = {key: np.mean(values) if values else 0.0 for key, values in train_metrics.items()}
            avg_train_loss = train_loss / len(dl_train)
            avg_train_metrics['loss'] = avg_train_loss
            avg_train_metrics['dice_lv'] = train_dice_score_lv / len(dl_train)
            # avg_train_dice_score = np.mean(train_dice_scores) if train_dice_scores else 0.0
            # avg_train_dice_score_lv = train_dice_score_lv / len(dl_train)

            avg_val_metrics = {}
            avg_val_dice = 0.0
            if dl_val:
                model.eval()
                # --- ИЗМЕНЕНИЕ 1: Создаем списки для всех метрик ---
                val_metrics = {
                    'acc': [],
                    'rec': [],
                    'prc': [],
                    'dice': [],
                    'iou': [],
                    'perr': [],
                    'bf1': [],                    
                }
                val_loss = 0.0
                with torch.no_grad():
                    for images, masks in tqdm(dl_val, desc=f"Epoch {epoch+1}/{cfg['train']['epochs']} [Val]"):
                        images, masks = images.to(device), masks.to(device)
                        with torch.amp.autocast(device_type=device_type, enabled=cfg['train']['amp']):
                            logits = model(images)
                        probs = torch.sigmoid(logits)
                        
                        # --- ИЗМЕНЕНИЕ 2: Считаем все метрики ---
                        # Метрики SMP (Dice, IoU)
                        val_main_metrics = main_metrics(probs, masks)
                        for k, v in val_main_metrics.items():
                            if k in val_metrics:
                                val_metrics[k].extend(v.cpu().numpy())

                        # val_metrics['dice'].extend(dice_score(probs, masks).cpu().numpy())
                        # val_metrics['iou'].extend(iou_score(probs, masks).cpu().numpy())
                        
                        # Метрика MONAI (BF1)
                        # Она принимает бинарные тензоры и возвращает одно число на батч
                        pred_bin = (probs > 0.5)
                        bf1 = boundary_f1_score(pred_bin, masks.bool(), boundary_eps=3) # boundary_eps можно вынести в конфиг
                        val_metrics['bf1'].append(bf1)

                        # val_metrics['pixel_error'].extend(pixel_error(pred_bin, masks.bool()).cpu().numpy())
                        
                        # Расчет лосса остается без изменений
                        batch_val_loss = loss_weights['dice'] * dice_loss(logits, masks) + loss_weights['focal'] * focal_loss(logits, masks)
                        val_loss += batch_val_loss.item()
                
                avg_val_loss = val_loss / len(dl_val)
                # --- ИЗМЕНЕНИЕ 3: Агрегируем все метрики ---
                avg_val_metrics = {key: np.mean(values) if values else 0.0 for key, values in val_metrics.items()}
                avg_val_metrics["loss"] = avg_val_loss
                avg_val_dice = avg_val_metrics['dice']

            # avg_val_dice = 0.0
            # if dl_val:
            #     model.eval()
            #     val_dices = []
            #     val_loss = 0.0
            #     with torch.no_grad():
            #         for images, masks in tqdm(dl_val, desc=f"Epoch {epoch+1}/{cfg['train']['epochs']} [Val]"):
            #             images, masks = images.to(device), masks.to(device)
            #             with torch.amp.autocast(device_type=device_type, enabled=cfg['train']['amp']):
            #                 logits = model(images)
            #             probs = torch.sigmoid(logits)
            #             batch_dice = dice_score(probs, masks).cpu().numpy()
            #             val_dices.extend(batch_dice)
            #             batch_val_loss = loss_weights['dice'] * dice_loss(logits, masks) + loss_weights['bce'] * bce_loss(logits, masks)
            #             val_loss += batch_val_loss.item()
            #     avg_val_loss = val_loss / len(dl_val)
            #     avg_val_dice = np.mean(val_dices) if val_dices else 0.0
            
            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Dice = {avg_val_dice:.4f}")
            log = {
                    # "train/loss": avg_train_loss,
                    # "train/dice": avg_train_dice_score,
                    # "train/dice_lv": avg_train_dice_score_lv,
                    **{f"train/{key}": value for key, value in avg_train_metrics.items()},
                    **{f"val/{key}": value for key, value in avg_val_metrics.items()},
                    "lr/decoder_adapter": optimizer.param_groups[0]['lr'] # Основной lr
                }
            if len(optimizer.param_groups) > 1:
                # Если есть вторая группа (энкодер), логируем и ее lr
                log["lr/encoder"] = optimizer.param_groups[1]['lr']

            mlflow.log_metrics(log, step=epoch)
    

            # --- Log Visuals and Adapter Weights ---
            if fixed_val_sample_paths:
                log_validation_visuals(
                    model=model,
                    val_sample_paths=fixed_val_sample_paths,
                    epoch=epoch,
                    mask_processing_cfg=cfg['data'].get('mask_processing'),
                    viz_config=viz_cfg,
                    device=device,
                    amp_enabled=cfg['train']['amp']
                )
            
            # Log adapter weights if the model uses one
            if cfg['model'].get('adapter', {}).get('use', False):
                adapter = get_adapter(model)
                if adapter:
                    # DEV: Передаем сам модуль адаптера, а не всю модель.
                    log_adapter_weights(
                        channel_adapter=adapter,
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