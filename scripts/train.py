# scripts/train.py

"""
Main script for training a segmentation model.

This script is a "dumb executor". It expects that all data preprocessing
(tiling, feature generation) and split generation has already been completed.

It takes a config and a seed as input, finds the corresponding `_tiles.txt`
split files, and runs the training loop.
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

def load_tile_ids_from_file(split_path: Path) -> list[str]:
    """Reads a list of tile IDs from a .txt file."""
    if not split_path.exists():
        print(f"Warning: Split file not found at {split_path}. Returning empty list.")
        return []
    with open(split_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

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
    # DEV: Логика загрузки данных стала предельно простой и явной.
    # Мы просто формируем путь к нужным файлам, используя `seed`.
    interim_root = Path(cfg['data']['interim_data_root'])
    splits_dir = interim_root / "splits" / f"seed_{args.seed}"

    train_tile_ids = load_tile_ids_from_file(splits_dir / "train_tiles.txt")
    val_tile_ids = load_tile_ids_from_file(splits_dir / "val_tiles.txt")

    if not train_tile_ids:
        raise ValueError(f"Training set is empty. Check if `train_tiles.txt` was generated correctly in {splits_dir}")

    ds_train = TilesDataset(interim_root / "tiles", train_tile_ids, augmentations=default_train_aug())
    ds_val = TilesDataset(interim_root / "tiles", val_tile_ids, augmentations=None) if val_tile_ids else None
    
    dl_train = DataLoader(ds_train, batch_size=cfg['train']['batch_size'], shuffle=True, num_workers=cfg['train']['num_workers'], pin_memory=(device_type == 'cuda'))
    dl_val = DataLoader(ds_val, batch_size=cfg['eval']['batch_size'], shuffle=False, num_workers=cfg['train']['num_workers'], pin_memory=(device_type == 'cuda')) if ds_val else None
    print(f"Data loaded: {len(ds_train)} train tiles, {len(ds_val) if ds_val else 0} validation tiles.")

    # --- 2. Model, Optimizer, Loss ---
    model = build_model(cfg).to(device)
    
    if args.init_checkpoint:
        print(f"Initializing model weights from checkpoint: {args.init_checkpoint}")
        model.load_state_dict(torch.load(args.init_checkpoint, map_location=device), strict=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['train']['optimizer']['lr'], weight_decay=cfg['train']['optimizer']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['train']['epochs'])
    
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

            # --- Validation Epoch ---
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
                avg_val_dice = np.mean(val_dices)
            
            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Dice = {avg_val_dice:.4f}")
            mlflow.log_metrics({"train_loss": avg_train_loss, "val_dice": avg_val_dice, "lr": optimizer.param_groups[0]['lr']}, step=epoch)

            if avg_val_dice > best_val_dice:
                best_val_dice = avg_val_dice
                mlflow.log_metric("best_val_dice", best_val_dice, step=epoch)
                checkpoint_path = Path("./best_model.pt")
                torch.save(model.state_dict(), checkpoint_path)
                mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoints")
                print(f"New best model saved with Dice: {best_val_dice:.4f}")

if __name__ == "__main__":
    main()