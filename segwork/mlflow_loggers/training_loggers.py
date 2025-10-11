# segwork\mlflow_loggers\training_loggers.py

"""
A collection of helper functions for logging specific artifacts and metrics
to MLflow during the training process.
"""
import torch
import torch.nn as nn
import mlflow
from pathlib import Path
import shutil
from typing import List, Dict, Any
import numpy as np
import cv2

from segwork.visualization.plotter import Plotter
from segwork.visualization.generators import generate_raw_panel, generate_overlay_panel
from segwork.models.adapter import ChannelAdapter
from segwork.data.dataset import binarize_mask

def log_validation_visuals(
    model: torch.nn.Module,
    val_sample_paths: List[Dict[str, Path]],
    epoch: int,
    mask_processing_cfg: str,
    viz_config: Dict[str, Any],    
    device: torch.device,
    amp_enabled: bool
):
    """
    Generates a panel of validation predictions and logs it as an artifact to MLflow.

    Args:
        model: The model being trained.
        val_sample_paths: A fixed list of dictionaries, each containing paths
                          to an image and its corresponding mask.
        epoch: The current epoch number.
        viz_config: The `train.validation_visualization` section from the config.
        device: The device to run inference on.
        amp_enabled: Flag indicating if Automatic Mixed Precision is active.
    """
    if not val_sample_paths:
        return
        
    print("Generating validation previews...")
    model.eval()
    
    viz_style = viz_config.get('style', {})
    plotter = Plotter(viz_style)
    
    with torch.no_grad():
        for i, paths in enumerate(val_sample_paths):
            try:
                # 1. Load data from paths
                image_stack = np.load(paths['image_path']).astype(np.float32)
                gt_mask_np_raw = cv2.imread(str(paths['mask_path']), cv2.IMREAD_GRAYSCALE)
                # gt_mask_np = (gt_mask_np > 127).astype(np.float32)
                gt_mask_np = binarize_mask(gt_mask_np_raw, mask_processing_cfg)

                # For visualization, we only show the first (raw) channel if the stack is multi-channel
                display_image_np = image_stack[0] if image_stack.ndim == 3 else image_stack
                
                # 2. Perform inference
                input_tensor = torch.from_numpy(image_stack).unsqueeze(0).to(device)
                with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                    logits = model(input_tensor)
                probs = torch.sigmoid(logits)
                pred_mask_np = probs.squeeze().cpu().numpy()

                # 3. Assemble the panel [Input | GT Overlay | Pred Overlay]
                plotter.add_panel(generate_raw_panel(display_image_np), f"Sample {i+1}")
                
                gt_color = viz_style.get('gt_color_rgb', [0, 170, 0])
                plotter.add_panel(generate_overlay_panel(display_image_np, gt_mask_np, color=gt_color), "Ground Truth")
                
                pred_color = viz_style.get('pred_color_rgb', [227, 27, 27])
                plotter.add_panel(generate_overlay_panel(display_image_np, pred_mask_np > 0.5, color=pred_color), "Prediction")
            except Exception as e:
                print(f"Warning: Failed to generate preview for sample {i+1}. Error: {e}")

    # Save to a temporary directory and log to MLflow
    if not plotter.panels:
        return
        
    temp_viz_dir = Path("./temp_train_viz")
    temp_viz_dir.mkdir(exist_ok=True)
    save_path = temp_viz_dir / f"validation_epoch_{epoch+1:03d}.png"
    
    plotter.render(
        save_path=save_path,
        suptitle=f"Validation Samples - Epoch {epoch+1}",
        force_cols=3, # We always show 3 panels per sample
        show=False
    )
    
    mlflow.log_artifact(str(save_path), artifact_path="validation_previews")
    shutil.rmtree(temp_viz_dir)

def log_adapter_weights(
    channel_adapter: torch.nn.Module,
    epoch: int,
    feature_bank_channels: List[str]
):
    """
    Calculates the importance of each input channel in a ChannelAdapter
    and logs it as a metric to MLflow.
    """
    if not isinstance(channel_adapter, ChannelAdapter):
        print(f"Warning: `log_adapter_weights` received an object of type "
              f"{type(channel_adapter)}, not ChannelAdapter. Skipping.")
        return

    with torch.no_grad():
        weights = channel_adapter.proj.weight.detach().abs()
        per_channel_importance = weights.mean(dim=0).squeeze().cpu().numpy()

    if per_channel_importance.size == 1:
        per_channel_importance = [per_channel_importance.item()]

    if len(per_channel_importance) != len(feature_bank_channels):
        print(f"Warning: Mismatch between adapter input channels ({len(per_channel_importance)}) "
              f"and feature bank channels ({len(feature_bank_channels)}). Skipping weight logging.")
        return

    metrics_to_log = {
        f"adapter_weights/{name}": importance
        for name, importance in zip(feature_bank_channels, per_channel_importance)
    }
    mlflow.log_metrics(metrics_to_log, step=epoch)

def log_gradient_norms(modules: Dict[str, nn.Module | None], global_step: int):
    """
    Calculates the L2 norm of gradients for specified model components and logs them.

    Args:
        modules: A dictionary mapping component names (e.g., 'encoder', 'decoder')
                 to the nn.Module instances.
        global_step: The current global training step (batch iteration).
    """
    metrics_to_log = {}
    
    for name, module in modules.items():
        # Пропускаем, если модуль не найден (например, нет адаптера)
        if module is None:
            continue
            
        # Собираем параметры с ненулевыми градиентами
        params_with_grad = [p for p in module.parameters() if p.grad is not None]
        
        if not params_with_grad:
            # Если градиентов нет (например, модуль заморожен), логируем 0.0
            norm = 0.0
        else:
            # Считаем общую L2 норму градиентов для этого модуля
            # DEV: Мы создаем плоский вектор из всех градиентов модуля и считаем его норму.
            # `torch.cat` - эффективный способ это сделать.
            all_grads = torch.cat([p.grad.detach().flatten() for p in params_with_grad])
            norm = torch.linalg.vector_norm(all_grads, 2.0).item()
            
        metrics_to_log[f'grad_norm/{name}'] = norm

    if metrics_to_log:
        mlflow.log_metrics(metrics_to_log, step=global_step)