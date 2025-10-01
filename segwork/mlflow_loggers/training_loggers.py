"""
A collection of helper functions for logging specific artifacts and metrics
to MLflow during the training process.
"""
import torch
import mlflow
from pathlib import Path
import shutil
from typing import List, Dict, Any
import numpy as np
import cv2

from segwork.visualization.plotter import Plotter
from segwork.visualization.generators import generate_raw_panel, generate_overlay_panel
from segwork.models.adapter import ChannelAdapter

def log_validation_visuals(
    model: torch.nn.Module,
    val_sample_paths: List[Dict[str, Path]],
    epoch: int,
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
                gt_mask_np = cv2.imread(str(paths['mask_path']), cv2.IMREAD_GRAYSCALE)
                gt_mask_np = (gt_mask_np > 127).astype(np.float32)

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
    model: torch.nn.Module,
    epoch: int,
    feature_bank_channels: List[str]
):
    """
    Finds a ChannelAdapter in the model, calculates the importance of each
    input channel, and logs it as a metric to MLflow for each epoch.
    """
    # Find the adapter module in the model
    channel_adapter = None
    if isinstance(model, torch.nn.Sequential) and isinstance(model[0], ChannelAdapter):
        channel_adapter = model[0]

    if channel_adapter is None:
        return # Do nothing if this is not an adapter-based model

    # Extract weights and calculate importance (mean absolute weight per input channel)
    with torch.no_grad():
        # Shape: (out_channels, in_channels, 1, 1)
        weights = channel_adapter.proj.weight.detach().abs()
        # Average across the output channels to get a single importance value per input channel
        per_channel_importance = weights.mean(dim=0).squeeze().cpu().numpy()

    if per_channel_importance.size == 1: # Handle the case of a single input channel
        per_channel_importance = [per_channel_importance.item()]

    if len(per_channel_importance) != len(feature_bank_channels):
        print(f"Warning: Mismatch between adapter input channels ({len(per_channel_importance)}) "
              f"and feature bank channels ({len(feature_bank_channels)}). Skipping weight logging.")
        return

    # Create a dictionary of metrics and log to MLflow
    metrics_to_log = {
        f"adapter_weights/{name}": importance
        for name, importance in zip(feature_bank_channels, per_channel_importance)
    }
    mlflow.log_metrics(metrics_to_log, step=epoch)