# segwork/mlflow_loggers/training_loggers.py

"""
A collection of helper functions for logging specific artifacts and metrics
to MLflow during the training process.
"""
import torch
import mlflow
from pathlib import Path
import shutil
from typing import List, Dict, Any

# DEV: Мы импортируем только то, что нужно этим конкретным функциям.
# Это делает модуль более независимым.
from segwork.visualization.plotter import Plotter
from segwork.visualization.generators import generate_raw_panel, generate_overlay_panel

def log_validation_visuals(
    model: torch.nn.Module,
    val_samples: List[tuple],
    epoch: int,
    viz_config: Dict[str, Any],
    device: torch.device,
    amp_enabled: bool
):
    """
    Generates a panel of validation predictions and logs it as an artifact to MLflow.

    Args:
        model: The model being trained.
        val_samples: A fixed list of (image_tensor, mask_tensor) tuples for visualization.
        epoch: The current epoch number.
        viz_config: The `train.validation_visualization` section from the config.
        device: The device to run inference on.
        amp_enabled: Flag indicating if Automatic Mixed Precision is active.
    """
    if not val_samples:
        return
        
    print("Generating validation previews...")
    model.eval()
    
    viz_style = viz_config.get('style', {})
    plotter = Plotter(viz_style)
    
    with torch.no_grad():
        for i, (image_tensor, mask_tensor) in enumerate(val_samples):
            # DEV: `image_tensor` и `mask_tensor` здесь - это полные тензоры,
            # возвращаемые `TilesDataset.__getitem__`.
            image_np = image_tensor.cpu().numpy()
            gt_mask_np = mask_tensor.cpu().numpy()
            
            # Если изображение многоканальное, для `generate_raw_panel` возьмем только первый (raw) канал.
            if image_np.shape[0] > 1:
                display_image_np = image_np[0]
            else:
                display_image_np = image_np.squeeze(0)

            # Делаем предсказание
            input_tensor = image_tensor.unsqueeze(0).to(device)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(input_tensor)
            probs = torch.sigmoid(logits)
            pred_mask_np = probs.squeeze().cpu().numpy()

            # Собираем панель [Input | GT Overlay | Pred Overlay]
            plotter.add_panel(generate_raw_panel(display_image_np), f"Sample {i+1} - Input")
            
            gt_color = viz_style.get('gt_color_rgb', [0, 170, 0])
            plotter.add_panel(generate_overlay_panel(display_image_np, gt_mask_np, color=gt_color), "Ground Truth")
            
            pred_color = viz_style.get('pred_color_rgb', [227, 27, 27])
            plotter.add_panel(generate_overlay_panel(display_image_np, pred_mask_np > 0.5, color=pred_color), f"Prediction (Epoch {epoch+1})")
    
    # Сохраняем во временную папку и логируем в MLflow
    temp_viz_dir = Path("./temp_train_viz")
    temp_viz_dir.mkdir(exist_ok=True)
    save_path = temp_viz_dir / f"validation_epoch_{epoch+1:03d}.png"
    
    plotter.render(
        save_path=save_path,
        suptitle=f"Validation Samples - Epoch {epoch+1}",
        force_cols=3, # Мы всегда показываем 3 панели на один сэмпл
        show=False
    )
    
    mlflow.log_artifact(str(save_path), artifact_path="validation_previews")
    shutil.rmtree(temp_viz_dir)