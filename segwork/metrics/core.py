# segwork/metrics/core.py
# segwork/metrics/core.py

"""
Core metrics for segmentation evaluation.

This module provides wrappers around standard, community-vetted metric
implementations from libraries like `segmentation-models-pytorch` and `monai`
to ensure reliability and comparability of results.
"""
import torch
import numpy as np

from typing import Dict

# --- Библиотечные импорты ---
import segmentation_models_pytorch.metrics as smp_metrics
from monai.metrics import compute_surface_dice

# =======================================================
# Overlap-based Metrics (using segmentation-models-pytorch)
# =======================================================

def dice_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Calculates the Dice score (F1 score) using the `segmentation-models-pytorch` backend.

    Args:
        pred: Predicted probabilities, shape (N, 1, H, W), float tensor in [0, 1].
        target: Ground truth mask, shape (N, 1, H, W), float tensor {0, 1}.
        eps: A small epsilon for numerical stability (handled by smp_metrics).

    Returns:
        A tensor of Dice scores for each item in the batch, shape (N,).
    """
    assert pred.shape == target.shape, "Prediction and target shapes must match."
    
    # smp_metrics ожидает target в формате long
    target_long = target.long()

    # get_stats работает с батчами и возвращает TP, FP, FN, TN для каждого элемента
    tp, fp, fn, tn = smp_metrics.get_stats(
        pred, target_long, mode='binary', threshold=0.5
    )
    
    # f1_score также работает с батчами, возвращая тензор нужной формы
    return smp_metrics.f1_score(tp, fp, fn, tn, reduction=None)

def iou_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Calculates the Intersection over Union (IoU) using the `segmentation-models-pytorch` backend.

    Args:
        pred: Predicted probabilities, shape (N, 1, H, W), float tensor in [0, 1].
        target: Ground truth mask, shape (N, 1, H, W), float tensor {0, 1}.
        eps: A small epsilon for numerical stability (handled by smp_metrics).

    Returns:
        A tensor of IoU scores for each item in the batch, shape (N,).
    """
    assert pred.shape == target.shape, "Prediction and target shapes must match."
    
    target_long = target.long()

    tp, fp, fn, tn = smp_metrics.get_stats(
        pred, target_long, mode='binary', threshold=0.5
    )
    
    return smp_metrics.iou_score(tp, fp, fn, tn, reduction=None)

# =======================================================
# Boundary-based Metrics (using MONAI)
# =======================================================

@torch.no_grad()
def boundary_f1_score(pred_bin: torch.Tensor, target_bin: torch.Tensor, boundary_eps: int = 3) -> float:
    """
    Calculates the Boundary F1 score (Surface Dice) using the `monai` backend.

    This metric is crucial for tasks where precise boundary delineation is important.
    It measures the agreement between predicted and ground truth boundaries.

    Args:
        pred_bin: Binary prediction mask (N, 1, H, W), bool or uint8 tensor.
        target_bin: Binary ground truth mask (N, 1, H, W), bool or uint8 tensor.
        boundary_eps: The tolerance in pixels for matching boundary points. This is
                      passed to MONAI's `class_thresholds`.

    Returns:
        A single float value representing the mean Boundary F1 score across the batch.
        MONAI's implementation returns an aggregated value, not per-image scores.
    """
    # MONAI ожидает бинарные тензоры типа float
    pred_float = pred_bin.float()
    target_float = target_bin.float()

    # MONAI работает с изотропным спейсингом по умолчанию (1.0, 1.0)
    surface_dice = compute_surface_dice(
        y_pred=pred_float,
        y=target_float,
        class_thresholds=[float(boundary_eps)],
        spacing=(1.0, 1.0) # Предполагаем 2D-изображения с пиксельным спейсингом 1x1
    )
    # compute_surface_dice возвращает тензор с одним элементом (среднее по батчу)
    return surface_dice.mean().item()


# =======================================================
# Pixel-based Metrics (can remain custom or use SMP)
# =======================================================

def pixel_error(pred_bin: torch.Tensor, target_bin: torch.Tensor) -> torch.Tensor:
    """
    Calculates the pixel error rate (1 - pixel accuracy) using the `segmentation-models-pytorch` backend.
    
    Args:
        pred_bin: Binary prediction mask (N, 1, H, W), bool or uint8 tensor.
        target_bin: Binary ground truth mask (N, 1, H, W), bool or uint8 tensor.

    Returns:
        A tensor of pixel error rates for each item in the batch.
    """
    assert pred_bin.shape == target_bin.shape

    # Используем smp_metrics.accuracy, который также работает с батчами
    target_long = target_bin.long()
    tp, fp, fn, tn = smp_metrics.get_stats(
        pred_bin.float(), target_long, mode='binary', threshold=0.5
    )
    
    accuracy = smp_metrics.accuracy(tp, fp, fn, tn, reduction=None)
    
    return 1.0 - accuracy

def main_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:

    target_long = target.long()

    tp, fp, fn, tn = smp_metrics.get_stats(
        pred, target_long, mode='binary', threshold=0.5
    )
    
    accuracy = smp_metrics.accuracy(tp, fp, fn, tn, reduction=None)
    recall = smp_metrics.recall(tp, fp, fn, tn, reduction=None)
    f1_score = smp_metrics.f1_score(tp, fp, fn, tn, reduction=None)
    iou_score = smp_metrics.iou_score(tp, fp, fn, tn, reduction=None)

    pixel_error = 1.0 - accuracy

    metrics = {"acc": accuracy, "rec": recall, 
               "dice": f1_score,
               "iou": iou_score, "perr": pixel_error}


    
    return metrics