# segwork/metrics/core.py

"""
Core metrics for segmentation evaluation.

This module contains implementations for common, widely used segmentation metrics
that can be calculated efficiently on PyTorch tensors or NumPy arrays.
"""
import torch
import numpy as np
from scipy.ndimage import distance_transform_edt

# =======================================================
# Overlap-based Metrics
# =======================================================

def dice_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Calculates the Dice score (F1 score) for semantic segmentation.

    Args:
        pred: Predicted probabilities, shape (N, 1, H, W), float tensor in [0, 1].
        target: Ground truth mask, shape (N, 1, H, W), float tensor {0, 1}.
        eps: A small epsilon to avoid division by zero.

    Returns:
        A tensor of Dice scores for each item in the batch, shape (N,).
    """
    # DEV: Эта реализация численно стабильна и работает на батчах.
    # Суммирование по `dim=[1,2,3]` "схлопывает" все измерения, кроме батча,
    # давая нам по одной метрике на каждый пример.
    assert pred.shape == target.shape, "Prediction and target shapes must match."
    
    intersection = (pred * target).sum(dim=[1, 2, 3])
    union = pred.sum(dim=[1, 2, 3]) + target.sum(dim=[1, 2, 3])
    
    dice = (2. * intersection + eps) / (union + eps)
    return dice

def iou_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Calculates the Intersection over Union (IoU) or Jaccard Index.

    Args:
        pred: Predicted probabilities, shape (N, 1, H, W), float tensor in [0, 1].
        target: Ground truth mask, shape (N, 1, H, W), float tensor {0, 1}.
        eps: A small epsilon to avoid division by zero.

    Returns:
        A tensor of IoU scores for each item in the batch, shape (N,).
    """
    assert pred.shape == target.shape, "Prediction and target shapes must match."

    intersection = (pred * target).sum(dim=[1, 2, 3])
    union = pred.sum(dim=[1, 2, 3]) + target.sum(dim=[1, 2, 3]) - intersection
    
    iou = (intersection + eps) / (union + eps)
    return iou

# =======================================================
# Boundary-based Metrics
# =======================================================

@torch.no_grad()
def boundary_f1_score(pred_bin: torch.Tensor, target_bin: torch.Tensor, boundary_eps: int = 2) -> np.ndarray:
    """
    Calculates the Boundary F1 score with a specified tolerance in pixels.

    This metric is crucial for tasks where precise boundary delineation is important.
    It measures the agreement between predicted and ground truth boundaries.

    Args:
        pred_bin: Binary prediction mask (N, 1, H, W), bool or uint8 tensor.
        target_bin: Binary ground truth mask (N, 1, H, W), bool or uint8 tensor.
        boundary_eps: The tolerance in pixels for matching boundary points.
                      A predicted boundary point is a true positive if it is within
                      this distance of a true boundary point.

    Returns:
        A NumPy array of Boundary F1 scores for each item in the batch.
    """
    # DEV: Эта метрика дорогая, так как требует вычислений на CPU с SciPy.
    # Мы используем `torch.no_grad()` и переводим данные на CPU.
    # `distance_transform_edt` - это "Euclidean Distance Transform",
    # которая эффективно вычисляет расстояние от каждого пикселя до ближайшего
    # "нулевого" пикселя.
    pred_np = pred_bin.squeeze(1).cpu().numpy().astype(np.uint8)
    target_np = target_bin.squeeze(1).cpu().numpy().astype(np.uint8)
    
    scores = []
    for i in range(pred_np.shape[0]):
        # Compute distance transforms for both prediction and ground truth
        pred_dist = distance_transform_edt(1 - pred_np[i])
        target_dist = distance_transform_edt(1 - target_np[i])

        # Find boundary pixels (where distance to background is 1)
        pred_boundary = (pred_dist == 1)
        target_boundary = (target_dist == 1)
        
        # --- Calculate Precision ---
        if pred_boundary.sum() == 0:
            # If no boundary is predicted, precision is perfect (1.0) only if
            # there was no true boundary to find. Otherwise, it's 0.
            precision = 1.0 if target_boundary.sum() == 0 else 0.0
        else:
            # Count how many predicted boundary pixels are within `boundary_eps`
            # of a true boundary.
            tp_precision = np.sum(target_dist[pred_boundary] <= boundary_eps)
            precision = tp_precision / pred_boundary.sum()
            
        # --- Calculate Recall ---
        if target_boundary.sum() == 0:
            # If no true boundary exists, recall is perfect (1.0) only if
            # no boundary was predicted. Otherwise, it's 0.
            recall = 1.0 if pred_boundary.sum() == 0 else 0.0
        else:
            # Count how many true boundary pixels are within `boundary_eps`
            # of a predicted boundary.
            tp_recall = np.sum(pred_dist[target_boundary] <= boundary_eps)
            recall = tp_recall / target_boundary.sum()
        
        # --- Calculate F1 Score ---
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        scores.append(f1)
        
    return np.array(scores)

# =======================================================
# Pixel-based Metrics
# =======================================================

def pixel_error(pred_bin: torch.Tensor, target_bin: torch.Tensor) -> torch.Tensor:
    """
    Calculates the pixel error rate (1 - pixel accuracy).
    
    Args:
        pred_bin: Binary prediction mask (N, 1, H, W), bool or uint8 tensor.
        target_bin: Binary ground truth mask (N, 1, H, W), bool or uint8 tensor.

    Returns:
        A tensor of pixel error rates for each item in the batch.
    """
    assert pred_bin.shape == target_bin.shape
    
    # DEV: Это простая метрика, но она может быть обманчивой при дисбалансе классов.
    # Если 99% изображения - фон, модель, предсказывающая только фон, будет иметь
    # 99% точности (1% ошибки), что звучит хорошо, но бесполезно.
    
    # Number of pixels that do not match
    incorrect_pixels = (pred_bin != target_bin).sum(dim=[1, 2, 3])
    
    # Total number of pixels
    total_pixels = pred_bin.shape[1] * pred_bin.shape[2] * pred_bin.shape[3]
    
    return incorrect_pixels / total_pixels