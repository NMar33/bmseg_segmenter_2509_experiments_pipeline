# src/metrics/core.py

"""
Core metrics for segmentation evaluation.
"""
import torch

def dice_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Calculates the Dice score.

    Args:
        pred: Predicted probabilities, shape (N, 1, H, W), float tensor [0, 1].
        target: Ground truth mask, shape (N, 1, H, W), float tensor {0, 1}.
        eps: A small epsilon to avoid division by zero.

    Returns:
        A tensor of Dice scores for each item in the batch, shape (N,).
    """
    # DEV: Эта реализация численно стабильна и работает на батчах.
    # Суммирование по `dim=[1,2,3]` "схлопывает" все измерения, кроме батча,
    # давая нам по одной метрике на каждый пример.
    intersection = (pred * target).sum(dim=[1, 2, 3])
    union = pred.sum(dim=[1, 2, 3]) + target.sum(dim=[1, 2, 3])
    
    dice = (2. * intersection + eps) / (union + eps)
    return dice