import torch
import torch.nn as nn


def validate(logits, target):
    assert logits.shape[-2:] == target.shape[-2:], f"{logits.shape[-2:]=} should equal {target.shape[-2:]=}"
    assert logits.device == target.device, f"{logits.device=} should equal {target.device=}"


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        validate(logits, target)
        logits = logits.view(-1)
        target = target.view(-1) + self.eps
        intersection = (logits * target).sum()
        union = (logits + target).sum()
        dice_score = 2.0 * intersection / (union + self.eps)
        return torch.mean(1.0 - dice_score)


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, smooth=1):
        super().__init__()
        self.alpha = alpha
        self.beta = 1 - alpha
        self.bias = smooth

    def forward(self, logits, target):
        validate(logits, target)
        logits = logits.view(-1)
        target = target.view(-1)
        true_positive = (logits * target).sum()
        false_positive = self.alpha * ((1 - target) * logits).sum()
        false_negative = self.beta * (target * (1 - logits)).sum()
        return 1 - (true_positive + self.bias) / (true_positive + false_positive + false_negative + self.bias)
