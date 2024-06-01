import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

def LipshitsForLayer(layer: nn.Module, name: str):
    lipschitz_constant = 1
    if "conv" in name:
        weights = layer.weight.detach()
        lipschitz_constant = torch.linalg.norm(weights.view(weights.shape[0], -1), dim=1).max()
    elif "lin" in name:
        weights = layer.weight.detach()
        singular_values = torch.svd(weights).S
        lipschitz_constant = singular_values.max()
    return lipschitz_constant

def lipshits(model: nn.Module):
    lipschitz_constants = []
    for name, module in model.named_modules():
        lipschitz_constants.append(LipshitsForLayer(module, name))
    # print(lipschitz_constants)
    return np.prod(lipschitz_constants)

def maxNorm(train_loader: DataLoader):
    max_diff_norm = 0
    max_norm = 0
    for batch in train_loader:
        images = batch["image"]  # Получаем изображения из батча
        labels = batch["label"]  # Получаем метки из батча
        if len(images) < 2:
            continue
        image_diffs = images[1:] - images[:-1]
        diff_norms = torch.norm(image_diffs.view(image_diffs.shape[0], -1), dim=1)
        image_norms = torch.norm(images.view(images.shape[0], -1), dim=1)
        batch_max_diff_norm = diff_norms.max().item()
        batch_max_norm = image_norms.max().item()
        max_diff_norm = max(max_diff_norm, batch_max_diff_norm)
        max_norm = max(max_norm, batch_max_norm)
    return max_norm, max_diff_norm
