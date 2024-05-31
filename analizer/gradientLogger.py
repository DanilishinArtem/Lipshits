import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np

class GradientLogger:
    def __init__(self, model: nn.Module, writer: SummaryWriter):
        self.model = model
        self.writer = writer
        self.gradient_hooks = []
        self.gradient_distributions = {}
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            hook = module.register_forward_hook(self._hook_fn(name))
            self.gradient_hooks.append(hook)

    def _hook_fn(self, layer_name: str):
        def hook(model, grad_input, grad_output):
            if grad_output[0] is not None:
                grad = grad_output[0].detach().cpu().numpy()
                self.gradient_distributions[layer_name] = grad
        return hook
    
    def log_gradients(self, epoch: int):
        for layer_name, grad in self.gradient_distributions.items():
            self.writer.add_histogram(f"Gradients/{layer_name}", grad, epoch)

    def close(self):
        self.writer.close()