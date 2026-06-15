import torch
import torch.nn as nn
from typing import Dict

class AdaptiveLossWeighter(nn.Module):
    """
    Self-Adaptive Loss Weighter using Homoscedastic Uncertainty.
    Based on Kendall et al. (2018) "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics".
    
    The loss for task i is: L_i * exp(-log_var_i) + log_var_i
    """
    def __init__(self, keys: list[str], initial_values: dict[str, float] | None = None):
        super().__init__()
        self.keys = keys
        
        # Initialize learnable log_vars
        self.log_vars = nn.ParameterDict()
        for key in keys:
            init_val = 0.0
            if initial_values and key in initial_values:
                # If we want to start with a specific weight W, exp(-log_var) = W => log_var = -log(W)
                # But if initial_values provided are raw W, we initialize log_var = -log(W).
                # However, the user agreed to initialize to 0 (which means weight = 1.0)
                init_val = initial_values[key]
            
            # Using nn.Parameter
            self.log_vars[key] = nn.Parameter(torch.tensor(init_val, dtype=torch.float32))

    def forward(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        total_loss = 0.0
        for key in self.keys:
            if key in losses:
                loss_val = losses[key]
                log_var = self.log_vars[key]
                # Formula: L * exp(-log_var) + log_var
                weight = torch.exp(-log_var)
                total_loss += loss_val * weight + log_var
        return total_loss
    
    def get_weights(self) -> dict[str, float]:
        """Return the effective weights exp(-log_var) for logging."""
        with torch.no_grad():
            return {key: torch.exp(-self.log_vars[key]).item() for key in self.keys}
