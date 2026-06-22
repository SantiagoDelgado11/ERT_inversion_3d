
import torch
import torch.nn as nn

class ConductivityNet(nn.Module):
    """
    Parametric Neural Network for the conductivity scalar field sigma_phi(r).
    Inputs: r = (x, y, z)
    Outputs: sigma (positive scalar)
    """
    def __init__(self, in_features=3, hidden_features=64, hidden_layers=4, out_features=1, min_conductivity=1e-5):
        super().__init__()
        
        self.min_conductivity = min_conductivity
        
        layers = []
        # Input layer
        layers.append(nn.Linear(in_features, hidden_features))
        layers.append(nn.SiLU())
        
        # Hidden layers
        for _ in range(hidden_layers):
            layers.append(nn.Linear(hidden_features, hidden_features))
            layers.append(nn.SiLU())
            
        # Output layer
        layers.append(nn.Linear(hidden_features, out_features))
        
        self.net = nn.Sequential(*layers)
        self.softplus = nn.Softplus() # Ensures positivity
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: Tensor of shape (N, 3) representing coordinates (x, y, z).
        Returns:
            sigma: Tensor of shape (N, 1) representing the positive conductivity.
        """
        raw_out = self.net(x)
        # Apply softplus to ensure strictly positive outputs, plus a small bias for numerical stability
        return self.softplus(raw_out) + self.min_conductivity
