import torch
import torch.nn as nn

class PotentialNet(nn.Module):
    """
    Parametric Neural Network for the scalar potential field u_theta(r).
    Inputs: r = (x, y, z) 
    Outputs: u
    """
    def __init__(self, in_features=3, hidden_features=64, hidden_layers=4, out_features=1):
        super().__init__()
        
        layers = []
        # Input layer
        layers.append(nn.Linear(in_features, hidden_features))
        layers.append(nn.SiLU()) # Smooth activation for 2nd order derivatives
        
        # Hidden layers
        for _ in range(hidden_layers):
            layers.append(nn.Linear(hidden_features, hidden_features))
            layers.append(nn.SiLU())
            
        # Output layer
        layers.append(nn.Linear(hidden_features, out_features))
        
        self.net = nn.Sequential(*layers)
        
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
            u: Tensor of shape (N, 1) representing the potential.
        """
        return self.net(x)
