import torch.nn as nn

class MLP(nn.Module):
    """
    Perceptrón Multicapa base con activación estricta (SiLU/Tanh).
    """
    def __init__(self, in_dim, hidden_layers, hidden_dim, out_dim, activation=nn.SiLU):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(activation())
        
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation())
            
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)
