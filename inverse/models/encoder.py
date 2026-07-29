import torch
import torch.nn as nn

class MeasurementEncoder(nn.Module):
    """
    PointNet-style encoder for ERT measurements.
    Processes a set of measurements (r_A, r_B, r_M, r_N, delta_V) independently
    and then applies global pooling to produce a permutation-invariant latent vector.
    """
    def __init__(self, in_features: int = 13, hidden_dim: int = 128, latent_dim: int = 256):
        super().__init__()
        
        # Point-wise MLP
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.SiLU()
        )
        
        # Optional second MLP after pooling for more capacity
        self.head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape (Batch, N_meas, in_features)
           in_features is usually 13: 
           r_A (3), r_B (3), r_M (3), r_N (3), delta_V (1)
        """
        # 1. Point-wise processing (shared weights across N_meas)
        # x is (B, N, 13). Linear applies to the last dimension, so output is (B, N, latent_dim)
        point_features = self.mlp(x)
        
        # 2. Global Pooling (Max pooling preserves features better in PointNets)
        global_feature, _ = torch.max(point_features, dim=1)  # (B, latent_dim)
        
        # 3. Final mapping
        latent = self.head(global_feature)  # (B, latent_dim)
        
        return latent
