import torch
import torch.nn as nn

class FourierFeatureMapping(nn.Module):
    """
    Mapeo de Características de Fourier para superar el sesgo espectral (Spectral Bias)
    y permitir el aprendizaje de altas frecuencias (bordes afilados de anomalías).
    """
    def __init__(self, in_features, mapping_size, scale=10.0, domain_scale=50.0):
        super().__init__()
        self.in_features = in_features
        self.mapping_size = mapping_size
        self.scale = scale
        self.domain_scale = domain_scale
        # Matriz B estática para el positional encoding
        self.register_buffer('B', torch.randn(in_features, mapping_size) * scale)
        
    def forward(self, x):
        # x: (batch_size, in_features)
        # Normalización espacial CRÍTICA para evitar oscilaciones de ruido de alta frecuencia
        x_norm = x / self.domain_scale
        x_proj = (2.0 * torch.pi * x_norm) @ self.B
        return torch.cat([x_norm, torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
