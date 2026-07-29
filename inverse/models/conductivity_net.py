import math
from typing import List, Optional

import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    Random Fourier Features (RFF) Positional Encoding.
    Evita el sesgo a ejes ortogonales (Grid Bias) de la codificación NeRF estándar,
    permitiendo que la red aprenda formas esféricas e isotrópicas.
    """
    def __init__(self, in_features: int, num_frequencies: int = 10, domain_scale: float = 50.0, sigma_rff: float = 1.0):
        super().__init__()
        self.in_features = in_features
        # Ajustamos a un buen número de frecuencias para 3D si viene con el valor por defecto de NeRF
        if num_frequencies == 10:
            num_frequencies = 64
            
        self.num_frequencies = num_frequencies
        self.domain_scale = domain_scale
        self.out_features = in_features + 2 * num_frequencies
        
        # Matriz de proyección aleatoria fija (B) para mapear coordenadas a frecuencias isotrópicas
        B = torch.randn(in_features, num_frequencies) * sigma_rff
        self.register_buffer('B', B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = (x / self.domain_scale) * torch.pi
        
        # Proyección en direcciones aleatorias
        proj = torch.matmul(x_norm, self.B)
        
        return torch.cat([x_norm, torch.sin(proj), torch.cos(proj)], dim=-1)


class ResidualBlock(nn.Module):
    """
    Bloque Residual con Normalización Configurable y SiLU.
    Diseñado para mejorar el flujo de gradientes en PINNs profundas.
    """
    def __init__(self, hidden_dim: int, activation=nn.SiLU, normalization: Optional[str] = 'WeightNorm', alpha: float = 0.1):
        super().__init__()
        
        # Scaling residual (learnable parameter) para estabilizar gradientes
        # Inicializado en un valor pequeño (e.g. 0.1) para priorizar el camino identity al inicio
        self.alpha = nn.Parameter(torch.tensor(alpha))
        
        linear1 = nn.Linear(hidden_dim, hidden_dim)
        linear2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Selección de esquema de normalización
        if normalization == 'WeightNorm':
            # Inicialización Kaiming
            nn.init.kaiming_normal_(linear1.weight, nonlinearity='relu')
            if linear1.bias is not None:
                nn.init.zeros_(linear1.bias)
                
            # Inicializar linear2 con valores pequeños en lugar de exactamente 0
            # para evitar gradientes nulos (vanishing gradients) en el bloque.
            nn.init.kaiming_normal_(linear2.weight, nonlinearity='relu')
            with torch.no_grad():
                linear2.weight.mul_(1e-3)
            if linear2.bias is not None:
                nn.init.zeros_(linear2.bias)

            self.linear1 = nn.utils.parametrizations.weight_norm(linear1)
            self.linear2 = nn.utils.parametrizations.weight_norm(linear2)
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
                
        elif normalization == 'LayerNorm':
            self.linear1 = linear1
            self.linear2 = linear2
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.norm2 = nn.LayerNorm(hidden_dim)
            
            nn.init.kaiming_normal_(self.linear1.weight, nonlinearity='relu')
            nn.init.kaiming_normal_(self.linear2.weight, nonlinearity='relu')
            with torch.no_grad():
                self.linear2.weight.mul_(1e-3)
            if self.linear2.bias is not None:
                nn.init.zeros_(self.linear2.bias)
                
        elif normalization is None:
            self.linear1 = linear1
            self.linear2 = linear2
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
            
            nn.init.kaiming_normal_(self.linear1.weight, nonlinearity='relu')
            nn.init.kaiming_normal_(self.linear2.weight, nonlinearity='relu')
            with torch.no_grad():
                self.linear2.weight.mul_(1e-3)
            if self.linear2.bias is not None:
                nn.init.zeros_(self.linear2.bias)
        else:
            raise ValueError(f"Unknown normalization: {normalization}")

        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        # Pre-LN para evitar que LayerNorm anule la inicialización a cero de linear2
        # Si norm1/norm2 son Identity (WeightNorm/None), esto equivale al orden estándar
        out = self.norm1(x)
        out = self.linear1(out)
        out = self.activation(out)
        
        out = self.norm2(out)
        out = self.linear2(out)
        
        return residual + self.alpha * out


class ResidualMLP(nn.Module):
    """
    Perceptrón Multicapa con conexiones residuales.
    """
    def __init__(self, in_dim: int, hidden_layers: int, hidden_dim: int, out_dim: int, activation=nn.SiLU, normalization: Optional[str] = 'WeightNorm', final_bias: float = 0.0):
        super().__init__()
        
        input_layer = nn.Linear(in_dim, hidden_dim)
        nn.init.kaiming_normal_(input_layer.weight, nonlinearity='relu')
        
        if normalization == 'WeightNorm':
            self.input_layer = nn.utils.parametrizations.weight_norm(input_layer)
        else:
            self.input_layer = input_layer
            
        blocks = []
        for _ in range(hidden_layers):
            blocks.append(ResidualBlock(hidden_dim, activation, normalization=normalization))
        self.blocks = nn.Sequential(*blocks)
        
        self.output_layer = nn.Linear(hidden_dim, out_dim)
        
        # Inicializamos la última capa para que sus salidas sean muy cercanas a final_bias.
        nn.init.uniform_(self.output_layer.weight, -1e-1, 1e-1)
        nn.init.constant_(self.output_layer.bias, final_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.blocks(x)
        return self.output_layer(x)


class ConductivityNet(nn.Module):
    """
    Red Neuronal sigma_theta que mapea coordenadas espaciales a conductividad local.
    Implementa una arquitectura Multi-Scale Residual PINN robusta para ERT 3D.
    """
    def __init__(
        self, 
        num_frequencies: int = 10,
        hidden_layers: int = 6,  # Capacidad aumentada por defecto para 3D
        hidden_dim: int = 256,   # Capacidad aumentada por defecto para 3D
        latent_dim: int = 256,   # Dimensión del vector latente del MeasurementEncoder
        sigma_min: float = 1e-4,
        sigma_max: float = 1.0,
        domain_scale: float = 50.0,
        sigma_rff: float = 4.0,
        normalization: Optional[str] = 'WeightNorm'
    ):
        super().__init__()
            
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        
        self.latent_dim = latent_dim
        
        # Mapeo de (x,y,z) con Positional Encoding NeRF
        self.fourier_map = PositionalEncoding(
            in_features=3, 
            num_frequencies=num_frequencies, 
            domain_scale=domain_scale,
            sigma_rff=sigma_rff
        )
        
        # Red Residual
        self.mlp = ResidualMLP(
            in_dim=self.fourier_map.out_features + latent_dim, 
            hidden_layers=hidden_layers, 
            hidden_dim=hidden_dim, 
            out_dim=1, 
            activation=nn.SiLU,
            final_bias=-4.595,
            normalization=normalization
        )

    def forward(self, coords: torch.Tensor, latent: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        coords: Tensor de coordenadas (N_points, 3) o (Batch, N_points, 3)
        latent: Tensor de características latentes (N_points, latent_dim) o (Batch, N_points, latent_dim)
        """
        ff = self.fourier_map(coords)
        
        if latent is not None:
            x = torch.cat([ff, latent], dim=-1)
        elif self.latent_dim == 0:
            x = ff
        else:
            # Fallback en caso de que no se provea latente (ej. para compatibilidad con código antiguo)
            # En práctica, si latent_dim > 0, esto fallará en self.mlp. Es mejor requerirlo.
            raise ValueError("latent must be provided when latent_dim > 0")

        raw_output = self.mlp(x)
        
        # Mapeo suave que garantiza que sigma siempre esté entre sigma_min y sigma_max
        # conservando un gradiente continuo (sin zonas muertas)
        sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * torch.sigmoid(raw_output)
        
        return sigma
