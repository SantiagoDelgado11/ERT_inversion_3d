import math
from typing import List, Optional

import torch
import torch.nn as nn

class MultiScaleFourierEncoding(nn.Module):
    """
    Mapeo de Características de Fourier Multiescala.
    Permite a la red aprender diferentes frecuencias espaciales concatenando
    las proyecciones en múltiples escalas.
    """
    def __init__(self, in_features: int, mapping_size: int, scales: List[float], domain_scale: float = 50.0):
        super().__init__()
        self.in_features = in_features
        self.domain_scale = domain_scale
        
        # Distribuimos el mapping_size total entre las escalas disponibles
        features_per_scale = max(1, mapping_size // len(scales))
        
        # Creamos una única matriz B para mayor eficiencia
        B = torch.randn(in_features, features_per_scale * len(scales))
        for i, scale in enumerate(scales):
            start = i * features_per_scale
            end = start + features_per_scale
            B[:, start:end] = B[:, start:end] * scale
            
        self.register_buffer('B', B)
        self.out_features = in_features + 2 * (features_per_scale * len(scales))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalización espacial CRÍTICA
        x_norm = x / self.domain_scale
        x_proj = (2.0 * math.pi * x_norm) @ self.B
        return torch.cat([x_norm, torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


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
            self.linear1 = nn.utils.weight_norm(linear1)
            self.linear2 = nn.utils.weight_norm(linear2)
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
            
            # Inicialización Kaiming
            nn.init.kaiming_normal_(self.linear1.weight_v, nonlinearity='relu')
            
            # Para WeightNorm, inicializamos weight_g a 0 para que la salida sea 0
            # Dejar weight_v aleatorio evita división por cero al calcular la norma
            nn.init.zeros_(self.linear2.weight_g)
            if self.linear2.bias is not None:
                nn.init.zeros_(self.linear2.bias)
                
        elif normalization == 'LayerNorm':
            self.linear1 = linear1
            self.linear2 = linear2
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.norm2 = nn.LayerNorm(hidden_dim)
            
            nn.init.kaiming_normal_(self.linear1.weight, nonlinearity='relu')
            nn.init.zeros_(self.linear2.weight)
            if self.linear2.bias is not None:
                nn.init.zeros_(self.linear2.bias)
                
        elif normalization is None:
            self.linear1 = linear1
            self.linear2 = linear2
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
            
            nn.init.kaiming_normal_(self.linear1.weight, nonlinearity='relu')
            nn.init.zeros_(self.linear2.weight)
            if self.linear2.bias is not None:
                nn.init.zeros_(self.linear2.bias)
        else:
            raise ValueError(f"Unknown normalization: {normalization}")

        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = self.linear1(x)
        out = self.norm1(out)
        out = self.activation(out)
        
        out = self.linear2(out)
        out = self.norm2(out)
        
        return residual + self.alpha * out


class ResidualMLP(nn.Module):
    """
    Perceptrón Multicapa con conexiones residuales.
    """
    def __init__(self, in_dim: int, hidden_layers: int, hidden_dim: int, out_dim: int, activation=nn.SiLU, normalization: Optional[str] = 'WeightNorm'):
        super().__init__()
        
        input_layer = nn.Linear(in_dim, hidden_dim)
        nn.init.kaiming_normal_(input_layer.weight, nonlinearity='relu')
        
        if normalization == 'WeightNorm':
            self.input_layer = nn.utils.weight_norm(input_layer)
        else:
            self.input_layer = input_layer
            
        blocks = []
        for _ in range(hidden_layers):
            blocks.append(ResidualBlock(hidden_dim, activation, normalization=normalization))
        self.blocks = nn.Sequential(*blocks)
        
        self.output_layer = nn.Linear(hidden_dim, out_dim)
        
        # Inicializamos la última capa para que sus salidas sean muy cercanas a 0.
        # Esto es crucial para la parametrización logarítmica, permitiendo que
        # al inicio delta ≈ 0 y sigma ≈ sigma_background.
        nn.init.uniform_(self.output_layer.weight, -1e-4, 1e-4)
        nn.init.zeros_(self.output_layer.bias)

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
        fourier_features: int = 256, 
        fourier_scale: float = 10.0, 
        hidden_layers: int = 6,  # Capacidad aumentada por defecto para 3D
        hidden_dim: int = 256,   # Capacidad aumentada por defecto para 3D
        fourier_scales: Optional[List[float]] = None,
        sigma_background: float = 1e-3,
        max_log_variation: float = 5.0
    ):
        super().__init__()
        
        if fourier_scales is None:
            # Escalas multiresolución por defecto, cubriendo desde variaciones suaves a anómalas
            fourier_scales = [1.0, 5.0, 10.0, 20.0, 40.0]
            
        self.sigma_background = sigma_background
        self.max_log_variation = max_log_variation
        
        # Almacenamos log(sigma_background) como un buffer para que resida en el device correcto 
        # sin ser entrenable y evitar calcularlo repetidas veces en el forward().
        self.register_buffer('log_sigma_bg', torch.tensor(math.log(sigma_background)))
        
        # Mapeo de (x,y,z) multiescala
        self.fourier_map = MultiScaleFourierEncoding(
            in_features=3, 
            mapping_size=fourier_features, 
            scales=fourier_scales
        )
        
        # Red Residual
        self.mlp = ResidualMLP(
            in_dim=self.fourier_map.out_features, 
            hidden_layers=hidden_layers, 
            hidden_dim=hidden_dim, 
            out_dim=1, 
            activation=nn.SiLU
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        coords: Tensor de coordenadas (batch_size, 3)
        """
        ff = self.fourier_map(coords)
        raw_output = self.mlp(ff)
        
        # 1. Restricción Suave y Diferenciable
        # Permite anomalías más conductivas o más resistivas sin destruir gradientes
        delta = self.max_log_variation * torch.tanh(raw_output)
        
        # 2. Perturbación en el Espacio Logarítmico
        log_sigma = self.log_sigma_bg + delta
        
        # 3. Positividad Estricta y Estabilidad Numérica
        sigma = torch.exp(log_sigma)
        
        return sigma
