import torch
import torch.nn as nn
from typing import Optional, List
from .conductivity_net import PositionalEncoding, ResidualMLP

class PotentialNet(nn.Module):
    """
    Red Neuronal u_phi que mapea coordenadas espaciales y fuentes a potencial eléctrico.
    Reutiliza componentes robustos de ConductivityNet para una arquitectura PINN estable.
    Aprovecha la geometría del problema integrando distancias a las fuentes.
    """
    def __init__(
        self, 
        num_frequencies: int = 10,
        hidden_layers: int = 5, 
        hidden_dim: int = 256, 
        domain_scale: float = 50.0,
        source_scale: float = 50.0,
        normalization: Optional[str] = 'WeightNorm'
    ):
        super().__init__()
            
        self.source_scale = source_scale
        
        # Mapeo NeRF de (x,y,z) reutilizado de ConductivityNet
        self.fourier_map = PositionalEncoding(
            in_features=3, 
            num_frequencies=num_frequencies,
            domain_scale=domain_scale
        )
        
        # Dimensiones de entrada para la MLP:
        # - ff: self.fourier_map.out_features
        # - source_coords normalizadas: 6
        # - r_A = coords - sourceA: 3
        # - r_B = coords - sourceB: 3
        # - ||r_A||: 1
        # - ||r_B||: 1
        # - 1 / (||r_A|| + eps): 1
        # - 1 / (||r_B|| + eps): 1
        # Total extra = 6 + 3 + 3 + 1 + 1 + 1 + 1 = 16
        in_dim = self.fourier_map.out_features + 16
        
        self.mlp = ResidualMLP(
            in_dim=in_dim,
            hidden_layers=hidden_layers,
            hidden_dim=hidden_dim,
            out_dim=1,
            activation=nn.SiLU,
            normalization=normalization
        )

    def forward(self, coords: torch.Tensor, source_coords: torch.Tensor) -> torch.Tensor:
        """
        coords: (batch_size, 3) coordenadas de evaluación (x, y, z)
        source_coords: (batch_size, 6) posiciones de dipolo inyector (xA, yA, zA, xB, yB, zB)
        
        Retorna:
        u: (batch_size, 1) Potencial eléctrico estimado
        """
        # 1. Características espaciales multiescala
        ff = self.fourier_map(coords)
        
        # 2. Variables geométricas relativas a las fuentes
        source_A = source_coords[..., :3]
        source_B = source_coords[..., 3:]
        
        # Normalizamos posiciones y distancias para la red utilizando source_scale
        # Esto elimina dependencias de unidades físicas absolutas y mejora la estabilidad
        r_A = (coords - source_A) / self.source_scale
        r_B = (coords - source_B) / self.source_scale
        
        # epsilon para evitar divisiones por cero y gradientes NaN en cercanías a la fuente
        eps_grad = 1e-8
        d_A = torch.sqrt(torch.sum(r_A**2, dim=-1, keepdim=True) + eps_grad)
        d_B = torch.sqrt(torch.sum(r_B**2, dim=-1, keepdim=True) + eps_grad)
        
        eps = 1e-3
        inv_d_A = 1.0 / (d_A + eps)
        inv_d_B = 1.0 / (d_B + eps)
        
        source_norm = source_coords / self.source_scale
        
        # 3. Concatenamos todas las representaciones (enriqueciendo el espacio de entrada)
        x = torch.cat([
            ff, 
            source_norm, 
            r_A, 
            r_B, 
            d_A, 
            d_B, 
            inv_d_A, 
            inv_d_B
        ], dim=-1)
        
        # 4. Paso por la MLP residual
        u = self.mlp(x)
        return u
