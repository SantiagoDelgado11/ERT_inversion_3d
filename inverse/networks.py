import torch
import torch.nn as nn
import torch.nn.functional as F

class FourierFeatureMapping(nn.Module):
    """
    Mapeo de Características de Fourier para superar el sesgo espectral (Spectral Bias)
    y permitir el aprendizaje de altas frecuencias (bordes afilados de anomalías).
    """
    def __init__(self, in_features, mapping_size, scale=1.0):
        super().__init__()
        self.in_features = in_features
        self.mapping_size = mapping_size
        self.scale = scale
        # Matriz B estática para el positional encoding
        self.register_buffer('B', torch.randn(in_features, mapping_size) * scale)
        
    def forward(self, x):
        # x: (batch_size, in_features)
        x_proj = (2.0 * torch.pi * x) @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

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

class ConductivityNet(nn.Module):
    """
    Red Neuronal sigma_theta que mapea coordenadas espaciales a conductividad local.
    """
    def __init__(self, fourier_features=256, fourier_scale=1.0, hidden_layers=4, hidden_dim=128):
        super().__init__()
        # Mapeo de (x,y,z)
        self.fourier_map = FourierFeatureMapping(in_features=3, mapping_size=fourier_features, scale=fourier_scale)
        
        # El mapeo de Fourier concatena sin y cos, por lo que la entrada es 2 * fourier_features
        self.mlp = MLP(in_dim=2 * fourier_features, 
                       hidden_layers=hidden_layers, 
                       hidden_dim=hidden_dim, 
                       out_dim=1, 
                       activation=nn.SiLU) # Estrictamente SiLU (no ReLU)

    def forward(self, coords):
        """
        coords: Tensor de coordenadas (batch_size, 3)
        """
        ff = self.fourier_map(coords)
        out = self.mlp(ff)
        # Forzamos conductividad estrictamente positiva
        sigma = F.softplus(out)
        return sigma

class PotentialNet(nn.Module):
    """
    Red Neuronal u_phi que mapea coordenadas espaciales y fuentes a potencial eléctrico.
    """
    def __init__(self, fourier_features=256, fourier_scale=1.0, hidden_layers=5, hidden_dim=256):
        super().__init__()
        # Mapeo de (x,y,z)
        self.fourier_map = FourierFeatureMapping(in_features=3, mapping_size=fourier_features, scale=fourier_scale)
        
        # Entrada: 2*fourier_features de coords + 6 de coordenadas fuente (xA,yA,zA, xB,yB,zB)
        self.mlp = MLP(in_dim=2 * fourier_features + 6,
                       hidden_layers=hidden_layers,
                       hidden_dim=hidden_dim,
                       out_dim=1,
                       activation=nn.SiLU) # Estrictamente SiLU (no ReLU)

    def forward(self, coords, source_coords):
        """
        coords: (batch_size, 3) coordenadas de evaluación (x, y, z)
        source_coords: (batch_size, 6) posiciones de dipolo inyector (xA, yA, zA, xB, yB, zB)
        """
        ff = self.fourier_map(coords)
        x = torch.cat([ff, source_coords], dim=-1)
        u = self.mlp(x)
        return u
