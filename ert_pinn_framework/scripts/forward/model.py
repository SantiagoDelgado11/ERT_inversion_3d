import torch
import torch.nn as nn

class PINN_3D(nn.Module):
    """
    Physics-Informed Neural Network para aproximar el potencial eléctrico en 3D.
    Entradas: (x, y, z)
    Salida: u (potencial)
    """
    def __init__(self, layers=[3, 64, 64, 64, 64, 1], activation=nn.Tanh()):
        super(PINN_3D, self).__init__()
        self.activation = activation
        
        self.linears = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.linears.append(nn.Linear(layers[i], layers[i+1]))
            
        # Inicialización de pesos
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, y, z):
        # Concatenamos las coordenadas espaciales
        inputs = torch.cat([x, y, z], dim=-1)
        
        out = inputs
        for i, layer in enumerate(self.linears[:-1]):
            out = self.activation(layer(out))
            
        # Capa final sin activación (regresión lineal pura para el potencial)
        out = self.linears[-1](out)
        return out
