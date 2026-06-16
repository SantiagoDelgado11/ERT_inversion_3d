"""Analytical conductivity network for the ERT PINN to inject synthetic anomalies."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class AnalyticalConductivity(nn.Module):
    def __init__(self, background_sigma: float, anomalies: list[dict]):
        """Initialize the analytical conductivity model.
        
        Args:
            background_sigma: The base background conductivity value.
            anomalies: A list of dictionaries defining anomalies.
        """
        super().__init__()
        self.background_sigma = float(background_sigma)
        self.anomalies = anomalies

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate the analytical conductivity at points x.
        
        Args:
            x: Tensor of shape (N, 3) with coordinates [x, y, z].
            
        Returns:
            Tensor of shape (N, 1) with conductivity values.
        """
        sigma = torch.full((x.shape[0], 1), self.background_sigma, device=x.device, dtype=x.dtype)
        
        for anomaly in self.anomalies:
            shape_type = anomaly.get("type", "sphere")
            val = float(anomaly.get("value", 1.0))
            sharpness = float(anomaly.get("sharpness", 50.0))
            
            if shape_type == "sphere":
                center = torch.tensor(anomaly["center"], device=x.device, dtype=x.dtype)
                radius = float(anomaly["radius"])
                dist = torch.norm(x - center.view(1, -1), dim=1, keepdim=True)
                
                # sigmoid((radius - dist) * sharpness) -> 1 inside, 0 outside
                mask = torch.sigmoid((radius - dist) * sharpness)
                sigma = mask * val + (1.0 - mask) * sigma
                
            elif shape_type == "box":
                mins = torch.tensor(anomaly["min"], device=x.device, dtype=x.dtype).view(1, -1)
                maxs = torch.tensor(anomaly["max"], device=x.device, dtype=x.dtype).view(1, -1)
                
                mask_min = torch.sigmoid((x - mins) * sharpness)
                mask_max = torch.sigmoid((maxs - x) * sharpness)
                
                # Product over all 3 spatial dimensions
                mask = torch.prod(mask_min * mask_max, dim=1, keepdim=True)
                sigma = mask * val + (1.0 - mask) * sigma
                
            elif shape_type == "layer":
                axis = int(anomaly.get("axis", 2)) # 0=x, 1=y, 2=z
                # Allow infinity defaults if one boundary is missing
                z_min = float(anomaly.get("min", -1e9))
                z_max = float(anomaly.get("max", 1e9))
                
                x_axis = x[:, axis:axis+1]
                mask_min = torch.sigmoid((x_axis - z_min) * sharpness)
                mask_max = torch.sigmoid((z_max - x_axis) * sharpness)
                
                mask = mask_min * mask_max
                sigma = mask * val + (1.0 - mask) * sigma
                
            else:
                raise ValueError(f"Unknown anomaly type: {shape_type}")
                
        return sigma
