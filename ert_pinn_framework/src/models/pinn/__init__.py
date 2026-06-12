"""PINN model components."""

from .electric_potential_network import PotentialNet
from .electrical_conductivity_network import ConductivityNet

__all__ = ["ConductivityNet", "PotentialNet"]
