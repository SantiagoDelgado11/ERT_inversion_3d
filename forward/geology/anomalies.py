from abc import ABC, abstractmethod
from typing import Dict, Any, Union
import numpy as np

class BaseAnomaly(ABC):
    """
    Clase base abstracta para todas las anomalías geológicas.
    Define la interfaz requerida por el generador de modelos de conductividad.
    """

    @abstractmethod
    def get_mask(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """
        Retorna una máscara booleana indicando qué puntos (X, Y, Z) pertenecen a la anomalía.
        
        Args:
            X (np.ndarray): Coordenadas X de la malla.
            Y (np.ndarray): Coordenadas Y de la malla.
            Z (np.ndarray): Coordenadas Z de la malla.
            
        Returns:
            np.ndarray: Arreglo booleano del mismo tamaño que X, Y, Z.
        """
        pass

    @property
    @abstractmethod
    def conductivity(self) -> float:
        """
        Retorna la conductividad de la anomalía en S/m.
        """
        pass

    def metadata(self) -> Dict[str, Any]:
        """
        Retorna la metadata asociada a la anomalía para ser serializada (e.g. HDF5).
        Por defecto retorna un diccionario vacío si no está implementada.
        """
        return {}


class LegacyAnomaly(BaseAnomaly):
    """
    Clase base para anomalías antiguas que fueron diseñadas almacenando resistividad.
    Se mantiene para compatibilidad con código existente, pero expone internamente conductividad.
    """
    def __init__(self, resistivity: float):
        if resistivity <= 0 or not np.isfinite(resistivity):
            raise ValueError(f"Resistivity must be positive and finite, got {resistivity}")
        self.resistivity = resistivity

    @property
    def conductivity(self) -> float:
        """
        Calcula la conductividad (S/m) al vuelo basándose en la resistividad almacenada.
        Evita duplicar estado en el objeto.
        """
        return 1.0 / self.resistivity


class Sphere(LegacyAnomaly):
    """Anomalía esférica legacy (usa resistividad)."""
    def __init__(self, resistivity: float, cx: float, cy: float, cz: float, radius: float):
        super().__init__(resistivity)
        self.cx = cx
        self.cy = cy
        self.cz = cz
        self.radius = radius

    def get_mask(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        dist_sq = (X - self.cx)**2 + (Y - self.cy)**2 + (Z - self.cz)**2
        return dist_sq <= self.radius**2


class Ellipsoid(LegacyAnomaly):
    """Anomalía elipsoidal legacy (usa resistividad)."""
    def __init__(self, resistivity: float, cx: float, cy: float, cz: float, rx: float, ry: float, rz: float):
        super().__init__(resistivity)
        self.cx = cx
        self.cy = cy
        self.cz = cz
        self.rx = rx
        self.ry = ry
        self.rz = rz

    def get_mask(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        dist_sq = ((X - self.cx)**2 / self.rx**2 + 
                   (Y - self.cy)**2 / self.ry**2 + 
                   (Z - self.cz)**2 / self.rz**2)
        return dist_sq <= 1.0


class Block(LegacyAnomaly):
    """Anomalía rectangular (bloque) legacy (usa resistividad)."""
    def __init__(self, resistivity: float, x_min: float, x_max: float, y_min: float, y_max: float, z_min: float, z_max: float):
        super().__init__(resistivity)
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max

    def get_mask(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        mask_x = (X >= self.x_min) & (X <= self.x_max)
        mask_y = (Y >= self.y_min) & (Y <= self.y_max)
        mask_z = (Z >= self.z_min) & (Z <= self.z_max)
        return mask_x & mask_y & mask_z


class SphereAnomaly(BaseAnomaly):
    """
    Clase para representar una única anomalía esférica.
    Opera internamente de manera exclusiva con conductividad (S/m), 
    la magnitud física nativa del forward solver. No hereda la deuda técnica de LegacyAnomaly.
    """
    def __init__(self, conductivity: float, cx: float, cy: float, cz: float, radius: float):
        if conductivity <= 0 or not np.isfinite(conductivity):
            raise ValueError(f"La conductividad debe ser positiva y finita, se recibió {conductivity}")
        if radius <= 0 or not np.isfinite(radius):
            raise ValueError(f"El radio debe ser positivo y finito, se recibió {radius}")
        if not np.all(np.isfinite([cx, cy, cz])):
            raise ValueError("Las coordenadas del centro deben ser valores finitos")
            
        self._conductivity = conductivity
        self.cx = cx
        self.cy = cy
        self.cz = cz
        self.radius = radius

    @property
    def conductivity(self) -> float:
        """Retorna la conductividad almacenada en S/m."""
        return self._conductivity

    def get_mask(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Identifica celdas de la malla contenidas dentro de la esfera."""
        dist_sq = (X - self.cx)**2 + (Y - self.cy)**2 + (Z - self.cz)**2
        return dist_sq <= self.radius**2

    def metadata(self) -> Dict[str, Union[str, float, tuple]]:
        """
        Retorna la metadata para inicializar la serialización en HDF5.
        
        Returns:
            Dict: Propiedades físicas y geométricas de la esfera.
        """
        return {
            "type": "sphere",
            "radius": self.radius,
            "conductivity": self.conductivity,
            "center": (self.cx, self.cy, self.cz)
        }
