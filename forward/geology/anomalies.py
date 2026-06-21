import numpy as np

class Anomaly:
    def __init__(self, resistivity):
        self.resistivity = resistivity

    def get_mask(self, X, Y, Z):
        raise NotImplementedError

class Sphere(Anomaly):
    def __init__(self, resistivity, cx, cy, cz, radius):
        super().__init__(resistivity)
        self.cx = cx
        self.cy = cy
        self.cz = cz
        self.radius = radius

    def get_mask(self, X, Y, Z):
        dist_sq = (X - self.cx)**2 + (Y - self.cy)**2 + (Z - self.cz)**2
        return dist_sq <= self.radius**2

class Ellipsoid(Anomaly):
    def __init__(self, resistivity, cx, cy, cz, rx, ry, rz):
        super().__init__(resistivity)
        self.cx = cx
        self.cy = cy
        self.cz = cz
        self.rx = rx
        self.ry = ry
        self.rz = rz

    def get_mask(self, X, Y, Z):
        dist_sq = ((X - self.cx)**2 / self.rx**2 + 
                   (Y - self.cy)**2 / self.ry**2 + 
                   (Z - self.cz)**2 / self.rz**2)
        return dist_sq <= 1.0

class Block(Anomaly):
    def __init__(self, resistivity, x_min, x_max, y_min, y_max, z_min, z_max):
        super().__init__(resistivity)
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max

    def get_mask(self, X, Y, Z):
        mask_x = (X >= self.x_min) & (X <= self.x_max)
        mask_y = (Y >= self.y_min) & (Y <= self.y_max)
        mask_z = (Z >= self.z_min) & (Z <= self.z_max)
        return mask_x & mask_y & mask_z
