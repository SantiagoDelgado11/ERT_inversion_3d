import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

class ERTDataset(Dataset):
    """
    Dataset para la Physics-Informed Neural Network (PINN).
    Lee datos de resistividad desde un HDF5 y genera dinámicamente los puntos
    de colocación (PDE, BC, Flux) requeridos para el entrenamiento físico.
    """
    def __init__(self, h5_filepath, n_pde=10000, n_bc_surf=2000, n_bc_inf=2000, n_flux=500, epsilon=0.5):
        self.h5_filepath = h5_filepath
        self.n_pde = n_pde
        self.n_bc_surf = n_bc_surf
        self.n_bc_inf = n_bc_inf
        self.n_flux = n_flux
        self.epsilon = epsilon
        
        # Dominio geológico basado en mesh.yaml (core)
        self.x_min, self.x_max = -50.0, 50.0
        self.y_min, self.y_max = -50.0, 50.0
        self.z_min, self.z_max = -50.0, 0.0
        
        # Abrimos el archivo en modo lectura y averiguamos el tamaño
        with h5py.File(self.h5_filepath, 'r') as f:
            self.n_samples = f['inputs/apparent_resistivity'].shape[0]

    def __len__(self):
        return self.n_samples
        
    def _sample_uniform(self, bounds, num_points):
        """Muestrea puntos uniformemente dentro de un cubo 3D definido por bounds"""
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
        x = torch.empty(num_points, 1).uniform_(xmin, xmax)
        y = torch.empty(num_points, 1).uniform_(ymin, ymax)
        z = torch.empty(num_points, 1).uniform_(zmin, zmax)
        return torch.cat([x, y, z], dim=1)
        
    def _sample_sphere(self, center, radius, num_points, half_sphere=True):
        """Muestrea puntos sobre la superficie de una (semi)esfera."""
        phi = torch.empty(num_points, 1).uniform_(0, 2 * np.pi)
        # Si half_sphere, z <= 0, entonces theta de pi/2 a pi
        if half_sphere:
            theta = torch.empty(num_points, 1).uniform_(np.pi/2, np.pi)
            area = 2 * np.pi * radius**2
        else:
            theta = torch.empty(num_points, 1).uniform_(0, np.pi)
            area = 4 * np.pi * radius**2
            
        x = center[0] + radius * torch.sin(theta) * torch.cos(phi)
        y = center[1] + radius * torch.sin(theta) * torch.sin(phi)
        z = center[2] + radius * torch.cos(theta)
        
        points = torch.cat([x, y, z], dim=1)
        normals = (points - center.unsqueeze(0)) / radius
        return points, normals, area

    def __getitem__(self, idx):
        # 1. Leer datos reales (data loss) de este sample
        with h5py.File(self.h5_filepath, 'r') as f:
            u_star_np = f['inputs/apparent_resistivity'][idx]  # [929]
            elec_pos_np = f['inputs/electrode_positions'][idx] # [929, 4, 3]
            
            # elec_pos_np tiene 4 electrodos: A, B, M, N
            r_A_all = torch.tensor(elec_pos_np[:, 0, :], dtype=torch.float32) # [929, 3]
            r_B_all = torch.tensor(elec_pos_np[:, 1, :], dtype=torch.float32) # [929, 3]
            r_M_all = torch.tensor(elec_pos_np[:, 2, :], dtype=torch.float32) # [929, 3]
            
            # Para los datos empíricos
            r_m = r_M_all
            u_star = torch.tensor(u_star_np, dtype=torch.float32).unsqueeze(1) # [929, 1]
            source = torch.cat([r_A_all, r_B_all], dim=1) # [929, 6]
            
            # Para evaluar la PDE, tomaremos el primer par A-B como representativo del sample
            r_A = r_A_all[0] # [3]
            r_B = r_B_all[0] # [3]

        # 2. PDE Samples
        bounds_pde = ((self.x_min, self.x_max), (self.y_min, self.y_max), (self.z_min, self.z_max))
        r_pde = self._sample_uniform(bounds_pde, self.n_pde)
        r_A_pde = r_A.unsqueeze(0).repeat(self.n_pde, 1)
        r_B_pde = r_B.unsqueeze(0).repeat(self.n_pde, 1)

        # 3. Neumann BC (z=0, excluyendo epsilon alrededor de electrodos)
        r_N = self._sample_uniform(((self.x_min, self.x_max), (self.y_min, self.y_max), (0.0, 0.0)), self.n_bc_surf)
        
        # 4. Dirichlet BC (fronteras lejanas: caras laterales y fondo)
        # Fondo (z_min)
        r_D_z = self._sample_uniform(((self.x_min, self.x_max), (self.y_min, self.y_max), (self.z_min, self.z_min)), self.n_bc_inf // 5)
        # X min/max
        r_D_x1 = self._sample_uniform(((self.x_min, self.x_min), (self.y_min, self.y_max), (self.z_min, self.z_max)), self.n_bc_inf // 5)
        r_D_x2 = self._sample_uniform(((self.x_max, self.x_max), (self.y_min, self.y_max), (self.z_min, self.z_max)), self.n_bc_inf // 5)
        # Y min/max
        r_D_y1 = self._sample_uniform(((self.x_min, self.x_max), (self.y_min, self.y_min), (self.z_min, self.z_max)), self.n_bc_inf // 5)
        r_D_y2 = self._sample_uniform(((self.x_min, self.x_max), (self.y_max, self.y_max), (self.z_min, self.z_max)), self.n_bc_inf // 5)
        r_D = torch.cat([r_D_z, r_D_x1, r_D_x2, r_D_y1, r_D_y2], dim=0)

        # 5. Flux Samples (Semiesferas alrededor de A y B en la superficie z=0)
        r_Bc_A, n_Bc_A, area_Bc = self._sample_sphere(r_A, self.epsilon, self.n_flux, half_sphere=True)
        r_Bc_B, n_Bc_B, _ = self._sample_sphere(r_B, self.epsilon, self.n_flux, half_sphere=True)

        return {
            'data': {'r_m': r_m, 'u_star': u_star, 'source': source},
            'pde': {'r': r_pde, 'r_A': r_A_pde, 'r_B': r_B_pde},
            'bc_neumann': {'r_N': r_N},
            'bc_dirichlet': {'r_D': r_D},
            'flux': {'r_Bc_A': r_Bc_A, 'n_Bc_A': n_Bc_A, 'r_Bc_B': r_Bc_B, 'n_Bc_B': n_Bc_B, 'area_Bc': area_Bc},
            'reg': {'r_reg': r_pde.clone()} # Reutilizamos puntos PDE para la penalización TV
        }
