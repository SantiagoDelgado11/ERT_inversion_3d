import h5py
import numpy as np
import torch
import yaml
from pathlib import Path
from torch.utils.data import Dataset


class ERTDataset(Dataset):
    """
    Dataset for the ERT PINN.

    The potential network is trained against voltage differences, not apparent
    resistivity. Existing HDF5 files may only store rho_a, so DeltaV is recovered
    from the geometric factor K when needed: DeltaV = rho_a / K for I = 1 A.
    """

    def __init__(
        self,
        h5_filepath,
        n_pde=10000,
        n_bc_surf=2000,
        n_bc_inf=2000,
        n_flux=500,
        epsilon=0.5,
    ):
        self.h5_filepath = h5_filepath
        self.n_pde = n_pde
        self.n_bc_surf = n_bc_surf
        self.n_bc_inf = n_bc_inf
        self.n_flux = n_flux
        self.epsilon = epsilon

        (
            self.x_min,
            self.x_max,
            self.y_min,
            self.y_max,
            self.z_min,
            self.z_max,
        ) = self._load_domain_bounds()

        with h5py.File(self.h5_filepath, "r", swmr=True) as f:
            self.n_samples = f["inputs/apparent_resistivity"].shape[0]

    def __len__(self):
        return self.n_samples

    def _load_domain_bounds(self):
        """Use the forward core mesh as the inverse collocation domain."""
        fallback = (-50.0, 50.0, -20.0, 20.0, -40.0, 0.0)
        config_path = Path(__file__).resolve().parents[1] / "forward" / "configs" / "mesh.yaml"
        try:
            with open(config_path, "r") as f:
                mesh_config = yaml.safe_load(f)["mesh"]
            x_half = 0.5 * mesh_config["nx"] * mesh_config["hx"]
            y_half = 0.5 * mesh_config["ny"] * mesh_config["hy"]
            z_depth = mesh_config["nz"] * mesh_config["hz"]
            return (-x_half, x_half, -y_half, y_half, -z_depth, 0.0)
        except Exception:
            return fallback

    @staticmethod
    def _calculate_geometric_factor(pos_A, pos_B, pos_M, pos_N):
        r_AM = np.linalg.norm(pos_A - pos_M)
        r_AN = np.linalg.norm(pos_A - pos_N)
        r_BM = np.linalg.norm(pos_B - pos_M)
        r_BN = np.linalg.norm(pos_B - pos_N)

        term_AM = 1.0 / r_AM if r_AM > 0 else 0.0
        term_AN = 1.0 / r_AN if r_AN > 0 else 0.0
        term_BM = 1.0 / r_BM if r_BM > 0 else 0.0
        term_BN = 1.0 / r_BN if r_BN > 0 else 0.0
        return 2 * np.pi / (term_AM - term_BM - term_AN + term_BN)

    def _sample_uniform(self, bounds, num_points):
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
        x = torch.empty(num_points, 1).uniform_(xmin, xmax)
        y = torch.empty(num_points, 1).uniform_(ymin, ymax)
        z = torch.empty(num_points, 1).uniform_(zmin, zmax)
        return torch.cat([x, y, z], dim=1)

    def _sample_pde_stratified(self, bounds, num_points):
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
        
        # 30% puntos globales uniformes
        n_global = int(0.3 * num_points)
        global_pts = self._sample_uniform(bounds, n_global)
        
        # 70% puntos enfocados en el volumen central profundo (zona de interés)
        n_focal = num_points - n_global
        focal_bounds = (
            (max(xmin, -25.0), min(xmax, 25.0)),
            (max(ymin, -25.0), min(ymax, 25.0)),
            (max(zmin, -30.0), min(zmax, -2.0)) # Z profundo
        )
        focal_pts = self._sample_uniform(focal_bounds, n_focal)
        
        return torch.cat([global_pts, focal_pts], dim=0)

    def _sample_source_coords(self, source_pool, num_points):
        idx = torch.randint(0, source_pool.shape[0], (num_points,))
        return source_pool[idx]

    def _sample_surface_excluding_electrodes(self, electrode_positions, num_points, exclusion_radius):
        if num_points == 0:
            return torch.empty(0, 3)

        electrodes = torch.tensor(electrode_positions, dtype=torch.float32)
        in_domain = (
            (electrodes[:, 0] >= self.x_min)
            & (electrodes[:, 0] <= self.x_max)
            & (electrodes[:, 1] >= self.y_min)
            & (electrodes[:, 1] <= self.y_max)
            & torch.isclose(electrodes[:, 2], torch.zeros_like(electrodes[:, 2]), atol=1e-5)
        )
        electrodes = electrodes[in_domain]
        if electrodes.numel() == 0:
            return self._sample_uniform(
                ((self.x_min, self.x_max), (self.y_min, self.y_max), (0.0, 0.0)),
                num_points,
            )

        accepted = []
        attempts = 0
        while sum(chunk.shape[0] for chunk in accepted) < num_points and attempts < 20:
            attempts += 1
            candidates = self._sample_uniform(
                ((self.x_min, self.x_max), (self.y_min, self.y_max), (0.0, 0.0)),
                max(num_points * 4, 128),
            )
            distances = torch.cdist(candidates[:, :2], electrodes[:, :2])
            keep = distances.min(dim=1).values > exclusion_radius
            accepted.append(candidates[keep])

        if accepted:
            points = torch.cat(accepted, dim=0)
            if points.shape[0] >= num_points:
                return points[:num_points]

        return self._sample_uniform(
            ((self.x_min, self.x_max), (self.y_min, self.y_max), (0.0, 0.0)),
            num_points,
        )

    def _sample_spheres_for_sources(self, source_coords, radius, half_sphere=True):
        num_points = source_coords.shape[0]
        phi = torch.empty(num_points, 1).uniform_(0, 2 * np.pi)
        if half_sphere:
            theta = torch.empty(num_points, 1).uniform_(np.pi / 2, np.pi)
            area = 2 * np.pi * radius**2
        else:
            theta = torch.empty(num_points, 1).uniform_(0, np.pi)
            area = 4 * np.pi * radius**2

        normals = torch.cat(
            [
                torch.sin(theta) * torch.cos(phi),
                torch.sin(theta) * torch.sin(phi),
                torch.cos(theta),
            ],
            dim=1,
        )

        center_A = source_coords[:, 0:3]
        center_B = source_coords[:, 3:6]
        return center_A + radius * normals, normals, center_B + radius * normals, normals, area

    def __getitem__(self, idx):
        with h5py.File(self.h5_filepath, "r", swmr=True) as f:
            rho_a_np = f["inputs/apparent_resistivity"][idx]
            elec_pos_np = f["inputs/electrode_positions"][idx]
            if "inputs/delta_v" in f:
                delta_v_np = f["inputs/delta_v"][idx]
            else:
                k_values = np.array(
                    [
                        self._calculate_geometric_factor(row[0], row[1], row[2], row[3])
                        for row in elec_pos_np
                    ],
                    dtype=np.float32,
                )
                delta_v_np = rho_a_np / k_values

        r_A_all = torch.tensor(elec_pos_np[:, 0, :], dtype=torch.float32)
        r_B_all = torch.tensor(elec_pos_np[:, 1, :], dtype=torch.float32)
        r_M_all = torch.tensor(elec_pos_np[:, 2, :], dtype=torch.float32)
        r_N_all = torch.tensor(elec_pos_np[:, 3, :], dtype=torch.float32)

        source = torch.cat([r_A_all, r_B_all], dim=1)
        source_pool = torch.tensor(np.unique(source.numpy(), axis=0), dtype=torch.float32)
        electrode_pool = np.unique(elec_pos_np.reshape(-1, 3), axis=0)

        bounds_pde = ((self.x_min, self.x_max), (self.y_min, self.y_max), (self.z_min, self.z_max))
        r_pde = self._sample_pde_stratified(bounds_pde, self.n_pde)
        source_pde = self._sample_source_coords(source_pool, self.n_pde)

        r_neumann = self._sample_surface_excluding_electrodes(
            electrode_pool,
            self.n_bc_surf,
            self.epsilon,
        )
        source_neumann = self._sample_source_coords(source_pool, r_neumann.shape[0])

        n_face = self.n_bc_inf // 5
        r_D_z = self._sample_uniform(((self.x_min, self.x_max), (self.y_min, self.y_max), (self.z_min, self.z_min)), n_face)
        r_D_x1 = self._sample_uniform(((self.x_min, self.x_min), (self.y_min, self.y_max), (self.z_min, self.z_max)), n_face)
        r_D_x2 = self._sample_uniform(((self.x_max, self.x_max), (self.y_min, self.y_max), (self.z_min, self.z_max)), n_face)
        r_D_y1 = self._sample_uniform(((self.x_min, self.x_max), (self.y_min, self.y_min), (self.z_min, self.z_max)), n_face)
        r_D_y2 = self._sample_uniform(((self.x_min, self.x_max), (self.y_max, self.y_max), (self.z_min, self.z_max)), n_face)
        r_dirichlet = torch.cat([r_D_z, r_D_x1, r_D_x2, r_D_y1, r_D_y2], dim=0)
        source_dirichlet = self._sample_source_coords(source_pool, r_dirichlet.shape[0])

        source_flux = self._sample_source_coords(source_pool, self.n_flux)
        r_Bc_A, n_Bc_A, r_Bc_B, n_Bc_B, area_Bc = self._sample_spheres_for_sources(
            source_flux,
            self.epsilon,
            half_sphere=True,
        )

        delta_v = torch.tensor(delta_v_np, dtype=torch.float32).unsqueeze(1)
        rho_a = torch.tensor(rho_a_np, dtype=torch.float32).unsqueeze(1)

        return {
            "data": {
                "r_m": r_M_all,
                "r_n": r_N_all,
                "delta_v": delta_v,
                "u_star": delta_v,
                "apparent_resistivity": rho_a,
                "source": source,
            },
            "pde": {
                "r": r_pde,
                "r_A": source_pde[:, 0:3],
                "r_B": source_pde[:, 3:6],
                "source": source_pde,
            },
            "bc_neumann": {"r_N": r_neumann, "source": source_neumann},
            "bc_dirichlet": {"r_D": r_dirichlet, "source": source_dirichlet},
            "flux": {
                "r_Bc_A": r_Bc_A,
                "n_Bc_A": n_Bc_A,
                "r_Bc_B": r_Bc_B,
                "n_Bc_B": n_Bc_B,
                "source_A": source_flux,
                "source_B": source_flux,
                "area_Bc": area_Bc,
            },
            "reg": {"r_reg": self._sample_uniform(bounds_pde, self.n_pde)},
        }
