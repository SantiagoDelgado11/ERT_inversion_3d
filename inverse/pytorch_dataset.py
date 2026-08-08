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
        csv_filepath,
        n_pde=10000,
        n_bc_surf=2000,
        n_bc_inf=2000,
        n_flux=500,
        epsilon=0.5,
    ):
        self.csv_filepath = csv_filepath
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

        self.n_samples = 1

    def __len__(self):
        return self.n_samples

    def _load_domain_bounds(self):
        """Use the forward core mesh as the inverse collocation domain."""
        # The generated campaign uses x,y in [0, 100], z=0 at the surface,
        # and negative z below the surface.  Keep this convention even when
        # the optional forward mesh configuration is not present.
        fallback = (0.0, 100.0, 0.0, 100.0, -50.0, 0.0)
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
        
        # 70% of the points focus on the central/deep part of the actual domain.
        n_focal = num_points - n_global
        x_center = 0.5 * (xmin + xmax)
        y_center = 0.5 * (ymin + ymax)
        z_center = 0.5 * (zmin + zmax)
        x_half = 0.25 * (xmax - xmin)
        y_half = 0.25 * (ymax - ymin)
        z_half = 0.3 * (zmax - zmin)
        focal_bounds = (
            (max(xmin, x_center - x_half), min(xmax, x_center + x_half)),
            (max(ymin, y_center - y_half), min(ymax, y_center + y_half)),
            (max(zmin, z_center - z_half), min(zmax, z_center + z_half)),
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
        import pandas as pd
        df = pd.read_csv(self.csv_filepath)

        required = {
            'A_x', 'A_y', 'A_z', 'B_x', 'B_y', 'B_z',
            'M_x', 'M_y', 'M_z', 'N_x', 'N_y', 'N_z', 'V', 'Rho_a'
        }
        missing = sorted(required.difference(df.columns))
        if missing:
            raise ValueError(f"measurements.csv no contiene columnas requeridas: {missing}")

        # B is intentionally empty in a pole-dipole survey.  Reusing A as a
        # sentinel keeps all source coordinates finite; PhysicsInformer treats
        # A==B as a pole (the second electrode is at infinity).
        pole_mask = df[['B_x', 'B_y', 'B_z']].isna().all(axis=1).to_numpy()
        partial_b = df[['B_x', 'B_y', 'B_z']].isna().any(axis=1).to_numpy() & ~pole_mask
        if partial_b.any():
            raise ValueError("Hay mediciones con coordenadas B parcialmente vacias")

        a_np = df[['A_x', 'A_y', 'A_z']].to_numpy(dtype=np.float32)
        b_np = df[['B_x', 'B_y', 'B_z']].to_numpy(dtype=np.float32)
        b_np[pole_mask] = a_np[pole_mask]

        rho_a_np = df['Rho_a'].to_numpy(dtype=np.float32)
        delta_v_np = df['V'].to_numpy(dtype=np.float32)
        if not np.isfinite(delta_v_np).all() or not np.isfinite(rho_a_np).all():
            raise ValueError("measurements.csv contiene V o Rho_a no finitos")

        # Apparent resistivity is an integrated response.  Use only its
        # low-resistivity contrast as a weak anomaly indicator; fitting rho_a
        # directly at every pseudo-point was collapsing the model to its
        # survey median.  The contrast is normalized robustly and clipped so
        # outliers cannot create unbounded conductivity.
        rho_reference = float(np.median(rho_a_np))
        rho_low = float(np.percentile(rho_a_np, 10.0))
        contrast = np.clip(
            (rho_reference - rho_a_np) / max(rho_reference - rho_low, 1e-6),
            0.0,
            1.0,
        )

        elec_pos_np = np.stack([
            a_np,
            b_np,
            df[['M_x', 'M_y', 'M_z']].values,
            df[['N_x', 'N_y', 'N_z']].values
        ], axis=1)

        r_A_all = torch.tensor(elec_pos_np[:, 0, :], dtype=torch.float32)
        r_B_all = torch.tensor(elec_pos_np[:, 1, :], dtype=torch.float32)
        r_M_all = torch.tensor(elec_pos_np[:, 2, :], dtype=torch.float32)
        r_N_all = torch.tensor(elec_pos_np[:, 3, :], dtype=torch.float32)

        # Pseudo-position used only as a weak data guide.  Apparent
        # resistivity is not local resistivity, so this must not replace the
        # PDE; it only tells the inversion where the survey sees a contrast.
        midpoint_np = 0.5 * (elec_pos_np[:, 2, :] + elec_pos_np[:, 3, :])
        pseudo_depth_np = 0.5 * np.linalg.norm(a_np - midpoint_np, axis=1)
        max_depth = max(-self.z_min - 1.0, 1.0)
        pseudo_depth_np = np.clip(pseudo_depth_np, 1.0, max_depth)
        r_sigma_data = midpoint_np.copy()
        # Pseudo-depth is a positive distance; the PINN domain points down.
        r_sigma_data[:, 2] = -pseudo_depth_np

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
                "r_sigma": torch.tensor(r_sigma_data, dtype=torch.float32),
                "log_sigma_target": (
                    -torch.log(torch.tensor(rho_reference, dtype=torch.float32))
                    + torch.tensor(contrast, dtype=torch.float32).unsqueeze(1)
                    * (
                        -torch.log(torch.tensor(rho_reference / 3.0, dtype=torch.float32))
                        + torch.log(torch.tensor(rho_reference, dtype=torch.float32))
                    )
                ),
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
