"""Acquisition geometry arrays module for 3D ERT.

This module provides vectorised implementations for various ERT array configurations
including Pole-Pole, Collinear Dipole-Dipole, Equatorial Dipole-Dipole, and L-Corner arrays
adapted for 3D grids.
"""

from __future__ import annotations

import torch
from torch import Tensor


class ERTGeometry:
    """Base class for robust, vectorized 3D ERT geometries."""

    def __init__(self, electrodes: Tensor):
        """
        Args:
            electrodes: (N, 3) tensor of electrode spatial coordinates (x, y, z).
        """
        self.electrodes = electrodes
        self.num_electrodes = electrodes.shape[0]

    def compute_distances(self, a_idx: Tensor, b_idx: Tensor, m_idx: Tensor, n_idx: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Calculates vectorized Euclidean distances r_AM, r_BM, r_AN, r_BN.
        Electrodes at infinity are represented by an index < 0.

        Args:
            a_idx, b_idx, m_idx, n_idx: 1D tensors of electrode indices.

        Returns:
            Tuple of distance tensors (r_AM, r_BM, r_AN, r_BN).
        """
        def get_pos(idx: Tensor) -> Tensor:
            # Map negative indices (infinity) to index 0 just to avoid out-of-bounds error
            # We will logically mask the distances out during K calculation.
            valid_idx = torch.where(idx >= 0, idx, torch.zeros_like(idx))
            return self.electrodes[valid_idx]

        a_pos = get_pos(a_idx)
        b_pos = get_pos(b_idx)
        m_pos = get_pos(m_idx)
        n_pos = get_pos(n_idx)

        r_am = torch.linalg.vector_norm(a_pos - m_pos, dim=-1)
        r_bm = torch.linalg.vector_norm(b_pos - m_pos, dim=-1)
        r_an = torch.linalg.vector_norm(a_pos - n_pos, dim=-1)
        r_bn = torch.linalg.vector_norm(b_pos - n_pos, dim=-1)

        return r_am, r_bm, r_an, r_bn

    def compute_geometric_factor(self, a_idx: Tensor, b_idx: Tensor, m_idx: Tensor, n_idx: Tensor) -> Tensor:
        """
        Calculates the geometric factor K = 2*pi / (1/r_AM - 1/r_BM - 1/r_AN + 1/r_BN).
        Electrodes with index < 0 are assumed to be at infinity (their 1/r contribution is 0).

        Args:
            a_idx, b_idx, m_idx, n_idx: 1D tensors of electrode indices.

        Returns:
            1D tensor of geometric factors (K) for each configuration.
        """
        r_am, r_bm, r_an, r_bn = self.compute_distances(a_idx, b_idx, m_idx, n_idx)

        def inv_r(r: Tensor, idx1: Tensor, idx2: Tensor) -> Tensor:
            mask = (idx1 < 0) | (idx2 < 0)
            # Avoid dividing by zero where the distance is legitimately close to zero
            # (which shouldn't happen unless electrodes overlap)
            safe_r = torch.where(r < 1e-12, torch.ones_like(r), r)
            return torch.where(mask, torch.zeros_like(r), 1.0 / safe_r)

        term1 = inv_r(r_am, a_idx, m_idx)
        term2 = inv_r(r_bm, b_idx, m_idx)
        term3 = inv_r(r_an, a_idx, n_idx)
        term4 = inv_r(r_bn, b_idx, n_idx)

        inv_k = term1 - term2 - term3 + term4
        k = torch.where(
            torch.abs(inv_k) > 1e-12, 
            2 * torch.pi / inv_k, 
            torch.inf * torch.ones_like(inv_k)
        )
        return k

    def generate_indices(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Generates indices (A, B, M, N) for the specific geometry."""
        raise NotImplementedError


class GridERTGeometry(ERTGeometry):
    """Base class for geometries that depend on a 2D regular grid layout of electrodes."""

    def __init__(self, electrodes: Tensor, nx: int, ny: int):
        super().__init__(electrodes)
        self.nx = nx
        self.ny = ny
        if nx * ny != self.num_electrodes:
            raise ValueError(f"Grid dimensions nx*ny ({nx}*{ny}) must match number of electrodes ({self.num_electrodes}).")

    def get_index(self, x: int, y: int) -> int:
        """Translates 2D grid logical coordinates into a 1D tensor index."""
        return y * self.nx + x


class PolePoleArray(ERTGeometry):
    """
    Pole-Pole Array.
    A and M iterate over the grid. B and N are at infinity.
    """

    def generate_indices(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        idx = torch.arange(self.num_electrodes, device=self.electrodes.device)
        grid_a, grid_m = torch.meshgrid(idx, idx, indexing='ij')
        a_flat = grid_a.flatten()
        m_flat = grid_m.flatten()

        mask = a_flat != m_flat
        a = a_flat[mask]
        m = m_flat[mask]
        b = torch.full_like(a, -1)
        n = torch.full_like(a, -1)
        return a, b, m, n


class Collinear3DDipoleDipoleArray(GridERTGeometry):
    """
    Collinear Dipole-Dipole array expanded for 3D grids.
    Can iterate over X-lines and Y-lines.
    """

    def generate_indices(self, a_spacing: int = 1, max_n: int = 6) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        a_idx, b_idx, m_idx, n_idx = [], [], [], []
        
        # Iteration over X-lines
        for y in range(self.ny):
            for x in range(self.nx):
                for n in range(1, max_n + 1):
                    A_x = x
                    B_x = x + a_spacing
                    M_x = B_x + n * a_spacing
                    N_x = M_x + a_spacing
                    if N_x < self.nx:
                        a_idx.append(self.get_index(A_x, y))
                        b_idx.append(self.get_index(B_x, y))
                        m_idx.append(self.get_index(M_x, y))
                        n_idx.append(self.get_index(N_x, y))

        # Iteration over Y-lines
        for x in range(self.nx):
            for y in range(self.ny):
                for n in range(1, max_n + 1):
                    A_y = y
                    B_y = y + a_spacing
                    M_y = B_y + n * a_spacing
                    N_y = M_y + a_spacing
                    if N_y < self.ny:
                        a_idx.append(self.get_index(x, A_y))
                        b_idx.append(self.get_index(x, B_y))
                        m_idx.append(self.get_index(x, M_y))
                        n_idx.append(self.get_index(x, N_y))

        device = self.electrodes.device
        return (torch.tensor(a_idx, device=device), 
                torch.tensor(b_idx, device=device), 
                torch.tensor(m_idx, device=device), 
                torch.tensor(n_idx, device=device))


class EquatorialDipoleDipoleArray(GridERTGeometry):
    """
    Equatorial (Cross-line) Dipole-Dipole Array.
    A and B form a dipole along a line. M and N form a dipole on an adjacent line.
    """

    def generate_indices(self, a_spacing: int = 1, dy_lines: int = 1) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        a_idx, b_idx, m_idx, n_idx = [], [], [], []
        
        for y in range(self.ny - dy_lines):
            y_ab = y
            y_mn = y + dy_lines
            for x_ab in range(self.nx - a_spacing):
                for x_mn in range(self.nx - a_spacing):
                    A_x, B_x = x_ab, x_ab + a_spacing
                    M_x, N_x = x_mn, x_mn + a_spacing
                    
                    a_idx.append(self.get_index(A_x, y_ab))
                    b_idx.append(self.get_index(B_x, y_ab))
                    m_idx.append(self.get_index(M_x, y_mn))
                    n_idx.append(self.get_index(N_x, y_mn))

        device = self.electrodes.device
        return (torch.tensor(a_idx, device=device), 
                torch.tensor(b_idx, device=device), 
                torch.tensor(m_idx, device=device), 
                torch.tensor(n_idx, device=device))


class LCornerArray(GridERTGeometry):
    """
    L-Corner Array for 3D grids.
    A, B are aligned on one axis and M, N are aligned on the perpendicular axis
    radiating from a specific corner/intersection point.
    """

    def generate_indices(self, a_spacing: int = 1, max_n: int = 6) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        a_idx, b_idx, m_idx, n_idx = [], [], [], []
        
        for cy in range(self.ny):
            for cx in range(self.nx):
                for n_x in range(1, max_n + 1):
                    for n_y in range(1, max_n + 1):
                        A_x = cx + n_x * a_spacing
                        B_x = A_x + a_spacing
                        
                        M_y = cy + n_y * a_spacing
                        N_y = M_y + a_spacing
                        
                        if B_x < self.nx and N_y < self.ny:
                            a_idx.append(self.get_index(A_x, cy))
                            b_idx.append(self.get_index(B_x, cy))
                            m_idx.append(self.get_index(cx, M_y))
                            n_idx.append(self.get_index(cx, N_y))

        device = self.electrodes.device
        return (torch.tensor(a_idx, device=device), 
                torch.tensor(b_idx, device=device), 
                torch.tensor(m_idx, device=device), 
                torch.tensor(n_idx, device=device))
