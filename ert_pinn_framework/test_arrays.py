import torch
import math
from src.acquisition.arrays import (
    PolePoleArray,
    Collinear3DDipoleDipoleArray,
    EquatorialDipoleDipoleArray,
    LCornerArray
)

def dummy_grid_electrodes():
    # A 5x5 grid with spacing 1
    nx, ny = 5, 5
    coords = []
    for y in range(ny):
        for x in range(nx):
            coords.append([float(x), float(y), 0.0])
    return torch.tensor(coords), nx, ny

def test_pole_pole_geometry():
    electrodes, _, _ = dummy_grid_electrodes()
    geom = PolePoleArray(electrodes)
    
    a_idx, b_idx, m_idx, n_idx = geom.generate_indices()
    assert a_idx.shape == b_idx.shape == m_idx.shape == n_idx.shape
    assert torch.all(b_idx == -1)
    assert torch.all(n_idx == -1)
    
    k = geom.compute_geometric_factor(a_idx, b_idx, m_idx, n_idx)
    r_am = torch.linalg.norm(electrodes[a_idx] - electrodes[m_idx], dim=-1)
    expected_k = 2 * math.pi * r_am
    assert torch.allclose(k, expected_k, atol=1e-5)
    print("Pole-Pole passed.")

def test_collinear_dipole_dipole():
    electrodes, nx, ny = dummy_grid_electrodes()
    geom = Collinear3DDipoleDipoleArray(electrodes, nx, ny)
    
    a_idx, b_idx, m_idx, n_idx = geom.generate_indices(a_spacing=1, max_n=2)
    k = geom.compute_geometric_factor(a_idx, b_idx, m_idx, n_idx)
    
    r_am = torch.linalg.norm(electrodes[a_idx] - electrodes[m_idx], dim=-1)
    r_bm = torch.linalg.norm(electrodes[b_idx] - electrodes[m_idx], dim=-1)
    r_an = torch.linalg.norm(electrodes[a_idx] - electrodes[n_idx], dim=-1)
    r_bn = torch.linalg.norm(electrodes[b_idx] - electrodes[n_idx], dim=-1)
    
    inv_k_manual = 1.0/r_am - 1.0/r_bm - 1.0/r_an + 1.0/r_bn
    expected_k = 2 * math.pi / inv_k_manual
    assert torch.allclose(k, expected_k, atol=1e-5)
    print("Collinear Dipole-Dipole passed.")

def test_equatorial_dipole_dipole():
    electrodes, nx, ny = dummy_grid_electrodes()
    geom = EquatorialDipoleDipoleArray(electrodes, nx, ny)
    
    a_idx, b_idx, m_idx, n_idx = geom.generate_indices(a_spacing=1, dy_lines=1)
    k = geom.compute_geometric_factor(a_idx, b_idx, m_idx, n_idx)
    
    r_am = torch.linalg.norm(electrodes[a_idx] - electrodes[m_idx], dim=-1)
    r_bm = torch.linalg.norm(electrodes[b_idx] - electrodes[m_idx], dim=-1)
    r_an = torch.linalg.norm(electrodes[a_idx] - electrodes[n_idx], dim=-1)
    r_bn = torch.linalg.norm(electrodes[b_idx] - electrodes[n_idx], dim=-1)
    
    inv_k_manual = 1.0/r_am - 1.0/r_bm - 1.0/r_an + 1.0/r_bn
    expected_k = 2 * math.pi / inv_k_manual
    assert torch.allclose(k, expected_k, atol=1e-5)
    print("Equatorial Dipole-Dipole passed.")

def test_l_corner_array():
    electrodes, nx, ny = dummy_grid_electrodes()
    geom = LCornerArray(electrodes, nx, ny)
    
    a_idx, b_idx, m_idx, n_idx = geom.generate_indices(a_spacing=1, max_n=2)
    k = geom.compute_geometric_factor(a_idx, b_idx, m_idx, n_idx)
    
    r_am = torch.linalg.norm(electrodes[a_idx] - electrodes[m_idx], dim=-1)
    r_bm = torch.linalg.norm(electrodes[b_idx] - electrodes[m_idx], dim=-1)
    r_an = torch.linalg.norm(electrodes[a_idx] - electrodes[n_idx], dim=-1)
    r_bn = torch.linalg.norm(electrodes[b_idx] - electrodes[n_idx], dim=-1)
    
    inv_k_manual = 1.0/r_am - 1.0/r_bm - 1.0/r_an + 1.0/r_bn
    expected_k = 2 * math.pi / inv_k_manual
    assert torch.allclose(k, expected_k, atol=1e-5)
    print("L-Corner passed.")

if __name__ == "__main__":
    test_pole_pole_geometry()
    test_collinear_dipole_dipole()
    test_equatorial_dipole_dipole()
    test_l_corner_array()
    print("All tests passed!")
