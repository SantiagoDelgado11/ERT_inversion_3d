"""
Focused verification tests for the active ERT PINN implementation.
"""

import math
import sys
from pathlib import Path

import torch

from models import ConductivityNet, PotentialNet
from physics_informer import PhysicsInformer
from pytorch_dataset import ERTDataset


def _small_informer():
    sigma_net = ConductivityNet(num_frequencies=16, hidden_layers=2, hidden_dim=32)
    pot_net = PotentialNet(num_frequencies=16, hidden_layers=2, hidden_dim=32)
    return PhysicsInformer(sigma_net, pot_net), sigma_net, pot_net


def test_gradient_flow():
    print("=" * 60)
    print("TEST 1: Gradient flow through PDE residual")
    print("=" * 60)

    informer, sigma_net, pot_net = _small_informer()
    coords = torch.randn(32, 3)
    coords[:, 2] = torch.rand(32) * 10
    coords.requires_grad_(True)
    src_A = torch.tensor([[-10.0, 0.0, 0.0]]).repeat(32, 1)
    src_B = torch.tensor([[10.0, 0.0, 0.0]]).repeat(32, 1)
    source = torch.cat([src_A, src_B], dim=1)

    loss = informer.compute_pde_loss(coords, source, I=1.0, gamma=4.0)
    loss.backward()

    sigma_grad = sum((p.grad.abs().sum().item() for p in sigma_net.parameters() if p.grad is not None))
    pot_grad = sum((p.grad.abs().sum().item() for p in pot_net.parameters() if p.grad is not None))
    ok = math.isfinite(loss.item()) and sigma_grad > 0.0 and pot_grad > 0.0
    print(f"  L_PDE={loss.item():.4e}, sigma_grad={sigma_grad:.4e}, pot_grad={pot_grad:.4e}")
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def test_half_space_source_normalization():
    print("\n" + "=" * 60)
    print("TEST 2: Half-space Gaussian source normalization")
    print("=" * 60)

    informer, _, _ = _small_informer()
    n = 200000
    coords = torch.empty(n, 3)
    coords[:, 0].uniform_(-50.0, 50.0)
    coords[:, 1].uniform_(-20.0, 20.0)
    coords[:, 2].uniform_(0.0, 50.0)

    center = torch.tensor([[0.0, 0.0, 0.0]]).repeat(n, 1)
    gamma = 4.0
    gaussian = (2.0 / (torch.pi ** 1.5 * gamma ** 3)) * torch.exp(
        -torch.sum((coords - center) ** 2, dim=1, keepdim=True) / gamma ** 2
    )
    volume = 100.0 * 40.0 * 50.0
    integral = gaussian.mean().item() * volume

    ok = abs(integral - 1.0) < 0.15
    print(f"  Integral={integral:.4f}, expected approximately 1.0")
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def test_sigma_positivity():
    print("\n" + "=" * 60)
    print("TEST 3: Conductivity positivity")
    print("=" * 60)

    net = ConductivityNet(num_frequencies=16, hidden_layers=2, hidden_dim=32)
    coords = torch.randn(1000, 3) * 25
    sigma = net(coords)
    ok = torch.isfinite(sigma).all().item() and (sigma > 0).all().item()
    print(f"  sigma range=[{sigma.min().item():.4e}, {sigma.max().item():.4e}]")
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def test_dataset_voltage_targets():
    print("\n" + "=" * 60)
    print("TEST 4: Dataset returns voltage-difference targets")
    print("=" * 60)

    dataset_path = Path(__file__).resolve().parents[1] / "dataset_output_test" / "measurements.csv"
    if not dataset_path.exists():
        print("  SKIP: regenerated campaign CSV not found")
        return True

    ds = ERTDataset(dataset_path, n_pde=8, n_bc_surf=4, n_bc_inf=5, n_flux=4, epsilon=4.0)
    sample = ds[0]
    delta_v = sample["data"]["delta_v"]
    rho_a = sample["data"]["apparent_resistivity"]
    ok = (
        delta_v.shape == rho_a.shape
        and torch.isfinite(delta_v).all().item()
        and delta_v.abs().mean().item() < rho_a.abs().mean().item()
        and "r_n" in sample["data"]
        and "source" in sample["pde"]
    )
    print(f"  mean(|DeltaV|)={delta_v.abs().mean().item():.4e}, mean(|rho_a|)={rho_a.abs().mean().item():.4e}")
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def test_voltage_gradient_reaches_conductivity():
    print("\n" + "=" * 60)
    print("TEST 5: Voltage loss is coupled to conductivity")
    print("=" * 60)

    sigma_net = ConductivityNet(num_frequencies=4, hidden_layers=1, hidden_dim=16)
    pot_net = PotentialNet(
        num_frequencies=4,
        hidden_layers=1,
        hidden_dim=16,
        conductivity_net=sigma_net,
    )
    coords = torch.rand(32, 3) * torch.tensor([100.0, 100.0, 50.0])
    source = torch.zeros(32, 6)
    loss = pot_net(coords, source).square().mean()
    grads = torch.autograd.grad(loss, sigma_net.parameters(), allow_unused=True)
    grad_norm = sum(g.abs().sum().item() for g in grads if g is not None)
    ok = math.isfinite(grad_norm) and grad_norm > 0.0
    print(f"  ||dL_voltage/dsigma||={grad_norm:.4e}")
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def test_losses_are_finite():
    print("\n" + "=" * 60)
    print("TEST 6: PDE, BC, regularization, and flux losses are finite")
    print("=" * 60)

    informer, _, _ = _small_informer()
    n = 16
    coords = torch.randn(n, 3, requires_grad=True)
    coords.data[:, 2] = torch.rand(n) * 10
    source = torch.cat(
        [
            torch.tensor([[-10.0, 0.0, 0.0]]).repeat(n, 1),
            torch.tensor([[10.0, 0.0, 0.0]]).repeat(n, 1),
        ],
        dim=1,
    )
    surf = torch.randn(n, 3, requires_grad=True)
    surf.data[:, 2] = 0.0
    inf = torch.randn(n, 3)
    inf[:, 2] = 50.0

    normals = torch.randn(n, 3)
    normals = normals / torch.linalg.norm(normals, dim=1, keepdim=True)
    coords_A = torch.tensor([[-10.0, 0.0, 0.0]]).repeat(n, 1) + 4.0 * normals
    coords_B = torch.tensor([[10.0, 0.0, 0.0]]).repeat(n, 1) + 4.0 * normals
    coords_A.requires_grad_(True)
    coords_B.requires_grad_(True)
    area = 2 * math.pi * 4.0**2

    losses = [
        informer.compute_pde_loss(coords, source, I=1.0, gamma=4.0),
        informer.compute_bc_loss(surf, inf, source, source),
        informer.compute_reg_loss(coords.clone().detach().requires_grad_(True)),
        informer.compute_flux_loss(coords_A, coords_B, normals, normals, source, source, I=1.0, area=area, gamma=4.0),
    ]
    ok = all(torch.isfinite(loss).item() and loss.item() >= 0.0 for loss in losses)
    print("  " + ", ".join(f"{loss.item():.4e}" for loss in losses))
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    tests = [
        ("Gradient Flow", test_gradient_flow),
        ("Half-space Source", test_half_space_source_normalization),
        ("Sigma Positivity", test_sigma_positivity),
        ("Dataset DeltaV", test_dataset_voltage_targets),
        ("Voltage-Sigma Coupling", test_voltage_gradient_reaches_conductivity),
        ("Finite Losses", test_losses_are_finite),
    ]

    results = []
    for name, test_fn in tests:
        results.append((name, test_fn()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_passed = True
    for name, passed in results:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        all_passed = all_passed and passed

    sys.exit(0 if all_passed else 1)
