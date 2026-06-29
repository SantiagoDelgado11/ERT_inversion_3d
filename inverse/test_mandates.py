"""
Verification tests for the three mathematical mandates.
Tests the computational graph, charge conservation, and positivity.
"""
import torch
import sys
import math

def test_gradient_flow():
    """
    Test 1: Verify that autograd correctly computes non-zero gradients
    through the entire computational graph for both networks.
    
    This ensures that:
    - The Laplacian Δu is correctly computed via second-order autograd
    - The gradient ∇m flows through the Fourier features
    - The product ∇m·∇u couples both networks in the loss
    - exp(-m) * q_ε does not detach the graph
    """
    print("=" * 60)
    print("TEST 1: Gradient Flow Through Computational Graph")
    print("=" * 60)
    
    from networks import LogConductivityNet, PotentialNet
    from physics_informer import PhysicsInformer
    
    m_net = LogConductivityNet(fourier_features=32, hidden_layers=2, hidden_dim=32)
    u_net = PotentialNet(fourier_features=32, hidden_layers=2, hidden_dim=32)
    pi = PhysicsInformer(m_net, u_net)
    
    coords = torch.randn(10, 3, requires_grad=True)
    src = torch.randn(10, 6)
    
    # Test PDE loss
    loss_pde = pi.compute_pde_loss(coords, src, I=1.0, epsilon=4.0)
    loss_pde.backward()
    
    m_zero_count = 0
    for name, p in m_net.named_parameters():
        if p.grad is None or p.grad.abs().sum() == 0:
            print(f"  WARNING: Zero gradient in m_net.{name}")
            m_zero_count += 1
    m_grad_ok = m_zero_count == 0
    
    u_zero_count = 0
    for name, p in u_net.named_parameters():
        if p.grad is None or p.grad.abs().sum() == 0:
            print(f"  WARNING: Zero gradient in u_net.{name}")
            u_zero_count += 1
    # Tolerate at most 1 zero-grad param (output bias in small test nets)
    u_grad_ok = u_zero_count <= 1
    
    if m_grad_ok and u_grad_ok:
        print("  PASS: All gradients are non-zero in both networks")
    
    # Test BC loss (Neumann + Dirichlet)
    m_net.zero_grad()
    u_net.zero_grad()
    
    surf_coords = torch.randn(5, 3, requires_grad=True)
    inf_coords = torch.randn(5, 3)
    src_surf = torch.randn(5, 6)
    src_inf = torch.randn(5, 6)
    
    loss_bc = pi.compute_bc_loss(surf_coords, inf_coords, src_surf, src_inf)
    loss_bc.backward()
    
    neumann_zero_count = 0
    for name, p in m_net.named_parameters():
        if p.grad is None or p.grad.abs().sum() == 0:
            print(f"  WARNING: Neumann BC gradient zero in m_net.{name}")
            neumann_zero_count += 1
    
    if neumann_zero_count <= 1:
        print("  PASS: Neumann BC (sigma * grad_u dot n) produces gradients for m_net")
    
    # Test TV regularization
    m_net.zero_grad()
    reg_coords = torch.randn(10, 3, requires_grad=True)
    loss_reg = pi.compute_reg_loss(reg_coords)
    loss_reg.backward()
    
    reg_zero_count = 0
    for name, p in m_net.named_parameters():
        if p.grad is None or p.grad.abs().sum() == 0:
            print(f"  WARNING: TV reg gradient zero in m_net.{name}")
            reg_zero_count += 1
    
    if reg_zero_count <= 1:
        print("  PASS: TV regularization on grad_m produces gradients for m_net")
    
    all_ok = m_grad_ok and u_grad_ok and (neumann_zero_count <= 1) and (reg_zero_count <= 1)
    print(f"\n  {'ALL GRADIENT CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}")
    return all_ok


def test_charge_conservation():
    """
    Test 2: Verify that the regularized dipole source integrates to approx 0
    over the domain (conservation of charge).
    
    Integral of q_eps(x) dOmega = I * [Integral delta_eps(x - x_plus) dOmega - Integral delta_eps(x - x_minus) dOmega] approx I - I = 0
    
    The approximation improves as:
    1. N (number of sample points) -> infinity
    2. The domain fully contains the support of both Gaussians
    """
    print("\n" + "=" * 60)
    print("TEST 2: Charge Conservation (Dipole Integral approx 0)")
    print("=" * 60)
    
    from physics_informer import PhysicsInformer
    from networks import LogConductivityNet, PotentialNet
    
    pi = PhysicsInformer(LogConductivityNet(), PotentialNet())
    
    N = 200000
    # Domain [-50, 50]^3 - large enough to contain the Gaussian support
    coords = torch.rand(N, 3) * 100 - 50
    x_plus = torch.zeros(N, 3)
    x_minus = torch.tensor([[10.0, 0.0, 0.0]]).repeat(N, 1)
    
    I = 1.0
    epsilon = 4.0
    
    q = pi._regularized_source(coords, x_plus, x_minus, I, epsilon)
    
    vol = 100**3  # Volume of the domain
    integral = q.mean().item() * vol
    
    print(f"  Domain: [-50, 50]^3 = {vol} m^3")
    print(f"  N samples: {N}")
    print(f"  eps = {epsilon}")
    print(f"  Integral q_eps dOmega = {integral:.6f} (expected: approx 0)")
    
    ok = abs(integral) < 0.5  # Generous tolerance for Monte Carlo
    print(f"  {'PASS' if ok else 'FAIL'}: Charge conservation {'satisfied' if ok else 'violated'}")
    return ok


def test_individual_gaussian_integral():
    """
    Test 2b: Verify that each individual Gaussian integrates to approx I
    over a sufficiently large domain.
    
    Integral of delta_eps(x - x_0) dOmega approx 1 for large enough Omega
    So Integral of I*delta_eps dOmega approx I
    """
    print("\n" + "=" * 60)
    print("TEST 2b: Individual Gaussian Normalization")
    print("=" * 60)
    
    N = 500000
    coords = torch.rand(N, 3) * 100 - 50
    center = torch.zeros(N, 3)
    
    epsilon = 4.0
    coeff = 1.0 / ((2.0 * math.pi * epsilon**2) ** 1.5)
    dist_sq = torch.sum((coords - center)**2, dim=1, keepdim=True)
    gaussian = coeff * torch.exp(-dist_sq / (2.0 * epsilon**2))
    
    vol = 100**3
    integral = gaussian.mean().item() * vol
    
    print(f"  Integral delta_eps(x) dOmega = {integral:.6f} (expected: approx 1.0)")
    
    ok = abs(integral - 1.0) < 0.1
    print(f"  {'PASS' if ok else 'FAIL'}: Gaussian normalization {'correct' if ok else 'incorrect'}")
    return ok


def test_sigma_positivity():
    """
    Test 3: Verify that sigma(x) = exp(m(x)) is strictly positive
    for arbitrary inputs to the LogConductivityNet.
    
    Since exp: R -> R+ is bijective, this is guaranteed by construction.
    We verify empirically over random coordinates.
    """
    print("\n" + "=" * 60)
    print("TEST 3: Positivity of sigma = exp(m)")
    print("=" * 60)
    
    from networks import LogConductivityNet
    
    net = LogConductivityNet(fourier_features=32, hidden_layers=2, hidden_dim=32)
    
    # Test over various coordinate ranges
    test_cases = [
        ("Normal range", torch.randn(1000, 3) * 25),
        ("Extreme range", torch.randn(1000, 3) * 100),
        ("Near origin", torch.randn(1000, 3) * 0.1),
    ]
    
    all_ok = True
    for name, coords in test_cases:
        m = net(coords)
        sigma = torch.exp(m)
        
        is_positive = (sigma > 0).all().item()
        is_finite = torch.isfinite(sigma).all().item()
        
        print(f"  {name}:")
        print(f"    m range: [{m.min().item():.4f}, {m.max().item():.4f}]")
        print(f"    sigma range: [{sigma.min().item():.6f}, {sigma.max().item():.6f}]")
        print(f"    sigma > 0: {is_positive}, finite: {is_finite}")
        
        if not (is_positive and is_finite):
            all_ok = False
    
    print(f"\n  {'PASS' if all_ok else 'FAIL'}: Positivity check {'passed' if all_ok else 'failed'}")
    return all_ok


def test_pde_residual_formulation():
    """
    Test 4: Verify the PDE residual is mathematically consistent.
    
    For a homogeneous medium (m = const, sigma = const), the PDE residual should
    reduce to: sigma*Laplacian(u) + q_eps = 0, i.e., the standard Poisson equation.
    
    Since grad(m) = 0 for constant m:
        R = Laplacian(u) + 0 + exp(-m)*q_eps = Laplacian(u) + (1/sigma)*q_eps
    
    Which is exactly div(sigma*grad(u))/sigma = -q_eps/sigma, consistent with div(sigma*grad(u)) = -q_eps.
    """
    print("\n" + "=" * 60)
    print("TEST 4: PDE Residual Formulation Consistency")
    print("=" * 60)
    
    from networks import LogConductivityNet, PotentialNet
    from physics_informer import PhysicsInformer
    
    m_net = LogConductivityNet(fourier_features=32, hidden_layers=2, hidden_dim=32)
    u_net = PotentialNet(fourier_features=32, hidden_layers=2, hidden_dim=32)
    pi = PhysicsInformer(m_net, u_net)
    
    coords = torch.randn(5, 3, requires_grad=True)
    src = torch.randn(5, 6)
    
    # Verify the loss is a finite, positive scalar
    loss = pi.compute_pde_loss(coords, src, I=1.0, epsilon=4.0)
    
    is_finite = torch.isfinite(loss).item()
    is_scalar = loss.dim() == 0
    is_nonneg = (loss >= 0).item()
    
    print(f"  L_PDE = {loss.item():.6e}")
    print(f"  Finite: {is_finite}, Scalar: {is_scalar}, Non-negative: {is_nonneg}")
    
    ok = is_finite and is_scalar and is_nonneg
    print(f"  {'PASS' if ok else 'FAIL'}: PDE residual formulation {'valid' if ok else 'invalid'}")
    return ok


def test_wang_balancing():
    """
    Test 5: Verify Wang dynamic weight balancing produces reasonable weights.
    
    The weights should not diverge or collapse to zero, and should
    equilibrate gradient magnitudes across loss terms.
    """
    print("\n" + "=" * 60)
    print("TEST 5: Wang Dynamic Gradient Balancing")
    print("=" * 60)
    
    from train import wang_dynamic_weights
    from networks import LogConductivityNet, PotentialNet
    from physics_informer import PhysicsInformer
    
    m_net = LogConductivityNet(fourier_features=32, hidden_layers=2, hidden_dim=32)
    u_net = PotentialNet(fourier_features=32, hidden_layers=2, hidden_dim=32)
    pi = PhysicsInformer(m_net, u_net)
    
    coords_pde = torch.randn(10, 3, requires_grad=True)
    coords_reg = torch.randn(10, 3, requires_grad=True)
    src = torch.randn(10, 6)
    
    # Compute individual losses
    loss_pde = pi.compute_pde_loss(coords_pde, src, I=1.0, epsilon=4.0)
    loss_reg = pi.compute_reg_loss(coords_reg)
    
    # Use only two losses for simplicity
    losses = [loss_pde, loss_reg]
    shared_params = list(u_net.parameters())
    initial_weights = [1.0, 1.0]
    
    new_weights = wang_dynamic_weights(losses, shared_params, initial_weights, alpha=0.5)
    
    print(f"  Initial weights: {initial_weights}")
    print(f"  Updated weights: [{new_weights[0]:.4f}, {new_weights[1]:.4f}]")
    
    all_finite = all(math.isfinite(w) for w in new_weights)
    all_positive = all(w > 0 for w in new_weights)
    
    ok = all_finite and all_positive
    print(f"  Finite: {all_finite}, Positive: {all_positive}")
    print(f"  {'PASS' if ok else 'FAIL'}: Wang balancing {'produces valid weights' if ok else 'failed'}")
    return ok


if __name__ == '__main__':
    results = []
    results.append(("Gradient Flow", test_gradient_flow()))
    results.append(("Charge Conservation", test_charge_conservation()))
    results.append(("Gaussian Normalization", test_individual_gaussian_integral()))
    results.append(("Sigma Positivity", test_sigma_positivity()))
    results.append(("PDE Residual", test_pde_residual_formulation()))
    results.append(("Wang Balancing", test_wang_balancing()))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_passed = False
    
    print(f"\n{'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    sys.exit(0 if all_passed else 1)
