import torch
import torch.optim as optim
from typing import Dict, Any, List
import wandb
from tqdm import tqdm
import math


# ═══════════════════════════════════════════════════════════════════════════════
# Mandato 3: Funciones de Balanceo Dinámico de Gradientes (Wang et al., 2021)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_grad_norm(loss: torch.Tensor, params: List[torch.nn.Parameter]) -> torch.Tensor:
    """
    Calcula la norma L2 del gradiente de `loss` respecto a `params`.

    Se usa retain_graph=True para permitir múltiples llamadas sobre
    distintos términos de pérdida sin destruir el grafo computacional.
    allow_unused=True porque no todos los parámetros contribuyen a
    todos los términos de pérdida (e.g., L_reg no depende de u_net).

    Args:
        loss: escalar — término de pérdida individual
        params: lista de parámetros respecto a los cuales computar gradientes

    Returns:
        norm: escalar — ||∇_params loss||_2
    """
    grads = torch.autograd.grad(
        loss, params,
        retain_graph=True,
        allow_unused=True,
        create_graph=False  # No necesitamos derivadas de segundo orden aquí
    )
    total_sq = torch.tensor(0.0, device=loss.device)
    for g in grads:
        if g is not None:
            total_sq = total_sq + g.detach().pow(2).sum()
    return torch.sqrt(total_sq + 1e-16)


def wang_dynamic_weights(
    losses: List[torch.Tensor],
    shared_params: List[torch.nn.Parameter],
    current_weights: List[float],
    alpha: float = 0.1
) -> List[float]:
    """
    Balanceo dinámico de pesos multi-tarea siguiendo Wang et al. (2021):
    "Understanding and Mitigating Gradient Flow Pathologies in
     Physics-Informed Neural Networks"

    Para cada término de pérdida L_k, calculamos:
        G_k = ||∇_{θ_shared} (w_k · L_k)||_2
        Ḡ = (1/K) Σ_k G_k
        λ̂_k = Ḡ / G_k

    Los pesos se actualizan con suavizado exponencial:
        w_k^{new} = (1 - α) · w_k^{old} + α · λ̂_k

    Esto equilibra la magnitud de los gradientes de cada término de pérdida,
    evitando que L_data domine sobre L_PDE (o viceversa).

    Args:
        losses: [L_data, L_pde, L_bc, L_reg, L_flux] — pérdidas individuales
        shared_params: parámetros de referencia para medir magnitudes de gradientes
        current_weights: [w_data, w_pde, w_bc, w_reg, w_flux] — pesos actuales
        alpha: tasa de suavizado exponencial (0 < α ≤ 1)

    Returns:
        new_weights: pesos actualizados
    """
    grad_norms = []
    for w_k, loss_k in zip(current_weights, losses):
        # Gradiente de w_k * L_k respecto a los parámetros compartidos
        weighted_loss = w_k * loss_k
        gn = compute_grad_norm(weighted_loss, shared_params)
        grad_norms.append(gn.item())

    mean_gn = sum(grad_norms) / len(grad_norms)

    new_weights = []
    for w_old, gn in zip(current_weights, grad_norms):
        ratio = mean_gn / (gn + 1e-16)
        w_new = (1.0 - alpha) * w_old + alpha * ratio
        new_weights.append(w_new)

    return new_weights


# ═══════════════════════════════════════════════════════════════════════════════
# Función Principal de Entrenamiento
# ═══════════════════════════════════════════════════════════════════════════════

def train_pinn(
    u_net: torch.nn.Module,
    m_net: torch.nn.Module,
    informer: Any,
    data_samples: Dict[str, torch.Tensor],
    pde_samples: Dict[str, torch.Tensor],
    bc_neumann_samples: Dict[str, torch.Tensor],
    bc_dirichlet_samples: Dict[str, torch.Tensor],
    flux_samples: Dict[str, torch.Tensor],
    reg_samples: Dict[str, torch.Tensor],
    weights: Dict[str, float],
    current_I: float,
    gamma: float,
    num_epochs_adam: int = 1000,
    num_epochs_lbfgs: int = 500,
    lr_u: float = 1e-3,
    lr_m: float = 5e-4,
    balance_every: int = 10,
    balance_alpha: float = 0.1,
    warmup_epochs: int = 500,
    device: str = 'cpu',
    use_wandb: bool = False
):
    """
    Entrenamiento conjunto de la cPINN con:
    - Mandato 2: Red m_net predice m(x) = ln σ(x)
    - Mandato 3: Tasas de aprendizaje desacopladas + balanceo dinámico de Wang

    Args:
        u_net: PotentialNet — red del potencial eléctrico
        m_net: LogConductivityNet — red del campo logarítmico de conductividad
        informer: PhysicsInformer — motor diferencial de residuos físicos
        data_samples: dict con 'r_m', 'u_star', 'source'
        pde_samples: dict con 'r', 'r_A', 'r_B'
        bc_neumann_samples: dict con 'r_N'
        bc_dirichlet_samples: dict con 'r_D'
        flux_samples: dict con 'r_Bc_A', 'n_Bc_A', 'r_Bc_B', 'n_Bc_B', 'area_Bc'
        reg_samples: dict con 'r_reg'
        weights: dict con 'w_data', 'w_pde', 'w_bc', 'w_reg', 'w_flux'
        current_I: corriente inyectada (Amperes)
        gamma: epsilon del dipolo gaussiano
        num_epochs_adam: épocas totales de Adam
        num_epochs_lbfgs: (reservado, no implementado aún)
        lr_u: learning rate para PotentialNet
        lr_m: learning rate para LogConductivityNet
        balance_every: recalcular pesos dinámicos cada N épocas
        balance_alpha: tasa de suavizado exponencial para balanceo
        warmup_epochs: épocas de warm-up antes del entrenamiento conjunto
        device: dispositivo de cómputo
        use_wandb: activar logging a Weights & Biases
    """
    u_net = u_net.to(device)
    m_net = m_net.to(device)

    # ─── Mandato 3: Optimizadores con tasas desacopladas ───
    optimizer_u = optim.Adam(u_net.parameters(), lr=lr_u)
    optimizer_m = optim.Adam(m_net.parameters(), lr=lr_m)
    scheduler_u = optim.lr_scheduler.CosineAnnealingLR(optimizer_u, T_max=num_epochs_adam)
    scheduler_m = optim.lr_scheduler.CosineAnnealingLR(optimizer_m, T_max=num_epochs_adam)

    def prepare(tensor: torch.Tensor, requires_grad: bool = False):
        t = tensor.to(device)
        return t.requires_grad_(True) if requires_grad else t

    # ─── Preparación de tensores ───
    r_m = prepare(data_samples['r_m'])
    u_star = prepare(data_samples['u_star'])
    r_pde = prepare(pde_samples['r'], requires_grad=True)

    r_A = prepare(pde_samples['r_A'])
    r_B = prepare(pde_samples['r_B'])

    # Coordenadas fuente (dim=6) para PotentialNet
    source_coords_pde = torch.cat([r_A, r_B], dim=-1)

    r_N = prepare(bc_neumann_samples['r_N'], requires_grad=True)
    r_D = prepare(bc_dirichlet_samples['r_D'])

    r_Bc_A = prepare(flux_samples['r_Bc_A'], requires_grad=True)
    n_Bc_A = prepare(flux_samples['n_Bc_A'])
    r_Bc_B = prepare(flux_samples['r_Bc_B'], requires_grad=True)
    n_Bc_B = prepare(flux_samples['n_Bc_B'])
    area_Bc = flux_samples['area_Bc']

    r_reg = prepare(reg_samples['r_reg'], requires_grad=True)

    # ─── Pesos iniciales de la función de pérdida ───
    w_data = weights.get('w_data', 1.0)
    w_pde = weights.get('w_pde', 1.0)
    w_bc = weights.get('w_bc', 1.0)
    w_flux = weights.get('w_flux', 1.0)
    w_reg = weights.get('w_reg', 1e-4)
    dynamic_weights = [w_data, w_pde, w_bc, w_reg, w_flux]

    # Valor de fondo en espacio logarítmico: m₀ = ln(σ₀) = ln(0.01) ≈ -4.605
    m0 = math.log(0.01)

    loss_dict = {}

    def compute_all_losses():
        """Evalúa todos los términos de la función de pérdida."""
        # 1. L_data — Ajuste a mediciones superficiales
        if 'source' in data_samples:
            source_data = prepare(data_samples['source'])
            u_pred = u_net(r_m, source_data)
            loss_data = torch.mean((u_pred - u_star)**2)
        else:
            loss_data = torch.tensor(0.0, device=device)

        # 2. L_PDE — Residuo: Δu + ∇m·∇u + e^{-m}·q_ε = 0
        loss_pde = informer.compute_pde_loss(r_pde, source_coords_pde, current_I, gamma)

        # 3. L_bc — Neumann (σ∇u·n = 0) + Dirichlet (u = 0)
        loss_bc = informer.compute_bc_loss(
            surface_coords=r_N,
            inf_coords=r_D,
            source_coords_surf=source_coords_pde[:r_N.shape[0]] if r_N.shape[0] > 0 else None,
            source_coords_inf=source_coords_pde[:r_D.shape[0]] if r_D.shape[0] > 0 else None
        )

        # 4. L_reg — Variación Total sobre ∇m
        loss_reg = informer.compute_reg_loss(r_reg)

        # 5. L_flux — Conservación de carga con σ = e^m
        loss_flux = informer.compute_flux_loss(
            r_Bc_A, r_Bc_B, n_Bc_A, n_Bc_B,
            source_coords_pde[:r_Bc_A.shape[0]], source_coords_pde[:r_Bc_B.shape[0]],
            current_I, area_Bc
        )

        return loss_data, loss_pde, loss_bc, loss_reg, loss_flux

    # ═══════════════════════════════════════════════════════════════════════
    # Bucle de Entrenamiento
    # ═══════════════════════════════════════════════════════════════════════
    print("Iniciando Entrenamiento Conjunto con Balanceo Dinámico de Gradientes")
    pbar_adam = tqdm(range(num_epochs_adam), desc="Adam")

    for epoch in pbar_adam:
        # ─── Learning rate warm-up lineal ───
        if epoch < warmup_epochs:
            warmup_factor = min(1.0, (epoch + 1) / warmup_epochs)
            for param_group in optimizer_u.param_groups:
                param_group['lr'] = lr_u * warmup_factor
            for param_group in optimizer_m.param_groups:
                param_group['lr'] = lr_m * warmup_factor
            current_lr_u = lr_u * warmup_factor
            current_lr_m = lr_m * warmup_factor
        else:
            scheduler_u.step()
            scheduler_m.step()
            current_lr_u = scheduler_u.get_last_lr()[0]
            current_lr_m = scheduler_m.get_last_lr()[0]

        if epoch < warmup_epochs:
            # ═══════════════════════════════════════════════════════════════
            # Fase de Warm-up: Estabilización en espacio logarítmico
            # ═══════════════════════════════════════════════════════════════

            # (a) Warm-up de m_net: m(x) → m₀ = ln(σ₀)
            optimizer_m.zero_grad()
            m_pred = m_net(r_reg)
            loss_warmup = torch.mean((m_pred - m0)**2)
            loss_warmup.backward()
            torch.nn.utils.clip_grad_norm_(m_net.parameters(), max_norm=1.0)
            optimizer_m.step()

            # (b) Warm-up del potencial sobre el medio homogéneo
            optimizer_u.zero_grad()
            loss_data, loss_pde, loss_bc, loss_reg, loss_flux = compute_all_losses()

            loss_total = (
                dynamic_weights[0] * loss_data
                + dynamic_weights[1] * loss_pde
                + dynamic_weights[2] * loss_bc
                + dynamic_weights[3] * loss_reg
                + dynamic_weights[4] * loss_flux
            )

            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(u_net.parameters(), max_norm=1.0)
            optimizer_u.step()

            loss_dict = {
                "loss_data": loss_data.item(), "loss_pde": loss_pde.item(),
                "loss_bc": loss_bc.item(), "loss_reg": loss_reg.item(),
                "loss_flux": loss_flux.item(), "loss_warmup_m": loss_warmup.item(),
                "loss_total": loss_total.item() + loss_warmup.item(),
                "w_data": dynamic_weights[0], "w_pde": dynamic_weights[1],
                "w_bc": dynamic_weights[2], "w_reg": dynamic_weights[3],
                "w_flux": dynamic_weights[4],
            }
        else:
            # ═══════════════════════════════════════════════════════════════
            # Fase Principal: Optimización Conjunta con Balanceo Dinámico
            # ═══════════════════════════════════════════════════════════════
            optimizer_u.zero_grad()
            optimizer_m.zero_grad()

            loss_data, loss_pde, loss_bc, loss_reg, loss_flux = compute_all_losses()
            losses = [loss_data, loss_pde, loss_bc, loss_reg, loss_flux]

            # ─── Mandato 3: Balanceo dinámico de Wang et al. ───
            if epoch % balance_every == 0:
                # Usamos parámetros de u_net como referencia para medir
                # la magnitud relativa de gradientes entre términos
                shared_params = list(u_net.parameters())
                dynamic_weights = wang_dynamic_weights(
                    losses, shared_params, dynamic_weights, alpha=balance_alpha
                )

            # Pérdida total ponderada
            loss_total = sum(
                w_k * l_k for w_k, l_k in zip(dynamic_weights, losses)
            )

            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(u_net.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(m_net.parameters(), max_norm=1.0)

            optimizer_u.step()
            optimizer_m.step()

            loss_dict = {
                "loss_data": loss_data.item(), "loss_pde": loss_pde.item(),
                "loss_bc": loss_bc.item(), "loss_reg": loss_reg.item(),
                "loss_flux": loss_flux.item(),
                "loss_total": loss_total.item(),
                "w_data": dynamic_weights[0], "w_pde": dynamic_weights[1],
                "w_bc": dynamic_weights[2], "w_reg": dynamic_weights[3],
                "w_flux": dynamic_weights[4],
            }

        if use_wandb:
            wandb.log({
                "epoch_adam": epoch,
                "lr_u": current_lr_u,
                "lr_m": current_lr_m,
                **loss_dict
            })
        pbar_adam.set_postfix(loss=f"{loss_dict.get('loss_total', 0):.4e}")

    return u_net, m_net
