import torch
import torch.optim as optim
from typing import Dict, Any

def train_pinn(
    u_net: torch.nn.Module, 
    sigma_net: torch.nn.Module, 
    informer: Any, 
    data_samples: Dict[str, torch.Tensor], 
    pde_samples: Dict[str, torch.Tensor], 
    bc_neumann_samples: Dict[str, torch.Tensor], 
    bc_dirichlet_samples: Dict[str, torch.Tensor],
    flux_samples: Dict[str, torch.Tensor], 
    reg_samples: Dict[str, torch.Tensor],
    weights: Dict[str, float], 
    current_I: float,
    epsilon: float,
    num_epochs_adam: int = 1000, 
    num_epochs_lbfgs: int = 500,
    lr: float = 1e-3, 
    device: str = 'cpu'
):
    u_net = u_net.to(device)
    sigma_net = sigma_net.to(device)
    
    # 1. Optimizador Adam para exploración macroscópica
    optimizer_adam = optim.Adam(list(u_net.parameters()) + list(sigma_net.parameters()), lr=lr)
    
    def prepare(tensor: torch.Tensor, requires_grad: bool = False):
        t = tensor.to(device)
        return t.requires_grad_(True) if requires_grad else t
        
    r_m = prepare(data_samples['r_m'])
    u_star = prepare(data_samples['u_star'])
    r_pde = prepare(pde_samples['r'], requires_grad=True)
    
    r_A = prepare(pde_samples['r_A'])
    r_B = prepare(pde_samples['r_B'])
    
    # CONCILIACIÓN TENSORIAL: Generamos el source_coords (dim=6) para PotentialNet
    source_coords_pde = torch.cat([r_A, r_B], dim=-1)
    
    r_N = prepare(bc_neumann_samples['r_N'], requires_grad=True)
    r_D = prepare(bc_dirichlet_samples['r_D'])
    
    r_Bc_A = prepare(flux_samples['r_Bc_A'], requires_grad=True)
    n_Bc_A = prepare(flux_samples['n_Bc_A'])
    r_Bc_B = prepare(flux_samples['r_Bc_B'], requires_grad=True)
    n_Bc_B = prepare(flux_samples['n_Bc_B'])
    area_Bc = flux_samples['area_Bc']
    
    r_reg = prepare(reg_samples['r_reg'], requires_grad=True)
    
    def closure():
        """El closure es obligatorio para permitir la ejecución de L-BFGS"""
        if torch.is_grad_enabled():
            optimizer_adam.zero_grad()
            
        # Data loss empírico
        if 'source' in data_samples:
            source_data = prepare(data_samples['source'])
            u_pred = u_net(r_m, source_data)
            loss_data = torch.mean((u_pred - u_star)**2)
        else:
            loss_data = torch.tensor(0.0, device=device)

        # Invocando la API real del physics_informer
        loss_pde = informer.compute_pde_loss(r_pde, source_coords_pde, current_I, epsilon)
        
        # Unificando las Condiciones de Frontera
        loss_bc = informer.compute_bc_loss(
            surface_coords=r_N, 
            inf_coords=r_D, 
            source_coords_surf=source_coords_pde[:r_N.shape[0]] if r_N.shape[0] > 0 else None,
            source_coords_inf=source_coords_pde[:r_D.shape[0]] if r_D.shape[0] > 0 else None
        )
        
        loss_reg = informer.compute_reg_loss(r_reg)
        loss_flux = informer.compute_flux_loss(
            r_Bc_A, r_Bc_B, n_Bc_A, n_Bc_B, 
            source_coords_pde[:r_Bc_A.shape[0]], source_coords_pde[:r_Bc_B.shape[0]], 
            current_I, area_Bc
        )
        
        # Suma ponderada con pesos fijos (Mitigación pasiva)
        loss_total = (weights.get('w_data', 1.0) * loss_data +
                      weights.get('w_pde', 1.0) * loss_pde +
                      weights.get('w_bc', 1.0) * loss_bc +
                      weights.get('w_reg', 1.0) * loss_reg +
                      weights.get('w_flux', 1.0) * loss_flux)
                      
        if loss_total.requires_grad:
            loss_total.backward()
            
        return loss_total

    # --- Fase 1: Adam ---
    print("Iniciando Fase 1: Entrenamiento inicial con Adam")
    for epoch in range(num_epochs_adam):
        optimizer_adam.step(closure)
        if epoch % 100 == 0:
            print(f"Adam Epoch {epoch}: Loss = {closure().item():.6e}")

    # --- Fase 2: L-BFGS ---
    print("Iniciando Fase 2: Ajuste fino con L-BFGS")
    optimizer_lbfgs = optim.LBFGS(
        list(u_net.parameters()) + list(sigma_net.parameters()),
        lr=1.0, max_iter=20, tolerance_grad=1e-7, tolerance_change=1e-9, history_size=50
    )
    
    for epoch in range(num_epochs_lbfgs):
        optimizer_lbfgs.step(closure)
        if epoch % 10 == 0:
            print(f"L-BFGS Epoch {epoch}: Loss = {closure().item():.6e}")

    return u_net, sigma_net
