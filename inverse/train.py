import torch
import torch.optim as optim
from typing import Dict, Any
import wandb
from tqdm import tqdm
import math

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
    lr: float = 1e-4, 
    device: str = 'cpu',
    use_wandb: bool = False
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
    
    dyn_w = {'pde': 1.0, 'bc': 100.0, 'flux': 10.0, 'reg': 1.0}
    alpha_lra = 0.9
    
    loss_dict = {}

    def closure():
        nonlocal loss_dict
        if torch.is_grad_enabled():
            optimizer_adam.zero_grad()
            
        # Data loss empírico
        if 'source' in data_samples:
            source_data = prepare(data_samples['source'])
            u_pred = u_net(r_m, source_data)
            
            # Aislamiento de la Pérdida de Datos: Evitar evaluación sobre los electrodos
            r_A_data = source_data[:, 0:3]
            r_B_data = source_data[:, 3:6]
            
            dist_sq_A_m = torch.sum((r_m - r_A_data)**2, dim=1, keepdim=True)
            dist_sq_B_m = torch.sum((r_m - r_B_data)**2, dim=1, keepdim=True)
            
            R_scale_sq = (3.0 * epsilon)**2
            w_data_A = torch.tanh(dist_sq_A_m / R_scale_sq)
            w_data_B = torch.tanh(dist_sq_B_m / R_scale_sq)
            w_data_mask = w_data_A * w_data_B
            
            loss_data = torch.mean(w_data_mask * (u_pred - u_star)**2)
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
        
        # Learning Rate Annealing (Wang et al., 2021)
        if not is_lbfgs and loss_data.requires_grad and epoch % 10 == 0:
            last_layer_u = u_net.mlp.network[-1].weight
            grad_data = torch.autograd.grad(loss_data, last_layer_u, retain_graph=True)[0]
            max_grad_data = torch.max(torch.abs(grad_data)) + 1e-8
            
            def update_dyn_w(key, loss_term):
                if loss_term.requires_grad:
                    grad_term = torch.autograd.grad(loss_term, last_layer_u, retain_graph=True)[0]
                    mean_grad_term = torch.mean(torch.abs(grad_term)) + 1e-8
                    hat_lambda = max_grad_data / mean_grad_term
                    # Clipping de lambda estricto para evitar multiplicadores infinitos
                    hat_lambda = torch.clamp(hat_lambda, max=100.0)
                    dyn_w[key] = alpha_lra * dyn_w[key] + (1.0 - alpha_lra) * hat_lambda.item()
            
            update_dyn_w('pde', loss_pde)
            update_dyn_w('bc', loss_bc)
            update_dyn_w('flux', loss_flux)
            
            last_layer_sigma = sigma_net.mlp.network[-1].weight
            if loss_reg.requires_grad and loss_pde.requires_grad:
                try:
                    grad_pde_sigma = torch.autograd.grad(loss_pde, last_layer_sigma, retain_graph=True, allow_unused=True)[0]
                    grad_reg = torch.autograd.grad(loss_reg, last_layer_sigma, retain_graph=True, allow_unused=True)[0]
                    if grad_pde_sigma is not None and grad_reg is not None:
                        max_grad_pde_sigma = torch.max(torch.abs(grad_pde_sigma)) + 1e-8
                        mean_grad_reg = torch.mean(torch.abs(grad_reg)) + 1e-8
                        hat_lambda_reg = max_grad_pde_sigma / mean_grad_reg
                        hat_lambda_reg = torch.clamp(hat_lambda_reg, max=100.0)
                        dyn_w['reg'] = alpha_lra * dyn_w['reg'] + (1.0 - alpha_lra) * hat_lambda_reg.item()
                except Exception:
                    pass

        # Suma ponderada con pesos dinámicos y base
        loss_total = (weights.get('w_data', 1.0) * loss_data +
                      dyn_w['pde'] * loss_pde +
                      dyn_w['bc'] * loss_bc +
                      dyn_w['reg'] * loss_reg +
                      dyn_w['flux'] * loss_flux)
                      
        loss_dict = {
            "loss_data": loss_data.item(),
            "loss_pde": loss_pde.item(),
            "loss_bc": loss_bc.item(),
            "loss_reg": loss_reg.item(),
            "loss_flux": loss_flux.item(),
            "loss_total": loss_total.item(),
            "lambda_pde": dyn_w['pde'],
            "lambda_bc": dyn_w['bc'],
            "lambda_flux": dyn_w['flux'],
            "lambda_reg": dyn_w['reg']
        }

        if loss_total.requires_grad:
            loss_total.backward()
            # Clipping Global Riguroso de Gradientes
            torch.nn.utils.clip_grad_norm_(list(u_net.parameters()) + list(sigma_net.parameters()), max_norm=1.0)
            
        return loss_total

    # --- Fase 1: Adam ---
    print("Iniciando Fase 1: Entrenamiento inicial con Adam")
    is_lbfgs = False
    pbar_adam = tqdm(range(num_epochs_adam), desc="Adam")
    for epoch in pbar_adam:
        # Warm-up lineal de 500 pasos
        warmup_steps = 500
        current_lr = lr * min(1.0, (epoch + 1) / warmup_steps)
        for param_group in optimizer_adam.param_groups:
            param_group['lr'] = current_lr
            
        optimizer_adam.step(closure)
        if use_wandb:
            wandb.log({"epoch_adam": epoch, "lr_adam": current_lr, **loss_dict})
        pbar_adam.set_postfix(loss=f"{loss_dict.get('loss_total', 0):.4e}")

    # --- Fase 2: L-BFGS ---
    print("Iniciando Fase 2: Ajuste fino con L-BFGS")
    is_lbfgs = True
    optimizer_lbfgs = optim.LBFGS(
        list(u_net.parameters()) + list(sigma_net.parameters()),
        lr=1.0, max_iter=20, tolerance_grad=1e-7, tolerance_change=1e-9, history_size=50
    )
    
    pbar_lbfgs = tqdm(range(num_epochs_lbfgs), desc="L-BFGS")
    for epoch in pbar_lbfgs:
        optimizer_lbfgs.step(closure)
        if use_wandb:
            wandb.log({"epoch_lbfgs": epoch, **loss_dict})
        pbar_lbfgs.set_postfix(loss=f"{loss_dict.get('loss_total', 0):.4e}")

    return u_net, sigma_net
