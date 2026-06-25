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
    gamma: float,
    num_epochs_adam: int = 1000, 
    num_epochs_lbfgs: int = 500,
    lr: float = 1e-4, 
    device: str = 'cpu',
    use_wandb: bool = False
):
    u_net = u_net.to(device)
    sigma_net = sigma_net.to(device)
    
    # 1. Optimizadores separados para Optimización Alternada
    optimizer_u = optim.Adam(u_net.parameters(), lr=lr)
    optimizer_sigma = optim.Adam(sigma_net.parameters(), lr=lr)
    scheduler_u = optim.lr_scheduler.CosineAnnealingLR(optimizer_u, T_max=num_epochs_adam)
    scheduler_sigma = optim.lr_scheduler.CosineAnnealingLR(optimizer_sigma, T_max=num_epochs_adam)
    
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
    
    # Usaremos escalado explícito estricto
    lambda_pde = 1.0
    lambda_bc = 1.0      # CRÍTICO: Debe ser fuerte para evitar fugas de corriente (Fantasmas)
    lambda_flux = 1.0    # CRÍTICO: Conservación de carga inyectada
    lambda_reg = 1e-4
    
    loss_dict = {}

    def compute_all_losses():
        # Data loss empírico con Depth Weighting
        if 'source' in data_samples:
            source_data = prepare(data_samples['source'])
            u_pred = u_net(r_m, source_data)
            
            # Aislamiento de la Pérdida de Datos: Evitar evaluación sobre los electrodos
            r_A_data = source_data[:, 0:3]
            r_B_data = source_data[:, 3:6]
            
            dist_sq_A_m = torch.sum((r_m - r_A_data)**2, dim=1, keepdim=True)
            dist_sq_B_m = torch.sum((r_m - r_B_data)**2, dim=1, keepdim=True)
            
            R_scale_sq = (3.0 * gamma)**2
            w_data_A = torch.tanh(dist_sq_A_m / R_scale_sq)
            w_data_B = torch.tanh(dist_sq_B_m / R_scale_sq)
            w_data_mask = w_data_A * w_data_B
            
            # Depth Weighting en Data Loss para prevenir overfitting de superficie
            z_data = r_m[:, 2:3]
            W_z_data = 1.0 / (torch.abs(z_data) + 2.0)**2.0
            
            loss_data = torch.mean(W_z_data * w_data_mask * (u_pred - u_star)**2)
        else:
            loss_data = torch.tensor(0.0, device=device)

        # Invocando la API real del physics_informer
        loss_pde = informer.compute_pde_loss(r_pde, source_coords_pde, current_I, gamma)
        
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
        
        return loss_data, loss_pde, loss_bc, loss_reg, loss_flux

    # --- Entrenamiento con Optimización Alternada ---
    print("Iniciando Entrenamiento con Optimización Alternada (Adam)")
    pbar_adam = tqdm(range(num_epochs_adam), desc="Adam")
    for epoch in pbar_adam:
        # Warm-up lineal de learning rate en las primeras 500 iteraciones
        if epoch < 500:
            current_lr = lr * min(1.0, (epoch + 1) / 500)
            for param_group in optimizer_u.param_groups:
                param_group['lr'] = current_lr
            for param_group in optimizer_sigma.param_groups:
                param_group['lr'] = current_lr
        else:
            scheduler_u.step()
            scheduler_sigma.step()
            current_lr = scheduler_u.get_last_lr()[0]
            
        if epoch <= 500:
            # --- Warm-up ---
            # 1. Forzar sigma a un background homogéneo (100 Ohm.m -> 0.01 S/m)
            optimizer_sigma.zero_grad()
            sigma_pred = sigma_net(r_reg)
            loss_warmup = torch.mean((sigma_pred - 0.01)**2)
            loss_warmup.backward()
            torch.nn.utils.clip_grad_norm_(sigma_net.parameters(), max_norm=1.0)
            optimizer_sigma.step()
            
            # 2. Entrenar u_net en el medio homogéneo para que aprenda el potencial base
            optimizer_u.zero_grad()
            loss_data, loss_pde, loss_bc, loss_reg, loss_flux = compute_all_losses()
            loss_u = (weights.get('w_data', 1.0) * loss_data + 
                      lambda_pde * loss_pde + 
                      lambda_bc * loss_bc + 
                      lambda_flux * loss_flux)
            loss_u.backward()
            torch.nn.utils.clip_grad_norm_(u_net.parameters(), max_norm=1.0)
            optimizer_u.step()
            
            loss_dict = {
                "loss_data": loss_data.item(), "loss_pde": loss_pde.item(), "loss_bc": loss_bc.item(),
                "loss_reg": loss_reg.item(), "loss_flux": loss_flux.item(), "loss_warmup": loss_warmup.item(),
                "loss_total": loss_u.item() + loss_warmup.item(),
                "lambda_pde": lambda_pde, "lambda_bc": lambda_bc, "lambda_flux": lambda_flux, "lambda_reg": lambda_reg
            }
        else:
            # --- Optimización Alternada ---
            # Paso 1: Actualizar u_net (Adaptar potencial a la conductividad actual y los datos)
            optimizer_u.zero_grad()
            loss_data, loss_pde, loss_bc, loss_reg, loss_flux = compute_all_losses()
            loss_u = (weights.get('w_data', 1.0) * loss_data + 
                      lambda_pde * loss_pde + 
                      lambda_bc * loss_bc + 
                      lambda_flux * loss_flux)
            loss_u.backward()
            torch.nn.utils.clip_grad_norm_(u_net.parameters(), max_norm=1.0)
            optimizer_u.step()
            
            # Paso 2: Actualizar sigma_net (Explicar potencial actual mediante la PDE y regularización)
            optimizer_sigma.zero_grad()
            loss_data_s, loss_pde_s, loss_bc_s, loss_reg_s, loss_flux_s = compute_all_losses()
            loss_sigma = (lambda_pde * loss_pde_s + 
                          lambda_flux * loss_flux_s + 
                          lambda_reg * loss_reg_s)
            loss_sigma.backward()
            torch.nn.utils.clip_grad_norm_(sigma_net.parameters(), max_norm=1.0)
            optimizer_sigma.step()
            
            loss_dict = {
                "loss_data": loss_data.item(), "loss_pde": loss_pde_s.item(), "loss_bc": loss_bc.item(),
                "loss_reg": loss_reg_s.item(), "loss_flux": loss_flux_s.item(), 
                "loss_total": loss_u.item() + loss_sigma.item(),
                "lambda_pde": lambda_pde, "lambda_bc": lambda_bc, "lambda_flux": lambda_flux, "lambda_reg": lambda_reg
            }

        if use_wandb:
            wandb.log({"epoch_adam": epoch, "lr_adam": current_lr, **loss_dict})
        pbar_adam.set_postfix(loss=f"{loss_dict.get('loss_total', 0):.4e}")

    return u_net, sigma_net
