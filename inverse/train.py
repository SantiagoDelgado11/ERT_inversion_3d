import logging
import math
from typing import Any, Dict
import torch
import torch.optim as optim
from tqdm import tqdm
import os

try:
    import wandb
except ImportError:
    wandb = None

logger = logging.getLogger(__name__)

def train_pinn(
    u_net: torch.nn.Module,
    sigma_net: torch.nn.Module,
    encoder: torch.nn.Module,
    informer: Any,
    dataloader: torch.utils.data.DataLoader,
    weights: Dict[str, float],
    current_I: float,
    gamma: float,
    num_epochs_adam: int = 1000,
    num_epochs_lbfgs: int = 500,
    lr: float = 1e-4,
    device: str = "cpu",
    use_wandb: bool = False,
    profile_training: bool = False,
    accumulation_steps: int = 1,
):
    if len(dataloader) == 0:
        raise ValueError("El dataloader está vacío.")

    u_net = u_net.to(device)
    sigma_net = sigma_net.to(device)

    # Tasas de aprendizaje seguras e idénticas
    optimizer_u = optim.Adam(u_net.parameters(), lr=1e-4) 
    optimizer_sigma = optim.Adam(list(sigma_net.parameters()) + list(encoder.parameters()), lr=1e-4)
    
    scheduler_u = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_u, T_0=200, T_mult=2, eta_min=1e-6)
    scheduler_sigma = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_sigma, T_0=200, T_mult=2, eta_min=1e-6)

    device_type = "cuda" if "cuda" in str(device) else "cpu"
    use_amp = device_type == "cuda"
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    def prepare(tensor: torch.Tensor, requires_grad: bool = False):
        t = tensor.to(device)
        return t.requires_grad_(True) if requires_grad else t

    def flatten_batch(tensor):
        if tensor is None: return None
        if tensor.dim() >= 2:
            return tensor.reshape(-1, *tensor.shape[2:])
        return tensor

    def expand_latent(l, n_pts):
        if l is None: return None
        return l.unsqueeze(1).expand(-1, n_pts, -1).reshape(-1, l.shape[-1])

    checkpoint_path = "checkpoint.pth"
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        print(f"\n[INFO] Reanudando entrenamiento desde {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        u_net.load_state_dict(checkpoint['u_net_state_dict'])
        sigma_net.load_state_dict(checkpoint['sigma_net_state_dict'])
        if encoder is not None and 'encoder_state_dict' in checkpoint:
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
        optimizer_u.load_state_dict(checkpoint['optimizer_u_state_dict'])
        optimizer_sigma.load_state_dict(checkpoint['optimizer_sigma_state_dict'])
        if 'scheduler_u_state_dict' in checkpoint:
            scheduler_u.load_state_dict(checkpoint['scheduler_u_state_dict'])
        if 'scheduler_sigma_state_dict' in checkpoint:
            scheduler_sigma.load_state_dict(checkpoint['scheduler_sigma_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    print("Iniciando entrenamiento SEGURO (Pesos Fijos + Grad Clipping)...")
    pbar_adam = tqdm(range(start_epoch, num_epochs_adam), desc="Adam", initial=start_epoch, total=num_epochs_adam)
    
    for epoch in pbar_adam:
        epoch_loss = 0.0
        num_batches = len(dataloader)
        
        pbar_batches = tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs_adam}", leave=False)
        for batch_idx, batch in enumerate(pbar_batches):
            data_samples = batch["data"]
            pde_samples = batch["pde"]
            reg_samples = batch["reg"]
            bc_neumann = batch.get("bc_neumann")
            bc_dirichlet = batch.get("bc_dirichlet")
            flux_data = batch.get("flux")
            
            _r_A = prepare(data_samples["source"][..., 0:3])
            _r_B = prepare(data_samples["source"][..., 3:6])
            _r_m = prepare(data_samples["r_m"])
            _r_n = prepare(data_samples["r_n"]) if "r_n" in data_samples else torch.zeros_like(_r_m)
            _delta_v = prepare(data_samples.get("delta_v", data_samples["u_star"]))
            
            _delta_v_scaled = torch.sign(_delta_v) * torch.log1p(torch.abs(_delta_v))
            encoder_input = torch.cat([_r_A, _r_B, _r_m, _r_n, _delta_v_scaled], dim=-1)
            latent = encoder(encoder_input)
            
            # --- Subsampling de PDE para memoria y estabilidad ---
            n_pde_target = 1000
            if pde_samples["r"].shape[1] > n_pde_target:
                idx = torch.randperm(pde_samples["r"].shape[1])[:n_pde_target]
                pde_samples["r"] = pde_samples["r"][:, idx, :]
                if "source" in pde_samples:
                    pde_samples["source"] = pde_samples["source"][:, idx, :]
            
            if reg_samples["r_reg"].shape[1] > n_pde_target:
                idx_reg = torch.randperm(reg_samples["r_reg"].shape[1])[:n_pde_target]
                reg_samples["r_reg"] = reg_samples["r_reg"][:, idx_reg, :]
            # -----------------------------------------------------

            r_m = flatten_batch(_r_m)
            r_n = flatten_batch(_r_n)
            source_data = flatten_batch(prepare(data_samples["source"]))
            target = flatten_batch(_delta_v)
            latent_data = expand_latent(latent, _r_m.shape[1])
            
            r_pde = flatten_batch(prepare(pde_samples["r"], requires_grad=True))
            source_pde = flatten_batch(prepare(pde_samples["source"])) if "source" in pde_samples else None
            latent_pde = expand_latent(latent, pde_samples["r"].shape[1])
            
            r_reg = flatten_batch(prepare(reg_samples["r_reg"], requires_grad=True))
            latent_reg = expand_latent(latent, reg_samples["r_reg"].shape[1])

            if bc_neumann is not None:
                r_neumann = flatten_batch(prepare(bc_neumann["r_N"], requires_grad=True))
                source_neumann = flatten_batch(prepare(bc_neumann["source"]))
                latent_neumann = expand_latent(latent, bc_neumann["r_N"].shape[1])
            else:
                r_neumann, source_neumann, latent_neumann = None, None, None
                
            if bc_dirichlet is not None:
                r_dirichlet = flatten_batch(prepare(bc_dirichlet["r_D"], requires_grad=True))
                source_dirichlet = flatten_batch(prepare(bc_dirichlet["source"]))
                latent_dirichlet = expand_latent(latent, bc_dirichlet["r_D"].shape[1])
            else:
                r_dirichlet, source_dirichlet, latent_dirichlet = None, None, None

            if flux_data is not None:
                r_Bc_A = flatten_batch(prepare(flux_data["r_Bc_A"], requires_grad=True))
                n_Bc_A = flatten_batch(prepare(flux_data["n_Bc_A"]))
                r_Bc_B = flatten_batch(prepare(flux_data["r_Bc_B"], requires_grad=True))
                n_Bc_B = flatten_batch(prepare(flux_data["n_Bc_B"]))
                source_A_flux = flatten_batch(prepare(flux_data["source_A"]))
                source_B_flux = flatten_batch(prepare(flux_data["source_B"]))
                
                area_Bc = flux_data["area_Bc"][0].item() if isinstance(flux_data["area_Bc"], torch.Tensor) else flux_data["area_Bc"]
                latent_Bc_A = expand_latent(latent, flux_data["r_Bc_A"].shape[1])
                latent_Bc_B = expand_latent(latent, flux_data["r_Bc_B"].shape[1])
            else:
                r_Bc_A, n_Bc_A, r_Bc_B, n_Bc_B, source_A_flux, source_B_flux, area_Bc, latent_Bc_A, latent_Bc_B = [None]*9

            # Forzar Z=0 en sensores
            r_m[:, 2] = 0.0
            if r_n is not None: r_n[:, 2] = 0.0
            source_data[:, 2] = 0.0
            source_data[:, 5] = 0.0

            with torch.autocast(device_type=device_type, enabled=use_amp, dtype=torch.float16 if device_type=="cuda" else torch.bfloat16):
                # Data Loss
                u_sec_m = u_net(r_m, source_data, latent=latent_data)
                u_pri_m = informer.compute_u_pri(r_m, source_data, current_I)
                u_tot_m = u_sec_m + u_pri_m
                
                if r_n is not None:
                    u_sec_n = u_net(r_n, source_data, latent=latent_data)
                    u_pri_n = informer.compute_u_pri(r_n, source_data, current_I)
                    u_tot_n = u_sec_n + u_pri_n
                    pred_u = u_tot_m - u_tot_n
                    pred_sec = u_sec_m - u_sec_n
                    target_sec = target - (u_pri_m - u_pri_n)
                else:
                    pred_u = u_tot_m
                    pred_sec = u_sec_m
                    target_sec = target - u_pri_m
                    
                # Aislamos la huella de la anomalía (Potencial Secundario)
                var_target_sec = torch.var(target_sec) + 1e-8
                loss_data = torch.mean((pred_sec - target_sec) ** 2) / var_target_sec

                # Física con Warm-up Lineal Seguro
                loss_pde = informer.compute_pde_loss(r_pde, source_pde, current_I, gamma, latent=latent_pde)
                loss_reg = informer.compute_reg_loss(r_reg, latent=latent_reg)
                loss_bc = informer.compute_bc_loss(r_neumann, r_dirichlet, source_neumann, source_dirichlet, latent_neumann, latent_dirichlet)
                loss_flux = informer.compute_flux_loss(r_Bc_A, n_Bc_A, r_Bc_B, n_Bc_B, source_A_flux, source_B_flux, current_I, area_Bc, latent_A=latent_Bc_A, latent_B=latent_Bc_B) if r_Bc_A is not None else torch.tensor(0.0, device=device_type)
                
                # Crecimiento suave hasta la época 100
                warmup = min(epoch / 100.0, 1.0)
                
                # Pesos estáticos: Sin Wang et al. para evitar explosiones
                w_pde = weights.get("w_pde", 1.0) * warmup
                w_reg = weights.get("w_reg", 0.01) * warmup
                w_bc = weights.get("w_bc", 1.0) * warmup
                w_flux = weights.get("w_flux", 1.0) * warmup
                
                loss_total = loss_data + (w_pde * loss_pde) + (w_reg * loss_reg) + (w_bc * loss_bc) + (w_flux * loss_flux)

            scaler.scale(loss_total / accumulation_steps).backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer_u)
                scaler.unscale_(optimizer_sigma)
                
                # CLIPPING ESTRICTO: La clave para que no vuelva a morir la red
                torch.nn.utils.clip_grad_norm_(u_net.parameters(), max_norm=5.0)
                torch.nn.utils.clip_grad_norm_(sigma_net.parameters(), max_norm=5.0)
                
                scaler.step(optimizer_u)
                scaler.step(optimizer_sigma)
                scaler.update()
                optimizer_u.zero_grad(set_to_none=True)
                optimizer_sigma.zero_grad(set_to_none=True)
                
            epoch_loss += loss_total.item()

        scheduler_u.step()
        scheduler_sigma.step()
        pbar_adam.set_postfix(loss=f"{epoch_loss/num_batches:.4e}", pde_w=f"{w_pde:.2f}")

        if use_wandb and wandb is not None:
            wandb.log({
                "epoch": epoch,
                "loss_total": epoch_loss / num_batches
            })

        # Guardado
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'u_net_state_dict': u_net.state_dict(),
                'sigma_net_state_dict': sigma_net.state_dict(),
                'encoder_state_dict': encoder.state_dict() if encoder is not None else None,
                'optimizer_u_state_dict': optimizer_u.state_dict(),
                'optimizer_sigma_state_dict': optimizer_sigma.state_dict(),
                'scheduler_u_state_dict': scheduler_u.state_dict(),
                'scheduler_sigma_state_dict': scheduler_sigma.state_dict(),
            }, checkpoint_path)

    return u_net, sigma_net, encoder