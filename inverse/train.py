import logging
import math
from typing import Any, Dict

import torch
import torch.optim as optim
from tqdm import tqdm

try:
    import wandb
except ImportError:  # pragma: no cover - optional experiment tracking
    wandb = None

logger = logging.getLogger(__name__)





def train_pinn(
    u_net: torch.nn.Module,
    sigma_net: torch.nn.Module,
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
):
    del num_epochs_lbfgs  # Reserved for a future second-stage optimizer.

    if len(dataloader) == 0:
        raise ValueError("El dataloader está vacío. No hay datos para entrenar.")

    u_net = u_net.to(device)
    sigma_net = sigma_net.to(device)

    # Optimizador conjunto para permitir el flujo simultáneo del gradiente
    optimizer_joint = optim.Adam(list(u_net.parameters()) + list(sigma_net.parameters()), lr=lr)
    
    # Se utiliza CosineAnnealingWarmRestarts para permitir escapar de mínimos locales
    # El scheduler se actualiza por época (no por batch) porque T_0=200 está dimensionado en épocas.
    scheduler_joint = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_joint, T_0=200, T_mult=2, eta_min=1e-6)

    def prepare(tensor: torch.Tensor, requires_grad: bool = False):
        t = tensor.to(device)
        return t.requires_grad_(True) if requires_grad else t

    base_weights = {
        "data_u": weights.get("w_data", 1.0),
        "data_sigma": weights.get("w_data", 1.0),
        "pde": weights.get("w_pde", 1.0),
        "bc": weights.get("w_bc", 1.0),
        "flux": weights.get("w_flux", 1.0),
        "reg": weights.get("w_reg", 1e-4)
    }

    # Gradient-based Adaptive Weighting (Wang et al., 2021)
    # ReLoBRaLo has been removed in favor of gradient pathology mitigation
    dynamic_lambdas = {k: 1.0 for k in base_weights.keys()}
    grad_ema_alpha = 0.9

    def _check_numerical_stability(loss_dict: Dict[str, torch.Tensor], model_outputs: Dict[str, torch.Tensor]):
        """Auditoría estricta antes de aplicar gradientes (Fail-Fast)."""
        for name, loss in loss_dict.items():
            if not torch.isfinite(loss):
                raise ValueError(f"Inestabilidad numérica detectada: {name} es {loss.item()}. Entrenamiento detenido.")
        
        for name, output in model_outputs.items():
            if not torch.isfinite(output).all():
                raise ValueError(f"Inestabilidad numérica detectada: Las predicciones de {name} contienen NaN o Inf.")

    def _log_layer_gradients(model: torch.nn.Module, prefix: str):
        """Registra normas L2 del gradiente por capa para detectar Vanishing Gradients locales."""
        layer_grads = {}
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                norm = param.grad.data.norm(2).item()
                layer_grads[f"grad_layer_{prefix}/{name}"] = norm
                if norm < 1e-10:
                    logger.warning(f"Capa crítica sin gradiente: {name} en {prefix} (norma: {norm:.2e})")
        return layer_grads

    def squeeze_dict(d):
        return {k: v[0] for k, v in d.items() if isinstance(v, torch.Tensor)} or d

    def _train_joint_step(dyn_weights, t):
        """
        Entrenamiento Conjunto (Joint Optimization) de PotentialNet y ConductivityNet.
        Permite el flujo simultáneo del gradiente a través del colector de soluciones físicas.
        """
        optimizer_joint.zero_grad()

        # Extraer tensores
        r_m = t["r_m"]
        r_n = t["r_n"]
        data_target = t["data_target"]
        source_data = t["source_data"]
        r_pde = t["r_pde"]
        source_coords_pde = t["source_coords_pde"]
        r_N = t["r_N"]
        source_coords_neumann = t["source_coords_neumann"]
        r_D = t["r_D"]
        source_coords_dirichlet = t["source_coords_dirichlet"]
        r_Bc_A = t["r_Bc_A"]
        n_Bc_A = t["n_Bc_A"]
        r_Bc_B = t["r_Bc_B"]
        n_Bc_B = t["n_Bc_B"]
        source_coords_flux_A = t["source_coords_flux_A"]
        source_coords_flux_B = t["source_coords_flux_B"]
        area_Bc = t["area_Bc"]
        r_reg = t["r_reg"]

        # Restricción física: Forzar Z=0 para mediciones de superficie
        # Esto elimina fugas geométricas en la superficie causadas por precisión flotante
        if r_m is not None:
            r_m = r_m.clone()
            r_m[:, 2] = 0.0
        if r_n is not None:
            r_n = r_n.clone()
            r_n[:, 2] = 0.0
        if source_data is not None:
            source_data = source_data.clone()
            source_data[:, 2] = 0.0
            if source_data.shape[1] == 6:
                source_data[:, 5] = 0.0

        def _compute_data_loss():
            if source_data is None:
                return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), None, None

            # --- SUBSAMPLING MEASUREMENTS FOR VRAM EFFICIENCY ---
            # ERT surveys often have ~1000 measurements per sample (e.g. 929).
            # Computing higher-order derivatives for 1000 measurements simultaneously 
            # requires ~12-14 GB of VRAM. We apply Stochastic Gradient Descent over
            # the measurements (mini-batching) to dramatically reduce VRAM.
            num_meas = source_data.shape[0]
            max_meas = 64
            if num_meas > max_meas:
                idx_meas = torch.randperm(num_meas, device=device)[:max_meas]
                sub_source = source_data[idx_meas]
                sub_r_m = r_m[idx_meas] if r_m is not None else None
                sub_r_n = r_n[idx_meas] if r_n is not None else None
                sub_target = data_target[idx_meas] if data_target is not None else None
                # Evaluamos el RMSE global sin gradientes para el logueo
                with torch.no_grad():
                    u_pri_m_full = informer.compute_u_pri(r_m, source_data, current_I)
                    u_tot_m_full = u_net(r_m, source_data) + u_pri_m_full
                    if r_n is not None:
                        u_pri_n_full = informer.compute_u_pri(r_n, source_data, current_I)
                        u_tot_n_full = u_net(r_n, source_data) + u_pri_n_full
                        pred_u_full = u_tot_m_full - u_tot_n_full
                    else:
                        pred_u_full = u_tot_m_full
            else:
                sub_source = source_data
                sub_r_m = r_m
                sub_r_n = r_n
                sub_target = data_target
                pred_u_full = None

            # 1. Pérdida estándar (ajusta u_net, que ahora predice u_sec)
            u_sec_m = u_net(sub_r_m, sub_source)
            u_pri_m = informer.compute_u_pri(sub_r_m, sub_source, current_I)
            u_tot_m = u_sec_m + u_pri_m
            
            if sub_r_n is not None:
                u_sec_n = u_net(sub_r_n, sub_source)
                u_pri_n = informer.compute_u_pri(sub_r_n, sub_source, current_I)
                u_tot_n = u_sec_n + u_pri_n
                pred_u = u_tot_m - u_tot_n
            else:
                pred_u = u_tot_m

            loss_data_u = torch.mean((pred_u - sub_target) ** 2)

            # 2. Pérdida de reciprocidad (ajusta sigma_net)
            # Desactivada temporalmente: La integral de Monte Carlo sobre todo el volumen 
            # tiene varianza infinita cerca de las singularidades de los electrodos, 
            # causando colapso en sigma_net.
            loss_data_sigma = torch.tensor(0.0, device=device, requires_grad=True)

            # Devolvemos las predicciones completas para el logging (RMSE) si aplicó subsampling
            final_pred = pred_u_full if pred_u_full is not None else pred_u
            return loss_data_u, loss_data_sigma, final_pred, data_target

        loss_data_u, loss_data_sigma, pred_data, true_data = _compute_data_loss()
        loss_pde = informer.compute_pde_loss(r_pde, source_coords_pde, current_I, gamma)
        loss_bc = informer.compute_bc_loss(
            surface_coords=r_N,
            inf_coords=r_D,
            source_coords_surf=source_coords_neumann if r_N.shape[0] > 0 else None,
            source_coords_inf=source_coords_dirichlet if r_D.shape[0] > 0 else None,
        )
        loss_reg = informer.compute_reg_loss(r_reg)
        loss_flux = informer.compute_flux_loss(
            r_Bc_A, r_Bc_B, n_Bc_A, n_Bc_B,
            source_coords_flux_A, source_coords_flux_B,
            current_I, area_Bc, gamma=gamma,
        )
        
        grad_norm_sigma_data = 0.0
        if loss_data_sigma.requires_grad:
            grads = torch.autograd.grad(loss_data_sigma, sigma_net.parameters(), allow_unused=True, retain_graph=True)
            for g in grads:
                if g is not None:
                    grad_norm_sigma_data += g.norm(2).item() ** 2
            grad_norm_sigma_data = math.sqrt(grad_norm_sigma_data)

        # Implementación de ponderación adaptativa basada en la magnitud de gradientes (Wang et al. 2021)
        if loss_pde.requires_grad and loss_data_u.requires_grad:
            shared_params = list(u_net.parameters())
            
            grads_data = torch.autograd.grad(loss_data_u, shared_params, retain_graph=True, allow_unused=True)
            gn_data = torch.sqrt(sum(g.pow(2).sum() for g in grads_data if g is not None) + 1e-8)
            
            grads_pde = torch.autograd.grad(loss_pde, shared_params, retain_graph=True, allow_unused=True)
            gn_pde = torch.sqrt(sum(g.pow(2).sum() for g in grads_pde if g is not None) + 1e-8)
            
            # Evitar que la PDE domine sobre los datos (escalamos PDE hacia Data)
            lambda_pde = gn_data.detach() / (gn_pde.detach() + 1e-8)
            
            # Actualización EMA del peso de la PDE
            dyn_weights["pde"] = grad_ema_alpha * dyn_weights["pde"] + (1 - grad_ema_alpha) * lambda_pde.item()

        loss_total = (dyn_weights["data_u"] * base_weights["data_u"] * loss_data_u + 
                      dyn_weights["data_sigma"] * base_weights["data_sigma"] * loss_data_sigma +
                      dyn_weights["pde"] * base_weights["pde"] * loss_pde + 
                      dyn_weights["bc"] * base_weights["bc"] * loss_bc + 
                      dyn_weights["flux"] * base_weights["flux"] * loss_flux +
                      dyn_weights["reg"] * base_weights["reg"] * loss_reg)

        # Métricas de evaluación para los datos
        with torch.no_grad():
            if pred_data is not None and true_data is not None:
                rmse_val = torch.sqrt(torch.mean((pred_data - true_data)**2)).item()
                pred_mean = torch.mean(pred_data)
                true_mean = torch.mean(true_data)
                cov = torch.mean((pred_data - pred_mean) * (true_data - true_mean))
                var_pred = torch.mean((pred_data - pred_mean)**2)
                var_true = torch.mean((true_data - true_mean)**2)
                corr_val = (cov / torch.sqrt(var_pred * var_true + 1e-8)).item()
            else:
                rmse_val = 0.0
                corr_val = 0.0

        # Comprobaciones de seguridad numérica
        with torch.no_grad():
            u_pred = u_net(r_m, source_data) if source_data is not None else torch.zeros(1, device=device)
            sigma_pred = sigma_net(r_reg)
        
        _check_numerical_stability(
            {"loss_total": loss_total, "loss_data_u": loss_data_u, "loss_data_sigma": loss_data_sigma, "loss_pde": loss_pde, "loss_bc": loss_bc, "loss_flux": loss_flux},
            {"u_pred": u_pred, "sigma_pred": sigma_pred}
        )

        loss_total.backward()

        # Extraer gradientes por capa
        u_layer_grads = _log_layer_gradients(u_net, "unet")
        sigma_layer_grads = _log_layer_gradients(sigma_net, "sigmanet")

        # Monitoreo de normas de gradientes y clipping
        grad_norm_u_pre_clip = torch.nn.utils.clip_grad_norm_(u_net.parameters(), max_norm=1.0)
        gn_u_pre = grad_norm_u_pre_clip.item() if isinstance(grad_norm_u_pre_clip, torch.Tensor) else grad_norm_u_pre_clip
        gn_u_post = min(gn_u_pre, 1.0)

        grad_norm_sigma_pre_clip = torch.nn.utils.clip_grad_norm_(sigma_net.parameters(), max_norm=1.0)
        gn_s_pre = grad_norm_sigma_pre_clip.item() if isinstance(grad_norm_sigma_pre_clip, torch.Tensor) else grad_norm_sigma_pre_clip
        gn_s_post = min(gn_s_pre, 1.0)
        
        optimizer_joint.step()
        
        return loss_total, loss_data_u, loss_data_sigma, loss_pde, loss_bc, loss_reg, loss_flux, gn_u_pre, gn_u_post, gn_s_pre, gn_s_post, u_layer_grads, sigma_layer_grads, grad_norm_sigma_data, rmse_val, corr_val

    print("Iniciando entrenamiento PINN con optimizacion conjunta (ReLoBRaLo)")
    pbar_adam = tqdm(range(num_epochs_adam), desc="Adam")

    consecutive_degenerate_epochs = 0
    dyn_weights = base_weights.copy()

    for epoch in pbar_adam:
        # Acumuladores de métricas de la época
        epoch_losses = {k: 0.0 for k in ["total", "data_u", "data_sigma", "pde", "bc", "reg", "flux"]}
        epoch_rmse = 0.0
        epoch_corr = 0.0
        epoch_gn_s_data = 0.0
        num_batches = len(dataloader)
        
        # Variables para gradientes y estadísticas
        last_gn_u_pre = last_gn_u_post = last_gn_s_pre = last_gn_s_post = 0.0
        last_u_layer_grads = {}
        last_sigma_layer_grads = {}
        epoch_sigma_preds = []

        # Barra de progreso interna para las 1000 iteraciones (batches) de la época
        pbar_batches = tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs_adam}", leave=False)
        for batch in pbar_batches:
            data_samples = squeeze_dict(batch["data"])
            pde_samples = squeeze_dict(batch["pde"])
            bc_neumann_samples = squeeze_dict(batch["bc_neumann"])
            bc_dirichlet_samples = squeeze_dict(batch["bc_dirichlet"])
            flux_samples = squeeze_dict(batch["flux"])
            flux_samples["area_Bc"] = batch["flux"]["area_Bc"][0].item()
            reg_samples = squeeze_dict(batch["reg"])
            
            # Preparar todos los tensores
            r_pde = prepare(pde_samples["r"], requires_grad=True)
            r_N = prepare(bc_neumann_samples["r_N"], requires_grad=True)
            r_D = prepare(bc_dirichlet_samples["r_D"])
            r_Bc_A = prepare(flux_samples["r_Bc_A"], requires_grad=True)
            r_Bc_B = prepare(flux_samples["r_Bc_B"], requires_grad=True)

            source_coords_pde = (
                prepare(pde_samples["source"])
                if "source" in pde_samples
                else torch.cat([prepare(pde_samples["r_A"]), prepare(pde_samples["r_B"])], dim=-1)
            )

            tensors = {
                "r_m": prepare(data_samples["r_m"]),
                "r_n": prepare(data_samples["r_n"]) if "r_n" in data_samples else None,
                "data_target": prepare(data_samples.get("delta_v", data_samples["u_star"])),
                "source_data": prepare(data_samples["source"]) if "source" in data_samples else None,
                "r_pde": r_pde,
                "source_coords_pde": source_coords_pde,
                "r_N": r_N,
                "source_coords_neumann": (
                    prepare(bc_neumann_samples["source"])
                    if "source" in bc_neumann_samples
                    else source_coords_pde[: r_N.shape[0]]
                ),
                "r_D": r_D,
                "source_coords_dirichlet": (
                    prepare(bc_dirichlet_samples["source"])
                    if "source" in bc_dirichlet_samples
                    else source_coords_pde[: r_D.shape[0]]
                ),
                "r_Bc_A": r_Bc_A,
                "n_Bc_A": prepare(flux_samples["n_Bc_A"]),
                "r_Bc_B": r_Bc_B,
                "n_Bc_B": prepare(flux_samples["n_Bc_B"]),
                "source_coords_flux_A": (
                    prepare(flux_samples["source_A"])
                    if "source_A" in flux_samples
                    else source_coords_pde[: r_Bc_A.shape[0]]
                ),
                "source_coords_flux_B": (
                    prepare(flux_samples["source_B"])
                    if "source_B" in flux_samples
                    else source_coords_pde[: r_Bc_B.shape[0]]
                ),
                "area_Bc": flux_samples["area_Bc"],
                "r_reg": prepare(reg_samples["r_reg"], requires_grad=True)
            }

            # Entrenamiento Conjunto del Batch
            res = _train_joint_step(dyn_weights, tensors)
            loss_total, loss_data_u, loss_data_sigma, loss_pde, loss_bc, loss_reg, loss_flux, gn_u_pre, gn_u_post, gn_s_pre, gn_s_post, u_layer_grads, sigma_layer_grads, grad_norm_sigma_data, rmse_val, corr_val = res
            
            # Acumular
            epoch_losses["total"] += loss_total.item()
            epoch_losses["data_u"] += loss_data_u.item()
            epoch_losses["data_sigma"] += loss_data_sigma.item()
            epoch_losses["pde"] += loss_pde.item()
            epoch_losses["bc"] += loss_bc.item()
            epoch_losses["reg"] += loss_reg.item()
            epoch_losses["flux"] += loss_flux.item()
            epoch_rmse += rmse_val
            epoch_corr += corr_val
            epoch_gn_s_data += grad_norm_sigma_data
            
            # Guardar último batch para logs de capas y chequeos
            last_gn_u_pre, last_gn_u_post, last_gn_s_pre, last_gn_s_post = gn_u_pre, gn_u_post, gn_s_pre, gn_s_post
            last_u_layer_grads = u_layer_grads
            last_sigma_layer_grads = sigma_layer_grads
            
            # Evaluamos conductividad para las estadísticas de la época
            with torch.no_grad():
                sigma_pred = sigma_net(tensors["r_reg"])
                epoch_sigma_preds.append(sigma_pred.detach().cpu())

        # Promediar métricas de la época
        for k in epoch_losses:
            epoch_losses[k] /= num_batches
        epoch_rmse /= num_batches
        epoch_corr /= num_batches
        epoch_gn_s_data /= num_batches

        # Calcular estadísticas de conductividad para toda la época
        with torch.no_grad():
            all_sigma_preds = torch.cat(epoch_sigma_preds, dim=0)
            epoch_sigma_stats = {
                "sigma_min": all_sigma_preds.min().item(),
                "sigma_max": all_sigma_preds.max().item(),
                "sigma_mean": all_sigma_preds.mean().item(),
                "sigma_std": all_sigma_preds.std().item(),
                "sigma_p5": torch.quantile(all_sigma_preds, 0.05).item(),
                "sigma_p95": torch.quantile(all_sigma_preds, 0.95).item(),
                "sigma_pred": all_sigma_preds
            }

        # Avisos de Vanishing Gradients Globales (del último batch)
        if last_gn_u_pre < 1e-10:
            logger.warning(f"Epoch {epoch}: ALERTA - Gradiente global desvanecido en PotentialNet (Norma: {last_gn_u_pre:.2e})")
        if last_gn_s_pre < 1e-10:
            logger.warning(f"Epoch {epoch}: ALERTA - Gradiente global desvanecido en ConductivityNet (Norma: {last_gn_s_pre:.2e})")

        # Los pesos se actualizan dinámicamente dentro de _train_joint_step
        # para aplicar la mitigación de la patología de gradientes en cada iteración.
        # Por lo tanto, no es necesario un actualizador por época aquí.
        
        # El scheduler se actualiza por época
        scheduler_joint.step()

        current_lr = optimizer_joint.param_groups[0]["lr"]

        # Diagnóstico de degeneración
        sigma_mean = epoch_sigma_stats["sigma_mean"]
        sigma_std = epoch_sigma_stats["sigma_std"]
        if sigma_std < 0.01 * abs(sigma_mean):
            consecutive_degenerate_epochs += 1
            if consecutive_degenerate_epochs >= 50:
                logger.warning(f"Epoch {epoch}: ALERTA - La conductividad está colapsando hacia una constante (std < 1% de mean).")
        else:
            consecutive_degenerate_epochs = 0

        # Cálculos de escala relativa para evaluación del balance
        ratio_data_pde = epoch_losses["data_u"] / (epoch_losses["pde"] + 1e-8)
        ratio_flux_pde = epoch_losses["flux"] / (epoch_losses["pde"] + 1e-8)
        ratio_reg_pde = epoch_losses["reg"] / (epoch_losses["pde"] + 1e-8)

        loss_dict = {
            "loss_data_u": epoch_losses["data_u"],
            "loss_data_sigma": epoch_losses["data_sigma"],
            "loss_data_unweighted": epoch_losses["data_u"],
            "RMSE_delta_V": epoch_rmse,
            "corr_delta_V": epoch_corr,
            "grad_norm_sigma_data": epoch_gn_s_data,
            "loss_pde": epoch_losses["pde"],
            "loss_bc": epoch_losses["bc"],
            "loss_reg": epoch_losses["reg"],
            "loss_flux": epoch_losses["flux"],
            "loss_total": epoch_losses["total"],
            "learning_rate": current_lr,
            "sigma_min": epoch_sigma_stats["sigma_min"],
            "sigma_max": epoch_sigma_stats["sigma_max"],
            "sigma_mean": sigma_mean,
            "sigma_std": sigma_std,
            "sigma_p5": epoch_sigma_stats["sigma_p5"],
            "sigma_p95": epoch_sigma_stats["sigma_p95"],
            "gradient_norm_u_pre": last_gn_u_pre,
            "gradient_norm_sigma_pre": last_gn_s_pre,
            "gradient_norm_u_post": last_gn_u_post,
            "gradient_norm_sigma_post": last_gn_s_post,
            "dyn_w_data_u": dyn_weights["data_u"],
            "dyn_w_data_sigma": dyn_weights["data_sigma"],
            "dyn_w_pde": dyn_weights["pde"],
            "dyn_w_bc": dyn_weights["bc"],
            "dyn_w_flux": dyn_weights["flux"],
            "dyn_w_reg": dyn_weights["reg"],
            "ratio_data_pde": ratio_data_pde,
            "ratio_flux_pde": ratio_flux_pde,
            "ratio_reg_pde": ratio_reg_pde,
            **last_u_layer_grads,
            **last_sigma_layer_grads
        }

        if use_wandb and wandb is not None:
            log_payload = {"epoch_adam": epoch, **loss_dict}
            
            # Registrar histograma completo de la conductividad cada ciertas épocas
            if epoch % 50 == 0:
                log_payload["sigma_histogram"] = wandb.Histogram(epoch_sigma_stats["sigma_pred"])
                
            wandb.log(log_payload)
            
        pbar_adam.set_postfix(loss=f"{epoch_losses['total']:.4e}", lr=f"{current_lr:.2e}")

    return u_net, sigma_net

