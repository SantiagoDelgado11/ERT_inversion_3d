import torch
import torch.autograd as autograd

class PhysicsInformer:
    def __init__(self, cond_net, pot_net):
        """
        IMPORTANTE: Asegúrate de que `pot_net` utilice funciones de activación 
        suaves y doblemente derivables (como SiLU/Swish o Tanh). 
        Si usas ReLU, las segundas derivadas (div_flux) serán cero y la red colapsará.
        """
        self.cond_net = cond_net
        self.pot_net = pot_net

    def compute_derivatives(self, coords, source_coords=None, latent=None, create_graph=True):
        coords.requires_grad_(True)
        sigma = self.cond_net(coords, latent=latent)
        
        grad_sigma = autograd.grad(
            outputs=sigma,
            inputs=coords,
            grad_outputs=torch.ones_like(sigma),
            create_graph=create_graph,
            retain_graph=True,
        )[0]
        ds_dx, ds_dy, ds_dz = grad_sigma[:, 0:1], grad_sigma[:, 1:2], grad_sigma[:, 2:3]

        if source_coords is not None:
            u_sec = self.pot_net(coords, source_coords, latent=latent)

            grad_u_sec = autograd.grad(
                outputs=u_sec,
                inputs=coords,
                grad_outputs=torch.ones_like(u_sec),
                create_graph=create_graph,
                retain_graph=True,
            )[0]

            du_dx, du_dy, du_dz = grad_u_sec[:, 0:1], grad_u_sec[:, 1:2], grad_u_sec[:, 2:3]

            flux_x = sigma * du_dx
            flux_y = sigma * du_dy
            flux_z = sigma * du_dz

            dflux_x = autograd.grad(
                outputs=flux_x,
                inputs=coords,
                grad_outputs=torch.ones_like(flux_x),
                create_graph=create_graph,
                retain_graph=True,
            )[0][:, 0:1]
            
            dflux_y = autograd.grad(
                outputs=flux_y,
                inputs=coords,
                grad_outputs=torch.ones_like(flux_y),
                create_graph=create_graph,
                retain_graph=True,
            )[0][:, 1:2]
            
            dflux_z = autograd.grad(
                outputs=flux_z,
                inputs=coords,
                grad_outputs=torch.ones_like(flux_z),
                create_graph=create_graph,
                retain_graph=True,
            )[0][:, 2:3]

            return {
                "sigma": sigma,
                "ds_dx": ds_dx, "ds_dy": ds_dy, "ds_dz": ds_dz,
                "u": u_sec,
                "div_flux": dflux_x + dflux_y + dflux_z,
                "du_dx": du_dx,
                "du_dy": du_dy,
                "du_dz": du_dz,
            }

        return {
            "sigma": sigma,
            "ds_dx": ds_dx,
            "ds_dy": ds_dy,
            "ds_dz": ds_dz,
        }

    def compute_grad_u_pri(self, coords, r_A, r_B, I, sigma_0=0.01, eps_src=1e-3):
        smooth_factor = eps_src**2
        r_dist_A = coords - r_A
        d_A = torch.sqrt(torch.sum(r_dist_A**2, dim=-1, keepdim=True) + smooth_factor)
        grad_A = -r_dist_A / (d_A**3 + 1e-8)
        
        r_dist_B = coords - r_B
        d_B = torch.sqrt(torch.sum(r_dist_B**2, dim=-1, keepdim=True) + smooth_factor)
        grad_B = -r_dist_B / (d_B**3 + 1e-8)
        
        return (I / (2 * torch.pi * sigma_0)) * (grad_A - grad_B)

    def compute_u_pri(self, coords, source_coords, I, sigma_0=0.01, eps_src=1e-3):
        if source_coords is None:
            return torch.zeros(coords.shape[0], 1, device=coords.device)
        r_A = source_coords[:, 0:3]
        r_B = source_coords[:, 3:6]
        smooth_factor = eps_src**2
        d_A = torch.sqrt(torch.sum((coords - r_A)**2, dim=-1, keepdim=True) + smooth_factor)
        d_B = torch.sqrt(torch.sum((coords - r_B)**2, dim=-1, keepdim=True) + smooth_factor)
        return (I / (2 * torch.pi * sigma_0)) * (1.0 / d_A - 1.0 / d_B)

    def compute_pde_loss(self, coords, source_coords, I, gamma, sigma_0=0.01, latent=None, create_graph=True, return_residuals=False):
        derivs = self.compute_derivatives(coords, source_coords, latent=latent, create_graph=create_graph)
        div_flux_sec = derivs["div_flux"]
        sigma = derivs["sigma"]  # <-- EXTRAEMOS SIGMA AQUÍ
        
        ds_dx, ds_dy, ds_dz = derivs["ds_dx"], derivs["ds_dy"], derivs["ds_dz"]
        grad_sigma = torch.cat([ds_dx, ds_dy, ds_dz], dim=-1)
        
        r_A = source_coords[:, 0:3]
        r_B = source_coords[:, 3:6]
        grad_u_pri = self.compute_grad_u_pri(coords, r_A, r_B, I, sigma_0)
        
        source_term = torch.sum(grad_sigma * grad_u_pri, dim=-1, keepdim=True)
        
        # ====================================================================
        # LA SOLUCIÓN ANTI-TRAMPA DEFINITIVA (Formulación Invariante)
        # ====================================================================
        # Al dividir el residuo crudo por sigma, bloqueamos matemáticamente
        # que la red pueda hacer trampa llevando la resistividad a 10,000.
        # Le sumamos 1e-8 por pura seguridad numérica.
        residual = (div_flux_sec + source_term) / (sigma + 1e-8)
        
        if return_residuals:
            return residual
            
        # Multiplicamos por 1e3 para mantener el gradiente vivo sin que explote
        scaled_residual = residual * 1e3
        
        return torch.nn.functional.mse_loss(scaled_residual, torch.zeros_like(scaled_residual))

    def compute_bc_loss(self, surface_coords, inf_coords, source_coords_surf, source_coords_inf, latent_surf=None, latent_inf=None):
        loss = None
        if surface_coords is not None and surface_coords.shape[0] > 0:
            derivs_surf = self.compute_derivatives(surface_coords, source_coords_surf, latent=latent_surf)
            flux_z = derivs_surf["sigma"] * derivs_surf["du_dz"]
            loss_neumann = torch.mean(flux_z ** 2)
            loss = loss_neumann if loss is None else loss + loss_neumann

        if inf_coords is not None and inf_coords.shape[0] > 0:
            u_inf = self.pot_net(inf_coords, source_coords_inf, latent=latent_inf)
            loss_dirichlet = torch.mean(u_inf**2)
            loss = loss_dirichlet if loss is None else loss + loss_dirichlet

        if loss is None:
            device = surface_coords.device if surface_coords is not None else inf_coords.device
            loss = torch.tensor(0.0, device=device)
        return loss

    def compute_reg_loss(self, coords, rho_bg=100.0, latent=None):
        derivs = self.compute_derivatives(coords, source_coords=None, latent=latent)
        ds_dx, ds_dy, ds_dz = derivs["ds_dx"], derivs["ds_dy"], derivs["ds_dz"]
        sigma = derivs["sigma"]

        # SOLUCIÓN: Calcular TV sobre el logaritmo de sigma en lugar de rho.
        # d(ln(sigma))/dx = ds_dx / sigma. Esto es estable y penaliza cambios relativos.
        safe_sigma = sigma + 1e-8
        dlog_sigma_dx = ds_dx / safe_sigma
        dlog_sigma_dy = ds_dy / safe_sigma
        dlog_sigma_dz = ds_dz / safe_sigma

        # Edge-Preserving TV (Huber norm approx)
        eps_tv = 1e-5
        tv_norm = torch.sqrt(dlog_sigma_dx**2 + dlog_sigma_dy**2 + dlog_sigma_dz**2 + eps_tv**2) - eps_tv
        
        # Depth weighting
        z = coords[:, 2:3]
        depth_weight = 1.0 / (1.0 + 0.2 * torch.abs(z))**2
        
        loss_tv = torch.mean(tv_norm * depth_weight)
        
        return loss_tv

    def compute_flux_loss(self, r_Bc_A, n_Bc_A, r_Bc_B, n_Bc_B, source_A, source_B, I, area_Bc, sigma_0=0.01, latent_A=None, latent_B=None):
        if r_Bc_A is None or r_Bc_B is None:
            return torch.tensor(0.0, device=r_Bc_A.device if r_Bc_A is not None else torch.device("cpu"))
        
        # Flux around electrode A (source, current = I)
        derivs_A = self.compute_derivatives(r_Bc_A, source_A, latent=latent_A)
        sigma_A = derivs_A["sigma"]
        grad_u_sec_A = torch.cat([derivs_A["du_dx"], derivs_A["du_dy"], derivs_A["du_dz"]], dim=-1)
        grad_u_pri_A = self.compute_grad_u_pri(r_Bc_A, source_A[:, 0:3], source_A[:, 3:6], I, sigma_0)
        grad_u_tot_A = grad_u_sec_A + grad_u_pri_A
        # n_Bc_A points outward from the hemisphere (into the ground)
        J_n_A = -sigma_A * torch.sum(grad_u_tot_A * n_Bc_A, dim=-1, keepdim=True)
        flux_A = area_Bc * torch.mean(J_n_A)
        loss_flux_A = (flux_A - I)**2

        # Flux around electrode B (sink, current = -I)
        derivs_B = self.compute_derivatives(r_Bc_B, source_B, latent=latent_B)
        sigma_B = derivs_B["sigma"]
        grad_u_sec_B = torch.cat([derivs_B["du_dx"], derivs_B["du_dy"], derivs_B["du_dz"]], dim=-1)
        grad_u_pri_B = self.compute_grad_u_pri(r_Bc_B, source_B[:, 0:3], source_B[:, 3:6], I, sigma_0)
        grad_u_tot_B = grad_u_sec_B + grad_u_pri_B
        J_n_B = -sigma_B * torch.sum(grad_u_tot_B * n_Bc_B, dim=-1, keepdim=True)
        flux_B = area_Bc * torch.mean(J_n_B)
        loss_flux_B = (flux_B - (-I))**2

        return loss_flux_A + loss_flux_B