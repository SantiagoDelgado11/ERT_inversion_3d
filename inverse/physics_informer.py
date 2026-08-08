import torch
import torch.autograd as autograd


class PhysicsInformer:
    def __init__(self, cond_net, pot_net):
        self.cond_net = cond_net
        self.pot_net = pot_net

    def compute_derivatives(self, coords, source_coords=None):
        coords.requires_grad_(True)
        sigma = self.cond_net(coords)
        
        grad_sigma = autograd.grad(
            outputs=sigma,
            inputs=coords,
            grad_outputs=torch.ones_like(sigma),
            create_graph=True,
            retain_graph=True,
        )[0]
        ds_dx, ds_dy, ds_dz = grad_sigma[:, 0:1], grad_sigma[:, 1:2], grad_sigma[:, 2:3]

        if source_coords is not None:
            u_sec = self.pot_net(coords, source_coords)

            grad_u_sec = autograd.grad(
                outputs=u_sec,
                inputs=coords,
                grad_outputs=torch.ones_like(u_sec),
                create_graph=True,
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
                create_graph=True,
                retain_graph=True,
            )[0][:, 0:1]
            dflux_y = autograd.grad(
                outputs=flux_y,
                inputs=coords,
                grad_outputs=torch.ones_like(flux_y),
                create_graph=True,
                retain_graph=True,
            )[0][:, 1:2]
            dflux_z = autograd.grad(
                outputs=flux_z,
                inputs=coords,
                grad_outputs=torch.ones_like(flux_z),
                create_graph=True,
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

    def compute_grad_u_pri(self, coords, r_A, r_B, I, sigma_0=0.01):
        eps = 1e-6
        r_dist_A = coords - r_A
        d_A = torch.sqrt(torch.sum(r_dist_A**2, dim=-1, keepdim=True) + eps)
        grad_A = -r_dist_A / (d_A**3 + eps)
        
        r_dist_B = coords - r_B
        d_B = torch.sqrt(torch.sum(r_dist_B**2, dim=-1, keepdim=True) + eps)
        grad_B = -r_dist_B / (d_B**3 + eps)

        # Pole sources are encoded by B == A by ERTDataset.  Their primary
        # field is a monopole, not a dipole whose two terms cancel.
        is_pole = (torch.linalg.norm(r_A - r_B, dim=-1, keepdim=True) < 1e-5).to(grad_A.dtype)
        grad_B = grad_B * (1.0 - is_pole)
        
        return (I / (2 * torch.pi * sigma_0)) * (grad_A - grad_B)

    def compute_u_pri(self, coords, source_coords, I, sigma_0=0.01):
        if source_coords is None:
            return torch.zeros(coords.shape[0], 1, device=coords.device)
        r_A = source_coords[:, 0:3]
        r_B = source_coords[:, 3:6]
        eps = 1e-6
        d_A = torch.sqrt(torch.sum((coords - r_A)**2, dim=-1, keepdim=True) + eps)
        d_B = torch.sqrt(torch.sum((coords - r_B)**2, dim=-1, keepdim=True) + eps)
        is_pole = (torch.linalg.norm(r_A - r_B, dim=-1, keepdim=True) < 1e-5).to(d_A.dtype)
        secondary = (1.0 - is_pole) / d_B
        return (I / (2 * torch.pi * sigma_0)) * (1.0 / d_A - secondary)

    def compute_pde_loss(self, coords, source_coords, I, gamma, sigma_0=0.01):
        derivs = self.compute_derivatives(coords, source_coords)
        div_flux_sec = derivs["div_flux"]
        
        ds_dx, ds_dy, ds_dz = derivs["ds_dx"], derivs["ds_dy"], derivs["ds_dz"]
        grad_sigma = torch.cat([ds_dx, ds_dy, ds_dz], dim=-1)
        
        r_A = source_coords[:, 0:3]
        r_B = source_coords[:, 3:6]
        grad_u_pri = self.compute_grad_u_pri(coords, r_A, r_B, I, sigma_0)
        
        source_term = torch.sum(grad_sigma * grad_u_pri, dim=-1, keepdim=True)
        
        residual = div_flux_sec + source_term
        return torch.mean(residual**2)

    def compute_bc_loss(self, surface_coords, inf_coords, source_coords_surf, source_coords_inf):
        loss = None
        if surface_coords is not None and surface_coords.shape[0] > 0:
            derivs_surf = self.compute_derivatives(surface_coords, source_coords_surf)
            flux_z = derivs_surf["sigma"] * derivs_surf["du_dz"]
            loss_neumann = torch.mean(flux_z ** 2)
            loss = loss_neumann if loss is None else loss + loss_neumann

        if inf_coords is not None and inf_coords.shape[0] > 0:
            u_inf = self.pot_net(inf_coords, source_coords_inf)
            loss_dirichlet = torch.mean(u_inf**2)
            loss = loss_dirichlet if loss is None else loss + loss_dirichlet

        if loss is None:
            device = surface_coords.device if surface_coords is not None else inf_coords.device
            loss = torch.tensor(0.0, device=device)
        return loss

    def compute_reg_loss(self, coords, rho_bg=100.0):
        derivs = self.compute_derivatives(coords, source_coords=None)
        ds_dx, ds_dy, ds_dz = derivs["ds_dx"], derivs["ds_dy"], derivs["ds_dz"]
        sigma = derivs["sigma"]

        # Regularize log(sigma), not rho.  The latter amplifies gradients in
        # low-conductivity background and suppresses the anomaly contrast.
        inv_sigma = 1.0 / torch.clamp(sigma, min=1e-6)
        dlog_dx = ds_dx * inv_sigma
        dlog_dy = ds_dy * inv_sigma
        dlog_dz = ds_dz * inv_sigma

        # Edge-preserving TV (Huber norm approximation).
        eps_tv = 1e-2
        tv_norm = torch.sqrt(dlog_dx**2 + dlog_dy**2 + dlog_dz**2 + eps_tv**2) - eps_tv
        loss_tv = torch.mean(tv_norm)
        
        # Background reference penalty
        rho = 1.0 / sigma
        loss_bg = torch.mean((rho - rho_bg)**2) * 1e-4
        
        return loss_tv + loss_bg

    def compute_flux_loss(self, *args, **kwargs):
        """Enforce the injected/extracted current on control hemispheres.

        The outward flux of ``sigma * grad(u_total)`` is ``-I`` around an
        injection pole and ``+I`` around a finite sink.  Pole surveys encode
        the sink at infinity as ``B == A``; in that case only the injection
        hemisphere is constrained.
        """
        if len(args) >= 8:
            r_A, r_B, n_A, n_B, source_A, source_B, current_I, area = args[:8]
        elif len(args) >= 6:
            r_A, r_B, n_A, n_B, source_A, source_B = args[:6]
            current_I = kwargs.get("I", kwargs.get("current_I", 1.0))
            area = kwargs.get("area")
        else:
            raise TypeError("compute_flux_loss requiere al menos 6 argumentos")
        if area is None:
            raise TypeError("compute_flux_loss requiere el area de la superficie")
        if r_A is None or r_A.shape[0] == 0:
            return torch.tensor(0.0, device=self._device_from(r_B))

        def integrated_flux(coords, normals, source):
            derivs = self.compute_derivatives(coords, source)
            grad_primary = self.compute_grad_u_pri(
                coords,
                source[:, 0:3],
                source[:, 3:6],
                current_I,
            )
            grad_total = torch.cat(
                [derivs["du_dx"], derivs["du_dy"], derivs["du_dz"]], dim=-1
            ) + grad_primary
            flux_density = derivs["sigma"] * torch.sum(grad_total * normals, dim=-1, keepdim=True)
            return flux_density.mean() * area

        flux_A = integrated_flux(r_A, n_A, source_A)
        loss = (flux_A + current_I) ** 2

        finite_sink = torch.linalg.norm(
            source_B[:, 0:3] - source_B[:, 3:6], dim=-1
        ) >= 1e-5
        if torch.any(finite_sink):
            flux_B = integrated_flux(
                r_B[finite_sink], n_B[finite_sink], source_B[finite_sink]
            )
            loss = loss + (flux_B - current_I) ** 2
        return loss

    @staticmethod
    def _device_from(tensor):
        return tensor.device if tensor is not None else torch.device("cpu")
