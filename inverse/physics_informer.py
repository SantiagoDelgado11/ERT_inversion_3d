import torch
import torch.autograd as autograd


class PhysicsInformer:
    """
    Differential engine for PDE, boundary, flux, and regularization losses.

    Sign convention used here:
        J = -sigma * grad(u)
        div(J) = q
        -div(sigma * grad(u)) = q
    Therefore the strong-form residual is div(sigma * grad(u)) + q = 0.
    """

    def __init__(self, cond_net, pot_net):
        self.cond_net = cond_net
        self.pot_net = pot_net

    def compute_derivatives(self, coords, source_coords=None):
        coords.requires_grad_(True)

        sigma = self.cond_net(coords)

        if source_coords is not None:
            u = self.pot_net(coords, source_coords)

            grad_u = autograd.grad(
                outputs=u,
                inputs=coords,
                grad_outputs=torch.ones_like(u),
                create_graph=True,
                retain_graph=True,
            )[0]

            du_dx, du_dy, du_dz = grad_u[:, 0:1], grad_u[:, 1:2], grad_u[:, 2:3]

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
                "u": u,
                "div_flux": dflux_x + dflux_y + dflux_z,
                "du_dx": du_dx,
                "du_dy": du_dy,
                "du_dz": du_dz,
            }

        grad_sigma = autograd.grad(
            outputs=sigma,
            inputs=coords,
            grad_outputs=torch.ones_like(sigma),
            create_graph=True,
            retain_graph=True,
        )[0]
        return {
            "sigma": sigma,
            "ds_dx": grad_sigma[:, 0:1],
            "ds_dy": grad_sigma[:, 1:2],
            "ds_dz": grad_sigma[:, 2:3],
        }

    def _gaussian_source(self, coords, source_pos, I, gamma):
        gamma_t = torch.as_tensor(gamma, dtype=coords.dtype, device=coords.device)
        current_t = torch.as_tensor(I, dtype=coords.dtype, device=coords.device)
        dist_sq = torch.sum((coords - source_pos) ** 2, dim=1, keepdim=True)
        coeff = current_t / ((2 * torch.pi) ** 1.5 * gamma_t**3)
        return coeff * torch.exp(-dist_sq / (2 * gamma_t**2))

    def _regularized_source(self, coords, r_A, r_B, I, gamma):
        """
        Smooth dipole source normalized over the physical half-space z <= 0.

        A full-space Gaussian centered on the air-ground interface contributes
        only half its mass inside the modeled earth. Surface electrodes therefore
        receive a factor of two so that each pole integrates to +/- I in-domain.
        """
        q_A = self._half_space_gaussian(coords, r_A, I, gamma)
        q_B = self._half_space_gaussian(coords, r_B, I, gamma)
        return q_A - q_B

    def _half_space_gaussian(self, coords, source_pos, I, gamma):
        source = self._gaussian_source(coords, source_pos, I, gamma)
        surface_factor = torch.where(
            torch.abs(source_pos[:, 2:3]) < 1e-6,
            torch.full_like(source, 2.0),
            torch.ones_like(source),
        )
        return surface_factor * source

    def compute_pde_loss(self, coords, source_coords, I, gamma):
        derivs = self.compute_derivatives(coords, source_coords)
        lhs = derivs["div_flux"]

        r_A = source_coords[:, 0:3]
        r_B = source_coords[:, 3:6]
        q_rhs = self._regularized_source(coords, r_A, r_B, I, gamma)

        residual = lhs + q_rhs
        return torch.mean(residual**2)

    def compute_bc_loss(self, surface_coords, inf_coords, source_coords_surf, source_coords_inf):
        loss = None

        if surface_coords is not None and surface_coords.shape[0] > 0:
            derivs_surf = self.compute_derivatives(surface_coords, source_coords_surf)
            loss_neumann = torch.mean(derivs_surf["du_dz"] ** 2)
            loss = loss_neumann if loss is None else loss + loss_neumann

        if inf_coords is not None and inf_coords.shape[0] > 0:
            u_inf = self.pot_net(inf_coords, source_coords_inf)
            loss_dirichlet = torch.mean(u_inf**2)
            loss = loss_dirichlet if loss is None else loss + loss_dirichlet

        if loss is None:
            device = surface_coords.device if surface_coords is not None else inf_coords.device
            loss = torch.tensor(0.0, device=device)
        return loss

    def compute_reg_loss(self, coords):
        derivs = self.compute_derivatives(coords, source_coords=None)
        ds_dx, ds_dy, ds_dz = derivs["ds_dx"], derivs["ds_dy"], derivs["ds_dz"]
        sigma = derivs["sigma"]

        # TV on resistivity rho = 1/sigma -> grad(rho) = -grad(sigma) / sigma^2
        sigma_sq = sigma ** 2
        drho_dx = -ds_dx / sigma_sq
        drho_dy = -ds_dy / sigma_sq
        drho_dz = -ds_dz / sigma_sq

        eps_tv = 1e-4
        tv_norm = torch.sqrt(drho_dx**2 + drho_dy**2 + drho_dz**2 + eps_tv**2)
        return torch.mean(tv_norm)

    def _gaussian_enclosed_fraction(self, radius, gamma, device, dtype):
        radius_t = torch.as_tensor(radius, dtype=dtype, device=device)
        gamma_t = torch.as_tensor(gamma, dtype=dtype, device=device)
        scaled = radius_t / gamma_t
        sqrt_two = torch.sqrt(torch.as_tensor(2.0, dtype=dtype, device=device))
        sqrt_two_over_pi = torch.sqrt(torch.as_tensor(2.0 / torch.pi, dtype=dtype, device=device))
        fraction = torch.erf(scaled / sqrt_two)
        fraction = fraction - sqrt_two_over_pi * scaled * torch.exp(-0.5 * scaled**2)
        return torch.clamp(fraction, min=0.0, max=1.0)

    def compute_flux_loss(
        self,
        coords_A,
        coords_B,
        normals_A,
        normals_B,
        source_coords_A,
        source_coords_B,
        I,
        area,
        gamma=None,
    ):
        device = coords_A.device if coords_A is not None else coords_B.device
        dtype = coords_A.dtype if coords_A is not None else coords_B.dtype
        loss = torch.tensor(0.0, device=device, dtype=dtype)

        area_t = torch.as_tensor(area, dtype=dtype, device=device)
        current_t = torch.as_tensor(I, dtype=dtype, device=device)
        if gamma is not None:
            radius = torch.sqrt(area_t / (2 * torch.pi))
            enclosed = self._gaussian_enclosed_fraction(radius, gamma, device, dtype)
        else:
            enclosed = torch.tensor(1.0, device=device, dtype=dtype)
        target_flux = current_t * enclosed / area_t

        if coords_A is not None and coords_A.shape[0] > 0:
            derivs_A = self.compute_derivatives(coords_A, source_coords_A)
            grad_u_A = torch.cat([derivs_A["du_dx"], derivs_A["du_dy"], derivs_A["du_dz"]], dim=1)
            flux_A = torch.sum(derivs_A["sigma"] * grad_u_A * normals_A, dim=1, keepdim=True)
            # For the injection pole, sigma*grad(u).n is negative because J=-sigma*grad(u).
            loss = loss + (torch.mean(flux_A) + target_flux) ** 2

        if coords_B is not None and coords_B.shape[0] > 0:
            derivs_B = self.compute_derivatives(coords_B, source_coords_B)
            grad_u_B = torch.cat([derivs_B["du_dx"], derivs_B["du_dy"], derivs_B["du_dz"]], dim=1)
            flux_B = torch.sum(derivs_B["sigma"] * grad_u_B * normals_B, dim=1, keepdim=True)
            loss = loss + (torch.mean(flux_B) - target_flux) ** 2

        return loss
