import torch
import torch.autograd as autograd

class PhysicsInformer:
    """
    Motor Diferencial para la evaluación de residuos físicos y regularización en la PINN.
    """
    def __init__(self, cond_net, pot_net):
        self.cond_net = cond_net
        self.pot_net = pot_net

    def compute_derivatives(self, coords, source_coords=None):
        """
        Extrae las derivadas espaciales usando autograd.grad con create_graph=True
        para permitir retropropagación sobre derivadas.
        coords: (batch_size, 3) - requiere requires_grad=True internamente.
        """
        coords.requires_grad_(True)
        
        # Forward pass: Conductividad
        sigma = self.cond_net(coords)
        
        # Forward pass: Potencial (si se proporcionan fuentes)
        if source_coords is not None:
            u = self.pot_net(coords, source_coords)
        else:
            u = None
        
        # 1. Gradiente de conductividad: nabla(sigma)
        grad_sigma = autograd.grad(
            outputs=sigma, 
            inputs=coords, 
            grad_outputs=torch.ones_like(sigma),
            create_graph=True,
            retain_graph=True
        )[0]
        ds_dx, ds_dy, ds_dz = grad_sigma[:, 0:1], grad_sigma[:, 1:2], grad_sigma[:, 2:3]

        if u is not None:
            # 2. Gradiente del potencial: nabla(u)
            grad_u = autograd.grad(
                outputs=u,
                inputs=coords,
                grad_outputs=torch.ones_like(u),
                create_graph=True,
                retain_graph=True
            )[0]
            du_dx, du_dy, du_dz = grad_u[:, 0:1], grad_u[:, 1:2], grad_u[:, 2:3]
            
            # 3. Laplaciano (segundas derivadas): nabla^2(u)
            d2u_dx2 = autograd.grad(
                outputs=du_dx, inputs=coords, grad_outputs=torch.ones_like(du_dx), create_graph=True, retain_graph=True
            )[0][:, 0:1]
            d2u_dy2 = autograd.grad(
                outputs=du_dy, inputs=coords, grad_outputs=torch.ones_like(du_dy), create_graph=True, retain_graph=True
            )[0][:, 1:2]
            d2u_dz2 = autograd.grad(
                outputs=du_dz, inputs=coords, grad_outputs=torch.ones_like(du_dz), create_graph=True, retain_graph=True
            )[0][:, 2:3]
            
            return {
                'sigma': sigma, 'u': u,
                'ds_dx': ds_dx, 'ds_dy': ds_dy, 'ds_dz': ds_dz,
                'du_dx': du_dx, 'du_dy': du_dy, 'du_dz': du_dz,
                'd2u_dx2': d2u_dx2, 'd2u_dy2': d2u_dy2, 'd2u_dz2': d2u_dz2
            }
        else:
            return {
                'sigma': sigma,
                'ds_dx': ds_dx, 'ds_dy': ds_dy, 'ds_dz': ds_dz
            }

    def _gaussian_source(self, coords, source_pos, I, epsilon):
        """
        Aproxima una carga puntual (delta de Dirac) mediante una Gaussiana.
        """
        dist_sq = torch.sum((coords - source_pos)**2, dim=1, keepdim=True)
        coeff = 1.0 / ((torch.sqrt(torch.tensor(torch.pi)) * epsilon)**3)
        return I * coeff * torch.exp(-dist_sq / (epsilon**2))

    def compute_pde_loss(self, coords, source_coords, I, epsilon):
        """
        Evalúa el residual de la PDE (Ecuación de Poisson):
        nabla . (sigma * nabla(u)) - q = 0
        """
        derivs = self.compute_derivatives(coords, source_coords)
        
        sigma = derivs['sigma']
        ds_dx, ds_dy, ds_dz = derivs['ds_dx'], derivs['ds_dy'], derivs['ds_dz']
        du_dx, du_dy, du_dz = derivs['du_dx'], derivs['du_dy'], derivs['du_dz']
        d2u_dx2, d2u_dy2, d2u_dz2 = derivs['d2u_dx2'], derivs['d2u_dy2'], derivs['d2u_dz2']
        
        # Laplaciano de u
        laplace_u = d2u_dx2 + d2u_dy2 + d2u_dz2
        # Producto punto nabla(sigma) . nabla(u)
        grad_s_dot_grad_u = ds_dx * du_dx + ds_dy * du_dy + ds_dz * du_dz
        
        # Divergencia: nabla(sigma).nabla(u) + sigma * nabla^2(u)
        lhs = grad_s_dot_grad_u + sigma * laplace_u
        
        # Fuente Gaussiana q = I * (delta_A - delta_B)
        r_A = source_coords[:, 0:3]
        r_B = source_coords[:, 3:6]
        q_A = self._gaussian_source(coords, r_A, I, epsilon)
        q_B = self._gaussian_source(coords, r_B, I, epsilon)
        q = q_A - q_B
        
        residual = lhs - q
        return torch.mean(residual**2)

    def compute_bc_loss(self, surface_coords, inf_coords, source_coords_surf, source_coords_inf):
        """
        Condición de Neumann: du/dz = 0 en z=0
        Condición de Dirichlet: u = 0 en fronteras lejanas
        """
        loss = 0.0
        
        # Neumann
        if surface_coords is not None and surface_coords.shape[0] > 0:
            derivs_surf = self.compute_derivatives(surface_coords, source_coords_surf)
            du_dz = derivs_surf['du_dz']
            loss_neumann = torch.mean(du_dz**2)
            loss += loss_neumann
            
        # Dirichlet
        if inf_coords is not None and inf_coords.shape[0] > 0:
            u_inf = self.pot_net(inf_coords, source_coords_inf)
            loss_dirichlet = torch.mean(u_inf**2)
            loss += loss_dirichlet
            
        return loss

    def compute_reg_loss(self, coords):
        """
        Total Variation (TV) en la conductividad.
        L_reg = mean(|ds/dx| + |ds/dy| + |ds/dz|)
        """
        derivs = self.compute_derivatives(coords, source_coords=None)
        ds_dx, ds_dy, ds_dz = derivs['ds_dx'], derivs['ds_dy'], derivs['ds_dz']
        
        tv_loss = torch.mean(torch.abs(ds_dx) + torch.abs(ds_dy) + torch.abs(ds_dz))
        return tv_loss

    def compute_flux_loss(self, coords_A, coords_B, normals_A, normals_B, source_coords_A, source_coords_B, I, area):
        """
        Conservación de carga local (L_flux).
        Integrando la ley de Ohm localmente alrededor de los electrodos.
        """
        loss = 0.0
        
        # Electrodo A
        if coords_A is not None and coords_A.shape[0] > 0:
            derivs_A = self.compute_derivatives(coords_A, source_coords_A)
            sigma_A = derivs_A['sigma']
            grad_u_A = torch.cat([derivs_A['du_dx'], derivs_A['du_dy'], derivs_A['du_dz']], dim=1)
            
            flux_A_pointwise = torch.sum(sigma_A * grad_u_A * normals_A, dim=1, keepdim=True)
            flux_A_mean = torch.mean(flux_A_pointwise)
            loss_flux_A = (flux_A_mean - I / area)**2
            loss += loss_flux_A
            
        # Electrodo B
        if coords_B is not None and coords_B.shape[0] > 0:
            derivs_B = self.compute_derivatives(coords_B, source_coords_B)
            sigma_B = derivs_B['sigma']
            grad_u_B = torch.cat([derivs_B['du_dx'], derivs_B['du_dy'], derivs_B['du_dz']], dim=1)
            
            flux_B_pointwise = torch.sum(sigma_B * grad_u_B * normals_B, dim=1, keepdim=True)
            flux_B_mean = torch.mean(flux_B_pointwise)
            loss_flux_B = (flux_B_mean + I / area)**2
            loss += loss_flux_B
            
        return loss
