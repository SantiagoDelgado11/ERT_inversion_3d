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
            
            # Gradiente del potencial: nabla(u)
            grad_u = autograd.grad(
                outputs=u,
                inputs=coords,
                grad_outputs=torch.ones_like(u),
                create_graph=True,
                retain_graph=True
            )[0]
            
            du_dx, du_dy, du_dz = grad_u[:, 0:1], grad_u[:, 1:2], grad_u[:, 2:3]
            
            # Flujo acoplado: J = sigma * nabla(u)
            # El autograd backpropagará naturalmente hacia sigma_net y pot_net al derivar esto
            flux_x = sigma * du_dx
            flux_y = sigma * du_dy
            flux_z = sigma * du_dz
            
            # Divergencia del Flujo: nabla . J
            dflux_x = autograd.grad(outputs=flux_x, inputs=coords, grad_outputs=torch.ones_like(flux_x), create_graph=True, retain_graph=True)[0][:, 0:1]
            dflux_y = autograd.grad(outputs=flux_y, inputs=coords, grad_outputs=torch.ones_like(flux_y), create_graph=True, retain_graph=True)[0][:, 1:2]
            dflux_z = autograd.grad(outputs=flux_z, inputs=coords, grad_outputs=torch.ones_like(flux_z), create_graph=True, retain_graph=True)[0][:, 2:3]
            
            div_flux = dflux_x + dflux_y + dflux_z
            
            return {
                'sigma': sigma, 'u': u,
                'div_flux': div_flux,
                'du_dx': du_dx, 'du_dy': du_dy, 'du_dz': du_dz
            }
        else:
            # 1. Gradiente de conductividad puro (para Regularización TV)
            grad_sigma = autograd.grad(
                outputs=sigma, 
                inputs=coords, 
                grad_outputs=torch.ones_like(sigma),
                create_graph=True,
                retain_graph=True
            )[0]
            ds_dx, ds_dy, ds_dz = grad_sigma[:, 0:1], grad_sigma[:, 1:2], grad_sigma[:, 2:3]
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
        
        # Lado izquierdo acoplado
        lhs = derivs['div_flux']
        
        # Fuente Gaussiana q = I * (delta_A - delta_B)
        r_A = source_coords[:, 0:3]
        r_B = source_coords[:, 3:6]
        q_A = self._gaussian_source(coords, r_A, I, epsilon)
        q_B = self._gaussian_source(coords, r_B, I, epsilon)
        q = q_A - q_B
        
        # Ponderación Espacial (Spatial Loss Weighting) w(x)
        # Atenúa fuertemente a 0 en las vecindades de r_A y r_B para evitar que
        # el MSE intente ajustar la infinidad de la fuente, relajando la carga del optimizador.
        dist_sq_A = torch.sum((coords - r_A)**2, dim=1, keepdim=True)
        dist_sq_B = torch.sum((coords - r_B)**2, dim=1, keepdim=True)
        alpha = 2.0 
        w_A = 1.0 - torch.exp(-dist_sq_A / (alpha * epsilon**2))
        w_B = 1.0 - torch.exp(-dist_sq_B / (alpha * epsilon**2))
        w_x = w_A * w_B
        
        residual = w_x * (lhs - q)
        
        # Usamos Huber Loss (Smooth L1) con el residual enmascarado
        return torch.nn.functional.huber_loss(residual, torch.zeros_like(residual), delta=100.0)

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
