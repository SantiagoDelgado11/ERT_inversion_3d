import torch
import math

class PhysicsInformer:
    """
    Differential calculation engine and loss tensor definition for 3D ERT PINN.
    """
    def __init__(self, u_net, sigma_net, epsilon=0.1, beta=1e-3):
        self.u_net = u_net
        self.sigma_net = sigma_net
        self.epsilon = epsilon
        self.beta = beta
        
    def _grad(self, y, x):
        """
        Compute the first order Jacobian (gradient) of y with respect to x.
        x must have requires_grad=True.
        """
        return torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=torch.ones_like(y),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
    def _laplacian_and_grad(self, y, x):
        """
        Compute the Laplacian (trace of Hessian) and the gradient of y with respect to x.
        """
        grad_y = self._grad(y, x)
        laplacian = torch.zeros_like(y)
        
        # Iterate over spatial dimensions (x, y, z)
        for d in range(x.shape[1]):
            grad_y_d = grad_y[:, d:d+1]
            grad_grad_y_d = self._grad(grad_y_d, x)
            laplacian += grad_grad_y_d[:, d:d+1]
            
        return laplacian, grad_y

    def source_term(self, r, r_A, r_B, I):
        """
        Compute the regularized dipole source q(r).
        Args:
            r: Coordinates (N, 3)
            r_A: Current injection electrode position (N, 3) or (1, 3)
            r_B: Current extraction electrode position (N, 3) or (1, 3)
            I: Current intensity
        """
        def delta_epsilon(r_eval, r_0):
            norm_sq = torch.sum((r_eval - r_0)**2, dim=1, keepdim=True)
            coeff = 1.0 / ( (self.epsilon * math.sqrt(math.pi))**3 )
            return coeff * torch.exp(-norm_sq / (self.epsilon**2))
            
        q = I * (delta_epsilon(r, r_A) - delta_epsilon(r, r_B))
        return q

    def pde_residual(self, r, r_A, r_B, I):
        """
        Compute the PDE residual r_PDE(r).
        """
        sigma = self.sigma_net(r)
        u = self.u_net(r)
        
        grad_sigma = self._grad(sigma, r)
        laplacian_u, grad_u = self._laplacian_and_grad(u, r)
        
        # Dot product: \nabla \sigma \cdot \nabla u
        grad_dot = torch.sum(grad_sigma * grad_u, dim=1, keepdim=True)
        
        q = self.source_term(r, r_A, r_B, I)
        
        residual = -(grad_dot + sigma * laplacian_u) - q
        return residual

    def loss_data(self, r_m, u_star):
        """
        L_data: Data misfit loss.
        """
        u_pred = self.u_net(r_m)
        return torch.mean((u_pred - u_star)**2)

    def loss_pde(self, r_pde, r_A, r_B, I):
        """
        L_PDE: Physics loss enforcing the Poisson equation.
        """
        res = self.pde_residual(r_pde, r_A, r_B, I)
        return torch.mean(res**2)

    def loss_bc_neumann(self, r_N, n_vec):
        """
        L_bc (Neumann): Zero current flux across topographic surface.
        Args:
            r_N: Points on Neumann boundary (N, 3)
            n_vec: Normal vectors at r_N (N, 3)
        """
        u = self.u_net(r_N)
        grad_u = self._grad(u, r_N)
        flux_surface = torch.sum(grad_u * n_vec, dim=1, keepdim=True)
        return torch.mean(flux_surface**2)

    def loss_bc_dirichlet(self, r_D):
        """
        L_bc (Dirichlet): Zero potential at infinite boundaries.
        Args:
            r_D: Points on Dirichlet boundary (N, 3)
        """
        u = self.u_net(r_D)
        return torch.mean(u**2)

    def loss_tv_reg(self, r_reg):
        """
        L_reg: Isotropic Total Variation (Huber/Charbonnier) regularization on conductivity.
        """
        sigma = self.sigma_net(r_reg)
        grad_sigma = self._grad(sigma, r_reg)
        norm_sq = torch.sum(grad_sigma**2, dim=1, keepdim=True)
        return torch.mean(torch.sqrt(norm_sq + self.beta**2))

    def loss_flux(self, r_Bc_A, n_Bc_A, r_Bc_B, n_Bc_B, I, area_Bc):
        """
        L_flux: Flux conservation around electrodes A and B.
        Args:
            r_Bc_A, r_Bc_B: Points on the control spheres (N, 3)
            n_Bc_A, n_Bc_B: Outward normal vectors (N, 3)
            I: Current intensity
            area_Bc: Surface area of the control sphere
        """
        # Flux around A (Source -> integral = I)
        sigma_A = self.sigma_net(r_Bc_A)
        u_A = self.u_net(r_Bc_A)
        grad_u_A = self._grad(u_A, r_Bc_A)
        flux_A_pointwise = sigma_A * torch.sum(grad_u_A * n_Bc_A, dim=1, keepdim=True)
        flux_A_mean = torch.mean(flux_A_pointwise)
        loss_A = (flux_A_mean - I / area_Bc)**2
        
        # Flux around B (Sink -> integral = -I)
        sigma_B = self.sigma_net(r_Bc_B)
        u_B = self.u_net(r_Bc_B)
        grad_u_B = self._grad(u_B, r_Bc_B)
        flux_B_pointwise = sigma_B * torch.sum(grad_u_B * n_Bc_B, dim=1, keepdim=True)
        flux_B_mean = torch.mean(flux_B_pointwise)
        loss_B = (flux_B_mean + I / area_Bc)**2
        
        return loss_A + loss_B
