import torch
import numpy as np

def compute_derivatives(u, x, y, z):
    """
    Calcula las derivadas de primer orden de u con respecto a x, y, z.
    """
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_z = torch.autograd.grad(u, z, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    return u_x, u_y, u_z

def gaussian_delta(x, y, z, r_src, sigma_g=0.2):
    """
    Aproximación Gaussiana 3D estable para la función Delta de Dirac.
    """
    x_s, y_s, z_s = r_src
    r_squared = (x - x_s)**2 + (y - y_s)**2 + (z - z_s)**2
    norm = 1.0 / ( (sigma_g * np.sqrt(2 * np.pi))**3 )
    return norm * torch.exp(-r_squared / (2 * sigma_g**2))

def pde_loss(u, x, y, z, sigma_tensor, r_A, r_B, current_I=1.0, sigma_g=0.2):
    """
    Residuo de la PDE principal: 
    - nabla . (sigma * nabla u) = I*delta(r - r_A) - I*delta(r - r_B)
    """
    u_x, u_y, u_z = compute_derivatives(u, x, y, z)
    
    # Flujo de corriente: J = sigma * grad(u)
    J_x = sigma_tensor * u_x
    J_y = sigma_tensor * u_y
    J_z = sigma_tensor * u_z
    
    # Divergencia del flujo: div(J)
    J_xx = torch.autograd.grad(J_x, x, grad_outputs=torch.ones_like(J_x), create_graph=True)[0]
    J_yy = torch.autograd.grad(J_y, y, grad_outputs=torch.ones_like(J_y), create_graph=True)[0]
    J_zz = torch.autograd.grad(J_z, z, grad_outputs=torch.ones_like(J_z), create_graph=True)[0]
    
    div_J = J_xx + J_yy + J_zz
    
    # Lado Izquierdo (LHS)
    lhs = -div_J
    
    # Lado Derecho (RHS): Fuente y sumidero
    delta_A = gaussian_delta(x, y, z, r_A, sigma_g)
    delta_B = gaussian_delta(x, y, z, r_B, sigma_g)
    rhs = current_I * (delta_A - delta_B)
    
    # Residuo de la ecuación
    residual = lhs - rhs
    return torch.mean(residual**2)

def neumann_boundary_loss(u_neumann, z_neumann):
    """
    Condición de contorno de Neumann en la superficie (z=0): du/dz = 0
    """
    u_z = torch.autograd.grad(u_neumann, z_neumann, grad_outputs=torch.ones_like(u_neumann), create_graph=True)[0]
    return torch.mean(u_z**2)

def dirichlet_boundary_loss(u_dirichlet):
    """
    Condición de contorno de Dirichlet en los bordes lejanos: u = 0
    """
    return torch.mean(u_dirichlet**2)

def total_loss(loss_pde, loss_neumann, loss_dirichlet, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0):
    """
    Suma ponderada de todas las pérdidas
    """
    return lambda_1 * loss_pde + lambda_2 * loss_neumann + lambda_3 * loss_dirichlet
