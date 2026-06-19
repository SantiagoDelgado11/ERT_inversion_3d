import torch

def sample_domain(num_points, x_range, y_range, z_range, device='cpu'):
    """
    Muestreo aleatorio uniforme dentro del volumen de dominio 3D.
    """
    x = (torch.rand(num_points, 1, device=device) * (x_range[1] - x_range[0]) + x_range[0]).requires_grad_(True)
    y = (torch.rand(num_points, 1, device=device) * (y_range[1] - y_range[0]) + y_range[0]).requires_grad_(True)
    z = (torch.rand(num_points, 1, device=device) * (z_range[1] - z_range[0]) + z_range[0]).requires_grad_(True)
    return x, y, z

def sample_neumann_boundary(num_points, x_range, y_range, z_surface=0.0, device='cpu'):
    """
    Muestreo en la superficie del suelo (z = 0), frontera Neumann.
    """
    x = (torch.rand(num_points, 1, device=device) * (x_range[1] - x_range[0]) + x_range[0]).requires_grad_(True)
    y = (torch.rand(num_points, 1, device=device) * (y_range[1] - y_range[0]) + y_range[0]).requires_grad_(True)
    z = (torch.ones(num_points, 1, device=device) * z_surface).requires_grad_(True)
    return x, y, z

def sample_dirichlet_boundary(num_points, x_range, y_range, z_range, device='cpu'):
    """
    Muestreo en las fronteras lejanas (laterales y fondo), frontera Dirichlet.
    """
    # Dividimos los puntos en las 5 caras exteriores (omitiendo z_max que es la superficie)
    points_per_face = num_points // 5
    
    x_list, y_list, z_list = [], [], []
    
    # Cara x_min
    x_list.append(torch.ones(points_per_face, 1, device=device) * x_range[0])
    y_list.append(torch.rand(points_per_face, 1, device=device) * (y_range[1] - y_range[0]) + y_range[0])
    z_list.append(torch.rand(points_per_face, 1, device=device) * (z_range[1] - z_range[0]) + z_range[0])
    
    # Cara x_max
    x_list.append(torch.ones(points_per_face, 1, device=device) * x_range[1])
    y_list.append(torch.rand(points_per_face, 1, device=device) * (y_range[1] - y_range[0]) + y_range[0])
    z_list.append(torch.rand(points_per_face, 1, device=device) * (z_range[1] - z_range[0]) + z_range[0])

    # Cara y_min
    x_list.append(torch.rand(points_per_face, 1, device=device) * (x_range[1] - x_range[0]) + x_range[0])
    y_list.append(torch.ones(points_per_face, 1, device=device) * y_range[0])
    z_list.append(torch.rand(points_per_face, 1, device=device) * (z_range[1] - z_range[0]) + z_range[0])
    
    # Cara y_max
    x_list.append(torch.rand(points_per_face, 1, device=device) * (x_range[1] - x_range[0]) + x_range[0])
    y_list.append(torch.ones(points_per_face, 1, device=device) * y_range[1])
    z_list.append(torch.rand(points_per_face, 1, device=device) * (z_range[1] - z_range[0]) + z_range[0])
    
    # Cara z_min (fondo)
    x_list.append(torch.rand(points_per_face, 1, device=device) * (x_range[1] - x_range[0]) + x_range[0])
    y_list.append(torch.rand(points_per_face, 1, device=device) * (y_range[1] - y_range[0]) + y_range[0])
    z_list.append(torch.ones(points_per_face, 1, device=device) * z_range[0])
    
    x = torch.cat(x_list, dim=0).requires_grad_(True)
    y = torch.cat(y_list, dim=0).requires_grad_(True)
    z = torch.cat(z_list, dim=0).requires_grad_(True)
    
    return x, y, z
