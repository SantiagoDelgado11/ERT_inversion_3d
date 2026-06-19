import torch
import torch.optim as optim
from ert_pinn_framework.scripts.forward.model import PINN_3D
from ert_pinn_framework.scripts.forward.physics_loss import pde_loss, neumann_boundary_loss, dirichlet_boundary_loss, total_loss
from ert_pinn_framework.scripts.forward.sampler import sample_domain, sample_neumann_boundary, sample_dirichlet_boundary

def get_conductivity(x, y, z):
    """
    Campo de conductividad conocido sigma(r).
    Por ahora se asume un medio homogéneo (ej. 0.1 S/m).
    """
    return torch.ones_like(x) * 0.1

def train_pinn():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo de entrenamiento: {device}")
    
    # Parámetros del Dominio
    x_range = [-10.0, 10.0]
    y_range = [-10.0, 10.0]
    z_range = [-10.0, 0.0]  # Profundidad de 10m
    
    # Posición de los electrodos
    r_A = (-2.0, 0.0, 0.0)  # Inyección
    r_B = ( 2.0, 0.0, 0.0)  # Extracción
    current_I = 1.0
    
    # Modelo
    model = PINN_3D().to(device)
    
    # Optimizador
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 1000
    N_domain = 5000
    N_neumann = 1000
    N_dirichlet = 1000
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # 1. Muestreo de puntos de colocación
        x_dom, y_dom, z_dom = sample_domain(N_domain, x_range, y_range, z_range, device)
        x_neu, y_neu, z_neu = sample_neumann_boundary(N_neumann, x_range, y_range, z_surface=0.0, device)
        x_dir, y_dir, z_dir = sample_dirichlet_boundary(N_dirichlet, x_range, y_range, z_range, device)
        
        # Evaluar la conductividad en los puntos de dominio
        sigma_dom = get_conductivity(x_dom, y_dom, z_dom)
        
        # 2. Predicciones (Forward pass)
        u_dom = model(x_dom, y_dom, z_dom)
        u_neu = model(x_neu, y_neu, z_neu)
        u_dir = model(x_dir, y_dir, z_dir)
        
        # 3. Cálculo de Pérdidas Físicas
        l_pde = pde_loss(u_dom, x_dom, y_dom, z_dom, sigma_dom, r_A, r_B, current_I)
        l_neu = neumann_boundary_loss(u_neu, z_neu)
        l_dir = dirichlet_boundary_loss(u_dir)
        
        loss = total_loss(l_pde, l_neu, l_dir, lambda_1=1.0, lambda_2=1.0, lambda_3=10.0) # Mayor peso a Dirichlet si es necesario
        
        # 4. Retropropagación (Backward)
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs} - Loss Total: {loss.item():.6f} "
                  f"(PDE: {l_pde.item():.6f}, Neumann: {l_neu.item():.6f}, Dirichlet: {l_dir.item():.6f})")
            
    print("Entrenamiento finalizado.")
    # Guardar modelo
    torch.save(model.state_dict(), "pinn_3d_forward.pth")
    print("Modelo guardado como pinn_3d_forward.pth")

if __name__ == "__main__":
    train_pinn()
