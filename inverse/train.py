import torch
import torch.optim as optim
from typing import Dict, Any

def train_pinn(
    u_net: torch.nn.Module, 
    sigma_net: torch.nn.Module, 
    informer: Any, 
    data_samples: Dict[str, torch.Tensor], 
    pde_samples: Dict[str, torch.Tensor], 
    bc_neumann_samples: Dict[str, torch.Tensor], 
    bc_dirichlet_samples: Dict[str, torch.Tensor],
    flux_samples: Dict[str, torch.Tensor], 
    reg_samples: Dict[str, torch.Tensor],
    weights: Dict[str, float], 
    current_I: float,
    num_epochs: int = 1000, 
    lr: float = 1e-3, 
    device: str = 'cpu'
):
    """
    Orchestrates the optimization of u_net and sigma_net.
    
    Args:
        u_net: Potential network u_theta.
        sigma_net: Conductivity network sigma_phi.
        informer: Instance of PhysicsInformer.
        data_samples: dict with 'r_m' and 'u_star'.
        pde_samples: dict with 'r', 'r_A', 'r_B'.
        bc_neumann_samples: dict with 'r_N', 'n_vec'.
        bc_dirichlet_samples: dict with 'r_D'.
        flux_samples: dict with 'r_Bc_A', 'n_Bc_A', 'r_Bc_B', 'n_Bc_B', 'area_Bc'.
        reg_samples: dict with 'r_reg'.
        weights: dict with 'w_data', 'w_pde', 'w_bc_N', 'w_bc_D', 'w_reg', 'w_flux'.
        current_I: Current intensity for the dipole.
        num_epochs: Number of epochs to train.
        lr: Learning rate for Adam optimizer.
        device: 'cpu' or 'cuda'.
    """
    u_net = u_net.to(device)
    sigma_net = sigma_net.to(device)
    
    # Joint optimization of both networks
    optimizer = optim.Adam(list(u_net.parameters()) + list(sigma_net.parameters()), lr=lr)
    
    # Helper to move tensors to device and enable gradients where needed
    def prepare(tensor: torch.Tensor, requires_grad: bool = False):
        t = tensor.to(device)
        if requires_grad:
            return t.requires_grad_(True)
        return t
        
    # Unpack and prepare data
    r_m = prepare(data_samples['r_m'])
    u_star = prepare(data_samples['u_star'])
    
    r_pde = prepare(pde_samples['r'], requires_grad=True)
    r_A = prepare(pde_samples['r_A'])
    r_B = prepare(pde_samples['r_B'])
    
    r_N = prepare(bc_neumann_samples['r_N'], requires_grad=True)
    n_vec = prepare(bc_neumann_samples['n_vec'])
    r_D = prepare(bc_dirichlet_samples['r_D'])
    
    r_Bc_A = prepare(flux_samples['r_Bc_A'], requires_grad=True)
    n_Bc_A = prepare(flux_samples['n_Bc_A'])
    r_Bc_B = prepare(flux_samples['r_Bc_B'], requires_grad=True)
    n_Bc_B = prepare(flux_samples['n_Bc_B'])
    area_Bc = flux_samples['area_Bc']
    
    r_reg = prepare(reg_samples['r_reg'], requires_grad=True)
    
    # Weights with default values
    w_data = weights.get('w_data', 1.0)
    w_pde = weights.get('w_pde', 1.0)
    w_bc_N = weights.get('w_bc_N', 1.0)
    w_bc_D = weights.get('w_bc_D', 1.0)
    w_reg = weights.get('w_reg', 1.0)
    w_flux = weights.get('w_flux', 1.0)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Calculate loss components
        loss_data = informer.loss_data(r_m, u_star)
        loss_pde = informer.loss_pde(r_pde, r_A, r_B, current_I)
        loss_bc_N = informer.loss_bc_neumann(r_N, n_vec)
        loss_bc_D = informer.loss_bc_dirichlet(r_D)
        loss_reg = informer.loss_tv_reg(r_reg)
        loss_flux = informer.loss_flux(r_Bc_A, n_Bc_A, r_Bc_B, n_Bc_B, current_I, area_Bc)
        
        # Total multiobjective loss
        loss_total = (w_data * loss_data +
                      w_pde * loss_pde +
                      w_bc_N * loss_bc_N +
                      w_bc_D * loss_bc_D +
                      w_reg * loss_reg +
                      w_flux * loss_flux)
                      
        loss_total.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Total Loss = {loss_total.item():.6e} "
                  f"(Data: {loss_data.item():.2e}, PDE: {loss_pde.item():.2e})")

    return u_net, sigma_net
