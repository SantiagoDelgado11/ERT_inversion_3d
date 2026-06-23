import torch
from torch.utils.data import DataLoader
from pytorch_dataset import ERTDataset
from networks import ConductivityNet, PotentialNet
from physics_informer import PhysicsInformer
from train import train_pinn
import wandb

import argparse

def main():
    parser = argparse.ArgumentParser(description="Entrenamiento PINN para ERT 3D")
    parser.add_argument("--w_data", type=float, default=1.0, help="Peso para el Data Loss")
    parser.add_argument("--w_pde", type=float, default=1e-4, help="Peso para el PDE Loss (Poisson)")
    parser.add_argument("--w_bc", type=float, default=10.0, help="Peso para Condiciones de Frontera (Hard Constraint)")
    parser.add_argument("--w_reg", type=float, default=0.1, help="Peso de Regularización (TV)")
    parser.add_argument("--w_flux", type=float, default=1e-2, help="Peso para el Flujo de Corriente")
    args = parser.parse_args()

    # Inicialización de Weights & Biases con la API key explícita
    wandb.login(key="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")
    wandb.init(
        project="ERT_PINN_3D", 
        name="baseline_training_run",
        config={
            "w_data": args.w_data,
            "w_pde": args.w_pde,
            "w_bc": args.w_bc,
            "w_reg": args.w_reg,
            "w_flux": args.w_flux
        }
    )

    # Configuración de Hardware
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Iniciando entrenamiento en: {device}")

    # Hiperparámetros físicos y de entrenamiento
    h5_filepath = '../forward/dataset/dataset_validation.h5'
    current_I = 1.0
    # Desviación estándar de la fuente: gamma >= 2*dx (dx=2.0)
    gamma = 4.0
    
    # Pesos de la Función de Pérdida pasados por argumento
    weights = {
        'w_data': args.w_data,
        'w_pde': args.w_pde,
        'w_bc': args.w_bc,
        'w_reg': args.w_reg,
        'w_flux': args.w_flux
    }

    # 1. Instanciar el Dataset y DataLoader
    print("Cargando dataset y generando puntos de colocación físicos...")
    # Usaremos lazy-loading: DataLoader extraerá una muestra a la vez
    dataset = ERTDataset(
        h5_filepath=h5_filepath, 
        n_pde=500, 
        n_bc_surf=100, 
        n_bc_inf=100, 
        n_flux=50, 
        epsilon=gamma
    )
    
    # Batch size 1 es usual en PINNs 3D debido al altísimo costo del Hessiano
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # 2. Inicializar Redes y Motor Físico
    sigma_net = ConductivityNet().to(device)
    pot_net = PotentialNet().to(device)
    informer = PhysicsInformer(sigma_net, pot_net)

    # Extraemos la primera muestra para la prueba inicial de entrenamiento
    print("Iniciando amortización / sobreajuste físico en el primer batch...")
    batch = next(iter(dataloader))
    
    # Extraer los diccionarios des-empaquetando la dimensión del batch (batch_size=1)
    def squeeze_dict(d):
        return {k: v[0] for k, v in d.items() if isinstance(v, torch.Tensor)} or d
        
    data_samples = squeeze_dict(batch['data'])
    pde_samples = squeeze_dict(batch['pde'])
    bc_neumann_samples = squeeze_dict(batch['bc_neumann'])
    bc_dirichlet_samples = squeeze_dict(batch['bc_dirichlet'])
    flux_samples = squeeze_dict(batch['flux'])
    flux_samples['area_Bc'] = batch['flux']['area_Bc'][0].item() # Escalar
    reg_samples = squeeze_dict(batch['reg'])

    # 3. Lanzar el entrenamiento
    trained_pot_net, trained_sigma_net = train_pinn(
        u_net=pot_net,
        sigma_net=sigma_net,
        informer=informer,
        data_samples=data_samples,
        pde_samples=pde_samples,
        bc_neumann_samples=bc_neumann_samples,
        bc_dirichlet_samples=bc_dirichlet_samples,
        flux_samples=flux_samples,
        reg_samples=reg_samples,
        weights=weights,
        current_I=current_I,
        gamma=gamma,
        num_epochs_adam=1000,
        num_epochs_lbfgs=500,
        lr=1e-3,
        device=device,
        use_wandb=True
    )
    
    print("Entrenamiento completado. Guardando pesos...")
    torch.save(trained_sigma_net.state_dict(), 'sigma_net.pth')
    torch.save(trained_pot_net.state_dict(), 'pot_net.pth')
    wandb.finish()

if __name__ == '__main__':
    main()
