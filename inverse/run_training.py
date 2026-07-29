import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models import ConductivityNet, PotentialNet, MeasurementEncoder
from physics_informer import PhysicsInformer
from pytorch_dataset import ERTDataset
from train import train_pinn

try:
    import wandb
except ImportError: 
    wandb = None

def main():
    parser = argparse.ArgumentParser(description="Entrenamiento PINN para ERT 3D")
    parser.add_argument("--w_data", type=float, default=1.0, help="Peso para el data loss")
    parser.add_argument("--w_pde", type=float, default=1.0, help="Peso para el PDE loss")
    parser.add_argument("--w_bc", type=float, default=10.0, help="Peso para condiciones de frontera")
    
    # CORRECCIÓN: Un pequeño valor por defecto para activar la regularización TV
    parser.add_argument("--w_reg", type=float, default=1e-4, help="Peso de regularizacion TV")
    
    parser.add_argument("--w_flux", type=float, default=1e-2, help="Peso para conservacion de flujo")
    parser.add_argument("--batch_size", type=int, default=4, help="Tamaño del batch para entrenar múltiples modelos")
    parser.add_argument("--num_workers", type=int, default=4, help="Número de workers para el DataLoader")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="Pasos de acumulación de gradientes")
    parser.add_argument("--epochs", type=int, default=1000, help="Número total de épocas para entrenar")
    
    # CORRECCIÓN: Añadido argumento para controlar la tasa de aprendizaje. 1e-3 suele ser inestable en PINNs 3D.
    parser.add_argument("--lr", type=float, default=5e-4, help="Tasa de aprendizaje (Learning Rate)")
    
    parser.add_argument("--compile", action="store_true", help="Activar torch.compile para acelerar (requiere PyTorch 2.0+)")
    parser.add_argument("--use_wandb", action="store_true", help="Activar logging en Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="ERT_PINN_3D")
    parser.add_argument("--wandb_name", type=str, default="baseline_training_run")
    parser.add_argument("--profile", action="store_true", help="Activar el perfilado con torch.profiler")
    args = parser.parse_args()

    if args.use_wandb:
        if wandb is None:
            raise ImportError("wandb no esta instalado, pero --use_wandb fue solicitado.")
        api_key = os.getenv("WANDB_API_KEY")
        if api_key:
            wandb.login(key=api_key)
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config={
                "w_data": args.w_data,
                "w_pde": args.w_pde,
                "w_bc": args.w_bc,
                "w_reg": args.w_reg,
                "w_flux": args.w_flux,
                "batch_size": args.batch_size,
                "lr": args.lr,
            },
        )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Iniciando entrenamiento en: {device}")

    repo_root = Path(__file__).resolve().parents[1]
    h5_filepath = repo_root / "forward" / "dataset.h5"
    if not h5_filepath.exists():
        h5_filepath = repo_root / "inverse" / "single_experiment_data.h5"

    current_I = 1.0
    gamma = 4.0

    weights = {
        "w_data": args.w_data,
        "w_pde": args.w_pde,
        "w_bc": args.w_bc,
        "w_reg": args.w_reg,
        "w_flux": args.w_flux,
    }

    print("Cargando dataset y generando puntos de colocacion fisicos...")
    dataset = ERTDataset(
        h5_filepath=h5_filepath,
        n_pde=1000, 
        n_bc_surf=200,
        n_bc_inf=200,
        n_flux=50,
        epsilon=gamma,
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0)
    )

    encoder = MeasurementEncoder(in_features=13, hidden_dim=128, latent_dim=128).to(device)
    sigma_net = ConductivityNet(hidden_layers=5, hidden_dim=256, latent_dim=128).to(device)
    pot_net = PotentialNet(num_frequencies=10, hidden_layers=5, hidden_dim=256, latent_dim=128).to(device)
    
    informer = PhysicsInformer(sigma_net, pot_net)

    if args.compile:
        print("Intentando compilar los modelos con torch.compile()...")
        try:
            encoder = torch.compile(encoder)
            sigma_net = torch.compile(sigma_net)
            pot_net = torch.compile(pot_net)
            print("Modelos compilados exitosamente.")
        except Exception as e:
            print(f"Advertencia: torch.compile falló, usando eager mode. Error: {e}")

    print("Iniciando entrenamiento PINN con todo el dataset...")
    trained_pot_net, trained_sigma_net, trained_encoder = train_pinn(
        u_net=pot_net,
        sigma_net=sigma_net,
        encoder=encoder,
        informer=informer,
        dataloader=dataloader,
        weights=weights,
        current_I=current_I,
        gamma=gamma,
        num_epochs_adam=args.epochs,
        num_epochs_lbfgs=500,
        lr=args.lr, # Usa la tasa de aprendizaje definida en los argumentos
        device=device,
        use_wandb=args.use_wandb,
        profile_training=args.profile,
        accumulation_steps=args.accumulation_steps,
    )

    print("Entrenamiento completado. Guardando pesos...")
    torch.save(trained_sigma_net.state_dict(), "sigma_net.pth")
    torch.save(trained_pot_net.state_dict(), "pot_net.pth")
    torch.save(trained_encoder.state_dict(), "encoder.pth")
    if args.use_wandb and wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()