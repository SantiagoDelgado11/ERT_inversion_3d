import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models import ConductivityNet, PotentialNet
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
    parser.add_argument("--w_data_sigma", type=float, default=0.0, help="Peso opcional de la guia pseudo-profunda de conductividad")
    parser.add_argument("--w_pde", type=float, default=1.0, help="Peso para el residual PDE")
    parser.add_argument("--w_bc", type=float, default=1.0, help="Peso para condiciones de frontera")
    parser.add_argument("--w_reg", type=float, default=1e-6, help="Peso de regularizacion TV sobre log(sigma)")
    parser.add_argument("--w_flux", type=float, default=1.0, help="Peso para conservacion de flujo")
    parser.add_argument("--use_wandb", action="store_true", help="Activar logging en Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="ERT_PINN_3D")
    parser.add_argument("--wandb_name", type=str, default="baseline_training_run")
    parser.add_argument("--csv", type=str, default="dataset_output/measurements.csv",
                        help="CSV de mediciones generado junto a campaign.h5")
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
                "w_data_sigma": args.w_data_sigma,
                "w_pde": args.w_pde,
                "w_bc": args.w_bc,
                "w_reg": args.w_reg,
                "w_flux": args.w_flux,
            },
        )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Iniciando entrenamiento en: {device}")

    repo_root = Path(__file__).resolve().parents[1]
    csv_filepath = Path(args.csv)
    if not csv_filepath.is_absolute():
        csv_filepath = repo_root / csv_filepath
    current_I = 1.0
    gamma = 4.0

    weights = {
        "w_data": args.w_data,
        "w_data_sigma": args.w_data_sigma,
        "w_pde": args.w_pde,
        "w_bc": args.w_bc,
        "w_reg": args.w_reg,
        "w_flux": args.w_flux,
    }

    print("Cargando dataset y generando puntos de colocacion fisicos...")
    dataset = ERTDataset(
        csv_filepath=csv_filepath,
        n_pde=500,
        n_bc_surf=100,
        n_bc_inf=100,
        n_flux=50,
        epsilon=gamma,
    )
    
    # IMPORTANTE: Para una inversión PINN, solo debemos entrenar sobre UN conjunto 
    # de mediciones (un "survey" específico). Extraemos la primera muestra (idx=0).
    subset = torch.utils.data.Subset(dataset, [0])
    dataloader = DataLoader(subset, batch_size=1, shuffle=True)

    sigma_net = ConductivityNet(hidden_layers=4, hidden_dim=128).to(device)
    pot_net = PotentialNet().to(device)
    informer = PhysicsInformer(sigma_net, pot_net)

    print("Iniciando entrenamiento PINN con todo el dataset...")
    trained_pot_net, trained_sigma_net = train_pinn(
        u_net=pot_net,
        sigma_net=sigma_net,
        informer=informer,
        dataloader=dataloader,
        weights=weights,
        current_I=current_I,
        gamma=gamma,
        num_epochs_adam=1000,
        num_epochs_lbfgs=1000,
        lr=1e-3,
        device=device,
        use_wandb=args.use_wandb,
    )

    print("Entrenamiento completado. Guardando pesos...")
    torch.save(trained_sigma_net.state_dict(), "sigma_net.pth")
    torch.save(trained_pot_net.state_dict(), "pot_net.pth")
    if args.use_wandb and wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
