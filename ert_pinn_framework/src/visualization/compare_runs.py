"""Script to compare loss histories and metrics across multiple ERT PINN runs."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from utils import load_loss_history, load_run_summary

def main():
    parser = argparse.ArgumentParser(description="Compare loss histories of multiple ERT PINN runs.")
    parser.add_argument("--runs", nargs="+", required=True, help="List of run directories to compare")
    parser.add_argument("--output", type=str, default="run_comparison.png", help="Output plot filename")
    
    args = parser.parse_args()
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()
    
    loss_keys = ["total", "pde", "bc", "flux"]
    titles = ["Total Loss", "PDE Loss", "Boundary Condition Loss", "Flux Loss"]
    
    plotted_runs = 0
    for run_path_str in args.runs:
        run_dir = Path(run_path_str)
        if not run_dir.exists() or not run_dir.is_dir():
            print(f"Warning: Run directory '{run_path_str}' does not exist. Skipping.")
            continue
            
        try:
            df = load_loss_history(run_dir)
        except FileNotFoundError as e:
            print(f"Warning: {e}. Skipping.")
            continue
            
        if "epoch" not in df.columns:
            print(f"Warning: No 'epoch' column in {run_dir.name}. Skipping.")
            continue
            
        for idx, key in enumerate(loss_keys):
            if key in df.columns:
                axs[idx].plot(df["epoch"], df[key], label=f"{run_dir.name}")
        
        plotted_runs += 1

    if plotted_runs == 0:
        print("No valid runs found to plot.")
        return

    for idx, key in enumerate(loss_keys):
        axs[idx].set_yscale("log")
        axs[idx].set_xlabel("Epochs")
        axs[idx].set_ylabel("Loss")
        axs[idx].set_title(titles[idx])
        axs[idx].legend()
        axs[idx].grid(True, which="both", ls="--", alpha=0.5)

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = Path.cwd() / args.output
        
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    
    print(f"Comparison plot saved to {out_path}")

if __name__ == "__main__":
    main()
