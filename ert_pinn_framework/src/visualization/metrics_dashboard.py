"""Script to parse multiple runs and summarize hyperparameter effects."""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from utils import load_run_summary

def flatten_dict(d: dict, parent_key: str = '', sep: str = '_') -> dict:
    """Flattens a nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def main():
    parser = argparse.ArgumentParser(description="Generate a metrics dashboard from multiple ERT PINN runs.")
    parser.add_argument("--results_dir", type=str, required=True, help="Parent directory containing run folders")
    parser.add_argument("--output_prefix", type=str, default="dashboard", help="Prefix for output plots/csv")
    
    args = parser.parse_args()
    
    parent_dir = Path(args.results_dir)
    if not parent_dir.exists() or not parent_dir.is_dir():
        print(f"Error: Results directory '{args.results_dir}' does not exist.")
        return

    run_data = []
    
    for run_dir in parent_dir.iterdir():
        if not run_dir.is_dir():
            continue
            
        try:
            summary = load_run_summary(run_dir)
        except FileNotFoundError:
            continue
            
        flat_summary = flatten_dict(summary)
        flat_summary["run_name"] = run_dir.name
        run_data.append(flat_summary)

    if not run_data:
        print(f"No valid run summaries found in '{args.results_dir}'.")
        return

    df = pd.DataFrame(run_data)
    
    # Extract some key columns if they exist
    cols_to_print = ["run_name"]
    for expected_col in [
        "final_metrics_total", 
        "final_metrics_pde",
        "final_metrics_bc",
        "final_metrics_data",
        "model_config_potential_hidden_dim", 
        "model_config_potential_num_hidden_layers",
        "inverse_config_conductivity_hidden_dim",
        "inverse_config_optimizer_lr"
    ]:
        if expected_col in df.columns:
            cols_to_print.append(expected_col)

    print("\n=== Hyperparameter & Metrics Summary ===")
    print(df[cols_to_print].to_string(index=False))
    
    csv_path = parent_dir / f"{args.output_prefix}_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nFull metrics saved to: {csv_path}")
    
    # Generate scatter plot: Learning Rate vs Final Total Loss (if available)
    if "inverse_config_optimizer_lr" in df.columns and "final_metrics_total" in df.columns:
        plt.figure(figsize=(8, 6))
        # Ensure numeric
        lr_vals = pd.to_numeric(df["inverse_config_optimizer_lr"], errors='coerce')
        loss_vals = pd.to_numeric(df["final_metrics_total"], errors='coerce')
        
        valid_mask = ~(lr_vals.isna() | loss_vals.isna())
        if valid_mask.any():
            plt.scatter(lr_vals[valid_mask], loss_vals[valid_mask], alpha=0.7, s=100)
            for i, txt in enumerate(df["run_name"][valid_mask]):
                plt.annotate(txt, (lr_vals[valid_mask].iloc[i], loss_vals[valid_mask].iloc[i]), 
                             xytext=(5, 5), textcoords='offset points')
            
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("Learning Rate")
            plt.ylabel("Final Total Loss")
            plt.title("Learning Rate vs Final Loss")
            plt.grid(True, which="both", ls="--", alpha=0.5)
            
            plot_path = parent_dir / f"{args.output_prefix}_lr_vs_loss.png"
            plt.savefig(plot_path, dpi=300)
            print(f"Plot saved to: {plot_path}")
            plt.close()

if __name__ == "__main__":
    main()
