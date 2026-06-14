"""Script to plot reconstructed conductivity from an ERT PINN experiment."""

import argparse
from pathlib import Path

from utils import load_predictions, plot_3d_scatter_slice

def main():
    parser = argparse.ArgumentParser(description="Plot reconstructed conductivity from an ERT PINN run.")
    parser.add_argument("--run", type=str, required=True, help="Path to the run directory (e.g., results/run_001)")
    parser.add_argument("--slice_axis", type=str, default="y", choices=["x", "y", "z"], help="Axis for 2D slice")
    parser.add_argument("--slice_value", type=float, default=0.0, help="Coordinate value for the slice")
    parser.add_argument("--cmap", type=str, default="viridis", help="Colormap to use")
    
    args = parser.parse_args()
    
    run_dir = Path(args.run)
    if not run_dir.exists() or not run_dir.is_dir():
        print(f"Error: Run directory '{args.run}' does not exist.")
        return

    try:
        data = load_predictions(run_dir)
    except FileNotFoundError as e:
        print(e)
        return

    if "conductivity" not in data:
        print(f"Error: No conductivity data found in predictions for {args.run}. Was this only a forward training run?")
        return

    points = data["points"]
    conductivity = data["conductivity"]
    
    output_path = run_dir / "plots" / f"conductivity_slice_{args.slice_axis}_{args.slice_value}.png"
    
    print(f"Plotting conductivity for run {run_dir.name}...")
    plot_3d_scatter_slice(
        points=points,
        values=conductivity.flatten(),
        title=f"Reconstructed Conductivity - {run_dir.name}",
        output_path=output_path,
        slice_axis=args.slice_axis,
        slice_value=args.slice_value,
        cmap=args.cmap,
        val_name="Conductivity (S/m)"
    )
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
