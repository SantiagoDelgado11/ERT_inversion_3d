"""Invert the generated pole-dipole campaign with the actual 3D ERT forward model.

This is intentionally separate from the PINN experiment: a free potential
network can fit voltage data without identifying conductivity.  Here every
candidate model is evaluated with the same SimPEG physics used to generate
measurements.csv.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from scipy.optimize import least_squares

import discretize
from simpeg import maps
import simpeg.electromagnetics.static.resistivity as dc

from dataset_generator.config.settings import AppConfig
from dataset_generator.geometry.terrain import GeometryManager
from dataset_generator.mesh.builder import MeshGenerator


def build_forward(df, cfg):
    # Use a fixed mesh during optimization.  Refining the mesh for each trial
    # would make the objective discontinuous and needlessly expensive.
    geometry = GeometryManager(cfg.domain)
    initial = geometry.generate_sphere(cfg.anomaly)
    all_electrodes = _electrode_positions(df)
    mesh_electrodes = all_electrodes[np.isfinite(all_electrodes).all(axis=1)]
    mesh = MeshGenerator(cfg.mesh, cfg.domain).build_tree_mesh(mesh_electrodes, initial)
    active = mesh.cell_centers[:, 2] >= -1e-5
    active_idx = np.flatnonzero(active)

    src_list = []
    ordered = []
    for a in np.unique(df["A_idx"].to_numpy()):
        rows = df[df["A_idx"] == a]
        m = rows[["M_x", "M_y", "M_z"]].to_numpy(float)
        n = rows[["N_x", "N_y", "N_z"]].to_numpy(float)
        src_list.append(dc.sources.Pole(
            [dc.receivers.Dipole(m, n)],
            location=rows[["A_x", "A_y", "A_z"]].iloc[0].to_numpy(float),
        ))
        ordered.extend(rows.index.to_list())

    survey = dc.Survey(src_list)
    try:
        from pymatsolver import Pardiso as solver
    except ImportError:
        from simpeg.utils import SolverLU as solver
    try:
        act_map = maps.InjectActiveCells(mesh, active_cells=active, value_inactive=1e-8)
    except TypeError:
        act_map = maps.InjectActiveCells(mesh, indActive=active, valInactive=1e-8)
    simulation = dc.Simulation3DCellCentered(
        mesh, survey=survey, sigmaMap=act_map, solver=solver
    )
    measured = df.loc[ordered, "R"].to_numpy(float)
    return mesh, active_idx, simulation, measured, ordered


def _electrode_positions(df):
    positions = {}
    for idx, row in df.iterrows():
        positions.setdefault(int(row.A_idx), np.array([row.A_x, row.A_y, row.A_z]))
        positions.setdefault(int(row.M_idx), np.array([row.M_x, row.M_y, row.M_z]))
        positions.setdefault(int(row.N_idx), np.array([row.N_x, row.N_y, row.N_z]))
    max_index = max(positions)
    electrodes = np.full((max_index + 1, 3), np.nan)
    for index, position in positions.items():
        electrodes[index] = position
    return electrodes


def model_from_parameters(mesh, active_idx, cfg, parameters):
    x, y, z, radius, rho_anomaly = parameters
    sigma = np.full(mesh.n_cells, 1.0 / cfg.domain.bg_resistivity)
    centers = mesh.cell_centers[active_idx]
    inside = np.sum((centers - np.array([x, y, z])) ** 2, axis=1) <= radius ** 2
    sigma[active_idx[inside]] = 1.0 / rho_anomaly
    return sigma


def make_plot(mesh, active_idx, sigma, parameters, output):
    # Interpolate the fitted model to the same regular grid used by the GT.
    x = np.linspace(0, 100, 50)
    y = np.linspace(0, 100, 50)
    z = np.linspace(0, 50, 25)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    active_centers = mesh.cell_centers[active_idx]
    from scipy.interpolate import NearestNDInterpolator
    rho = 1.0 / NearestNDInterpolator(active_centers, sigma[active_idx])(points)
    rho = rho.reshape((len(x), len(y), len(z)))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    norm = LogNorm(vmin=1, vmax=10000)
    images = [
        (rho[:, :, len(z) // 2].T, [0, 100, 0, 100], "XY"),
        (rho[:, len(y) // 2, :].T, [0, 100, 50, 0], "XZ"),
        (rho[len(x) // 2, :, :].T, [0, 100, 50, 0], "YZ"),
    ]
    for ax, (image, extent, name) in zip(axes, images):
        im = ax.imshow(image, extent=extent, origin="upper" if name != "XY" else "lower",
                       cmap="jet", norm=norm, aspect="equal")
        ax.set_title(f"Ajuste paramétrico ({name})")
        fig.colorbar(im, ax=ax, label=r"Resistividad ($\Omega\cdot m$)")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="dataset_output/measurements.csv")
    parser.add_argument("--max_nfev", type=int, default=12)
    parser.add_argument("--max_measurements", type=int, default=2000,
                        help="Submuestreo para acelerar la inversión; 0 usa todas")
    parser.add_argument("--output", default="inverse/parametric_result")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if args.max_measurements > 0 and len(df) > args.max_measurements:
        indices = np.linspace(0, len(df) - 1, args.max_measurements, dtype=int)
        df = df.iloc[np.unique(indices)].reset_index(drop=True)
        print(f"Usando {len(df)} mediciones estratificadas de la campaña")
    if not {"A_idx", "M_idx", "N_idx", "R", "Rho_a"}.issubset(df.columns):
        raise ValueError("measurements.csv no tiene las columnas de la campaña ERT")
    cfg = AppConfig()
    # Inversion mesh is deliberately coarser than the generation mesh.  The
    # anomaly is parametrized analytically, so fine local refinement is not
    # needed and would make each objective evaluation prohibitively slow.
    cfg.mesh.core_cell_size = 5.0
    cfg.mesh.padding_cells = 3
    mesh, active_idx, simulation, measured, ordered = build_forward(df, cfg)
    log_measured = np.log(np.maximum(np.abs(measured), 1e-12))

    def residual(parameters):
        sigma = model_from_parameters(mesh, active_idx, cfg, parameters)
        predicted = simulation.dpred(sigma[active_idx])
        # Log residual balances the many small and few large resistances.
        return np.log(np.maximum(np.abs(predicted), 1e-12)) - log_measured

    x0 = np.array([50.0, 50.0, 25.0, 10.0, 30.0])
    lower = np.array([15.0, 15.0, 3.0, 4.0, 10.0])
    upper = np.array([85.0, 85.0, 47.0, 25.0, 200.0])
    result = least_squares(residual, x0, bounds=(lower, upper),
                           max_nfev=args.max_nfev, verbose=1, diff_step=0.05)
    sigma = model_from_parameters(mesh, active_idx, cfg, result.x)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output.with_suffix(".npz"), parameters=result.x, sigma=sigma,
             cell_centers=mesh.cell_centers)
    with output.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump({"x": result.x.tolist(), "cost": float(result.cost),
                   "rmse_log_R": float(np.sqrt(2 * result.cost / len(measured))),
                   "nfev": result.nfev}, handle, indent=2)
    make_plot(mesh, active_idx, sigma, result.x, output.with_suffix(".png"))
    print("Parámetros ajustados [x, y, z, radio, rho]:", result.x)
    print("RMSE log(R):", np.sqrt(2 * result.cost / len(measured)))
    print("Resultados:", output.with_suffix(".json"))


if __name__ == "__main__":
    main()
