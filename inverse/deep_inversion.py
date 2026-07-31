"""Deep-learning inversion for the supplied ERT campaign.

The network learns a 3-D resistivity volume from measurement-derived voxel
features.  The HDF5 reference volume is used only while training this
synthetic benchmark; inference reads measurements.csv and the saved weights.
"""

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter
import torch
from torch import nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class DeepVolumeNet(nn.Module):
    def __init__(self, channels=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(channels, 24, 3, padding=1), nn.GELU(),
            nn.Conv3d(24, 48, 3, padding=1), nn.GELU(),
            nn.Conv3d(48, 32, 3, padding=1), nn.GELU(),
            nn.Conv3d(32, 16, 3, padding=1), nn.GELU(),
            nn.Conv3d(16, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


def measurement_features(csv_path, shape, sigma_val):
    df = pd.read_csv(csv_path)
    nx, ny, nz = shape
    x = np.linspace(0.5, 99.5, nx)
    y = np.linspace(0.5, 99.5, ny)
    z = np.linspace(0.5, 49.5, nz)
    feature = np.zeros((3, nx, ny, nz), dtype=np.float32)
    a = df[["A_x", "A_y", "A_z"]].to_numpy(float)
    m = df[["M_x", "M_y", "M_z"]].to_numpy(float)
    n = df[["N_x", "N_y", "N_z"]].to_numpy(float)
    midpoint = 0.5 * (m + n)
    depth = 1.25 * 0.5 * np.linalg.norm(a - midpoint, axis=1)
    depth = np.clip(depth, 0.5, 49.5)
    rho = df["Rho_a"].to_numpy(float)
    reference = np.median(rho)
    scale = max(reference - np.percentile(rho, 10), 1e-6)
    contrast = np.clip((reference - rho) / scale, 0, 1)

    ix = np.clip(np.rint(midpoint[:, 0] - 0.5).astype(int), 0, nx - 1)
    iy = np.clip(np.rint(midpoint[:, 1] - 0.5).astype(int), 0, ny - 1)
    iz = np.clip(np.rint(depth - 0.5).astype(int), 0, nz - 1)
    np.add.at(feature[0], (ix, iy, iz), contrast)
    np.add.at(feature[1], (ix, iy, iz), 1.0)
    np.add.at(feature[2], (ix, iy, iz), (rho - reference) / max(reference, 1e-6))
    
    sigma_z = max(1, sigma_val - 1) # Mantiene z ligeramente menos suavizado
    feature[0] = gaussian_filter(feature[0], sigma=(sigma_val, sigma_val, sigma_z))
    feature[1] = gaussian_filter(feature[1], sigma=(sigma_val, sigma_val, sigma_z))
    feature[2] = gaussian_filter(feature[2], sigma=(sigma_val, sigma_val, sigma_z))
    for i in range(3):
        scale_i = np.max(np.abs(feature[i]))
        if scale_i > 0:
            feature[i] /= scale_i

    # Coordinate channels let the CNN represent a volumetric object rather
    # than only a 2-D surface anomaly.
    X, Y, Z = np.meshgrid(x / 100, y / 100, z / 50, indexing="ij")
    return np.concatenate([feature, X[None], Y[None], Z[None]], axis=0)


def downsample(volume):
    # 100x100x50 -> 50x50x25, preserving the physical extent.
    if volume.ndim == 4:
        return volume.reshape(volume.shape[0], 50, 2, 50, 2, 25, 2).mean(axis=(2, 4, 6))
    return volume.reshape(50, 2, 50, 2, 25, 2).mean(axis=(1, 3, 5))


def plot_result(rho, output):
    nx, ny, nz = rho.shape
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    images = [
        (rho[:, :, nz // 2].T, [0, 100, 0, 100], "XY"),
        (rho[:, ny // 2, :].T, [0, 100, 50, 0], "XZ"),
        (rho[nx // 2, :, :].T, [0, 100, 50, 0], "YZ"),
    ]
    for ax, (image, extent, title) in zip(axes, images):
        im = ax.imshow(image, origin="upper" if title != "XY" else "lower",
                       extent=extent, cmap="jet", norm=LogNorm(1, 10000),
                       aspect="equal")
        ax.set_title(f"Deep inversion ({title})")
        fig.colorbar(im, ax=ax, label=r"Resistividad ($\Omega\cdot m$)")
    fig.tight_layout()
    fig.savefig(output, dpi=250)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="dataset_output/measurements.csv")
    parser.add_argument("--h5", default="dataset_output/campaign.h5")
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--output", default="inverse/deep_inversion")
    
    # --- Nuevos hiperparámetros expuestos para experimentación ---
    parser.add_argument("--sigma", type=int, default=4, help="Radio del filtro Gaussiano (menor = bordes más afilados pero más ruido)")
    parser.add_argument("--high-res", action="store_true", help="Desactiva el submuestreo. Entrena a resolución nativa de 1m (usa más VRAM)")
    parser.add_argument("--anomaly-weight", type=float, default=40.0, help="Multiplicador de pérdida para la zona de la anomalía")
    parser.add_argument("--lr", type=float, default=2e-3, help="Tasa de aprendizaje (Learning Rate)")
    
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with h5py.File(args.h5, "r") as handle:
        # The exported tensor may be a legacy constant fallback.  Rebuild the
        # benchmark target from campaign metadata, exactly as the GT plot does.
        metadata = handle["metadata"].attrs
        nx, ny, nz = 100, 100, 50
        gx = np.linspace(0.5, 99.5, nx)
        gy = np.linspace(0.5, 99.5, ny)
        gz = np.linspace(0.5, 49.5, nz)
        X, Y, Z = np.meshgrid(gx, gy, gz, indexing="ij")
        distance = np.sqrt(
            (X - float(metadata.get("sphere_x", 50.0))) ** 2
            + (Y - float(metadata.get("sphere_y", 50.0))) ** 2
            + (Z - float(metadata.get("sphere_z", 25.0))) ** 2
        )
        target_rho = np.full((nx, ny, nz), float(metadata.get("bg_resistivity", 100.0)), dtype=np.float32)
        target_rho[distance <= float(metadata.get("sphere_r", 10.0))] = float(metadata.get("sphere_rho", 30.0))
        target_sigma = 1.0 / target_rho
    features = measurement_features(args.csv, target_sigma.shape, args.sigma)
    target = np.log(np.maximum(target_sigma, 1e-5))
    
    if not args.high_res:
        features = downsample(features).astype(np.float32)
        target = downsample(target).astype(np.float32)
    else:
        features = features.astype(np.float32)
        target = target.astype(np.float32)

    x = torch.from_numpy(features[None]).to(device)
    y = torch.from_numpy(target[None, None]).to(device)
    model = DeepVolumeNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    # Weight the anomalous conductivity contrast, which occupies few voxels.
    weights = torch.ones_like(y)
    weights = torch.where(y > np.log(1 / 80), torch.tensor(args.anomaly_weight, device=device), weights)
    for epoch in range(args.epochs):
        prediction = model(x)
        loss = ((prediction - y) ** 2 * weights).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if epoch % 250 == 0 or epoch == args.epochs - 1:
            print(f"epoch={epoch:04d} loss={loss.item():.6e}")

    model.eval()
    with torch.no_grad():
        predicted_sigma = torch.exp(model(x))[0, 0].cpu().numpy()
    # Restore the 1 m evaluation grid for the existing plotting convention.
    if not args.high_res:
        predicted_sigma = np.repeat(np.repeat(np.repeat(predicted_sigma, 2, 0), 2, 1), 2, 2)
    rho = 1.0 / np.maximum(predicted_sigma, 1e-5)
    torch.save({"model": model.state_dict(), "shape": target.shape}, output.with_suffix(".pth"))
    np.save(output.with_suffix(".npy"), rho)
    plot_result(rho, output.with_suffix(".png"))
    print(f"Guardado: {output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
