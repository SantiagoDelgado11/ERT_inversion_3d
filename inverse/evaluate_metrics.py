import argparse
import h5py
import numpy as np
import pandas as pd
from pathlib import Path

def evaluate(npy_path, h5_path):
    # Cargar la predicción
    rho_pred = np.load(npy_path)
    
    # Use the exact discretized ground truth saved with the campaign.  Rebuilding
    # an ideal sphere can disagree with TreeMesh interpolation at the boundary.
    with h5py.File(h5_path, "r") as handle:
        gt = handle["ground_truth_conductivity"]
        sigma_true = np.asarray(gt["conductivity_tensor"], dtype=np.float32)
        gx = np.asarray(gt["grid_x"], dtype=np.float32)
        gy = np.asarray(gt["grid_y"], dtype=np.float32)
        gz = np.asarray(gt["grid_z"], dtype=np.float32)
        X, Y, Z = np.meshgrid(gx, gy, gz, indexing="ij")
        nx, ny, nz = sigma_true.shape
        metadata = handle["metadata"].attrs
        sx = float(metadata.get("sphere_x", 50.0))
        sy = float(metadata.get("sphere_y", 50.0))
        sz = float(metadata.get("sphere_z", 25.0))
        sr = float(metadata.get("sphere_r", 10.0))
        srho = float(metadata.get("sphere_rho", 30.0))
        bg_rho = float(metadata.get("bg_resistivity", 100.0))
        
        target_rho = 1.0 / np.maximum(sigma_true, 1e-8)
        anomaly_mask = target_rho < np.sqrt(bg_rho * srho)
        
    # Si la predicción se hizo a baja resolución, la interpolamos para poder comparar peras con peras
    if rho_pred.shape != target_rho.shape:
        from scipy.ndimage import zoom
        factors = tuple(t / p for t, p in zip(target_rho.shape, rho_pred.shape))
        rho_pred = zoom(rho_pred, factors, order=1)
        
    if rho_pred.shape != target_rho.shape:
        print(f"Ignorando {npy_path}: Dimensiones incompatibles {rho_pred.shape}")
        return
        
    # Es mucho más estable matemáticamente comparar conductividades (S/m) que resistividades
    # para evitar que un píxel mal predicho con 10,000 ohm-m arruine toda la métrica.
    sigma_pred = 1.0 / np.maximum(rho_pred, 1e-5)
    sigma_true = 1.0 / target_rho
    
    # 1. RMSE Global
    rmse_global = np.sqrt(np.mean((sigma_pred - sigma_true)**2))
    
    # 2. RMSE de la Anomalía (El más importante: ¿Qué tan bien recuperó la esfera?)
    rmse_anomaly = np.sqrt(np.mean((sigma_pred[anomaly_mask] - sigma_true[anomaly_mask])**2))
    
    # 3. RMSE del Fondo (¿Introdujo artefactos "fantasma" donde no había nada?)
    rmse_bg = np.sqrt(np.mean((sigma_pred[~anomaly_mask] - sigma_true[~anomaly_mask])**2))
    
    # 4. Error Geométrico (Centro de masa y Volumen)
    # Definimos como "anomalía predicha" todo lo que esté por debajo del promedio logarítmico entre fondo y esfera
    threshold_rho = np.exp((np.log(bg_rho) + np.log(srho)) / 2.0)
    pred_mask = rho_pred < threshold_rho
    
    print(f"=== Evaluacion de: {Path(npy_path).name} ===")
    print(f"RMSE Global:            {rmse_global:.2e} S/m")
    print(f"RMSE en la ANOMALÍA:    {rmse_anomaly:.2e} S/m  <-- (Más bajo es mejor resolución)")
    print(f"RMSE en el Fondo:       {rmse_bg:.2e} S/m  <-- (Más bajo = menos artefactos/ruido)")
    
    if np.any(pred_mask):
        com_x = np.average(X[pred_mask])
        com_y = np.average(Y[pred_mask])
        com_z = np.average(Z[pred_mask])
        error_com = np.sqrt((com_x - sx)**2 + (com_y - sy)**2 + (com_z - sz)**2)
        
        true_vol = np.sum(anomaly_mask)
        pred_vol = np.sum(pred_mask)
        vol_error_pct = (pred_vol - true_vol) / true_vol * 100.0
        
        print(f"Centro de masa (Real):  X={sx:.1f}, Y={sy:.1f}, Z={sz:.1f}")
        print(f"Centro de masa (Pred):  X={com_x:.1f}, Y={com_y:.1f}, Z={com_z:.1f}")
        print(f"Error de Localización:  {error_com:.2f} metros")
        print(f"Error Volumétrico:      {vol_error_pct:+.1f} %")
    else:
        print("Error de Localización:  [Fallo Crítico] La red colapsó y no detectó la anomalía.")
        
    print("-" * 50)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("npy_files", nargs="+", help="Rutas a los archivos .npy generados por la red")
    parser.add_argument("--h5", default="dataset_output/campaign.h5", help="Ruta al Ground Truth original")
    args = parser.parse_args()
    
    for f in args.npy_files:
        evaluate(f, args.h5)
