import h5py
import pandas as pd
import numpy as np
import argparse

def export_sample_to_csv(h5_file, sample_idx=0, output_csv="dataset_sample.csv"):
    print(f"Abriendo {h5_file} y extrayendo muestra {sample_idx}...")
    
    with h5py.File(h5_file, 'r') as f:
        rho_a = f['inputs']['apparent_resistivity'][sample_idx]
        electrodes = f['inputs']['electrode_positions'][sample_idx]
        
        # electrodes tiene forma (1175, 4, 3) 
        # donde 4 son los electrodos A, B, M, N y 3 son las coordenadas x, y, z
        
        rows = []
        for i in range(len(rho_a)):
            A = electrodes[i, 0]
            B = electrodes[i, 1]
            M = electrodes[i, 2]
            N = electrodes[i, 3]
            
            rows.append({
                'A_x': A[0], 'A_y': A[1], 'A_z': A[2],
                'B_x': B[0], 'B_y': B[1], 'B_z': B[2],
                'M_x': M[0], 'M_y': M[1], 'M_z': M[2],
                'N_x': N[0], 'N_y': N[1], 'N_z': N[2],
                'rho_a': rho_a[i]
            })
            
        df = pd.DataFrame(rows)
        df.to_csv(output_csv, index=False)
        print(f"Muestra {sample_idx} exportada exitosamente a {output_csv} con {len(df)} mediciones.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exporta una muestra de HDF5 a CSV")
    parser.add_argument("--h5_file", type=str, default="ert3d_dataset_final.h5", help="Ruta al archivo HDF5")
    parser.add_argument("--sample_idx", type=int, default=0, help="Indice de la muestra a exportar")
    parser.add_argument("--output_csv", type=str, default="dataset_sample_0.csv", help="Nombre del archivo CSV de salida")
    
    args = parser.parse_args()
    export_sample_to_csv(args.h5_file, args.sample_idx, args.output_csv)
