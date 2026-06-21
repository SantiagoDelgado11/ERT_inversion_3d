import pandas as pd
import argparse
from dataset.generator import generate_single_sample

def export_sample_to_csv(output_csv="synthetic_measurements.csv"):
    print("Generating a synthetic sample...")
    sample = generate_single_sample(seed=42)
    
    measurements = sample['measurements']
    electrodes = sample['electrodes']
    
    rows = []
    for m in measurements:
        A_pos = electrodes[m['A']]
        B_pos = electrodes[m['B']] if m['B'] != -1 else [1000.0, 0.0, 0.0] # Infinity approx
        M_pos = electrodes[m['M']]
        N_pos = electrodes[m['N']]
        
        rows.append({
            'A_x': A_pos[0], 'A_y': A_pos[1], 'A_z': A_pos[2],
            'B_x': B_pos[0], 'B_y': B_pos[1], 'B_z': B_pos[2],
            'M_x': M_pos[0], 'M_y': M_pos[1], 'M_z': M_pos[2],
            'N_x': N_pos[0], 'N_y': N_pos[1], 'N_z': N_pos[2],
            'K': m['K'],
            'delta_v': m['delta_v'],
            'rho_a': m['rho_a']
        })
        
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Dataset of {len(df)} surface measurements successfully exported to {output_csv}")
    
    # Export anomalies
    anomalies = sample['anomalies']
    labels = []
    for a in anomalies:
        label = {'type': type(a).__name__, 'resistivity': a.resistivity}
        if type(a).__name__ == 'Sphere':
            label.update({'cx': a.cx, 'cy': a.cy, 'cz': a.cz, 'radius': a.radius})
        elif type(a).__name__ == 'Ellipsoid':
            label.update({'cx': a.cx, 'cy': a.cy, 'cz': a.cz, 'rx': a.rx, 'ry': a.ry, 'rz': a.rz})
        elif type(a).__name__ == 'Block':
            label.update({'x_min': a.x_min, 'x_max': a.x_max, 'y_min': a.y_min, 'y_max': a.y_max, 'z_min': a.z_min, 'z_max': a.z_max})
        labels.append(label)
        
    labels_csv = output_csv.replace(".csv", "_labels.csv")
    pd.DataFrame(labels).to_csv(labels_csv, index=False)
    print(f"Anomaly parameters exported to {labels_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export ERT Sample to CSV")
    parser.add_argument("--output", type=str, default="synthetic_measurements.csv", help="Output CSV filename")
    args = parser.parse_args()
    
    export_sample_to_csv(args.output)
