import h5py
import sys

def check_h5(filepath):
    try:
        with h5py.File(filepath, 'r') as f:
            print(f"--- HDF5 File: {filepath} ---")
            print("Apparent Resistivity shape:", f['inputs/apparent_resistivity'].shape)
            if 'inputs/delta_v' in f:
                print("Delta V shape:", f['inputs/delta_v'].shape)
            print("Electrode positions shape:", f['inputs/electrode_positions'].shape)
            if 'labels/true_conductivity_3d' in f:
                print("True Conductivity 3D shape:", f['labels/true_conductivity_3d'].shape)
            if 'labels/true_resistivity_3d' in f:
                print("Legacy mislabeled Conductivity 3D shape:", f['labels/true_resistivity_3d'].shape)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")

if __name__ == "__main__":
    filepath = sys.argv[1] if len(sys.argv) > 1 else 'test_dataset.h5'
    check_h5(filepath)
