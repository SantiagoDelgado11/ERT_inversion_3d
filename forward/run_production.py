import os
import sys
import subprocess
import h5py
import time

import argparse

parser = argparse.ArgumentParser(description="Resilient Production Runner")
parser.add_argument("--samples", type=int, default=10000)
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--cores", type=int, default=4)
parser.add_argument("--output", type=str, default="dataset/dataset_10k.h5")
args = parser.parse_args()

total_samples = args.samples
batch_size = args.batch_size
cores = args.cores
output_file = args.output

def get_current_samples(filepath):
    if not os.path.exists(filepath):
        return 0
    try:
        with h5py.File(filepath, 'r') as f:
            if 'inputs/apparent_resistivity' in f:
                return f['inputs/apparent_resistivity'].shape[0]
            return 0
    except Exception as e:
        print(f"Warning reading HDF5: {e}")
        return 0

def run():
    os.makedirs("dataset", exist_ok=True)
    while True:
        current = get_current_samples(output_file)
        if current >= total_samples:
            print(f"Goal Completed! {current} samples successfully generated.")
            break
            
        remaining = total_samples - current
        print(f"\n[GOAL AGENT] Resuming generation. Current samples: {current}, Remaining: {remaining}")
        
        cmd = [
            sys.executable, "generate_dataset.py",
            "--samples", str(remaining),
            "--batch_size", str(batch_size),
            "--cores", str(cores),
            "--output", output_file,
            "--seed_offset", str(current)
        ]
        
        try:
            process = subprocess.Popen(cmd)
            process.wait()
            
            if process.returncode == 0:
                print("[GOAL AGENT] Generation process finished properly.")
                current_check = get_current_samples(output_file)
                if current_check >= total_samples:
                    break
            else:
                print(f"[GOAL AGENT] Process crashed (OOM or other error, code {process.returncode}). Restarting in 5 seconds...")
                time.sleep(5)
                
        except Exception as e:
            print(f"[GOAL AGENT] Error executing script: {e}. Restarting in 5 seconds...")
            time.sleep(5)

if __name__ == "__main__":
    run()
