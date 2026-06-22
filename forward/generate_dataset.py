import argparse
import multiprocessing as mp
from dataset.generator import generate_single_sample
from dataset.hdf5_writer import HDF5Writer
from tqdm import tqdm
import os
import yaml
from mesh.mesh_generator import generate_mesh
import gc

def worker(seed):
    try:
        # Load configs per process to avoid serialization issues
        with open("configs/survey.yaml", 'r') as f:
            config_survey = yaml.safe_load(f)['survey']
        with open("configs/geology.yaml", 'r') as f:
            config_geology = yaml.safe_load(f)['geology']
            
        # Passing mesh implicitly or regenerating it per worker
        # since it's cheap to generate
        mesh = generate_mesh()
        
        sample = generate_single_sample(
            seed=seed, 
            mesh=mesh,
            config_geology=config_geology,
            config_survey=config_survey
        )
        return sample
    except Exception as e:
        print(f"Error in worker {seed}: {e}")
        return None
    finally:
        # Force garbage collection to prevent memory leaks from SimPEG matrices
        gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ERT 3D Dataset")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for saving checkpoints")
    parser.add_argument("--cores", type=int, default=4, help="Number of CPU cores to use")
    parser.add_argument("--output", type=str, default="dataset.h5", help="Output HDF5 file")
    parser.add_argument("--seed_offset", type=int, default=0, help="Offset for random seeds to avoid duplicates across runs")
    
    args = parser.parse_args()
    
    # Pre-generate mesh just to log details
    mesh = generate_mesh()
    print(f"Mesh has {mesh.nC} cells, Nodes: {mesh.nN}")
    print(f"Starting generation of {args.samples} samples using {args.cores} cores.")
    
    writer = HDF5Writer(args.output)
    
    seeds = [args.seed_offset + i for i in range(args.samples)]
    batch = []
    
    # Using maxtasksperchild=1 forces Python to kill and restart the worker process
    # after every task, releasing all RAM memory entirely back to the OS.
    with mp.Pool(processes=args.cores, maxtasksperchild=1) as pool:
        for sample in tqdm(pool.imap_unordered(worker, seeds), total=args.samples):
            if sample is not None:
                batch.append(sample)
                
            if len(batch) >= args.batch_size:
                writer.append_batch(batch)
                batch = []
                
        # Append remaining
        if batch:
            writer.append_batch(batch)
            
    writer.close()
    print(f"Dataset saved to {args.output}")
