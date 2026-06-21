import h5py
import numpy as np
import os

class HDF5Writer:
    def __init__(self, filepath, max_samples=None):
        self.filepath = filepath
        self.file = h5py.File(filepath, 'a')
        self._initialized = False

    def _initialize_datasets(self, sample):
        sigma = sample['sigma']
        measurements = sample['measurements']
        
        num_cells = sigma.shape[0]
        num_meas = len(measurements)
        
        # Initialize datasets with unlimited maxshape for appending
        self.file.create_dataset('sigma', shape=(0, num_cells), maxshape=(None, num_cells), dtype='float32', compression='gzip')
        self.file.create_dataset('rho_a', shape=(0, num_meas), maxshape=(None, num_meas), dtype='float32', compression='gzip')
        self.file.create_dataset('delta_v', shape=(0, num_meas), maxshape=(None, num_meas), dtype='float32', compression='gzip')
        
        # Assume electrode layout and sequence is fixed per dataset for simplicity
        electrodes = sample['electrodes']
        self.file.create_dataset('electrodes', data=electrodes, dtype='float32')
        
        A_idx = np.array([m['A'] for m in measurements], dtype='int32')
        B_idx = np.array([m['B'] for m in measurements], dtype='int32')
        M_idx = np.array([m['M'] for m in measurements], dtype='int32')
        N_idx = np.array([m['N'] for m in measurements], dtype='int32')
        
        self.file.create_dataset('sequence_A', data=A_idx, dtype='int32')
        self.file.create_dataset('sequence_B', data=B_idx, dtype='int32')
        self.file.create_dataset('sequence_M', data=M_idx, dtype='int32')
        self.file.create_dataset('sequence_N', data=N_idx, dtype='int32')
        
        self._initialized = True

    def append_batch(self, samples):
        if not samples:
            return
            
        if not self._initialized:
            # Check if datasets already exist from previous runs
            if 'sigma' in self.file:
                self._initialized = True
            else:
                self._initialize_datasets(samples[0])
                
        num_new = len(samples)
        
        sigma_batch = np.array([s['sigma'] for s in samples])
        rho_a_batch = np.array([[m['rho_a'] for m in s['measurements']] for s in samples])
        delta_v_batch = np.array([[m['delta_v'] for m in s['measurements']] for s in samples])
        
        # Resize and append
        current_size = self.file['sigma'].shape[0]
        new_size = current_size + num_new
        
        self.file['sigma'].resize(new_size, axis=0)
        self.file['sigma'][current_size:new_size] = sigma_batch
        
        self.file['rho_a'].resize(new_size, axis=0)
        self.file['rho_a'][current_size:new_size] = rho_a_batch
        
        self.file['delta_v'].resize(new_size, axis=0)
        self.file['delta_v'][current_size:new_size] = delta_v_batch
        
        self.file.flush()

    def close(self):
        self.file.close()
