import h5py
import numpy as np
import yaml
import os

class HDF5Writer:
    def __init__(self, filepath, max_samples=None):
        self.filepath = filepath
        self.file = h5py.File(filepath, 'a')
        self._initialized = False
        
        with open("configs/mesh.yaml", 'r') as f:
            self.mesh_config = yaml.safe_load(f)['mesh']
            
        self.pad_x = self.mesh_config['pad_x']
        self.nx = self.mesh_config['nx']
        self.pad_y = self.mesh_config['pad_y']
        self.ny = self.mesh_config['ny']
        self.pad_z = self.mesh_config['pad_z']
        self.nz = self.mesh_config['nz']

    def _initialize_datasets(self, sample):
        num_meas = len(sample['measurements'])
        
        # inputs/apparent_resistivity -> shape (0, num_meas)
        self.file.create_dataset('inputs/apparent_resistivity', shape=(0, num_meas), maxshape=(None, num_meas), dtype='float32', compression='gzip')
        
        # inputs/electrode_positions -> shape (0, num_meas, 4, 3)
        self.file.create_dataset('inputs/electrode_positions', shape=(0, num_meas, 4, 3), maxshape=(None, num_meas, 4, 3), dtype='float32', compression='gzip')
        
        # labels/true_resistivity_3d -> shape (0, nx, ny, nz)
        self.file.create_dataset('labels/true_resistivity_3d', shape=(0, self.nx, self.ny, self.nz), maxshape=(None, self.nx, self.ny, self.nz), dtype='float32', compression='gzip')
        
        self._initialized = True

    def _extract_core_sigma(self, sigma):
        nx_total = self.nx + 2 * self.pad_x
        ny_total = self.ny + 2 * self.pad_y
        nz_total = self.nz + self.pad_z
        # Reshape using Fortran order as typical for discretize/simpeg
        sigma_3d = sigma.reshape((nx_total, ny_total, nz_total), order='F')
        core_sigma = sigma_3d[self.pad_x:self.pad_x+self.nx, 
                              self.pad_y:self.pad_y+self.ny, 
                              self.pad_z:self.pad_z+self.nz]
        return core_sigma
        
    def _get_electrode_pos(self, sample):
        electrodes = sample['electrodes']
        measurements = sample['measurements']
        num_meas = len(measurements)
        pos = np.zeros((num_meas, 4, 3), dtype=np.float32)
        
        for i, m in enumerate(measurements):
            pos_A = electrodes[m['A']]
            pos_B = electrodes[m['B']] if m['B'] != -1 else pos_A + np.array([1000, 0, 0])
            pos_M = electrodes[m['M']]
            pos_N = electrodes[m['N']]
            pos[i] = [pos_A, pos_B, pos_M, pos_N]
            
        return pos

    def append_batch(self, samples):
        if not samples:
            return
            
        if not self._initialized:
            if 'inputs/apparent_resistivity' in self.file:
                self._initialized = True
            else:
                self._initialize_datasets(samples[0])
                
        num_new = len(samples)
        
        # Prepare batches
        rho_a_batch = np.array([[m['rho_a'] for m in s['measurements']] for s in samples], dtype=np.float32)
        pos_batch = np.array([self._get_electrode_pos(s) for s in samples], dtype=np.float32)
        core_sigma_batch = np.array([self._extract_core_sigma(s['sigma']) for s in samples], dtype=np.float32)
        
        # Resize and append
        current_size = self.file['inputs/apparent_resistivity'].shape[0]
        new_size = current_size + num_new
        
        self.file['inputs/apparent_resistivity'].resize(new_size, axis=0)
        self.file['inputs/apparent_resistivity'][current_size:new_size] = rho_a_batch
        
        self.file['inputs/electrode_positions'].resize(new_size, axis=0)
        self.file['inputs/electrode_positions'][current_size:new_size] = pos_batch
        
        self.file['labels/true_resistivity_3d'].resize(new_size, axis=0)
        self.file['labels/true_resistivity_3d'][current_size:new_size] = core_sigma_batch
        
        self.file.flush()

    def close(self):
        self.file.close()
