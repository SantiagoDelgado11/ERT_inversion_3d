import h5py
import torch
from torch.utils.data import Dataset

class ERTDataset(Dataset):
    def __init__(self, h5_filepath, transform=None):
        self.h5_filepath = h5_filepath
        self.transform = transform
        # Open in read mode
        self.file = h5py.File(h5_filepath, 'r')
        self.length = self.file['sigma'].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Read from HDF5
        rho_a = torch.tensor(self.file['rho_a'][idx])
        sigma = torch.tensor(self.file['sigma'][idx])
        
        sample = {'rho_a': rho_a, 'sigma': sigma}
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
        
    def get_electrodes(self):
        return torch.tensor(self.file['electrodes'][:])
        
    def get_sequence(self):
        A = self.file['sequence_A'][:]
        B = self.file['sequence_B'][:]
        M = self.file['sequence_M'][:]
        N = self.file['sequence_N'][:]
        return A, B, M, N

    def close(self):
        self.file.close()
