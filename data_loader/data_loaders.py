from torchvision import datasets, transforms
from base import BaseDataLoader
import torch
from torch.utils.data import Dataset
import numpy as np

class VeloDataset(Dataset):
    def __init__(self, data_dir, train=True):
        data_obj = np.load(data_dir)
        self.Ux_sz = data_obj['Ux_sz'].T
        self.Sx_sz = data_obj['Sx_sz'].T
        self.velo = data_obj['velo'].T # shape (1448, 1720) to shape (1720, 1448)
        
        self.Ux_sz = torch.tensor(self.Ux_sz, dtype=torch.float32)
        self.Sx_sz = torch.tensor(self.Sx_sz, dtype=torch.float32)
        self.velo = torch.tensor(self.velo, dtype=torch.float32)
        print('velo data shape:', self.velo.shape)

    def __len__(self):
        return len(self.Ux_sz)  # 1720

    def __getitem__(self, i):
        data_dict = {
            "Ux_sz": self.Ux_sz[i],
            "Sx_sz": self.Sx_sz[i],
            "velo": self.velo[i]
        }
        return data_dict

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class VeloDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.data_dir = data_dir
        self.dataset = VeloDataset(self.data_dir, train=training)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


if __name__ == '__main__':
    VeloDataset('./data/DG_norm_genes.npz')