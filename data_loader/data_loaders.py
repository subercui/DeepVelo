from torchvision import datasets, transforms
from base import BaseDataLoader
import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.metrics import pairwise_distances
import dgl

class VeloDataset(Dataset):
    def __init__(self, data_dir, train=True):
        data_obj = np.load(data_dir)
        self.Ux_sz = data_obj['Ux_sz'].T
        self.Sx_sz = data_obj['Sx_sz'].T
        self.velo = data_obj['velo'].T # shape (1448, 1720) to shape (1720, 1448)
        

        self.S_tp1 = self.Sx_sz + self.velo
        dist = pairwise_distances(self.S_tp1, self.Sx_sz)  # (1720, 1720)
        ind = np.argsort(dist, axis=1)[:, :30]  # (1720, 30)
        for i in range(self.velo.shape[0]):
            self.velo[i] = np.mean(self.Sx_sz[ind[i]], axis=0) - self.Sx_sz[i]
        
        # build the knn graph in the original space
        dist = pairwise_distances(self.Sx_sz, self.Sx_sz)  # (1720, 1720)
        ind = np.argsort(dist, axis=1)[:, :20]  # (1720, 20)
        self.g = self.build_graph(ind)

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

    def build_graph(self, ind):
        """ind (N,k) contains neighbor index"""
        print('building graph')
        g = dgl.DGLGraph()
        g.add_nodes(len(self.Ux_sz))
        edge_list = []
        for i in range(ind.shape[0]):
            for j in range(ind.shape[1]):
                edge_list.append((i,ind[i,j]))
        # add edges two lists of nodes: src and dst
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        # edges are directional in DGL; make them bi-directional
        g.add_edges(dst, src)
        return g

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