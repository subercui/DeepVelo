from torchvision import datasets, transforms
from base import BaseDataLoader
import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.metrics import pairwise_distances
import dgl
from dgl.contrib.sampling import NeighborSampler

class VeloDataset(Dataset):
    def __init__(self, data_dir, train=True, type='average', topC=30, topG=16):
        data_obj = np.load(data_dir)
        self.Ux_sz = data_obj['Ux_sz'].T
        self.Sx_sz = data_obj['Sx_sz'].T
        self.velo = data_obj['velo'].T # shape (1448, 1720) to shape (1720, 1448)
        self.topG = topG
        N_cell, N_gene = self.Sx_sz.shape
        

        self.S_tp1 = self.Sx_sz + self.velo
        dist = pairwise_distances(self.S_tp1, self.Sx_sz)  # (1720, 1720)
        ind = np.argsort(dist, axis=1)[:, :topC]  # (1720, 30)
        # update the velocity target vectors
        if type == 'average':
            self.velo = np.zeros(self.Sx_sz.shape, dtype=np.float32)
            for i in range(self.velo.shape[0]):
                self.velo[i] = np.mean(self.Sx_sz[ind[i]], axis=0) - self.Sx_sz[i]
        if type == 'min':
            # provide the vector to downstream neighbors as candidates
            # TODO(Haotian): temporaryly store the vectors here. Future: store the index ind
            # and compute the vectors on the fly during training, which can be more
            # memory efficient.
            self.velo = self.Sx_sz[ind] - np.expand_dims(self.Sx_sz, 1)

            # self.velo = np.zeros(self.Sx_sz.shape+(topC,), dtype=np.float32)
            # for i in range(self.velo.shape[0]):
            #     self.velo[i] = self.Sx_sz[ind[i]] - self.Sx_sz[i]
        elif type == 'mimic':
            pass
        
        # build the knn graph in the original space
        dist = pairwise_distances(self.Sx_sz, self.Sx_sz)  # (1720, 1720)
        ind = np.argsort(dist, axis=1)[:, :20]  # (1720, 20)
        self.g = self.build_graph(ind)
        self.ind = ind

        # # build the gene neighbors indices
        # self.neighbors_per_gene = np.zeros([N_cell, topG, N_gene], dtype=np.float32)
        # for g in range(N_gene):
        #     # FIXME(Haotian): for each gene, it has the problem that a lot of the genes are 0, 
        #     # thus the closest are other 0s, which makes no sense a lot of times
        #     dist = pairwise_distances(self.Sx_sz[:, g:g+1], self.Sx_sz[:, g:g+1])
        #     # import ipdb; ipdb.set_trace()
        #     self.neighbors_per_gene[:,:,g] = np.argsort(dist, axis=1)[:, :topG]

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

    def gen_neighbor_batch(self, size):
        indices = np.random.random_integers(0, high=len(self)-1, size=size)
        # self.neighbors_per_gene is the neighbor indices for all cells, shape (N_cells, topG, genes)

        # TODO(Haotian): try the per gene version
        # Here since the per gene version encounters the 0 gene count bug, we first use the per cell version
        # which is using self.ind
        return self.ind[indices, :self.topG].flatten()

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
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, 
                num_workers=1, training=True, type='average', topC=30, topG=16):
        self.data_dir = data_dir
        self.dataset = VeloDataset(self.data_dir, train=training, type=type, topC=topC, topG=topG)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class VeloNeighborSampler(NeighborSampler, BaseDataLoader):
    """
    minibatch neighbor sampler using DGL NeighborSampler
    """
    def __init__(
        self,
        data_dir,
        batch_size,
        num_neighbors,
        num_hops,
        shuffle=True,
        validation_split=0.0,
        num_workers=32,
        training=True
    ):
        self.data_dir = data_dir
        self.dataset = VeloDataset(self.data_dir, train=training)
        # FIXME: the split_validation here is not working as in the BaseDataLoader
        # BaseDataLoader.__init__(self, self.dataset, batch_size, shuffle, validation_split, num_workers)
        
        g = self.dataset.g
        norm = 1. / g.in_degrees().float().unsqueeze(1)
        g.ndata['Ux_sz'] = self.dataset.Ux_sz 
        g.ndata['Sx_sz'] = self.dataset.Sx_sz
        g.ndata['velo'] = self.dataset.velo
        g.ndata['norm'] = norm
        # need to set to readonly for nodeflow
        g.readonly()

        NeighborSampler.__init__(
            self,
            g, batch_size,
            num_neighbors,
            neighbor_type='in',
            shuffle=shuffle,
            num_workers=num_workers,
            num_hops=num_hops,
            #  seed_nodes=train_nid
        )

    #FIXME: the split_validation here is not working as in the BaseDataLoader
    def split_validation(self):
        return None

    def __len__(self):
        return self.dataset.__len__()


if __name__ == '__main__':
    VeloDataset('./data/DG_norm_genes.npz')
    VeloNeighborSampler(
        './data/DG_norm_genes.npz',
        32,
        15,
        4
    )
