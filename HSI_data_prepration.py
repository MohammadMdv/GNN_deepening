import numpy as np
import scipy.io as scio
import scipy.sparse as sp
from torch_geometric.data import Dataset, Data
import torch


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


class HSI_dataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        self.ALL_X = scio.loadmat('HSI_data/ALL_X.mat')
        self.ALL_Y = scio.loadmat('HSI_data/ALL_Y.mat')
        self.ALL_L = scio.loadmat('HSI_data/ALL_L.mat')

        self.ALL_L = self.ALL_L['ALL_L']
        self.ALL_X = self.ALL_X['ALL_X']
        self.ALL_Y = self.ALL_Y['ALL_Y']
        # converting ALL_L to numpy array

        self.ALL_Y = convert_to_one_hot(self.ALL_Y - 1, 16)
        self.ALL_Y = self.ALL_Y.T
        super(HSI_dataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return ['ALL_X.mat', 'ALL_Y.mat', 'ALL_L.mat']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def get_adjacency_matrix(self, adj):
        adj = sp.csc_matrix.toarray(adj)
        row, col = np.where(adj)
        coo = np.array(list(zip(row, col)))
        coo = np.reshape(coo, (2, -1))
        return torch.tensor(coo, dtype=torch.long)

    def process(self):
        self.ALL_X = np.concatenate((self.ALL_X, np.zeros((self.ALL_X.shape[0], 1233))), axis=1)

        data = Data(x=torch.tensor(self.ALL_X, dtype=torch.float),
                    y=torch.tensor(self.ALL_Y, dtype=torch.float),
                    edge_index=torch.tensor(self.get_adjacency_matrix(self.ALL_L), dtype=torch.long))
        torch.save(data, self.processed_paths[0])

    def len(self):
        return len(self.ALL_Y)

    def get(self, idx):
        data = torch.load(self.processed_paths[0])
        return data


# dataset = HSI_dataset(root="HSI_data/")
