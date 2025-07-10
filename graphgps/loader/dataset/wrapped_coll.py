import os
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
import torch
from torch_geometric.transforms import RadiusGraph
from tqdm import tqdm

class WrappedColl(InMemoryDataset):
    def __init__(self, root, radius=5.0, max_neighbors=50, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.radius_graph = RadiusGraph(r=radius, max_num_neighbors=max_neighbors)
        self._load_all()

    def _load_all(self):
        self.data_list = []
        self.splits = {}

        for split in ['train', 'val', 'test']:
            npz_path = os.path.join(self.root, f'coll_{split}.npz')
            data = np.load(npz_path)
            R = data['R']
            Z = data['Z']
            E = data['E']
            F = data['F']
            N = data['N']

            offset = 0
            indices = []
            for i in tqdm(range(len(N)), desc=f'Loading {split}'):
                num_atoms = N[i]
                pos = torch.tensor(R[offset:offset + num_atoms], dtype=torch.float32)
                z = torch.tensor(Z[offset:offset + num_atoms], dtype=torch.long)
                y = torch.tensor(E[i:i+1], dtype=torch.float32)
                force = torch.tensor(F[offset:offset + num_atoms], dtype=torch.float32)

                x = z.unsqueeze(1)
                data_obj = Data(x=x, pos=pos, z=z, y=y, force=force)
                data_obj = self.radius_graph(data_obj)

                self.data_list.append(data_obj)
                indices.append(len(self.data_list) - 1)
                offset += num_atoms

            self.splits[split] = indices

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

    def get_idx_split(self):
        return self.splits
