from torch_geometric.datasets import QM9
import torch_geometric.transforms as T
import copy
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
import numpy as np
import os
from sklearn.linear_model import LinearRegression
import torch

def precompute_qm9_normalization(ds, output_path='qm9_atomref.npz'):
    prop_id = 7                    # U0 (eV) in the PyG loader
    Z, ptr = ds.data.z.numpy(), ds.slices['z']
    y = ds.data.y[:, prop_id].numpy()            # (n_mol,)

    elements = [1, 6, 7, 8, 9]                        # H,C,N,O,F
    X = np.zeros((len(ptr)-1, len(elements)))
    for i in range(len(ptr)-1):
        atoms = Z[ptr[i]:ptr[i+1]]
        #Number of atoms of each type (for each molecule i)
        X[i]  = [(atoms == Z0).sum() for Z0 in elements]

    lin = LinearRegression().fit(X, y)
    baseline  = lin.predict(X)
    residuals = y - baseline

    # replace targets in–memory; now train on residuals
    ds.data.y[:, prop_id] = torch.tensor(residuals, dtype=torch.float32)
    # np.savez(output_path, coef=lin.coef_, intercept=lin.intercept_)
    return {'coef': lin.coef_, 'intercept': lin.intercept_}

class WrappedQM9(InMemoryDataset):
    def __init__(self, root: str, name: str, train: bool = True, transform=None, pre_transform=None, pre_filter=None, radius: float = 5.0, num_neighbors: int = 12):
        super().__init__(root, transform, pre_transform, pre_filter)
        # self.compute_edge_indices = T.RadiusGraph(r=radius, max_num_neighbors=num_neighbors)

        self.compute_edge_indices_norm = T.Compose([T.RadiusGraph(r=radius, max_num_neighbors=num_neighbors), T.Distance(norm=False)])
        self.name = name
        self.train = train
        self.qm9_dataset = QM9(root=root, transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)
        self._load_normalized_parameters()


    def len(self):
        return 1000;
        return len(self.qm9_dataset)

    def get(self, idx):
        if self._data:
            return super().get(idx)
        
        if not hasattr(self, '_data_list') or self._data_list is None:
            self._data_list = self.len() * [None]
        elif self._data_list[idx] is not None:
            return copy.copy(self._data_list[idx])
    
        element = self.qm9_dataset[idx]
        atom_numbers = element.z
        kcalmol_energy = self._normalize_energy(self._extract_energy(element))

        element = self.compute_edge_indices_norm(element)

        encapsulated_data = Data(
            x=atom_numbers.unsqueeze(1),
            pos=element.pos,
            edge_index=element.edge_index,
            edge_attr=element.edge_attr,
            edge_weight=element.edge_attr.view(-1),
            y=kcalmol_energy
        )

        return encapsulated_data

    def _extract_energy(self, element):
        return element.y[:,7]
    
    def _normalize_energy(self, energy):
        #Ev to kcal/mol
        return energy * 23.0621

    def _load_normalized_parameters(self, path='qm9_atomref.npz'):
        precompute_qm9_normalization(self.qm9_dataset, path)
        # if not os.path.exists(path):
        #     precompute_qm9_normalization(self.qm9_dataset, path)
        # else:
        #     print(f"Loading normalized parameters from {path}")
        #     coef, intercept = np.load(path)['coef'], np.load(path)['intercept']
        #     self.qm9_dataset.data.y[:, 7] = self.qm9_dataset.data.y[:, 7] * coef + intercept


    def get_idx_split(self):
        """ Get dataset splits.

        Returns:
            Dict with 'train', 'val', 'test', splits indices.
        """
        # random split from self.data_df
        all_indices = list(range(self.len()))
        num_train = int(0.8 * len(all_indices))
        num_val = int(0.1 * len(all_indices))
        num_test = len(all_indices) - num_train - num_val
        train_indices = all_indices[:num_train]
        val_indices = all_indices[num_train:num_train + num_val]
        test_indices = all_indices[num_train + num_val:]
        return {
            'train': train_indices,
            'val': val_indices,
            'test': test_indices
        }