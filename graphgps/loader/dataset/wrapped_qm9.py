from torch_geometric.datasets import QM9
import torch_geometric.transforms as T
import copy
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset

class WrappedQM9(InMemoryDataset):
    def __init__(self, root: str, name: str, train: bool = True, transform=None, pre_transform=None, pre_filter=None, radius: float = 5.0, num_neighbors: int = 12):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.compute_edge_indices = T.RadiusGraph(r=radius, max_num_neighbors=num_neighbors)
        self.name = name
        self.train = train
        self.qm9_dataset = QM9(root=root, transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)

    def len(self):
        return 1000

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

        element = self.compute_edge_indices(element)
        row, col = element.edge_index
        edge_weight = (element.pos[row] - element.pos[col]).norm(dim=-1)

        element.edge_weight = edge_weight

        encapsulated_data = Data(
            x=atom_numbers.unsqueeze(1),
            pos=element.pos,
            edge_index=element.edge_index,
            edge_weight=edge_weight,
            y=kcalmol_energy
        )

        return encapsulated_data

    def _extract_energy(self, element):
        return element.y[:,7]
    
    def _normalize_energy(self, energy):
        #Ev to kcal/mol
        return energy * 23.0621





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