import copy
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
import torch
from openqdc import DES370K
import torch_geometric.transforms as T
from tqdm import tqdm


def normalize_des370k_energy(dataset, train_indices):
    mean_energy = 0.0
    for idx in tqdm(train_indices, desc="Computing mean energy"):
        item = dataset[idx]
        # Get the first non-NaN energy value
        energies = torch.tensor(item.energies)
        valid_energies = energies[~torch.isnan(energies)]
        if valid_energies.numel() > 0:
            energy = valid_energies[0]
        else:
            print(f"Warning: No valid energy found for index {idx}. Using 0.0 as default.")
            energy = torch.tensor(0.0)
        mean_energy += energy.item()
    mean_energy /= len(train_indices)
    return mean_energy


class WrappedDES370K(InMemoryDataset):
    def __init__(self, root: str, train: bool = True, transform=None, pre_transform=None, pre_filter=None, radius: float = 10.0, num_neighbors: int = 32):
        """
        Initializes the custom dataset.

        Args:
            root (str): Root directory where the dataset should be saved.
            train (bool, optional): If True, loads the training dataset, otherwise the test dataset. Defaults to True.
            transform (callable, optional): Transform applied to data object before access. Defaults to None.
            pre_transform (callable, optional): Transform applied before saving data. Defaults to None.
            pre_filter (callable, optional): Filter determining dataset inclusion. Defaults to None.
        """
        super().__init__(root, transform, pre_transform, pre_filter)

        self.train = train
        # Load the original DES370K dataset
        self.des370k_dataset = DES370K(energy_unit="kcal/mol", distance_unit="ang")

        self.compute_edge_indices_norm = T.Compose([
            T.RadiusGraph(r=radius, max_num_neighbors=num_neighbors),
            T.Distance(norm=False)
        ])

        self.splits = self.get_idx_split()
        train_indices = self.splits['train']
        self.mean_energy = normalize_des370k_energy(self.des370k_dataset, train_indices=train_indices)

    @property
    def raw_file_names(self):
        return self.des370k_dataset.raw_file_names

    @property
    def processed_file_names(self):
        return [f'encapsulated_DES370K_{"train" if self.train else "test"}.pt']

    def len(self):
        return len(self.des370k_dataset)

    def get(self, idx):
        if self._data:
            return super().get(idx)

        if not hasattr(self, '_data_list') or self._data_list is None:
            self._data_list = self.len() * [None]
        elif self._data_list[idx] is not None:
            return copy.copy(self._data_list[idx])

        original_data = self.des370k_dataset[idx]
        original_data.pos = torch.tensor(original_data.positions)
        original_data.batch = torch.zeros(original_data.atomic_numbers.shape[0], dtype=torch.long)
        original_data = self.compute_edge_indices_norm(original_data)

        energies = torch.tensor(original_data.energies)
        valid_energies = energies[~torch.isnan(energies)]
        if valid_energies.numel() > 0:
            energy = valid_energies[0]
        else:
            print(f"Warning: No valid energy found for index {idx}. Using 0.0 as default.")
            energy = torch.tensor(self.mean_energy)
        normalized_energy = energy - self.mean_energy

        encapsulated_data = Data(
            x=torch.tensor(original_data.atomic_numbers).unsqueeze(1),
            pos=original_data.pos,
            edge_index=original_data.edge_index,
            edge_attr=original_data.edge_attr,
            edge_weight=original_data.edge_attr.view(-1),
            y=normalized_energy.squeeze(),
            z=torch.tensor(original_data.atomic_numbers),
        )

        return encapsulated_data

    def get_idx_split(self):
        if hasattr(self, 'splits') and self.splits is not None:
            return self.splits

        total_len = self.len()
        all_indices = np.arange(total_len)
        np.random.shuffle(all_indices)

        num_train = int(0.8 * total_len)
        num_val = int(0.1 * total_len)

        self.splits = {
            'train': all_indices[:num_train].tolist(),
            'val': all_indices[num_train:num_train + num_val].tolist(),
            'test': all_indices[num_train + num_val:].tolist()
        }

        return self.splits
