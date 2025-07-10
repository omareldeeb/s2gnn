import torch.nn as nn
from torch_geometric.nn import global_add_pool   # or torch_scatter.scatter_add
from torch_geometric.graphgym import register_head
from torch_geometric.graphgym import cfg

@register_head('per_atom_head')
class PerAtomHead(nn.Module):
    def __init__(self, dim_in, dim_out, is_first):
        super().__init__()
        hidden_dim = cfg.gnn.dim_inner
        self.atom_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=True)
        )

    def forward(self, batch):
        e_atom = self.atom_head(batch.x).squeeze(-1)

        e_mol = global_add_pool(e_atom, batch.batch)

        return e_mol, batch.y
