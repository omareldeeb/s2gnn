import torch
import torch.nn as nn
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_head
from torch_scatter import scatter

from graphgps.layer.gemnet.atom_update_block import OutputBlock


@register_head('gemnet_graph')
class GemNetGraphHead(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        assert 'mlp_rbf_out' in kwargs, "mlp_rbf_out must be provided in kwargs"

        self.emb_size_atom  = cfg.gnn.emb_size_atom
        self.emb_size_edge  = cfg.gnn.emb_size_edge
        self.emb_size_rbf   = cfg.gnn.emb_size_rbf
        self.num_atom = cfg.gnn.num_atom
        self.act = cfg.gnn.act

        self.mlp_rbf_out = kwargs['mlp_rbf_out']
        self.out_block = OutputBlock(
            emb_size_atom=self.emb_size_atom,
            emb_size_edge=self.emb_size_edge,
            emb_size_rbf=self.emb_size_rbf,
            nHidden=self.num_atom,
            num_targets=1,
            activation=self.act,
            direct_forces=not cfg.derive_forces,
            output_init="HeOrthogonal",
            name=f"OutputBlock_{hex(id(self))}" # We don't need the name but I'm assuming it needs to be globally unique?
        )

    def forward(self, batch):
        h = batch.x
        m = batch.edge_attr
        src, dst = batch.edge_index

        rbf = batch.rbf
        rbf_out = self.mlp_rbf_out(rbf)

        E, F = self.out_block(h, m, rbf_out, dst)
        E += batch.E
        F += batch.F

        nMolecules = torch.max(batch.batch) + 1
        E = scatter(E, batch.batch, dim=0, dim_size=nMolecules, reduce="add")

        if not cfg.derive_forces:
            nAtoms = h.shape[0]
            v = batch.v  # Get the direction vectors from the batch
            # map forces in edge directions
            F_ji = F * v  # (nEdges, 3)
            F_j = scatter(F_ji, dst, dim=0, dim_size=nAtoms, reduce="add")
            # (nAtoms, num_targets, 3)

            return (E, F_j), batch.y
        
        return E, batch.y