import numpy as np
import torch
from torch_geometric.graphgym.register import register_edge_encoder
from torch_geometric.graphgym.config import cfg

from graphgps.layer.gemnet.basis_layers import BesselBasisLayer
from .base_layers import Dense


class AtomEmbedding(torch.nn.Module):
    """
    Initial atom embeddings based on the atom type

    Parameters
    ----------
        emb_size: int
            Atom embeddings size
    """

    def __init__(self, emb_size, name=None):
        super().__init__()
        self.emb_size = emb_size

        # Atom embeddings: We go up to Pu (94). Use 93 dimensions because of 0-based indexing
        self.embeddings = torch.nn.Embedding(93, emb_size)
        # init by uniform distribution
        torch.nn.init.uniform_(self.embeddings.weight, a=-np.sqrt(3), b=np.sqrt(3))

    def forward(self, Z):
        """
        Returns
        -------
            h: Tensor, shape=(nAtoms, emb_size)
                Atom embeddings.
        """
        h = self.embeddings(Z - 1)  # -1 because Z.min()=1 (==Hydrogen)
        return h

class EdgeBatchTransform(torch.nn.Module):
    def __init__(self, bessel_basis_layer: BesselBasisLayer, name=None):
        super().__init__()
        self.rbf_basis = bessel_basis_layer#BesselBasisLayer(cfg.gnn.num_radial, cfg.gnn.cutoff)

        self.name = name
        
    def forward(self, batch):
        atom_embeddings = batch.x
        pos = batch.pos
        edge_index = batch.edge_index
        src, dst = edge_index
        r = pos[src] - pos[dst]  # shape=(nEdges, 3)
        r = torch.norm(r, dim=-1)  # shape=(nEdges,)
        m_rbf = self.rbf_basis(r)  # shape=(nEdges, nFeatures)
        idnb_a = src  # shape=(nEdges,)
        idnb_c = dst  # shape=(nEdges,)

        return atom_embeddings, m_rbf, idnb_a, idnb_c



@register_edge_encoder('GemEdge')
class InitialEdgeEmbedding(torch.nn.Module):
    """
    Edge embedding based on the concatenation of atom embeddings and subsequent dense layer.

    Parameters
    ----------
        atom_features: int
            Embedding size of the atom embeddings.
        edge_features: int
            Embedding size of the edge embeddings.
        out_features: int
            Embedding size after the dense layer.
        activation: str
            Activation function used in the dense layer.
    """
    #TODO: this is unwieldy, either make it work with DI or use custom handler for GemNet initial embeddings
    def __init__(
        self, dim_emb: int, **args
    ):
        super().__init__()
        atom_features = cfg.gnn.emb_size_atom
        edge_features = cfg.gnn.num_radial
        out_features =  cfg.gnn.emb_size_edge
        activation = cfg.gnn.edge_emb_act

        self.batch_transform = EdgeBatchTransform(
            bessel_basis_layer=BesselBasisLayer(
                num_radial=cfg.gnn.num_radial,
                cutoff=cfg.gnn.cutoff,
            ),
        )

        in_features = 2 * atom_features + edge_features
        self.dense = Dense(in_features, out_features, activation=activation, bias=False)

    def forward(self, batch):
        """
        Returns
        -------
            m_ca: Tensor, shape=(nEdges, emb_size)
                Edge embeddings.
        """
        # m_rbf: shape (nEdges, nFeatures)
        # in embedding block: m_rbf = rbf ; In interaction block: m_rbf = m_ca

        h, m_rbf, idnb_a, idnb_c = self.batch_transform(batch)  # h: shape=(nAtoms, emb_size_atom), m_rbf: shape=(nEdges, nFeatures), idnb_a: shape=(nEdges,), idnb_c: shape=(nEdges,)

        h_a = h[idnb_a]  # shape=(nEdges, emb_size)
        h_c = h[idnb_c]  # shape=(nEdges, emb_size)

        m_ca = torch.cat([h_a, h_c, m_rbf], dim=-1)  # (nEdges, 2*emb_size+nFeatures)
        m_ca = self.dense(m_ca)  # (nEdges, emb_size)
        batch.edge_attr = m_ca
        return batch


class EdgeEmbedding(torch.nn.Module):
    """
    Edge embedding based on the concatenation of atom embeddings and subsequent dense layer.

    Parameters
    ----------
        atom_features: int
            Embedding size of the atom embeddings.
        edge_features: int
            Embedding size of the edge embeddings.
        out_features: int
            Embedding size after the dense layer.
        activation: str
            Activation function used in the dense layer.
    """
    def __init__(
        self, emb_size_atom=None, emb_size_edge=None, out_size=None, activation=None, name=None
    ):
        super().__init__()
        atom_features = emb_size_atom if emb_size_atom is not None else cfg.gnn.emb_size_atom
        edge_features = emb_size_edge if emb_size_edge is not None else cfg.gnn.emb_size_edge
        out_features = out_size if out_size is not None else cfg.gnn.dim_inner
        activation = activation if activation is not None else cfg.gnn.edge_emb_act

        in_features = 2 * atom_features + edge_features
        self.dense = Dense(in_features, out_features, activation=activation, bias=False)

    def forward(self, h, m_rbf, idnb_a, idnb_c):
        """
        Returns
        -------
            m_ca: Tensor, shape=(nEdges, emb_size)
                Edge embeddings.
        """
        h_a = h[idnb_a]  # shape=(nEdges, emb_size)
        h_c = h[idnb_c]  # shape=(nEdges, emb_size)

        m_ca = torch.cat([h_a, h_c, m_rbf], dim=-1)  # (nEdges, 2*emb_size+nFeatures)
        m_ca = self.dense(m_ca)  # (nEdges, emb_size)
        return m_ca
