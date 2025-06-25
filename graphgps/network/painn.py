# paiNN_torch.py
# --------------------------------------------------
from __future__ import annotations
from typing import Callable, List, Sequence, Tuple
import copy
import math
import torch
import torch_geometric.transforms as T
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.data import DataLoader
from torch_geometric.transforms import Compose, RadiusGraph, Distance
from graphgps.encoder.distance_embedding import DistanceEmbedding
from graphgps.encoder.embedding_node_encoder import EmbeddingNodeEncoder
from graphgps.encoder.scalar_embedding import ScalarEmbedding
from graphgps.loader.dataset.wrapped_md17 import WrappedMD17
from graphgps.loader.dataset.wrapped_qm9 import WrappedQM9


Tensor = torch.Tensor


# -----------------------------------------------------------------------------#
#  Helpers                                                                     #
# -----------------------------------------------------------------------------#
def cosine_cutoff(r: Tensor, cutoff: float) -> Tensor:
    """
    Behler-style cosine cutoff.

    Parameters
    ----------
    r : (N_atoms, N_atoms) pairwise distances
    cutoff : float
    """
    f = 0.5 * (torch.cos(math.pi * r / cutoff) + 1.0)
    return f * (r < cutoff).float()          # JAX: jnp.where(...)


class MLP(nn.Module):
    """Simple feed-forward network with optional zero-initialised head."""
    def __init__(
        self,
        sizes: Sequence[int],
        activation: Callable[[Tensor], Tensor] = F.silu,
        init_last_layer_to_zero: bool = False,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        for in_f, out_f in zip(sizes[:-1], sizes[1:]):
            lin = nn.Linear(in_f, out_f)
            if init_last_layer_to_zero and out_f == sizes[-1]:
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)
            layers += [lin]
        self.layers = nn.ModuleList(layers)
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        *hidden, last = self.layers
        for layer in hidden:
            x = self.activation(layer(x))
        return last(x)


# -----------------------------------------------------------------------------#
#  Core blocks                                                                 #
# -----------------------------------------------------------------------------#
class ScalarFilter(nn.Module):
    """
    φ_ij  =  W(dist_ij)   with   W(dist) = Dense( sin(nπ r / r_cut) )

    Output shape: (N_atoms, N_atoms, 3F)
    """

    def __init__(
        self,
        cutoff_dist: float,
        atom_features: int,
        radial_basis_fn: int,
    ):
        super().__init__()
        self.cutoff_dist = cutoff_dist
        self.radial_basis_fn = radial_basis_fn
        self.prefactors = nn.Parameter(          # register as buffer (not trainable)
            torch.arange(1, radial_basis_fn + 1).float() * math.pi / cutoff_dist,
            requires_grad=False,
        )
        self.lin = nn.Linear(radial_basis_fn, 3 * atom_features)

    def forward(self, dists: Tensor) -> Tensor:
        # JAX: jnp.sin(dists[...,None] * self.prefactors)
        x = torch.sin(dists.unsqueeze(-1) * self.prefactors)
        return self.lin(x)                       # (N, N, 3F)


class MessageBlock(nn.Module):
    """
    Computes messages  m_{ij}  → (Δs_i, Δv_i)

    JAX reference: see MessageBlock in original code.
    """

    def __init__(
        self,
        atom_features: int,
        cutoff_dist: float,
        radial_basis_fn: int,
    ):
        super().__init__()
        self.atom_features = atom_features
        self.cutoff_dist = cutoff_dist
        self.scalar_filter = ScalarFilter(
            cutoff_dist, atom_features, radial_basis_fn
        )
        self.phi = nn.Sequential(
            nn.Linear(atom_features, atom_features),
            nn.SiLU(),
            nn.Linear(atom_features, 3 * atom_features),
        )

    def forward(
        self,
        s: Tensor,                  # (N_atoms, F)
        v: Tensor,                  # (N_atoms, F, 3)
        dr: Tensor,                 # (N_atoms, N_atoms, 3)
        atom_mask: Tensor,          # (N_atoms,)  bool / {0,1}
    ) -> Tuple[Tensor, Tensor]:
        d = dr.norm(dim=-1)                              # distances

        phi_j = self.phi(s)                              # (N, 3F)
        W_ij = self.scalar_filter(d)                     # (N, N, 3F)
        f_cut = cosine_cutoff(d, self.cutoff_dist)[..., None]

        msg = torch.einsum(                              # 'jf, ijf, j -> ijf'
            'jf,ijf,j->ijf', phi_j, W_ij * f_cut, atom_mask.float()
        )

        # Split message channels
        F = self.atom_features
        msg_ss, msg_vv, msg_vs = torch.split(msg, F, dim=-1)

        ds = msg_ss.sum(dim=1)                           # (N, F)

        e_ij = dr / (d + 1e-9).unsqueeze(-1)             # unit vectors
        dv = (
            torch.einsum('jfv,ijf->ifv', v, msg_vv) +
            torch.einsum('ijv,ijf->ifv', e_ij, msg_vs)
        )                                                # (N, F, 3)
        return ds, dv


class UpdateBlock(nn.Module):
    """
    Atom-wise feature update (Δs, Δv).
    """

    def __init__(self, atom_features: int):
        super().__init__()
        F = atom_features
        # weight: (F_out, F_in) with F_out = F_in = F
        self.W_vv = nn.Parameter(torch.empty(F, F))
        self.W_uv = nn.Parameter(torch.empty(F, F))
        nn.init.xavier_uniform_(self.W_vv)
        nn.init.xavier_uniform_(self.W_uv)

        self.mlp = nn.Sequential(
            nn.Linear(2 * F, F),
            nn.SiLU(),
            nn.Linear(F, 3 * F),
        )

    def forward(self, s: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        F = v.shape[1]
        Vv = torch.einsum('ifv,gf->igv', v, self.W_vv)   # (N, F, 3)
        Uv = torch.einsum('ifv,gf->igv', v, self.W_uv)   # (N, F, 3)

        norm_Vv = Vv.norm(dim=-1)                        # (N, F)
        scalar_prod = (Vv * Uv).sum(dim=-1)              # (N, F)

        x = torch.cat([s, norm_Vv], dim=-1)              # (N, 2F)
        atm_rep = self.mlp(x)                            # (N, 3F)

        ds = atm_rep[:, :F] + atm_rep[:, F:2*F] * scalar_prod
        dv = atm_rep[:, 2*F:].unsqueeze(-1) * Uv         # broadcast on vector dim
        return ds, dv


# -----------------------------------------------------------------------------#
#  PaiNN network                                                               #
# -----------------------------------------------------------------------------#
class PaiNN(nn.Module):
    """
    Polarizable atom-interaction neural network (PyTorch version).

    Parameters
    ----------
    node_feature_dim : int
        F – number of scalar/vector channels per atom.
    out_node_feature_dim : int
        F_out – returned per-atom scalar channels (vector part is returned unchanged).
    cutoff : float
        Distance cutoff in Å.
    layers : int
        Number of message/update blocks (L).
    n_radial_basis : int, default=20
        Radial basis functions.
    readout_activation : callable, default=torch.nn.functional.silu
        Activation for the graph-level head.
    """

    def __init__(
        self,
        node_feature_dim: int,
        out_node_feature_dim: int,
        cutoff: float,
        layers: int,
        n_radial_basis: int = 20,
        readout_activation: Callable[[Tensor], Tensor] = F.silu,
    ):
        super().__init__()
        F = node_feature_dim

        self.layers = nn.ModuleList(
            [
                nn.ModuleDict({
                    'msg':MessageBlock(F, cutoff, n_radial_basis),
                    'upd':UpdateBlock(F),
                    # separate LayerNorms to match JAX behaviour
                    'ln_s':nn.LayerNorm(F),
                    'ln_v':nn.LayerNorm(F),        # applied to ‖v‖, see forward()
                })
                for _ in range(layers)
            ]
        )

        # Read-out MLP - same as original
        self.readout = MLP(
            [F, F, 1],
            activation=readout_activation,
            init_last_layer_to_zero=True,
        )

        # Final linear on scalars (optional)
        self.out_lin = nn.Linear(F, out_node_feature_dim, bias=True)
        self.scalar_embedding = ScalarEmbedding(dim_emb=F)
        self.vector_embedding = DistanceEmbedding(cutoff=cutoff, radial_basis_fn=n_radial_basis, out_dim=F)

    # ------------------------------------------------------------------ #
    #   Forward                                                          #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        s: Tensor,                # (N_atoms, F)      – scalar channels
        v: Tensor,                # (N_atoms, F, 3)   – vector channels
        pos: Tensor,              # (N_atoms, 3)      – Å
        atom_mask: Tensor,        # (N_atoms,) {0,1}
        distances: Tensor,        # (N_edges, 1)
        directions: Tensor,       # (N_edges, 3)
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        dr = pos.unsqueeze(0) - pos.unsqueeze(1)       # (N, N, 3)

        s = self.scalar_embedding(s)
        # v = self.vector_embedding(v)

        # initial normalisation (LayerNorm on atom dimension)et        s = nn.LayerNorm(s.shape[-1])(s)
        v_norm = v.norm(dim=-1)
        v = v * (nn.LayerNorm(v_norm.shape[-1])(v_norm) / (v_norm + 1e-9)).unsqueeze(-1)

        for layer in self.layers:
            ds_msg, dv_msg = layer['msg'](s, v, dr, atom_mask)
            s = layer['ln_s'](s + ds_msg)

            v = v + dv_msg
            v_norm = v.norm(dim=-1)
            v = v * (layer['ln_v'](v_norm) / (v_norm + 1e-9)).unsqueeze(-1)

            ds_up, dv_up = layer['upd'](s, v)
            s = layer['ln_s'](s + ds_up)

            v = v + dv_up
            v_norm = v.norm(dim=-1)
            v = v * (layer['ln_v'](v_norm) / (v_norm + 1e-9)).unsqueeze(-1)

        # Graph-level readout (sum over atoms)

        s = s.reshape(32, -1, 384)
        graph_readout = self.readout(s)
        total_energy = torch.einsum('ijk,ijk->i', graph_readout, s)

        # Optional per-atom projection
        s_out = self.out_lin(s)
        return total_energy, (s_out, v)

def preprocess_batch(batch):
    mask = torch.ones(batch.num_nodes, dtype=torch.bool)
    s = batch.z.view(-1, 1).float()
    v = torch.zeros(s.size(0), s.size(1), 3)  # v_i = 0
    pos = batch.pos


    src, dest = batch.edge_index
    r_ij = pos[src] - pos[dest]
    distances = torch.norm(r_ij, dim=1, keepdim=True)
    directions = r_ij / (distances + 1e-9)
    return s, v, pos, mask, distances, directions

def train(model, loader, optimizer, device, mean_energy):
    model.train()
    criterion = nn.MSELoss()
    
    sum_loss = 0.0
    batch_count = 0
    for batch in tqdm(loader, desc='Train', unit='batch'):
        batch = batch.to(device)
        optimizer.zero_grad()
        targets = batch.energy.view(-1).to(device) - mean_energy
        s, v, pos, mask, distances, directions = preprocess_batch(batch)
        preds_norm = model(s, v, pos, mask, distances, directions)
        pred = preds_norm[0]
        loss = criterion(pred, targets)
        loss.backward()
        optimizer.step()
        batch_count += 1
        sum_loss += loss.item()
        avg_loss = sum_loss / batch_count
        tqdm.write(f"Batch {batch_count}: Loss: {loss.item():.4f} | Avg Loss: {avg_loss:.4f}")
    return sum_loss / batch_count

def test(model, loader, device, mean_energy):
    model.eval()
    
    total_mae = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc='Test', unit='batch'):
            batch = batch.to(device)
            preds_norm = model(batch)
            preds = preds_norm + mean_energy
            total_mae += (preds - batch.energy.view(-1).to(device)).abs().sum().item()
    return total_mae / len(loader.dataset)

