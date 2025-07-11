"""Message passing / spatially parametrized filters for S2GNNs.

Most of this file contains default components that allow for an additional
batch dimension where the graph structure is the same but features differ."""

import logging

import torch
from torch import nn
import torch_geometric.nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.models.schnet import CFConv
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.graphgym.models.layer import GeneralLayer
from torch_geometric.utils import add_remaining_self_loops, scatter
from torch_sparse import SparseTensor
import torch.nn.functional as F
import torch_scatter as scatter

from graphgps.layer.gat_conv_layer import GATConv
from graphgps.layer.gatedgcn_layer import GatedGCNLayer
from torch_geometric.nn.models.schnet import SchNet, InteractionBlock as SchnetInteractionBlock
from torch_geometric.nn.models.schnet import ShiftedSoftplus

from graphgps.layer.gemnet.base_layers import Dense
from graphgps.layer.gemnet.basis_layers import BesselBasisLayer, SphericalBasisLayer
from graphgps.layer.gemnet.efficient import EfficientInteractionDownProjection
from graphgps.layer.gemnet.embedding_block import AtomEmbedding, EdgeEmbedding
from graphgps.layer.gemnet.interaction_block import InteractionBlock, InteractionBlockTripletsOnly
from graphgps.loader.gemnet.utils import compute_triplets
from graphgps.network.gemnet import GemNet
from graphgps.layer.gemnet.atom_update_block import OutputBlock
# from torch_geometric.nn.models.dimenet import BesselBasisLayer, SphericalBasisLayer



def directed_norm(edge_index, n_nodes, edge_weight=None, add_self_loops=True,
                  flow='source_to_target'):
    """
    Taken from https://github.com/emalgorithm/directed-graph-neural-network/blob/main/src/datasets/data_utils.py

    Applies the normalization for directed graphs:
        \mathbf{D}_{out}^{-1/2} \mathbf{A} \mathbf{D}_{in}^{-1/2}.
    """
    assert flow in ['source_to_target', 'target_to_source']

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=torch.float32,
                                 device=edge_index.device)

    if add_self_loops:
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, 1, n_nodes)

    if flow == 'source_to_target':
        col, row = edge_index
    else:
        row, col = edge_index

    in_deg = scatter(edge_weight, col, dim=0, dim_size=n_nodes, reduce='sum')
    in_deg_inv_sqrt = in_deg.pow_(-0.5)
    in_deg_inv_sqrt.masked_fill_(in_deg_inv_sqrt == float("inf"), 0.0)

    out_deg = scatter(edge_weight, row, dim=0, dim_size=n_nodes, reduce='sum')
    out_deg_inv_sqrt = out_deg.pow_(-0.5)
    out_deg_inv_sqrt.masked_fill_(out_deg_inv_sqrt == float("inf"), 0.0)

    edge_weight = out_deg_inv_sqrt[row] * edge_weight * in_deg_inv_sqrt[col]
    return edge_index, edge_weight


class FeatureBatchSpatialGraphConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True,
                 normalize: bool = True, add_self_loops: bool = True,
                 batchnorm: bool = False, adj_dtype=torch.float32):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.add_self_loops = add_self_loops
        self.adj_dtype = adj_dtype

        self.lin = nn.Linear(in_channels, out_channels, bias=False)

        if bias and not batchnorm:
            self.bias = nn.Parameter(
                torch.zeros(out_channels), requires_grad=True)
        else:
            self.register_parameter('bias', None)

        self.batchnorm = None
        if batchnorm:
            self.batchnorm = nn.BatchNorm1d(out_channels)

    def adj(self, batch):
        if 'adj_norm' in batch:
            return batch
        edge_index = batch.edge_index
        edge_index = torch.concatenate(
            (edge_index, torch.flip(edge_index, [0,])), dim=-1)
        edge_attr = torch.ones_like(edge_index[0], dtype=self.adj_dtype)
        adj = SparseTensor.from_edge_index(
            edge_index, edge_attr,
            (int(batch.num_nodes), int(batch.num_nodes)))
        adj = adj.coalesce()

        # Move to cpu to save memory
        batch.edge_index = edge_index.to('cpu', non_blocking=True)

        if self.normalize:
            adj = gcn_norm(adj, add_self_loops=self.add_self_loops)
        batch.adj_norm = adj

        return batch

    def forward(self, batch):
        batch = self.adj(batch)

        x = batch.x
        shape = x.shape

        x = self.lin(x).view(shape[0], -1)  # Reshape required for TPUGraphs
        y = batch.adj_norm @ x

        if self.batchnorm is not None:
            y = y.view(-1, x.shape[-1])  # Reshape required for TPUGraphs
            y = self.batchnorm(y)

        y = y.view(*shape[:-1], -1)  # Reshape required for TPUGraphs

        if self.bias is not None:
            y = y + self.bias

        batch.x = y
        return batch


class FeatureBatchSpatialDirectionalGraphConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True,
                 normalize: str = 'dir', add_self_loops: bool = True,
                 use_edge_attr: bool = False, adj_dtype=torch.float32,
                 dir_aggr='cat'):
        super().__init__()
        if dir_aggr == 'cat':
            assert out_channels % 2 == 0, f'Dim {out_channels} must be even'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.add_self_loops = add_self_loops
        self.use_edge_attr = use_edge_attr
        self.adj_dtype = adj_dtype
        self.dir_aggr = dir_aggr

        if dir_aggr.startswith('mean'):
            lin_out_channels, lin_bias = out_channels, bias
        elif dir_aggr == 'cat':
            lin_out_channels, lin_bias = out_channels // 2, False

        self.lin_forward = nn.Linear(
            in_channels, lin_out_channels, bias=lin_bias)
        self.lin_backward = nn.Linear(
            in_channels, lin_out_channels, bias=lin_bias)

        if dir_aggr == 'cat' and bias:
            self.bias = nn.Parameter(
                torch.zeros(out_channels), requires_grad=True)
        else:
            self.register_parameter('bias', None)

    def adj(self, batch):
        if 'adj_forward' in batch and 'adj_backward' in batch:
            return batch
        if self.normalize == 'gcn':
            forw = gcn_norm(batch.edge_index,
                            add_self_loops=self.add_self_loops,
                            flow="source_to_target")
            (forward_col, forward_row), forward_weight = forw
            backw = gcn_norm(batch.edge_index,
                             add_self_loops=self.add_self_loops,
                             flow="target_to_source")
            (backward_row, backward_col), backward_weight = backw
        elif self.normalize == 'dir':
            forw = directed_norm(batch.edge_index, batch.num_nodes,
                                 flow="source_to_target",
                                 add_self_loops=self.add_self_loops)
            (forward_col, forward_row), forward_weight = forw
            backward_col = forward_row
            backward_row = forward_col
            backward_weight = forward_weight
        else:
            if self.add_self_loops:
                edge_index = add_remaining_self_loops(
                    batch.edge_index, None, 1, batch.num_nodes)[0]
            forward_col, forward_row = edge_index
            backward_row, backward_col = forward_col, forward_row
            forward_weight = backward_weight = torch.ones_like(
                forward_row, dtype=self.adj_dtype)
        batch.adj_forward = (forward_row, forward_col), forward_weight
        batch.adj_backward = (backward_row, backward_col), backward_weight

        return batch

    def propagate(self, x, adj, projection, edge_attr=None):
        (source, target), edge_weight = adj
        n_nodes = x.shape[0]
        x = projection(x)[source]
        if edge_attr is not None:
            x = x * edge_attr[:, None]
        edge_weight = edge_weight.reshape(
            [edge_weight.shape[0]] + [1] * (x.ndim - 1))
        y = scatter(x * edge_weight, target, dim_size=n_nodes)
        return y

    def forward(self, batch):
        batch = self.adj(batch)

        x = batch.x

        edge_attr_forward = (
            batch.edge_attr_forward if self.use_edge_attr else None)
        y_forward = self.propagate(x, batch.adj_forward, self.lin_forward,
                                   edge_attr_forward)
        edge_attr_backward = (
            batch.edge_attr_backward if self.use_edge_attr else None)
        y_backward = self.propagate(x, batch.adj_backward, self.lin_backward,
                                    edge_attr_backward)

        if self.dir_aggr == 'cat':
            y = torch.concatenate((y_forward, y_backward), dim=-1)
        elif self.dir_aggr == 'mean':
            y = (y_forward + y_backward) / 2

        if self.bias is not None:
            y = y + self.bias

        batch.x = y
        return batch




class GaussianSmearing(torch.nn.Module):
    def __init__(self, start: float, stop: float, num_gaussians: int):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        # dist: [num_edges]
        dist_expanded = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * dist_expanded.pow(2))


class SharedGemNetProjections(nn.Module):
    def __init__(self, num_radial, num_spherical, emb_size_rbf, emb_size_cbf):
        super().__init__()
        self.mlp_rbf3 = Dense(num_radial, emb_size_rbf, activation=None, bias=False)
        self.mlp_cbf3 = EfficientInteractionDownProjection(
            num_spherical, num_radial, emb_size_cbf)
        self.mlp_rbf_h = Dense(num_radial, emb_size_rbf, activation=None, bias=False)
        self.mlp_rbf_out = Dense(num_radial, emb_size_rbf, activation=None, bias=False)


def compute_distances(batch):
        pos = batch.pos
        edge_index = batch.edge_index

        src, dst = edge_index

        vec = pos[dst] - pos[src]
        dist = vec.norm(dim=1)  # (nEdges,)
        v = vec / (dist[:, None] + 1e-8)  # Normalize to get direction
        return dist, v

class GemNetInteractionBlockGNNLayer(nn.Module):
    def __init__(self, layer_config, *, shared_projections: SharedGemNetProjections = None, **args):
        super().__init__()

        # ——— dimensions ————————————————————————————————
        self.emb_size_atom  = cfg.gnn.emb_size_atom
        self.emb_size_edge  = cfg.gnn.emb_size_edge
        self.emb_size_quad  = 0                   # not used
        self.emb_size_rbf   = cfg.gnn.emb_size_rbf
        self.emb_size_cbf   = cfg.gnn.emb_size_cbf
        self.emb_size_bil_trip = cfg.gnn.emb_size_bil_trip
        self.emb_size_trip  = cfg.gnn.emb_size_trip  # down-projection size
        self.num_radial = cfg.gnn.num_radial
        self.num_spherical = cfg.gnn.num_spherical
        self.num_before_skip = cfg.gnn.num_before_skip
        self.num_after_skip = cfg.gnn.num_after_skip
        self.num_concat = cfg.gnn.num_concat
        self.num_atom = cfg.gnn.num_atom
        self.act = cfg.gnn.act
        self.cutoff = cfg.gnn.cutoff
        # ——— basis layers ————————————————————————————————


        self.rbf_basis  = BesselBasisLayer(self.num_radial, cutoff=self.cutoff)
        self.cbf_basis3 = SphericalBasisLayer(self.num_spherical, self.num_radial, 
                                              cutoff=self.cutoff, efficient=True)
        # shared MLPs exactly like GemNet
        if shared_projections is not None:  # use pre-defined shared projections
            self.mlp_rbf3  = shared_projections.mlp_rbf3
            self.mlp_cbf3  = shared_projections.mlp_cbf3
            self.mlp_rbf_h = shared_projections.mlp_rbf_h
            self.mlp_rbf_out = shared_projections.mlp_rbf_out
        else:  # create new shared projections
            self.mlp_rbf3  = Dense(self.num_radial, self.emb_size_rbf, activation=None, bias=False)
            self.mlp_cbf3  = EfficientInteractionDownProjection(
                                self.num_spherical, self.num_radial, self.emb_size_cbf)
            self.mlp_rbf_h = Dense(self.num_radial, self.emb_size_rbf, activation=None, bias=False)
            self.mlp_rbf_out = Dense(self.num_radial, self.emb_size_rbf, activation=None, bias=False)

        # atom / edge embedding blocks (copied from GemNet)
        # self.atom_emb = AtomEmbedding(self.emb_size_atom)
        # self.edge_emb = EdgeEmbedding(self.emb_size_atom, self.num_radial,
        #                               self.emb_size_edge)

        # InteractionBlock itself
        self.block = InteractionBlockTripletsOnly(
            emb_size_atom     = self.emb_size_atom,
            emb_size_edge     = self.emb_size_edge,
            emb_size_trip     = self.emb_size_trip,
            emb_size_quad     = 0,                 # not used
            emb_size_rbf      = self.emb_size_rbf,
            emb_size_cbf      = self.emb_size_cbf,
            emb_size_bil_trip = self.emb_size_trip,
            num_before_skip   = self.num_before_skip,
            num_after_skip    = self.num_after_skip,
            num_concat        = self.num_concat,
            num_atom          = self.num_atom,
            activation        = self.act,
        )

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
        pos        = batch.pos                  # (nAtoms,3)
        edge_index = batch.edge_index           # (2,nEdges)
        Z          = batch.x                    # we assume x already is atomic numbers
                                               # (OR supply batch.z, whatever you loaded)

        src, dst = edge_index                  # src=c , dst=a
        # nEdges   = src.numel()
        nAtoms   = Z.size(0)

        # 1) pair-wise geometry
        dist, v = compute_distances(batch)  # (nEdges,)
        rbf  = self.rbf_basis(dist)            # (nEdges,num_radial)
        batch.rbf = rbf # TODO: Hack to avoid recomputing this in the final output block
        batch.v = v  # same here

        # 2) triplet indices
        id3_expand_ba, id3_reduce_ca, id_swap, Kidx3 = compute_triplets(
            edge_index, num_nodes=nAtoms)

        # 3) angles + CBF basis  (GemNet takes cos(angle) inside SphericalBasisLayer)
        R_ca = pos[src[id3_reduce_ca]] - pos[dst[id3_reduce_ca]]
        R_ba = pos[src[id3_expand_ba]] - pos[dst[id3_expand_ba]]
        angles = GemNet.calculate_neighbor_angles(R_ca, R_ba)   # (nTriplets,)

        cbf_basis = self.cbf_basis3(dist, angles, id3_reduce_ca, Kidx3) # tuple (rbf_env, sph2)

        # 4) shared down-projections
        rbf3  = self.mlp_rbf3(rbf)                  # (nEdges, emb_size_rbf)
        cbf3  = self.mlp_cbf3(cbf_basis)            # (nEdges, emb_size_cbf)
        rbf_h = self.mlp_rbf_h(rbf)                 # (nEdges, emb_size_rbf)
        rbf_out = self.mlp_rbf_out(rbf)               # (nEdges, emb_size_rbf)

        # 5) initial atom + edge embeddings
        # h = self.atom_emb(Z)                        # (nAtoms, emb_size_atom)
        h=Z
        m = batch.edge_attr  # (nEdges, emb_size_edge) ; use edge_attr directly if it is already provided

        # TODO: assuming direct_forces=False so ignore the forces here
        E, F = self.out_block(h, m, rbf_out, dst)
        if hasattr(batch, 'E') and hasattr(batch, 'F'):
            batch.E += E
            batch.F += F
        else:
            batch.E = E
            batch.F = F

        h, m = self.block(
            h=h, m=m,
            rbf3=rbf3, cbf3=cbf3, Kidx3=Kidx3,
            id_swap=id_swap,
            id3_expand_ba=id3_expand_ba,
            id3_reduce_ca=id3_reduce_ca,
            rbf_h=rbf_h,
            id_c=src, id_a=dst
        )
        # 6) update batch
        batch.x = h
        batch.edge_attr = m
        return batch

class GCNConvGNNLayer(nn.Module):

    def __init__(self, layer_config: LayerConfig, add_self_loops: bool = True,
                 normalize: str = 'dir', make_undirected: bool = True,
                 is_first: bool = True, use_edge_attr: bool = False,
                 with_node_residual: bool = True, overwrite_x: bool = True,
                 **kwargs):
        if with_node_residual:
            assert layer_config.dim_in == layer_config.dim_out, 'Due to residual'

        super().__init__()
        self.with_node_residual = with_node_residual
        self.use_edge_attr = use_edge_attr
        self.overwrite_x = overwrite_x

        n_gaussians = 50
        cutoff = 5.0
        n_filters = 64
        self.distance_expansion = GaussianSmearing(0.0, cutoff, n_gaussians)

        dim_in, dim_out = layer_config.dim_in, layer_config.dim_out
        has_bias = layer_config.has_bias
        if use_edge_attr:
            self.edge_nn = nn.Sequential(
                nn.Linear(n_gaussians, n_filters),
                nn.ReLU(),
                nn.Linear(n_filters, n_filters)
            )

        gcn_normalize = True if normalize == 'dir' else False
        # self.conv = GCNConv(dim_in, dim_out, bias=has_bias, normalize=gcn_normalize)
        self.conv = CFConv(in_channels=dim_in, out_channels=dim_out, num_filters=n_filters, nn=self.edge_nn, cutoff=cutoff)

        self.dropout = nn.Dropout(layer_config.dropout)
        self.activation = register.act_dict[layer_config.act]()

    def forward(self, batch):
        if self.use_edge_attr:
            smeared_edge_attr = self.distance_expansion(batch.edge_attr)
            y = self.conv(batch.x, batch.edge_index, edge_weight=batch.edge_weight, edge_attr=smeared_edge_attr)
        else:
            y = self.conv(batch.x, batch.edge_index)
        y = self.dropout(self.activation(y))

        if self.with_node_residual:
            y = batch.x + y

        if self.overwrite_x:
            batch.x = y
            return batch
        else:
            return y

class GatedGCNConvGNNLayer(nn.Module):

    def __init__(self, layer_config: LayerConfig,
                 with_node_residual: bool = True, overwrite_x: bool = True,
                 **kwargs):
        if with_node_residual and layer_config.dim_in != layer_config.dim_out:
            logging.warning(
                f'Residual not possible with dim_in={layer_config.dim_in} not '
                f'equal to dim_out={layer_config.dim_out}')
            with_node_residual = False
        # TODO: Currently only usable with overwrite_x.
        # TODO: Make MPGNN implementations more consistent
        # TODO: Currently always with ReLU activation
        super().__init__()
        self.with_node_residual = with_node_residual
        dim = layer_config.dim_in
        self.conv = GatedGCNLayer(dim, dim, dropout=layer_config.dropout,
                                  residual=self.with_node_residual)

    def forward(self, batch):
        return self.conv(batch)


class GATConvGNNLayer(nn.Module):

    def __init__(self, layer_config: LayerConfig, add_self_loops: bool = True,
                 is_first: bool = True, is_last: bool = True,
                 overwrite_x: bool = True,
                 with_node_residual: bool = False, **kwargs):
        super().__init__()
        self.is_first = is_first
        self.is_last = is_last
        self.overwrite_x = overwrite_x

        cfg_gat = cfg.gnn.gatconv
        self.with_linear = cfg_gat.with_linear
        heads = cfg_gat.num_heads
        dim_in = layer_config.dim_in
        dim_out = layer_config.dim_out

        if is_first:  # housekeeping for first layer
            self.pre_dropout = nn.Dropout(cfg_gat.pre_dropout)
            if (cfg.dataset.node_encoder
                    and cfg.dataset.node_encoder_name == 'OGBNArxivNode'
                    and cfg.gnn.layers_pre_mp == 0):
                dim_in = cfg.share.dim_in

        if is_last and cfg.gnn.head == 'transductive_node_dummy':
            dim_out = gat_dim_out = cfg.share.dim_out
            self.bias_last = nn.Parameter(torch.Tensor(1, gat_dim_out))
        else:
            assert dim_out % heads == 0
            dim_out = dim_out // heads

        if with_node_residual and dim_out * heads != dim_in:
            logging.warning(
                f'Residual not possible with dim_in={dim_in} not '
                f'equal to dim_out={dim_out * heads}')
            with_node_residual = False
        self.with_node_residual = with_node_residual

        # Choose from different implementation to mimix existing baselines etc.
        if cfg_gat.backend == 'PyG':
            self.conv = GATConv(dim_in,
                                dim_out,
                                num_heads=heads,
                                negative_slope=cfg_gat.negative_slope,
                                attn_dropout=cfg_gat.attn_dropout,
                                norm=cfg_gat.norm)
        elif cfg_gat.backend == 'PyG_plain':
            self.conv = torch_geometric.nn.GATConv(
                dim_in, dim_out, heads=heads, concat=True,
                add_self_loops=False, bias=False)
            self.convert_to_sparse_tensor = True
        else:
            raise ValueError(
                f"GATConv backend '{cfg_gat.backend}' not supported!")

        if self.with_linear:
            self.linear = nn.Linear(dim_in, heads * dim_out, bias=False)

        if not (is_last and cfg.gnn.head == 'transductive_node_dummy'):
            self.bn = nn.BatchNorm1d(heads * dim_out)
            self.activation = register.act_dict[layer_config.act]()
            self.dropout = nn.Dropout(cfg_gat.feat_dropout)

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'bias_last'):
            nn.init.zeros_(self.bias_last)

    def forward(self, batch):

        y = self.pre_dropout(batch.x) if self.is_first else batch.x

        conv = self.conv(y, batch.edge_index)

        if self.with_linear:
            linear = self.linear(batch.x).view(conv.shape)
            y = conv + linear
        else:
            y = conv

        if self.is_last and cfg.gnn.head == 'transductive_node_dummy':
            y = y.mean(1)
            y = y + self.bias_last
        else:
            y = y.flatten(1)
            y = self.bn(y)
            y = self.dropout(self.activation(y))

        if self.with_node_residual:
            y = y + batch.x
        if self.overwrite_x:
            batch.x = y
            return batch
        else:
            return y


class FeatureBatchGNNLayer(nn.Module):

    def __init__(self, layer_config: LayerConfig, add_self_loops: bool = True,
                 normalize: str = 'dir', make_undirected: bool = True,
                 is_first: bool = True, use_edge_attr: bool = False,
                 with_node_residual: bool = True, overwrite_x: bool = True,
                 dir_aggr: str = 'cat', **kwargs):
        if with_node_residual and layer_config.dim_in != layer_config.dim_out:
            logging.warning(
                f'Residual not possible with dim_in={layer_config.dim_in} not '
                f'equal to dim_out={layer_config.dim_out}')
            with_node_residual = False

        super().__init__()
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.make_undirected = make_undirected
        self.use_edge_attr = use_edge_attr
        self.with_node_residual = with_node_residual
        self.overwrite_x = overwrite_x

        dim_in = layer_config.dim_in
        dim_out = layer_config.dim_out
        has_bias = layer_config.has_bias
        if self.make_undirected:
            assert not self.use_edge_attr
            self.conv = FeatureBatchSpatialGraphConv(
                dim_in, dim_out, bias=has_bias, normalize=self.normalize,
                batchnorm=layer_config.has_batchnorm)
        else:
            self.conv = FeatureBatchSpatialDirectionalGraphConv(
                dim_in, dim_out, bias=has_bias, normalize=self.normalize,
                use_edge_attr=self.use_edge_attr, dir_aggr=dir_aggr)

        self.dropout = nn.Dropout(layer_config.dropout)
        self.activation = register.act_dict[layer_config.act]()

    def forward(self, batch):
        x_residual = batch.x
        batch = self.conv(batch)
        y = self.dropout(self.activation(batch.x))
        if self.with_node_residual:
            y = y + x_residual
        if self.overwrite_x:
            batch.x = y
            return batch
        else:
            return y


class InteractionBlockGNNLayer(nn.Module):
    def __init__(self, layer_config: LayerConfig,
                 with_node_residual: bool = True, overwrite_x: bool = True,
                 **kwargs):
        if with_node_residual and layer_config.dim_in != layer_config.dim_out:
            logging.warning(
                f'Residual not possible with dim_in={layer_config.dim_in} not '
                f'equal to dim_out={layer_config.dim_out}')
            with_node_residual = False

        super().__init__()
        self.with_node_residual = with_node_residual
        self.overwrite_x = overwrite_x

        # TOOO: make configurable
        n_gaussians = 50
        cutoff = 10.0
        n_filters = 64
        self.distance_expansion = GaussianSmearing(0.0, cutoff, n_gaussians)

        dim_in, dim_out = layer_config.dim_in, layer_config.dim_out
        self.interaction = SchnetInteractionBlock(
            dim_out,
            n_gaussians,
            n_filters,
            cutoff
        )
        self.lin1 = nn.Linear(dim_out, dim_out)
        self.dropout = nn.Dropout(layer_config.dropout)
        self.activation = register.act_dict[layer_config.act]()

    def forward(self, batch):
        y = batch.x

        distances, _ = compute_distances(batch)  # (nEdges,)
        # Assumes batch has edge_index and edge_weight computed using RadiusGraph
        edge_attr = self.distance_expansion(distances)
        y = self.interaction(y, batch.edge_index, distances, edge_attr)

        y = self.lin1(y)
        y = self.dropout(self.activation(y))

        if self.with_node_residual:
            y = batch.x + y

        if self.overwrite_x:
            batch.x = y
            return batch
        else:
            return y

def GNNLayer(layer_config, *args, **kwargs):
    """
    Wrapper for a GNN layer.
    """
    return GeneralLayer(cfg.gnn.layer_type, layer_config=layer_config)


class Dropout1d(nn.Dropout1d):

    def forward(self, batch):
        if isinstance(batch, torch.Tensor):
            batch = super().forward(batch)
        else:
            batch.x = super().forward(batch.x)
        return batch


class Residual(nn.Module):

    def __init__(self, field_name='x', buffer_name='residual_sum') -> None:
        super().__init__()
        self.field_name = field_name
        self.buffer_name = buffer_name

    def forward(self, batch):
        if self.buffer_name not in batch:
            batch[self.buffer_name] = []
        batch[self.buffer_name].append(batch[self.field_name])
        return batch
