import numpy as np
from torch_geometric.graphgym.register import register_node_encoder
import torch
import torch.nn as nn


@register_node_encoder('Embedding')
class EmbeddingNodeEncoder(nn.Module):
    """Node encoder that uses an embedding layer to encode node features.

    Args:
        dim_in (int): The input dimension.
        dim_emb (int): The output embedding dimension.
    """

    def __init__(self, dim_emb: int, dim_in: int = 118):
        super().__init__()
        self.dim_in = dim_emb
        self.dim_emb = dim_emb
        # We go all the way to Pu (94). Use 93 dimensions because of 0-based indexing
        self.embedding = nn.Embedding(93, dim_emb)
        nn.init.uniform_(self.embedding.weight, a=-np.sqrt(3), b=np.sqrt(3))

    def forward(self, batch):
        # -1 because Z.min()=1 (==Hydrogen)
        batch.x = self.embedding.to(batch.x.device)(batch.x.squeeze(-1).long() - 1)
        return batch