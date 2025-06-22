from torch_geometric.graphgym.register import register_node_encoder
import torch.nn as nn


@register_node_encoder('Embedding')
class EmbeddingNodeEncoder(nn.Module):
    """Node encoder that uses an embedding layer to encode node features.

    Args:
        dim_in (int): The input dimension.
        dim_emb (int): The output embedding dimension.
    """

    def __init__(self, dim_emb: int, dim_in: int = 9):
        super().__init__()
        self.dim_in = dim_emb
        self.dim_emb = dim_emb
        self.embedding = nn.Embedding(9, dim_emb)
        nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, batch):
        batch.x = self.embedding(batch.x.squeeze(-1).long())
        return batch