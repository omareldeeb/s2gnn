import torch
import torch.nn as nn

class DistanceEmbedding(nn.Module):
    def __init__(self, cutoff=5.0, radial_basis_fn=20, out_dim=384):
        super().__init__()
        self.cutoff = cutoff
        self.freqs = nn.Parameter(torch.arange(1, radial_basis_fn + 1) * torch.pi / cutoff, requires_grad=False)
        self.linear = nn.Linear(radial_basis_fn, out_dim)

    def forward(self, distances):  # (N_edges, 1)
        f_cut = 0.5 * (torch.cos(torch.clamp(distances / self.cutoff, max=1.0) * torch.pi) + 1.0)  # (N_edges, 1)
        x = torch.sin(distances * self.freqs)  # (N_edges, radial_basis_fn)
        return self.linear(x) * f_cut  # (N_edges, out_dim)
