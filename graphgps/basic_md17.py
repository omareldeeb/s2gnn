import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from torch_geometric.datasets import MD17
from torch_geometric.transforms import Compose, RadiusGraph, Distance
from torch_geometric.nn import global_add_pool
from torch_geometric.nn.models.schnet import CFConv
from tqdm import tqdm

# RBF expansion module inspired by SchNet
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

class GraphCFConvModel(nn.Module):
    def __init__(
        self,
        node_dim=64,
        num_layers=3,
        num_gaussians=50,
        num_filters=64,
        cutoff=5.0,
        dropout=0.1
    ):
        super().__init__()
        # Embedding for atomic number features
        self.node_encoder = nn.Linear(1, node_dim)
        # Cutoff distance for envelope
        self.cutoff = cutoff
        # RBF distance expansion
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)
        # Kernel network: maps RBF vector to filter weights
        self.kernel_nn = nn.Sequential(
            nn.Linear(num_gaussians, num_filters),
            nn.ReLU(),
            nn.Linear(num_filters, num_filters)
        )
        # CFConv layers
        self.convs = nn.ModuleList([
            CFConv(
                in_channels=node_dim,
                out_channels=node_dim,
                num_filters=num_filters,
                nn=self.kernel_nn,
                cutoff=self.cutoff
            ) for _ in range(num_layers)]
        )
        # Activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # Final regression layers
        self.norm = nn.LayerNorm(node_dim)
        self.mlp = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, 1)
        )

    def forward(self, data):
        # Node features
        x = data.z.view(-1, 1).float()
        x = self.node_encoder(x)
        # Raw distances and RBF
        dist = data.edge_attr.view(-1).float()
        edge_attr = self.distance_expansion(dist)  # [num_edges, num_gaussians]
        # CFConv layers with envelope
        for conv in self.convs:
            x = conv(x, data.edge_index, edge_weight=dist, edge_attr=edge_attr)
            x = self.relu(x)
            x = self.dropout(x)
        # Pooling and regression
        x = self.norm(x)
        x = global_add_pool(x, data.batch)
        return self.mlp(x).squeeze(-1)


def train(model, loader, optimizer, device, mean_energy):
    model.train()
    criterion = nn.MSELoss()
    
    sum_loss = 0.0
    batch_count = 0
    for batch in tqdm(loader, desc='Train', unit='batch'):
        batch = batch.to(device)
        optimizer.zero_grad()
        targets = batch.energy.view(-1).to(device) - mean_energy
        preds_norm = model(batch)
        loss = criterion(preds_norm, targets)
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


if __name__ == '__main__':
    torch.manual_seed(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    radius = 5.0
    transform = Compose([RadiusGraph(r=radius, max_num_neighbors=32), Distance(norm=False)])
    dataset = MD17(root='data/MD17', name='aspirin', transform=transform)

    split = int(len(dataset) * 0.8)
    train_dataset = dataset[:split]
    test_dataset = dataset[split:]

    # Mean-normalization
    all_train_energies = torch.cat([d.energy.view(-1) for d in train_dataset])
    mean_energy = all_train_energies.mean().to(device)
    print(f"Mean training energy: {mean_energy:.4f}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    model = GraphCFConvModel(cutoff=radius).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1, 101):
        print(f"Epoch {epoch:03d}")
        train_loss = train(model, train_loader, optimizer, device, mean_energy)
        test_mae = test(model, test_loader, device, mean_energy)
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Test MAE: {test_mae:.4f}")