import torch
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from graphgps.loader.dataset.wrapped_md17 import WrappedMD17
from graphgps.network.painn import PaiNN, train, test
from torch_geometric.datasets import MD17


if __name__ == '__main__':
    torch.manual_seed(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    radius = 5.0
    transform = T.Compose([T.RadiusGraph(r=radius, max_num_neighbors=32), T.Distance(norm=False)])
    dataset = MD17(root='data/MD17', name='ethanol', transform=transform)

    split = int(len(dataset) * 0.8)
    train_dataset = dataset[:split]
    test_dataset = dataset[split:]

    # Mean-normalization
    # all_train_energies = torch.cat([d.energy.view(-1) for d in train_dataset])
    # mean_energy = all_train_energies.mean().to(device)
    # print(f"Mean training energy: {mean_energy:.4f}")
    mean_energy = -97195.9314

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    model = PaiNN(cutoff=radius, node_feature_dim=384, out_node_feature_dim=384, layers=2, n_radial_basis=20).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1, 101):
        print(f"Epoch {epoch:03d}")
        train_loss = train(model, train_loader, optimizer, device, mean_energy)
        test_mae = test(model, test_loader, device, mean_energy)
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Test MAE: {test_mae:.4f}")