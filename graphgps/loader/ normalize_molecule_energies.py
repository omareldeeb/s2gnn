import numpy as np, torch
from torch_geometric.datasets import QM9
from sklearn.linear_model import LinearRegression


def precompute_qm9_normalization(ds, output_path='qm9_atomref.npz'):
    prop_id = 7                    # U0 (eV) in the PyG loader
    Z, ptr = ds.data.z.numpy(), ds.data.ptr.numpy()
    y = ds.data.y[:, prop_id].numpy()            # (n_mol,)

    elements = [1, 6, 7, 8, 9]                        # H,C,N,O,F
    X = np.zeros((len(ptr)-1, len(elements)))
    for i in range(len(ptr)-1):
        atoms = Z[ptr[i]:ptr[i+1]]
        X[i]  = [(atoms == Z0).sum() for Z0 in elements]

    lin = LinearRegression().fit(X, y)
    baseline  = lin.predict(X)
    residuals = y - baseline

    # replace targets in–memory; now train on residuals
    ds.data.y[:, prop_id] = torch.tensor(residuals, dtype=torch.float32)
    np.savez(output_path, coef=lin.coef_, intercept=lin.intercept_)
    return {'coef': lin.coef_, 'intercept': lin.intercept_}