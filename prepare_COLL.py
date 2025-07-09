import os
import numpy as np
from ase.io import read
from graphgps.loader.gemnet.data_container import DataContainer

def process_structures(structures, mean_energy):
    all_R, all_Z, all_E, all_F, all_N = [], [], [], [], []

    for atoms in structures:
        all_R.append(atoms.get_positions())
        all_Z.append(atoms.get_atomic_numbers())
        all_E.append([atoms.get_potential_energy()])
        all_F.append(atoms.get_forces())
        all_N.append(len(atoms))

    E_raw = np.array(all_E).flatten()
    E_centered = E_raw - mean_energy

    R = np.concatenate(all_R, axis=0)
    Z = np.concatenate(all_Z, axis=0)
    F = np.concatenate(all_F, axis=0)
    N = np.array(all_N)

    return R, Z, E_centered, F, N


def save_npz(path, R, Z, E, F, N):
    np.savez(
        path,
        R=R.astype(np.float32),
        Z=Z.astype(np.int32),
        E=E.astype(np.float32),
        F=F.astype(np.float32),
        N=N.astype(np.int32),
    )
    print(f"Saved: {path}")


if __name__ == '__main__':
    base_path = "./datasets/Coll"
    save_dir = "./datasets/Coll"
    os.makedirs(save_dir, exist_ok=True)

    print("Reading train set...")
    train_structs = read(os.path.join(base_path, "coll_v1.2_AE_train.xyz"), index=":")
    train_energies = np.array([atoms.get_potential_energy() for atoms in train_structs])
    mean_energy = np.mean(train_energies)
    print(f"Train set mean energy: {mean_energy:.6f} eV")

    R, Z, E, F, N = process_structures(train_structs, mean_energy)
    save_npz(os.path.join(save_dir, "coll_train.npz"), R, Z, E, F, N)

    print("Reading val set...")
    val_structs = read(os.path.join(base_path, "coll_v1.2_AE_val.xyz"), index=":")
    R, Z, E, F, N = process_structures(val_structs, mean_energy)
    save_npz(os.path.join(save_dir, "coll_val.npz"), R, Z, E, F, N)

    print("Reading test set...")
    test_structs = read(os.path.join(base_path, "coll_v1.2_AE_test.xyz"), index=":")
    R, Z, E, F, N = process_structures(test_structs, mean_energy)
    save_npz(os.path.join(save_dir, "coll_test.npz"), R, Z, E, F, N)

    try:
        print("\nLoading coll_train.npz with DataContainer...")
        dc = DataContainer(
            path=os.path.join(save_dir, "coll_train.npz"),
            cutoff=5.0,
            int_cutoff=5.0
        )
        print("Success ✅")
    except Exception as e:
        print(f"DataContainer error: {e}")
