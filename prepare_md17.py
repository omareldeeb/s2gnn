import os
import numpy as np
import torch
from torch_geometric.datasets import MD17
from graphgps.loader.gemnet.data_container import DataContainer
#import torch sparse
import torch_sparse
def preprocess_md17_for_mlff(
    root_path: str,
    dataset_name: str,
    save_path: str
) -> None:
    """
    Loads a dataset from the PyTorch Geometric MD17 collection,
    processes it by subtracting the mean energy, and saves it in
    a .npz format compatible with the provided DataContainer class.

    Args:
        root_path (str): The directory where the MD17 dataset will be
                         stored.
        dataset_name (str): The name of the specific molecule from the
                            MD17 dataset to process (e.g., 'Aspirin').
        save_path (str): The path where the output .npz file will be saved.
    """
    # Load the specified MD17 dataset
    dataset = MD17(root=root_path, name=dataset_name)

    # Initialize lists to store the data for all molecules
    all_R = []
    all_Z = []
    all_E = []
    all_F = []
    all_N = []

    # Iterate through each molecule in the dataset to collect data
    for data in dataset:
        all_R.append(data.pos.numpy())
        all_Z.append(data.z.numpy())
        all_E.append(data.energy.numpy())
        all_F.append(data.force.numpy())
        all_N.append(data.z.shape[0])

    # --- Energy Adjustment ---
    # Convert energy list to a NumPy array to calculate the mean
    E_raw = np.array(all_E).flatten()

    # Calculate the average energy across all molecules in the dataset
    mean_energy = np.mean(E_raw)
    print(f"Calculated and subtracted mean energy: {mean_energy:.6f} eV")

    # Subtract the mean energy from each molecule's energy (mean centering)
    E_centered = E_raw - mean_energy
    # -----------------------

    # Concatenate the other lists into single NumPy arrays
    R = np.concatenate(all_R, axis=0)
    Z = np.concatenate(all_Z, axis=0)
    F = np.concatenate(all_F, axis=0)
    N = np.array(all_N)

    # Save the processed data to a .npz file
    np.savez(
        save_path,
        R=R.astype(np.float32),
        Z=Z.astype(np.int32),
        E=E_centered.astype(np.float32),  # Save the mean-centered energies
        F=F.astype(np.float32),
        N=N.astype(np.int32),
    )
    print(f"Successfully preprocessed '{dataset_name}' and saved to '{save_path}'")

if __name__ == '__main__':
    # Define the root directory for the dataset and the desired save path
    ROOT = './md17_data'
    SAVE_FILE = './md17_aspirin_mean_centered.npz'
    DATASET_NAME = 'aspirin'

    # Create the root directory if it doesn't exist
    if not os.path.exists(ROOT):
        os.makedirs(ROOT)

    # Preprocess the dataset
    preprocess_md17_for_mlff(
        root_path=ROOT,
        dataset_name=DATASET_NAME,
        save_path=SAVE_FILE
    )

    # Example of how to use the generated file with the DataContainer
    # Assuming the DataContainer class from the initial prompt is defined here
    try:
        data_container = DataContainer(
            path=SAVE_FILE,
            cutoff=5.0,
            int_cutoff=5.0
        )
        print(f"\nSuccessfully loaded the preprocessed data from '{SAVE_FILE}' with DataContainer.")
        # The energies 'data_container.E' will now be mean-centered.
        # print(f"Average energy in DataContainer: {np.mean(data_container.E):.6f} eV (should be close to 0)")
    except NameError:
        print("\nSkipping DataContainer loading example because the class is not defined in this script.")
    except Exception as e:
        print(f"\nAn error occurred while loading with DataContainer: {e}")