import h5py
import numpy as np


def print_h5_contents(file_path):
    with h5py.File(file_path, "r") as f:

        def print_group(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"Dataset: {name}")
                print(f"  Shape: {obj.shape}")
                print(f"  Dtype: {obj.dtype}")
                print(f"mean: {np.mean(obj)}")
                print(f"std: {np.std(obj)}")
                print("\n")
            elif isinstance(obj, h5py.Group):
                print(f"Group: {name}")

        # Recursively visit all groups and datasets
        f.visititems(print_group)


# Usage
file_path = "logs/latent_planning/Isaac-Reach-Franka-v0/hdf_dataset.hdf5"
print_h5_contents(file_path)
