import h5py
import numpy as np

np.set_printoptions(precision=2, suppress=True)


def print_h5_contents(file_path):
    with h5py.File(file_path, "r") as f:

        def print_group(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  Name: {obj.name}")
                print(f"  Shape: {obj.shape}")
                print(f"  Dtype: {obj.dtype}")
                # print(f"  min: {np.min(obj, axis=(0,1))}")
                # print(f"  max: {np.max(obj, axis=(0,1))}")
                print("\n")
            elif isinstance(obj, h5py.Group):
                print(f"Group: {name}")

        # Recursively visit all groups and datasets
        f.visititems(print_group)


# Usage
file_path = "logs/latent_planning_record/hdf_dataset.hdf5"
print_h5_contents(file_path)
