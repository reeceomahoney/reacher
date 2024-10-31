import h5py
import logging
import os
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

log = logging.getLogger(__name__)


def get_dataloaders(
    data_directory: str,
    train_fraction: float,
    train_batch_size: int,
    test_batch_size: int,
    num_workers: int,
):
    # build path
    current_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_path = current_dir + "/../../" + data_directory
    log.info(f"Loading data from {data_directory}")

    data = {}
    # load data
    with h5py.File(dataset_path, "r") as f:

        def load_dataset(name, obj):
            if isinstance(obj, h5py.Dataset):
                data[name] = obj[:]

        f.visititems(load_dataset)

    # (B, T, D)
    for k, v in data.items():
        data[k] = torch.from_numpy(v).transpose(0, 1)

    # build obs
    obs = torch.cat((data["data/obs/joint_pos"], data["data/obs/ee_state"]), dim=-1)
    obs = obs.reshape(-1, obs.shape[-1])

    # Build the datasets
    dataset = TensorDataset(obs)
    train_set, test_set = random_split(dataset, [train_fraction, 1 - train_fraction])

    log.info(f"<Dataset> Shape: {obs.shape}")

    # Build the dataloaders
    train_dataloader = DataLoader(
        train_set,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_set,
        batch_size=test_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader, obs
