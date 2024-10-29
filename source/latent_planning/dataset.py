import h5py
import logging
import numpy as np
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
    dataset_type: str = "isaac",
):
    # build path
    current_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_path = current_dir + "/../../" + data_directory
    log.info(f"Loading data from {data_directory}")

    if dataset_type == "isaac":
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
        obs = torch.cat((data["data/obs/joint_pos"], data["data/obs/ee_pos"]), dim=-1)
        obs = obs.reshape(-1, obs.shape[-1])

        # Build the datasets
        dataset = TensorDataset(obs)
        train_set, test_set = random_split(
            dataset, [train_fraction, 1 - train_fraction]
        )

    elif dataset_type == "hung":
        with open(os.path.join(data_directory, "train.dat"), "rb") as f:
            np_train = np.load(f)[:, :-3]  # remove goal
        with open(os.path.join(data_directory, "test.dat"), "rb") as f:
            np_test = np.load(f)[:, :-3]

        train = torch.from_numpy(np_train).to(torch.float32)
        test = torch.from_numpy(np_test).to(torch.float32)
        obs = train
        train_set = TensorDataset(train)
        test_set = TensorDataset(test)

    # TODO: use pretty table
    log.info(f"Dataset size | Observations: {obs.shape} ")

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
