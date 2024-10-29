import h5py
import logging
import numpy as np
import os
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split

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
    obs = torch.cat((data["data/obs/joint_pos"], data["data/obs/ee_pos"]), dim=-1)
    obs = obs.reshape(-1, obs.shape[-1])

    # TODO: use pretty table
    log.info(f"Dataset size | Observations: {obs.shape} ")

    # Build the datasets
    dataset = TensorDataset(obs)
    train_set, test_set = random_split(dataset, [train_fraction, 1 - train_fraction])

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


class RobotStateDataset(Dataset):

    def __init__(
        self,
        data_dir,
        train,
        train_data_name="free_space_100k_train.dat",
        test_data_name="free_space_10k_test.dat",
    ):

        self.train = train

        with open(os.path.join(data_dir, train_data_name), "rb") as f:
            np_train_val = np.load(f)

        with open(os.path.join(data_dir, test_data_name), "rb") as f:
            np_test = np.load(f)

        size = np_train_val.shape[0]

        size_train = int(0.8 * size)
        size_val = size - size_train
        size_test = np_test.shape[0]

        np_train = np_train_val[:size_train, :]
        np_val = np_train_val[size_train:, :]

        # normalise data
        input_dim = np_train.shape[1]
        mean_train = np_train.mean(axis=0).reshape((1, input_dim))
        std_train = np_train.std(axis=0).reshape((1, input_dim))

        # If all the values are the same, set them to 0 after normalisation.
        # std_train[std_train < EPSILON] = 1.0

        np_train -= np.tile(mean_train, (size_train, 1))
        np_train /= np.tile(std_train, (size_train, 1))

        np_val -= np.tile(mean_train, (size_val, 1))
        np_val /= np.tile(std_train, (size_val, 1))

        np_test -= np.tile(mean_train, (size_test, 1))
        np_test /= np.tile(std_train, (size_test, 1))

        if self.train == 0:
            self.robot_data = torch.tensor(np_train, dtype=torch.float32)
        elif self.train == 1:
            self.robot_data = torch.tensor(np_val, dtype=torch.float32)
        else:
            self.robot_data = torch.tensor(np_test, dtype=torch.float32)

        self.np_train = np_train
        self.np_val = np_val
        self.np_test = np_test
        self.mean_train = mean_train
        self.std_train = std_train

    def __len__(self):
        return self.robot_data.shape[0]

    def __getitem__(self, index):
        jpos_ee_xyz = self.robot_data[index, :-3]
        # goal_xyz = self.robot_data[index, -3:]
        return jpos_ee_xyz

    def get_np_train(self):
        return self.np_train

    def get_np_val(self):
        return self.np_val

    def get_np_test(self):
        return self.np_test

    def get_mean_train(self):
        return self.mean_train

    def get_std_train(self):
        return self.std_train

    def get_all_data(self):
        return self.robot_data[:, :-3]


def get_hung_dataset():
    kwargs = {"num_workers": 1, "pin_memory": True}

    train_dataset = RobotStateDataset("logs/hung/", train=0)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1024, shuffle=True, **kwargs
    )

    val_dataset = RobotStateDataset("logs/hung/", train=1)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1024, shuffle=True, **kwargs
    )

    obs = train_dataset.get_all_data()

    return train_loader, val_loader, obs
