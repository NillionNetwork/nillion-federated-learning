import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

NUM_CLIENTS = 10


class MyDataset(Dataset):
    def __init__(
        self,
        num_samples: int,
        num_features: int,
    ):
        super(MyDataset, self).__init__()
        np.random.seed(42)

        self.values = np.random.rand(num_samples, num_features).astype(np.float32)
        self.labels = self.values[:, 0].reshape(
            num_samples, 1
        )  # first feature is the label

    def __len__(self):
        return len(self.values)  # number of samples in the dataset

    def __getitem__(self, index):
        return self.values[index], self.labels[index]


def load_datasets(num_clients: int):

    dataset = MyDataset(1000, 10)  # 1000 train samples, 10 features

    # Split training set into `num_clients` partitions to simulate different local datasets
    partition_size = len(dataset) // num_clients
    lengths = [partition_size] * num_clients
    datasets = random_split(dataset, lengths, torch.Generator().manual_seed(42))

    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for ds in datasets:
        len_val = len(ds) // 10  # 10 % validation set
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
        trainloaders.append(DataLoader(ds_train, batch_size=32, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=32))

    return trainloaders, valloaders
