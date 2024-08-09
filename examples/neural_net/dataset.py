import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from nillion_fl.logs import logger


class NeuralNetDataset(Dataset):
    def __init__(
        self,
        num_samples: int,
        num_features: int,
    ):
        super(NeuralNetDataset, self).__init__()
        np.random.seed(42)

        # Generate random features
        self.features = np.random.randn(num_samples, num_features).astype(np.float32)

        # Generate labels based on a linear combination of features
        weights = np.random.randn(num_features)
        logger.info(f"True weights: {weights}")
        logits = np.dot(self.features, weights)
        self.labels = 1 / (1 + np.exp(-logits))
        self.labels = self.labels.reshape(-1, 1).astype(np.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


def load_datasets(
    num_clients: int,
    batch_size: int = 32,
    num_samples: int = 1000,
    num_features: int = 10,
):
    dataset = NeuralNetDataset(
        num_samples, num_features
    )  # 1000 samples, 10 features

    # Split dataset into `num_clients` partitions
    partition_size = len(dataset) // num_clients
    lengths = [partition_size] * num_clients
    datasets = random_split(dataset, lengths, torch.Generator().manual_seed(42))

    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for ds in datasets:
        len_val = len(ds) // 10  # 10% validation set
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        trainloader = None
        valloader = None
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
        if batch_size == 0:
            trainloader = DataLoader(ds_train, shuffle=True)
            valloader = DataLoader(ds_val)
        else:
            trainloader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
            valloader = DataLoader(ds_val, batch_size=batch_size)
        trainloaders.append(trainloader)
        valloaders.append(valloader)

    return trainloaders, valloaders
