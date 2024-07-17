from collections import OrderedDict

import numpy as np
import torch
from torch import nn, optim

from examples.logistic_regression.dataset import load_datasets
from examples.logistic_regression.model import Net
from nillion_fl.client import FederatedLearningClient

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Training on {DEVICE} using PyTorch {torch.__version__}")

torch.manual_seed(42)


class NillionFLClient(FederatedLearningClient):

    def __init__(self, net, trainloader, valloader, config):
        super(NillionFLClient, self).__init__(net, trainloader, valloader)
        self.config = config
        self.criterion = nn.BCELoss()
        self.optimizer = optim.SGD(
            self.net.parameters(), lr=self.config["learning_rate"]
        )

    def get_parameters(self) -> np.ndarray:
        """
        Takes the parameters of the network and returns them as a single
        one-dimensional array.

        Returns:
            np.ndarray: The parameters of the network as a single one-dimensional array.
        """
        params = [val.cpu().numpy() for _, val in self.net.state_dict().items()]
        layers_concat = np.concatenate([layer.flatten() for layer in params])
        return layers_concat

    def set_parameters(self, parameters: np.ndarray):
        """
        Gets a single one-dimensional array of parameters and sets the network's
        parameters to these values.

        Args:
            parameters (np.ndarray): The parameters of the network as a single one-dimensional array

        Returns:
            None

        """
        start = 0
        new_state_dict = OrderedDict()
        for key, val in self.net.state_dict().items():
            end = start + np.prod(val.shape)
            new_state_dict[key] = torch.tensor(
                parameters[start:end].reshape(val.shape)
            ).to(DEVICE)
            start = end
        self.net.load_state_dict(new_state_dict, strict=True)

    def train(self):
        """
        Trains the network for the given number of epochs using the given optimizer and criterion.

        Returns:
            None
        """
        epochs = self.config["epochs"]

        for epoch in range(epochs):
            for batch_inputs, batch_labels in self.trainloader:
                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.net(batch_inputs)
                loss = self.criterion(outputs, batch_labels)

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    def fit(self, parameters=None):
        if parameters is not None:
            self.set_parameters(parameters)
        self.train()
        return self.get_parameters()


def run():

    NUM_CLIENTS = 2
    trainloaders, valloaders = load_datasets(NUM_CLIENTS)
    net = Net()

    client = NillionFLClient(
        net, trainloaders[0], valloaders[0], {"epochs": 5, "learning_rate": 0.001}
    )
    client.start_client()


if __name__ == "__main__":
    run()
