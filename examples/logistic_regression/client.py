import argparse
from collections import OrderedDict

import numpy as np
import torch
from torch import nn, optim

from examples.logistic_regression.dataset import load_datasets
from examples.logistic_regression.model import LogisticRegression as Net
from nillion_fl.client import FederatedLearningClient
from nillion_fl.logs import logger

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.debug(f"Training on {DEVICE} using PyTorch {torch.__version__}")

# torch.manual_seed(42)


class NillionFLClient(FederatedLearningClient):

    def __init__(self, net, trainloader, valloader, config):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.num_parameters = sum(
            p.numel() for p in net.parameters() if p.requires_grad
        )
        self.config = config
        self.criterion = nn.BCELoss()
        self.optimizer = optim.SGD(
            self.net.parameters(), lr=self.config["learning_rate"]
        )

        super(NillionFLClient, self).__init__(self.num_parameters)

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
            self.net.train()
            total_loss = 0
            for batch_inputs, batch_labels in self.trainloader:
                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.net(batch_inputs)
                loss = self.criterion(outputs, batch_labels)

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.trainloader)
            logger.warning(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    def fit(self, parameters=None):
        if parameters is not None:
            self.set_parameters(parameters)
        self.train()
        self.local_evaluate()
        return self.get_parameters()

    def local_evaluate(self):
        """
        Evaluates the network on the validation dataset and returns the accuracy.

        Returns:
            float: The accuracy of the network on the validation dataset.
        """
        import time

        self.net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.valloader:
                predicted = self.net(inputs)
                labels = (labels > 0.5).float()
                predicted = (predicted > 0.5).float()
                total += predicted.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        logger.warning(f"Validation Accuracy: {accuracy:.2f}%")
        time.sleep(2)
        return accuracy


def run(client_id):
    NUM_PARAMETERS = 1000  # Number of features in our dataset
    NUM_CLIENTS = 2
    net = Net(NUM_PARAMETERS)

    # Generate data
    trainloaders, valloaders = load_datasets(
        NUM_CLIENTS, batch_size=0, num_features=NUM_PARAMETERS
    )  # We're using only one client for this example

    client = NillionFLClient(
        net,
        trainloaders[client_id],
        valloaders[client_id],
        {"epochs": 1, "learning_rate": 0.001},
    )
    client.start_client()


def main():
    parser = argparse.ArgumentParser(
        description="Run the client program with a client ID",
        epilog="Example: python client.py 1",
    )
    parser.add_argument(
        "client_id",
        type=int,
        help="The client ID number (a value from 0 to 9 representing the client_id)",
    )
    args = parser.parse_args()
    run(args.client_id)


if __name__ == "__main__":
    main()
