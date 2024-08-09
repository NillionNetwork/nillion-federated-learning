import argparse
from collections import OrderedDict

import numpy as np
import torch
from torch import nn, optim

from examples.logistic_regression.dataset import load_datasets
from examples.logistic_regression.model import LogisticRegression as Net
from nillion_fl.pytorch import PytorchFLClient
from nillion_fl.logs import logger

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.debug(f"Training on {DEVICE} using PyTorch {torch.__version__}")

class NillionFLClient(PytorchFLClient):

    def __init__(self, net, trainloader, valloader, config):
        super(NillionFLClient, self).__init__(net, trainloader, valloader)

        self.config = config
        self.criterion = nn.BCELoss()
        self.optimizer = optim.SGD(
            self.net.parameters(), lr=self.config["learning_rate"]
        )

 

    def train(self):
        """
        Trains the network for the given number of epochs using the given optimizer and criterion.

        Returns:
            None
        """
        epochs = self.config["epochs"]
        logger.warning(f"Training for {epochs} epochs")
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
        return accuracy


def run(client_id):
    NUM_CLIENTS = 2       # Number of clients in the federated network
    NUM_PARAMETERS = 1000  # Number of parameters for the Logistic Regression
    NUM_SAMPLES = 10000  # Number of samples for the Logistic Regression dataset for all clients
    net = Net(NUM_PARAMETERS) 

    # Generate data
    trainloaders, valloaders = load_datasets(
        NUM_CLIENTS, batch_size=0, num_samples=NUM_SAMPLES, num_features=NUM_PARAMETERS
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
        help="The client ID number to determine which fraction of the dataset to use",
    )
    args = parser.parse_args()
    run(args.client_id)


if __name__ == "__main__":
    main()
