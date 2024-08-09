import argparse
from collections import OrderedDict

import numpy as np
import torch
from torch import nn, optim

from examples.conv_net.dataset import load_datasets
from examples.conv_net.model import NeuralNet as Net
from nillion_fl.logs import logger
from nillion_fl.pytorch import PytorchFLClient

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.debug(f"Training on {DEVICE} using PyTorch {torch.__version__}")


class NillionFLClient(PytorchFLClient):

    def __init__(self, net, trainloader, valloader, config):
        super(NillionFLClient, self).__init__(net, trainloader, valloader)

        self.config = config
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    def train(self):
        """
        Trains the network for the given number of epochs using the given optimizer and criterion.

        Returns:
            None
        """
        epochs = self.config["epochs"]
        logger.warning(f"Training for {epochs} epochs")
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0
    
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss:.4f}")

    def local_evaluate(self):
        """
        Evaluates the network on the validation dataset and returns the accuracy.

        Returns:
            float: The accuracy of the network on the validation dataset.
        """
        import time

        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in self.valloader:
                images, labels = data
                # calculate outputs by running images through the network
                outputs = self.net(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        logger.warning(f"Accuracy of the network on the {total} validation images: {100 * correct / total}%")
        return correct / total


def run(client_id):
    NUM_CLIENTS = 2  # Number of clients in the federated network
    NUM_PARAMETERS = 750  # Number of input parameters for the Neural Network
    net = Net()

    # Generate data
    trainloaders, valloaders = load_datasets(
        NUM_CLIENTS
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
