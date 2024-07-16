from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from nillion_fl.client import FederatedLearningClient

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
)


class NillionFLClient(FederatedLearningClient):

    def __init__(self, net, trainloader, valloader, config):
        super(NillionFLClient, self).__init__(net, trainloader, valloader)
        self.config = config

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
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.net.parameters())
        self.net.train()
        for epoch in range(epochs):
            correct, total, epoch_loss = 0, 0, 0.0
            for images, labels in self.trainloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = self.net(images)
                loss = criterion(self.net(images), labels)
                loss.backward()
                optimizer.step()
                # Metrics
                epoch_loss += loss
                total += labels.size(0)
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            epoch_loss /= len(self.trainloader.dataset)
            epoch_acc = correct / total
            print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")

    def fit(self, parameters):
        self.set_parameters(parameters)
        self.train()
        return self.get_parameters()

    def store_secrets(self, program_id, user_id, batch_size, num_parties):
        print(f"[Client {self.cid}] evaluate, config: {config}")
        self.set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


def run():
    client = NillionFLClient(None, None, None)
    client.start_client()


if __name__ == "__main__":
    run()
