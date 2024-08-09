""" PyTorch-based Federated Learning client for training and evaluating models. """

# Import necessary modules
from collections import OrderedDict

import numpy as np
import torch

from nillion_fl.client import FederatedLearningClient
from nillion_fl.logs import logger

# Set the device for PyTorch
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.debug("Training on %s using PyTorch %s", DEVICE, torch.__version__)


class PytorchFLClient(FederatedLearningClient):
    """
    A PyTorch-based Federated Learning client for training and evaluating models.
    """

    # pylint: disable=too-many-arguments
    def __init__(self, net, trainloader, valloader, host="localhost", port=50051):
        """
        Initialize the Federated Learning client.

        Args:
            net: The neural network to be trained and evaluated.
            trainloader: DataLoader for the training data.
            valloader: DataLoader for the validation data.
            host: The host of the server to connect to.
            port: The port of the server to connect to.
        """
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.num_parameters = sum(
            p.numel() for p in net.parameters() if p.requires_grad
        )
        super().__init__(host=host, port=port, num_parameters=self.num_parameters)

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
        Sets the network's parameters from a single one-dimensional array.

        Args:
            parameters (np.ndarray): The parameters of the network
                as a single one-dimensional array.

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

        self.local_evaluate()

    def train(self):
        """
        Trains the network using the provided training data.

        Returns:
            None
        """
        # Placeholder for the training logic

    def fit(self, parameters=None):
        """
        Fits the model using the provided parameters, trains the model, and evaluates it.

        Args:
            parameters (np.ndarray, optional): The parameters to set before training.

        Returns:
            np.ndarray: The updated parameters of the network.
        """
        if parameters is not None:
            self.set_parameters(parameters)
        self.train()
        self.local_evaluate()
        return self.get_parameters()

    def local_evaluate(self):
        """
        Evaluates the network on the validation dataset.

        Returns:
            float: The accuracy of the network on the validation dataset.
        """
        # Placeholder for the evaluation logic
