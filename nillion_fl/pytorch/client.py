import argparse
from collections import OrderedDict

import numpy as np
import torch
from torch import nn, optim

from examples.logistic_regression.model import LogisticRegression as Net
from nillion_fl.client import FederatedLearningClient
from nillion_fl.logs import logger

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.debug(f"Training on {DEVICE} using PyTorch {torch.__version__}")


class PytorchFLClient(FederatedLearningClient):

    def __init__(self, net, trainloader, valloader, host = "localhost", port = 50051):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.num_parameters = sum(
            p.numel() for p in net.parameters() if p.requires_grad
        )
        super(PytorchFLClient, self).__init__(host=host, port=port, num_parameters=self.num_parameters)

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
        pass

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
        pass


