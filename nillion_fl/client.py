from net_client import FederatedLearningClient
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict
import numpy as np
import torch


class NillionFLClient(object):

    def __init__(self):
        self.thread = None
        self._net_client = None

    @property
    def net_client(self):
        if not hasattr(self, '_net_client'):
            raise ValueError("net_client is not set")
        return self._net_client

    @property.setter
    def net_client(self, value):
        if value is None:
            raise ValueError("net_client cannot be None")

        if self._net_client is not None:
            raise ValueError("net_client is already set")

        self._net_client = value

    def start_client(self, config):
        # Create the
        self.net_client = FederatedLearningClient(config['host'], config['port'])
        self.net_client.register_client() # Creates the client_id and token

        while True:
            self.net_client.schedule_learning_iteration()

            self.fit()

        
    def fit(self):
        pass

    def fit_batch(self):
        pass
        self.active_streams.add(stream_id)


class PyTorchFLClient(NillionFLClient):

    def __init__(self, net, trainloader, valloader, config):
        super(PyTorchFLClient, self).__init__(net, trainloader, valloader, config)

    def get_parameters(self) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]
    
    def set_parameters(net, parameters: List[np.ndarray]):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)
        def fit(self, parameters, config):
        print(f"[Client {self.cid}] fit, config: {config}")
        print("SETTING PARAMETERS")
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=1)
        return get_parameters(self.net), len(self.trainloader), {}
        ### HERE WE STORE TO THE NILLION NETWORK
        store_id = store_weights(self.net)
        if not isinstance(store_id, list):
            store_id = [store_id]

        return store_id, len(store_id), {}

    def evaluate(self, parameters, config):
        print(f"[Client {self.cid}] evaluate, config: {config}")
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


def client_fn(cid) -> NillionFlowerClient:
    net = Net().to(DEVICE)
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]
    return NillionFlowerClient(cid, net, trainloader, valloader).to_client()