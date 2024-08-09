from nillion_fl.server import FederatedLearningServer


class PytorchFLServer(FederatedLearningServer):
    """
    A federated learning server class for PyTorch models.
    """

    def __init__(self, host: str = "localhost", port: int = 50051, config=None):
        """
        Initialize the PytorchFLServer with necessary parameters.

        Args:
            host (str): The server host.
            port (int): The server port.
            config (dict): Configuration dictionary containing 'num_parties',
                and 'batch_size'.
        """
        super().__init__(host, port, config)