import asyncio
import signal
import threading
import time
import uuid

import grpc
import numpy as np

import nillion_fl.fl_net.fl_service_pb2 as fl_pb2
import nillion_fl.fl_net.fl_service_pb2_grpc as fl_pb2_grpc
from nillion_fl.logs import logger, uuid_str
from nillion_fl.nillion_network.client import NillionNetworkClient


class FederatedLearningClient:
    """
    A client class for participating in federated learning.
    """

    def __init__(self, num_parameters):
        """
        Initialize the FederatedLearningClient.

        Args:
            net: The neural network model.
            trainloader: Data loader for training data.
            valloader: Data loader for validation data.
        """
        self.responses = []
        self.client_info = None
        self.nillion_client = None
        self.parameters = None
        self.store_secret_thread = None
        self.num_parameters = num_parameters

        # locks for critical sections
        self.responses_lock = threading.Lock()
        self.stop_event = threading.Event()

    def register_client(self):
        """
        Register the client with the federated learning server.
        """
        request = fl_pb2.RegisterRequest(
            model_size=self.num_parameters,
        )
        self.client_info = self.stub.RegisterClient(request)
        logger.debug(
            f"""
            Registered with client_id: {self.client_info.client_id}, 
            token: {uuid_str(self.client_info.token)}, 
            num_parties: {self.client_info.num_parties}
            """
        )
        self.nillion_client = NillionNetworkClient(
            self.client_info.client_id, self.client_info.num_parties
        )

    def schedule_learning_iteration(self):
        """
        Schedule and manage learning iterations with the server.

        Returns:
            None
        """

        def client_request_sender():
            """
            Generator function to send client requests to the server.
            """
            logger.debug("[CLIENT] Sending initial message")
            yield fl_pb2.StoreIDs(
                store_id="", party_id="", token=self.client_info.token, batch_id=-1
            )

            while True:
                response = None
                with self.responses_lock:
                    if len(self.responses) > 0:
                        response = self.responses.pop(0)
                        logger.error(
                            f"[CLIENT] Sending store id response for batch: {response.batch_id} "
                        )
                if response is not None:
                    yield response

            logger.debug("[CLIENT][SEND] STOP")

        # Start the learning iteration process
        learning_requests = self.stub.ScheduleLearningIteration(client_request_sender())

        for learning_request in learning_requests:
            logger.debug("[CLIENT] Received learning request")
            if learning_request.program_id == "-1":
                logger.warning("Received STOP training request")
                learning_requests.cancel()
                break

            logger.debug(f"Learning Request: {learning_request}")

            self.learning_iteration(learning_request)

        return None

    def learning_iteration(self, learning_request):
        """
        Perform a single learning iteration.

        Args:
            learning_request: The learning request from the server.
        """
        parameters = self.fit(self.parameters)
        expected_results = len(parameters) // learning_request.batch_size

        def store_secrets_thread():
            self.store_secrets(
                parameters,
                learning_request.program_id,
                learning_request.user_id,
                learning_request.batch_size,
            )

        self.store_secret_thread = threading.Thread(target=store_secrets_thread)
        self.store_secret_thread.start()

        new_parameters = asyncio.run(
            self.nillion_client.get_compute_result(expected_results)
        )

        new_parameters = sorted(new_parameters, key=lambda x: x[0])
        new_parameters = np.concatenate([x[1] for x in new_parameters])
        self.parameters = new_parameters

        logger.debug(f"Waiting for thread to join()...")
        self.store_secret_thread.join()
        self.store_secret_thread = None
        logger.info(f"New Parameters: {len(self.parameters)}")

    def fit(self, parameters):
        """
        Fit the model using the current parameters.

        Args:
            parameters: Current model parameters.

        Returns:
            numpy.ndarray: Updated model parameters.
        """
        logger.debug("Fitting...")
        # Dummy parameters and function
        if parameters is None:
            return np.ones(10)
        else:
            return parameters + 0.5

    def store_secrets(self, parameters, program_id, user_id, batch_size):
        """
        Store the parameters as secrets in the Nillion Network.

        Args:
            parameters (numpy.ndarray): Model parameters to store.
            program_id (str): ID of the program.
            user_id (str): ID of the user.
            batch_size (int): Size of each batch.
        """
        # Create a batch of maximum batch size of the parameters vector
        remainder = len(parameters) % batch_size
        if remainder > 0:
            parameters = np.pad(parameters, (0, batch_size - remainder))

        secret_name = chr(ord("A") + self.client_info.client_id)

        logger.debug(f"Parameters: {divmod(len(parameters), batch_size)}")
        for batch_start in range(0, len(parameters), batch_size):
            if self.stop_event.is_set():
                logger.warning("Stopping store_secrets operation...")
                break
            batch_id = batch_start // batch_size
            batch = parameters[batch_start : batch_start + batch_size]

            store_id = asyncio.run(
                self.nillion_client.store_array(batch, secret_name, program_id, user_id)
            )

            # Prepare a response
            with self.responses_lock:
                logger.debug(
                    f"Storing secret {secret_name} with batch {np.concatenate([batch[:3], batch[-3:]])} {batch.shape}, store_id {store_id}, party_id {self.nillion_client.party_id}, batch_id {batch_id}, token {self.client_info.token}"
                )
                self.responses.append(
                    fl_pb2.StoreIDs(
                        store_id=store_id,
                        party_id=self.nillion_client.party_id,
                        token=self.client_info.token,
                        batch_id=batch_id,
                    )
                )

    def start_client(self, host="localhost", port=50051):
        """
        Start the federated learning client.

        Args:
            host (str): Server host address. Default is "localhost".
            port (int): Server port number. Default is 50051.
        """
        try:
            # Establish a gRPC channel and create a stub
            self.channel = grpc.insecure_channel(f"{host}:{port}")
            self.stub = fl_pb2_grpc.FederatedLearningServiceStub(self.channel)
            self.client_info = None
            self.register_client()  # Creates the client_id and token
            self.schedule_learning_iteration()
        except KeyboardInterrupt:
            logger.warning("Client stopping...")
        finally:
            self.stop_event.set()
            if self.store_secret_thread and self.store_secret_thread.is_alive():
                self.store_secret_thread.join()
            if self.channel:
                self.channel.close()


def main():
    """
    Main function to create and start a FederatedLearningClient.
    """
    client = FederatedLearningClient(None, None, None)
    client.start_client()


if __name__ == "__main__":
    main()
