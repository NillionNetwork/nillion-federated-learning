import asyncio
import signal
import threading
import time
import uuid

import grpc
import numpy as np

import nillion_fl.network.fl_service_pb2 as fl_pb2
import nillion_fl.network.fl_service_pb2_grpc as fl_pb2_grpc
from nillion_fl.logs import logger, uuid_str


class FLClientCore(object):
    """
    A client class for participating in federated learning.
    """

    def __init__(self, host="localhost", port=50051):
        """
        Initialize the FederatedLearningClient.

        Args:
            net: The neural network model.
            trainloader: Data loader for training data.
            valloader: Data loader for validation data.
        """
        self.responses = []
        self.__client_info = None

        self.channel = grpc.insecure_channel(f"{host}:{port}")
        self.stub = fl_pb2_grpc.FederatedLearningServiceStub(self.channel)

        # locks for critical sections
        self.responses_lock = threading.Lock()


    @property
    def client_info(self):
        """
        Get the client information.

        Returns:
            fl_pb2.ClientInfo: The client information.
        """
        if self.__client_info is None:
            raise ValueError("Client not registered")
        return self.__client_info

    def register_client(self, num_parameters):
        """
        Register the client with the federated learning server.
        """
        request = fl_pb2.RegisterRequest(
            model_size=num_parameters,
        )
        self.__client_info = self.stub.RegisterClient(request)
        logger.debug(
            f"""
            Registered with client_id: {self.client_info.client_id}, 
            token: {uuid_str(self.client_info.token)}, 
            num_parties: {self.client_info.num_parties}
            """
        )

    def initialize_stream(self):
        """
        Initialize the stream with the server.

        """
        self.send_store_id(store_id="", party_id="", token=self.client_info.token, batch_id=-1)

    def send_store_id(self, store_id: str, party_id: str, token: str, batch_id: int):
        """
        Send a response to the server.

        Args:
            store_id (str): The store ID.
            party_id (str): The party ID.
            token (str): The token.
            batch_id (int): The batch ID.
        """
        with self.responses_lock:
            self.responses.append(
                fl_pb2.StoreIDs(
                    store_id=store_id, party_id=store_id, token=token, batch_id=batch_id
                )
            )

    def client_request_sender(self):
        """
        Generator function to send client requests to the server.
        """
        logger.debug("[CLIENT] Sending initial message")
        self.initialize_stream() # Inputs a initial store_id, party_id, token, and batch_id
        while True:
            response = None
            with self.responses_lock:
                if len(self.responses) > 0:
                    response = self.responses.pop(0)
                    logger.error(
                        f"[CLIENT] Sending store id response for batch: {response.batch_id} "
                    )
            # This needs to be here to avoid a deadlock
            if response is not None:
                yield response

        logger.debug("[CLIENT][SEND] STOP")

    def schedule_learning_iteration(self, callback: callable):
        """
        Schedule and manage learning iterations with the server.

        Returns:
            None
        """

        # Start the learning iteration process
        learning_requests = self.stub.ScheduleLearningIteration(
            self.client_request_sender()
        )

        for learning_request in learning_requests:
            logger.debug("[CLIENT] Received learning request")
            if learning_request.program_id == "-1":
                logger.warning("Received STOP training request")
                learning_requests.cancel()
                break

            logger.debug(f"Learning Request: {learning_request}")

            callback(learning_request)
            
        return None


    def start_learning(self, callback: callable):
        """
        Start the federated learning client.

        Args:
            host (str): Server host address. Default is "localhost".
            port (int): Server port number. Default is 50051.
        """
        try:
            self.schedule_learning_iteration(callback)
        except KeyboardInterrupt:
            logger.warning("Client stopping...")
        finally:
            if self.channel:
                self.channel.close()

