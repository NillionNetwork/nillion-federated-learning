"""
This module contains the FederatedLearningServer class which handles federated learning operations
using gRPC.
"""

import time
from concurrent import futures

import grpc

import nillion_fl.network.fl_service_pb2_grpc as fl_pb2_grpc
from nillion_fl.core.server.client_manager import ClientManager
from nillion_fl.core.server.learning_manager import LearningManager
from nillion_fl.core.server.nillion_integration import NillionServerIntegration
from nillion_fl.logs import logger


class FederatedLearningServer(fl_pb2_grpc.FederatedLearningServiceServicer):
    """
    A gRPC servicer class for handling federated learning operations.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, host: str = "localhost", port: int = 50051, config=None):
        """
        Initialize the FederatedLearningServer with necessary parameters.

        Args:
            host (str): The server host.
            port (int): The server port.
            config (dict): Configuration dictionary containing 'num_parties',
                           'program_number', and 'batch_size'.
        """
        if config is None:
            config = {"num_parties": 2, "program_number": 10, "batch_size": 1000}

        self.host = host
        self.port = port

        # Initialize ClientManager to manage client registrations and communications
        self.client_manager = ClientManager(config["num_parties"])

        # Initialize NillionServerIntegration for federated learning integration
        self.nillion_integration = NillionServerIntegration(
            config["num_parties"], config["program_number"], config["batch_size"]
        )

        # Initialize LearningManager to handle the learning iterations
        self.learning_manager = LearningManager(
            config["num_parties"],
            config["batch_size"],
            self.client_manager,
            self.nillion_integration,
        )

    def RegisterClient(self, request, context):
        """
        Register a new client to the federated learning server.

        Args:
            request: The request object containing client details.
            context: The context for the RPC call.

        Returns:
            response_message: The response message after registration.
        """
        self.learning_manager.model_size, response_message = (
            self.client_manager.register_client(request, context)
        )
        return response_message

    def ScheduleLearningIteration(self, request_iterator, context):
        """
        Schedule a new learning iteration for the clients.

        Args:
            request_iterator: An iterator over incoming requests.
            context: The context for the RPC call.

        Yields:
            message: Messages to be sent to the clients during the learning iteration.
        """
        stream_id = self.learning_manager.schedule_learning_iteration(
            request_iterator, context
        )
        yield from self.learning_manager.learning_iteration_message_loop(stream_id)

    def serve(self):
        """
        Start the gRPC server and listen for incoming requests.

        This method sets up the gRPC server, adds the FederatedLearningServiceServicer
        to it, and starts the server. It keeps the server running until interrupted.
        """
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        fl_pb2_grpc.add_FederatedLearningServiceServicer_to_server(self, server)
        server.add_insecure_port(f"{self.host}:{self.port}")

        server.start()
        logger.debug("Server started. Listening on port %d.", self.port)
        try:
            while True:
                time.sleep(86400)  # Sleep for a day to keep the server running
        except KeyboardInterrupt:
            logger.debug("Server stopping...")
            self.learning_manager.stop()  # Stop the learning manager
            logger.debug("All client threads stopped")
            server.stop(1)  # Stop the server
