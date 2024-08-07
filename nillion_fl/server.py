import time
from concurrent import futures

import grpc

import nillion_fl.network.fl_service_pb2_grpc as fl_pb2_grpc
from nillion_fl.core.server.client_manager import ClientManager
from nillion_fl.core.server.learning_manager import LearningManager
from nillion_fl.core.server.nillion_integration import NillionServerIntegration
from nillion_fl.logs import logger, uuid_str


class FederatedLearningServer(fl_pb2_grpc.FederatedLearningServiceServicer):
    """
    A servicer class for handling federated learning operations.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 50051,
        num_parties=2,
        program_number=10,
        batch_size=1000,
    ):
        self.host = host
        self.port = port

        self.client_manager = ClientManager(num_parties)
        self.nillion_integration = NillionServerIntegration(
            num_parties, program_number, batch_size
        )

        self.learning_manager = LearningManager(
            num_parties,
            batch_size,
            self.client_manager,
            self.nillion_integration,
        )

    def RegisterClient(self, request, context):
        self.learning_manager.model_size, response_message = (
            self.client_manager.register_client(request, context)
        )
        return response_message

    def ScheduleLearningIteration(self, request_iterator, context):
        stream_id = self.learning_manager.schedule_learning_iteration(
            request_iterator, context
        )
        message = not None
        yield from self.learning_manager.learning_iteration_message_loop(stream_id)
        # Main server loop to answer requests from the client
        try:
            while self.client_manager.is_client(stream_id):
                # If there are messages to send to the client, yield them
                if not self.client_manager.has_messages(stream_id):
                    message = self.clients_queue[stream_id].get(timeout=5)
                    logger.debug(
                        f"[SERVER][{uuid_str(stream_id)}] Sending message: {message}"
                    )
                    yield message

                    # Wait for learning to complete or timeout
                    if not self.learning_complete.wait():  # 60 seconds timeout
                        logger.warning(
                            f"[SERVER][{uuid_str(stream_id)}] Learning iteration timed out"
                        )
                        self.client_threads[stream_id].join()
                        self.end_learning_for_all_clients()

                    # Reset the event for the next iteration
                    self.learning_complete.clear()

        except grpc.RpcError as e:
            logger.error(
                f"[SERVER][{uuid_str(stream_id)}] RPC Error in message sending: {e}"
            )
        finally:
            self.handle_client_disconnect(stream_id)

    def serve(self):
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        fl_pb2_grpc.add_FederatedLearningServiceServicer_to_server(self, server)
        server.add_insecure_port(f"{self.host}:{self.port}")

        server.start()
        logger.debug("Server started. Listening on port 50051.")
        try:
            while True:
                time.sleep(86400)
        except KeyboardInterrupt:
            logger.debug("Server stopping...")
            self.learning_manager.stop()
            logger.debug("All client threads stopped")
            server.stop(1)
