import asyncio
import logging
import threading
import time
import traceback
import uuid
from concurrent import futures
from enum import Enum
from queue import Queue

import grpc

import nillion_fl.fl_net.fl_service_pb2 as fl_pb2
import nillion_fl.fl_net.fl_service_pb2_grpc as fl_pb2_grpc
from nillion_fl.logs import logger, uuid_str
from nillion_fl.nillion_network.server import (MAX_SECRET_BATCH_SIZE,
                                               FedAvgNillionNetworkServer)


# Enum to represent the state of a client
class ClientState(Enum):
    INITIAL = 0
    READY = 1
    END = 2


class FederatedLearningServicer(fl_pb2_grpc.FederatedLearningServiceServicer):
    """
    A servicer class for handling federated learning operations.
    """

    def __init__(self, num_parties, program_number=10, batch_size=1000):
        """
        Initialize the FederatedLearningServicer.

        Args:
            num_parties (int): The total number of parties participating in federated learning.
            program_batch (int): The num size for the Nillion program.
        """
        self.num_parties = num_parties
        self.batch_size = int(MAX_SECRET_BATCH_SIZE / self.num_parties)
        self.batch_size = (
            batch_size  # The size of the batch to be processed in the Nillion Network
        )
        self.program_number = program_number  # The number of the program to be executed in the Nillion Network

        # Dictionaries to manage client information and state
        self.clients = {}  # token -> client_id
        self.active_streams = {}  # stream_id -> token
        self.client_threads = {}  # stream_id -> thread
        self.clients_queue = {}  # stream_id -> Queue of messages
        self.iteration_values = {}  # stream_id -> received values for current iteration
        self.model_size = None  # The size of the model in trainings
        self.processed_batches = (
            0  # Number of processed batches in the current iteration
        )
        self.batches_per_iteration: int = (
            None  # The number of batches we need to process per iteration
        )
        # Thread synchronization
        self.lock = threading.Lock()
        self.learning_in_progress = False
        self.learning_complete = threading.Event()

        self.compute_threads = []
        self.compute_lock = threading.Lock()

        # Initialize the Nillion Network server
        self.nillion_server = FedAvgNillionNetworkServer(num_parties, program_number)
        self.nillion_server.compile_program(self.batch_size)
        self.program_id = asyncio.run(self.nillion_server.store_program())

    def __del__(self):
        """
        Destructor to ensure all clients end learning when the servicer is destroyed.
        """
        self.end_learning_for_all_clients()

    def __str__(self):
        """
        Return a string representation of the FederatedLearningServicer.

        Returns:
            str: A string containing the servicer's current state.
        """
        return f"FederatedLearningServicer(\n num_parties={self.num_parties},\n clients={self.clients},\n active_streams={self.active_streams},\n client_threads={self.client_threads},\n clients_queue={self.clients_queue}, \n iteration_values={self.iteration_values},\n learning_in_progress={self.learning_in_progress}\n)"

    def __repr__(self):
        """
        Return a string representation of the FederatedLearningServicer.

        Returns:
            str: Same as __str__ method.
        """
        return self.__str__()

    def is_valid_token(self, token):
        """
        Check if a given token is valid.

        Args:
            token (str): The token to validate.

        Returns:
            bool: True if the token is valid, False otherwise.
        """
        return token in self.clients

    def is_initial_request(self, request):
        """
        Check if a request is an initial request.

        Args:
            request: The request to check.

        Returns:
            bool: True if it's an initial request, False otherwise.
        """
        return (
            len(request.store_id) == 0
            and request.party_id == ""
            and request.token != ""
            and request.batch_id == -1
        )

    def RegisterClient(self, request, context):
        """
        Register a new client.

        Args:
            request: The registration request.
            context: The gRPC context.

        Returns:
            fl_pb2.ClientInfo: Client information including client_id, token, and num_parties.
        """
        with self.lock:
            # For the first client, set the model size
            if self.model_size is None:
                self.model_size = request.model_size
                self.batches_per_iteration, sum_1 = divmod(
                    self.model_size, self.batch_size
                )
                if sum_1 != 0:
                    self.batches_per_iteration += 1
            # Check if the model size matches the first client model size
            elif self.model_size != request.model_size:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Model size mismatch.")
                return fl_pb2.ClientInfo(
                    client_id=-1, token="", num_parties=self.num_parties
                )
            if len(self.clients) >= self.num_parties:
                context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
                context.set_details("The maximum number of clients has been reached.")
                return fl_pb2.ClientInfo(
                    client_id=-1, token="", num_parties=self.num_parties
                )

            client_id = len(self.clients)
            token = str(uuid.uuid4())
            self.clients[token] = client_id
            logger.debug(
                f"[SERVER] Registering client [{client_id}] with token: {uuid_str(token)}"
            )
            return fl_pb2.ClientInfo(
                client_id=client_id, token=token, num_parties=self.num_parties
            )

    def schedule_learning_iteration_messages(self):
        """
        Schedule learning iteration messages for all connected clients.
        """
        logger.debug("Scheduling learning iteration for all connected clients")
        self.learning_in_progress = True
        for client_stream_id in self.clients_queue.keys():
            self.clients_queue[client_stream_id].put(
                fl_pb2.ScheduleRequest(
                    program_id=self.program_id,
                    user_id=self.nillion_server.user_id,
                    batch_size=self.batch_size,
                    num_parties=self.num_parties,
                )
            )
        logger.debug("Scheduled learning iteration for all connected clients")

    def InitialState(self, stream_id, request):
        """
        Handle the initial state of a client connection.

        Args:
            stream_id (str): The unique identifier for the client stream.
            request: The initial request from the client.
        """
        logger.info(f"[{uuid_str(stream_id)}] Received InitialState")
        logger.debug(f"[{uuid_str(stream_id)}] Received initial request: {request}")
        if self.is_initial_request(request):
            with self.lock:
                self.active_streams[stream_id] = request.token
                if len(self.active_streams) == self.num_parties:
                    self.schedule_learning_iteration_messages()
        else:
            logger.warning(
                f"[{uuid_str(stream_id)}] Invalid initial request: {request}"
            )

    def LearningState(self, stream_id, request):
        """
        Handle the learning state of a client connection.

        Args:
            stream_id (str): The unique identifier for the client stream.
            request: The learning state request from the client.

        Returns:
            ClientState: The next state of the client.
        """
        logger.info(f"[{uuid_str(stream_id)}] Received StoreIDs")
        logger.debug(
            f"[{uuid_str(stream_id)}] Received LearningState request: {request}"
        )

        # Need to protect the shared state with a lock
        with self.lock:
            # Check if the client is in the learning state
            # If the client is in the learning state, store the received values
            # If all clients have sent their values, trigger the aggregation
            # If the client is not in the learning state, return an error
            if self.learning_in_progress:
                if request.batch_id not in self.iteration_values:
                    self.iteration_values[request.batch_id] = {}
                if stream_id not in self.iteration_values[request.batch_id]:
                    self.iteration_values[request.batch_id][stream_id] = request
                    logger.debug(
                        f"[{uuid_str(stream_id)}] LearningState: {len(self.iteration_values[request.batch_id])}/{self.num_parties} values for batch {request.batch_id} received"
                    )
                else:
                    raise ValueError(
                        f"Received repeated batch_id {request.batch_id} for stream_id: {uuid_str(stream_id)}"
                    )
            else:
                return ClientState.END

            # Only current batch can have reached the num_parties
            # We check for this one.

            if len(self.iteration_values[request.batch_id]) == self.num_parties:
                values = self.iteration_values.pop(
                    request.batch_id
                )  # We have all the values for this batch, thus we remove it from the iteration_values
                logger.info(
                    f"[{uuid_str(stream_id)}] Triggering Aggregation for batch {request.batch_id}"
                )
                store_ids = [request.store_id for request in values.values()]
                # Ordered based on client_id: party_id (that's why token is sent on each request).
                party_ids = {
                    self.clients[request.token]: request.party_id
                    for request in values.values()
                }
                logger.debug(party_ids)

                def run_async_task(nillion_server, store_ids, party_ids, batch_id):
                    asyncio.run(nillion_server.compute(store_ids, party_ids, batch_id))

                # In your main program:
                # thread = threading.Thread(target=run_async_task, args=(self.nillion_server, store_ids, party_ids, request.batch_id))
                # self.compute_threads.append(thread)
                # thread.start()
                asyncio.run(
                    self.nillion_server.compute(store_ids, party_ids, request.batch_id)
                )
                self.processed_batches += 1

            if self.processed_batches == self.batches_per_iteration:
                # for thread in self.compute_threads:
                #     thread.join()
                # self.compute_threads = []
                self.processed_batches = 0
                self.schedule_learning_iteration_messages()
                self.learning_complete.set()

        logger.debug(
            f"[{uuid_str(stream_id)}] Finished LearningState request: {len(self.iteration_values)}/{self.num_parties} values received"
        )
        return ClientState.READY

    def ScheduleLearningIteration(self, request_iterator, context):
        """
        Schedule and manage learning iterations for a client.

        Args:
            request_iterator: An iterator of client requests.
            context: The gRPC context.

        Yields:
            Messages to be sent to the client.
        """
        # Create a unique stream id for each client
        stream_id = str(uuid.uuid4())
        self.clients_queue[stream_id] = Queue()

        # Define the client request handler function
        def client_request_handler():
            client_state = ClientState.INITIAL
            logger.debug(
                f"[SERVER][{uuid_str(stream_id)}] Handling client request thread started"
            )
            try:
                for request in request_iterator:
                    if not self.is_valid_token(request.token):
                        logger.warning(
                            f"[SERVER][{uuid_str(stream_id)}] Invalid token: {uuid_str(request.token)}"
                        )
                        continue

                    if client_state is ClientState.INITIAL:
                        self.InitialState(stream_id, request)
                        client_state = ClientState.READY
                    elif client_state is ClientState.READY:
                        client_state = self.LearningState(stream_id, request)
                    if client_state is ClientState.END:
                        break

            except grpc.RpcError as e:
                logger.error(f"[SERVER][{uuid_str(stream_id)}] RPC Error: {e}")
            except Exception as e:
                error = traceback.format_exc()
                logger.error(f"[SERVER][{uuid_str(stream_id)}] Error: {error}")
            finally:
                logger.error(
                    f"[SERVER][{uuid_str(stream_id)}] Handling client request thread stopped"
                )
                self.handle_client_disconnect(stream_id)

        logger.debug(
            f"[SERVER][{uuid_str(stream_id)}] Starting client request handler thread"
        )

        # Start the client request handler thread
        self.client_threads[stream_id] = threading.Thread(
            target=client_request_handler, daemon=True
        )
        self.client_threads[stream_id].start()

        # Server management routine
        logger.debug(f"[SERVER][{uuid_str(stream_id)}] Starting server loop")

        # Main server loop to answer requests from the client
        try:
            while stream_id in self.clients_queue:
                # If there are messages to send to the client, yield them
                if not self.clients_queue[stream_id].empty():
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

    def handle_client_disconnect(self, stream_id):
        """
        Handle client disconnection and clean up resources.

        Args:
            stream_id (str): The unique identifier for the client stream.
        """
        with self.lock:
            if stream_id in self.active_streams:
                token = self.active_streams.pop(stream_id)
                self.clients.pop(token, None)
            if stream_id in self.client_threads:
                self.client_threads.pop(stream_id)
            if stream_id in self.clients_queue:
                self.clients_queue.pop(stream_id)
            for batch_id in self.iteration_values:
                if stream_id in self.iteration_values[batch_id]:
                    self.iteration_values[batch_id].pop(stream_id)
        logger.debug(
            f"[SERVER][{uuid_str(stream_id)}] Client disconnected and cleaned up"
        )
        # If a client disconnects during learning, end the learning for all clients
        self.end_learning_for_all_clients()

    def end_learning_for_all_clients(self):
        """
        End the learning process for all connected clients.
        """
        logger.debug("Ending learning iteration for all connected clients")
        with self.lock:
            self.learning_in_progress = False
            for client_stream_id in self.clients_queue.keys():
                self.clients_queue[client_stream_id].put(
                    fl_pb2.ScheduleRequest(
                        program_id="-1",  # Use -1 to signal end of learning
                        user_id="",
                        batch_size=0,
                        num_parties=0,
                    )
                )
        self.learning_complete.set()
        logger.debug("Ended learning iteration for all connected clients")

    def stop(self):
        """
        Stop all client threads and end the learning process.
        """
        threads = []
        with self.lock:
            self.learning_in_progress = False
            threads = list(self.client_threads.values())
        for thread in threads:
            thread.join()

    def serve(self):
        """
        Start the gRPC server and listen for client connections.
        """
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        fl_pb2_grpc.add_FederatedLearningServiceServicer_to_server(self, server)
        server.add_insecure_port("[::]:50051")

        server.start()
        logger.debug("Server started. Listening on port 50051.")
        try:
            while True:
                time.sleep(86400)
        except KeyboardInterrupt:
            logger.debug("Server stopping...")
            self.stop()
            logger.debug("All client threads stopped")
            server.stop(1)


def main():
    """
    Main function to start the Federated Learning Server.
    """
    FederatedLearningServicer(num_parties=2).serve()


if __name__ == "__main__":
    main()
