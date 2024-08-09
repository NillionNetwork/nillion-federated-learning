""" Manages the learning iterations and client interactions during training. """

import threading
import traceback
import uuid

import grpc

import nillion_fl.network.fl_service_pb2 as fl_pb2
from nillion_fl.core.server.client_state import ClientState
from nillion_fl.logs import logger, uuid_str


class LearningManager:
    """
    Manages the learning iterations and client interactions during training.

    Attributes:
        num_parties (int): The number of parties involved in the learning process.
        batch_size (int): The size of each batch in the learning process.
        iteration_values (dict): Stores values for each batch ID during learning.
        processed_batches (int): Number of processed batches in the current iteration.
        batches_per_iteration (int): Number of batches to process per iteration.
        learning_in_progress (bool): Indicates if learning is currently in progress.
        learning_complete (threading.Event): Event used to signal completion of learning.
        lock (threading.Lock): Lock for synchronizing access to shared resources.
        client_manager (ClientManager): Manages client connections and communication.
        nillion_integration (NillionIntegration): Handles integration with Nillion network.
        __model_size (int): The size of the model being processed.
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(self, num_parties, batch_size, client_manager, nillion_integration):
        """
        Initializes the LearningManager with the given parameters.

        Args:
            num_parties (int): The number of parties involved in the learning process.
            batch_size (int): The size of each batch in the learning process.
            client_manager (ClientManager): Manages client connections and communication.
            nillion_integration (NillionIntegration): Handles integration with Nillion network.
        """
        self.num_parties = num_parties
        self.batch_size = batch_size
        self.iteration_values = {}
        self.processed_batches = 0
        self.batches_per_iteration = None
        self.learning_in_progress = False
        self.learning_complete = threading.Event()
        self.lock = threading.Lock()

        self.client_manager = client_manager
        self.nillion_integration = nillion_integration

        self.__model_size = None

    @property
    def model_size(self):
        """
        Returns the size of the model.

        Returns:
            int: The size of the model.
        """
        return self.__model_size

    @model_size.setter
    def model_size(self, value):
        """
        Sets the size of the model and calculates the number of batches per iteration.

        Args:
            value (int): The size of the model.

        Raises:
            ValueError: If the model size is changed after being set.
        """
        if self.__model_size is None:
            self.__model_size = value
            self.batches_per_iteration, remainder = divmod(
                self.model_size, self.batch_size
            )
            if remainder != 0:
                self.batches_per_iteration += 1
        elif self.__model_size != value:
            raise ValueError("Model size cannot be changed once set")

    # pylint: disable=unused-argument
    def schedule_learning_iteration(self, request_iterator, context):
        """
        Schedules a new learning iteration and starts handling client requests.

        Args:
            request_iterator: Iterator of client requests.
            context: The gRPC context for setting response codes and details.

        Returns:
            str: The ID of the new client stream.
        """
        stream_id = str(uuid.uuid4())
        self.client_manager.new_client(stream_id)

        def client_request_handler():
            client_state = ClientState.INITIAL
            logger.debug(
                "[SERVER][%s] Handling client request thread started",
                uuid_str(stream_id),
            )
            try:
                for request in request_iterator:
                    if not self.client_manager.is_valid_token(request.token):
                        logger.warning(
                            "[SERVER][%s] Invalid token: %s",
                            uuid_str(stream_id),
                            uuid_str(request.token),
                        )
                        continue

                    if client_state is ClientState.INITIAL:
                        self.handle_initial_state(stream_id, request)
                        client_state = ClientState.READY
                    elif client_state is ClientState.READY:
                        client_state = self.handle_learning_state(stream_id, request)
                    if client_state is ClientState.END:
                        break
            except grpc.RpcError as e:
                logger.error("[SERVER][%s] RPC Error: %s", uuid_str(stream_id), e)
            except Exception:  # pylint: disable=broad-exception-caught
                error = traceback.format_exc()
                logger.error("[SERVER][%s] Error: %s", uuid_str(stream_id), error)
            finally:
                logger.error(
                    "[SERVER][%s] Handling client request thread stopped",
                    uuid_str(stream_id),
                )
                self.handle_client_disconnect(stream_id)

        thread = threading.Thread(target=client_request_handler, daemon=True)
        self.client_manager.client_threads[stream_id] = thread
        thread.start()
        return stream_id

    def learning_iteration_message_loop(self, stream_id):
        """
        Handles the message loop for a learning iteration,
        sending messages to clients and handling timeouts.

        Args:
            stream_id (str): The ID of the client stream.

        Yields:
            Message: Messages to be sent to the client.
        """
        logger.debug("[SERVER][%s] Starting server loop", uuid_str(stream_id))
        try:
            while self.client_manager.is_client(stream_id):
                if self.client_manager.has_messages(stream_id):
                    message = self.client_manager.get_last_message(stream_id)
                    logger.debug(
                        "[SERVER][%s] Sending message: %s", uuid_str(stream_id), message
                    )
                    yield message

                    if not self.learning_complete.wait():
                        logger.warning(
                            "[SERVER][%s] Learning iteration timed out",
                            uuid_str(stream_id),
                        )
                        self.client_manager.end_stream(stream_id)
                        self.end_learning_for_all_clients()

                    self.learning_complete.clear()
        except grpc.RpcError as e:
            logger.error(
                "[SERVER][%s] RPC Error in message sending: %s", uuid_str(stream_id), e
            )
        finally:
            self.handle_client_disconnect(stream_id)

    def handle_initial_state(self, stream_id, request):
        """
        Handles the initial state of a client,
        setting up necessary data and scheduling learning messages.

        Args:
            stream_id (str): The ID of the client stream.
            request: The initial client request.
        """
        self.client_manager.new_stream(stream_id, request.token)
        with self.lock:
            if len(self.client_manager.active_streams) == self.num_parties:
                self.schedule_learning_iteration_messages()

    def handle_learning_state(self, stream_id, request):
        """
        Handles the learning state of a client, processing batch data and triggering aggregation.

        Args:
            stream_id (str): The ID of the client stream.
            request: The client request containing batch data.

        Returns:
            ClientState: The new state of the client.
        """
        with self.lock:
            if self.learning_in_progress:
                if request.batch_id not in self.iteration_values:
                    self.iteration_values[request.batch_id] = {}
                if stream_id not in self.iteration_values[request.batch_id]:
                    self.iteration_values[request.batch_id][stream_id] = request
                    logger.debug(
                        "[%s] LearningState: %d/%d values for batch %d received",
                        uuid_str(stream_id),
                        len(self.iteration_values[request.batch_id]),
                        self.num_parties,
                        request.batch_id,
                    )
                else:
                    raise ValueError(
                        f"Received repeated batch_id {request.batch_id} for stream_id:{uuid_str(stream_id)}"  # fmt: off # pylint: disable=line-too-long
                    )
            else:
                return ClientState.END

            if len(self.iteration_values[request.batch_id]) == self.num_parties:
                self.processed_batches += 1
                logger.info(
                    "[%s] Triggering Aggregation for batch %d [%d/%d]",
                    uuid_str(stream_id),
                    request.batch_id,
                    self.processed_batches,
                    self.batches_per_iteration,
                )
                values = self.iteration_values.pop(request.batch_id)
                store_ids = [request.store_id for request in values.values()]
                party_ids = {
                    self.client_manager.get_client_id(request.token): request.party_id
                    for request in values.values()
                }
                self.nillion_integration.compute(store_ids, party_ids, request.batch_id)

            if self.processed_batches == self.batches_per_iteration:
                logger.info("All batches processed. Scheduling next iteration")
                self.processed_batches = 0
                self.schedule_learning_iteration_messages()
                self.learning_complete.set()

        return ClientState.READY

    def schedule_learning_iteration_messages(self):
        """
        Schedules messages for all clients to start a new learning iteration.
        """
        self.learning_in_progress = True
        for client_stream_id in self.client_manager.stream_ids:
            self.client_manager.send(
                client_stream_id,
                fl_pb2.ScheduleRequest(  # fmt: off # pylint: disable=no-member
                    program_id=self.nillion_integration.program_id,
                    user_id=self.nillion_integration.user_id,
                    batch_size=self.batch_size,
                    num_parties=self.num_parties,
                ),
            )

    def handle_client_disconnect(self, stream_id):
        """
        Handles a client disconnection, cleaning up associated data.

        Args:
            stream_id (str): The ID of the disconnected client stream.
        """
        with self.lock:
            for batch_iteration_values in self.iteration_values.items():
                if stream_id in batch_iteration_values:
                    batch_iteration_values.pop(stream_id)

        self.client_manager.handle_client_disconnect(stream_id)

    def end_learning_for_all_clients(self):
        """
        Ends the learning process for all clients and sends termination messages.

        """
        with self.lock:
            self.learning_in_progress = False
            for client_stream_id in self.client_manager.stream_ids:
                self.client_manager.send(
                    client_stream_id,
                    fl_pb2.ScheduleRequest(  # fmt: off # pylint: disable=no-member
                        program_id="-1",
                        user_id="",
                        batch_size=0,
                        num_parties=0,
                    ),
                )
        self.learning_complete.set()

    def stop(self):
        """
        Stops the learning process and ends all client streams.
        """
        with self.lock:
            self.learning_in_progress = False
        self.client_manager.end_all_streams()
