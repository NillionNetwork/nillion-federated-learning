import threading
import traceback
import uuid
from queue import Queue

import grpc

import nillion_fl.network.fl_service_pb2 as fl_pb2
from nillion_fl.core.server.client_state import ClientState
from nillion_fl.logs import logger, uuid_str


class LearningManager:
    def __init__(self, num_parties, batch_size, client_manager, nillion_integration):
        self.num_parties = num_parties
        self.batch_size = batch_size
        self.iteration_values = {}
        self.processed_batches = (
            0  # Number of processed batches in the current iteration
        )
        self.batches_per_iteration: int = (
            None  # The number of batches we need to process per iteration
        )
        self.learning_in_progress = False
        self.learning_complete = threading.Event()
        self.lock = threading.Lock()

        self.client_manager = client_manager
        self.nillion_integration = nillion_integration

        self.__model_size = None

    @property
    def model_size(self):
        return self.__model_size

    @model_size.setter
    def model_size(self, value):
        if self.__model_size is None:
            self.__model_size = value
            self.batches_per_iteration, sum_1 = divmod(self.model_size, self.batch_size)
            if sum_1 != 0:
                self.batches_per_iteration += 1
        elif self.__model_size != value:
            raise ValueError("Model size cannot be changed once set")

    def schedule_learning_iteration(self, request_iterator, context):
        stream_id = str(uuid.uuid4())
        self.client_manager.new_client(stream_id)

        def client_request_handler():
            client_state = ClientState.INITIAL
            logger.debug(
                f"[SERVER][{uuid_str(stream_id)}] Handling client request thread started"
            )
            try:
                for request in request_iterator:
                    if not self.client_manager.is_valid_token(request.token):
                        logger.warning(
                            f"[SERVER][{uuid_str(stream_id)}] Invalid token: {uuid_str(request.token)}"
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
                logger.error(f"[SERVER][{uuid_str(stream_id)}] RPC Error: {e}")
            except Exception as e:
                error = traceback.format_exc()
                logger.error(f"[SERVER][{uuid_str(stream_id)}] Error: {error}")
            finally:
                logger.error(
                    f"[SERVER][{uuid_str(stream_id)}] Handling client request thread stopped"
                )
                self.handle_client_disconnect(stream_id)

        self.client_manager.client_threads[stream_id] = threading.Thread(
            target=client_request_handler, daemon=True
        )
        self.client_manager.client_threads[stream_id].start()
        return stream_id

    def learning_iteration_message_loop(self, stream_id):
        logger.debug(f"[SERVER][{uuid_str(stream_id)}] Starting server loop")
        try:
            while self.client_manager.is_client(stream_id):
                if self.client_manager.has_messages(stream_id):
                    message = self.client_manager.get_last_message(stream_id)
                    logger.debug(
                        f"[SERVER][{uuid_str(stream_id)}] Sending message: {message}"
                    )
                    yield message

                    if not self.learning_complete.wait(timeout=180):
                        logger.warning(
                            f"[SERVER][{uuid_str(stream_id)}] Learning iteration timed out"
                        )
                        self.client_manager.end_stream(stream_id)

                        self.end_learning_for_all_clients()

                    self.learning_complete.clear()
        except grpc.RpcError as e:
            logger.error(
                f"[SERVER][{uuid_str(stream_id)}] RPC Error in message sending: {e}"
            )
        finally:
            self.handle_client_disconnect(stream_id)
        return None

    def handle_initial_state(self, stream_id, request):
        self.client_manager.new_stream(stream_id, request.token)
        with self.lock:
            if len(self.client_manager.active_streams) == self.num_parties:
                self.schedule_learning_iteration_messages()

    def handle_learning_state(self, stream_id, request):
        with self.lock:
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

            if len(self.iteration_values[request.batch_id]) == self.num_parties:
                self.processed_batches += 1
                logger.info(
                    f"[{uuid_str(stream_id)}] Triggering Aggregation for batch {request.batch_id} [{self.processed_batches}/{self.batches_per_iteration}]"
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
        self.learning_in_progress = True
        for client_stream_id in self.client_manager.stream_ids:
            self.client_manager.clients_queue[client_stream_id].put(
                fl_pb2.ScheduleRequest(
                    program_id=self.nillion_integration.program_id,
                    user_id=self.nillion_integration.user_id,
                    batch_size=self.batch_size,
                    num_parties=self.num_parties,
                )
            )

    def handle_client_disconnect(self, stream_id):
        with self.lock:
            for batch_id in self.iteration_values:
                if stream_id in self.iteration_values[batch_id]:
                    self.iteration_values[batch_id].pop(stream_id)

        self.client_manager.handle_client_disconnect(stream_id)

    def end_learning_for_all_clients(self):
        with self.lock:
            self.learning_in_progress = False
            for client_stream_id in self.client_manager.stream_ids:
                self.client_manager.send(
                    client_stream_id,
                    fl_pb2.ScheduleRequest(
                        program_id="-1",
                        user_id="",
                        batch_size=0,
                        num_parties=0,
                    ),
                )
        self.learning_complete.set()

    def stop(self):
        threads = []
        with self.lock:
            self.learning_in_progress = False
        self.client_manager.end_all_streams()
