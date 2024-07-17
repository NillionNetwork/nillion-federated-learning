import logging
import threading
import time
import uuid
from concurrent import futures
from enum import Enum
from queue import Queue

import grpc
import asyncio

import nillion_fl.fl_net.fl_service_pb2 as fl_pb2
import nillion_fl.fl_net.fl_service_pb2_grpc as fl_pb2_grpc
from nillion_fl.logs import logger, uuid_str
from nillion_fl.nillion_network.server import FedAvgNillionNetworkServer, MAX_SECRET_BATCH_SIZE


class ClientState(Enum):
    INITIAL = 0
    READY = 1
    END = 2


class FederatedLearningServicer(fl_pb2_grpc.FederatedLearningServiceServicer):

    def __init__(self, num_parties):
        # A dictionary mapping a token to a client_id
        self.clients = {}
        # A dictionary mapping a stream id to a token
        self.active_streams = {}
        # A dictionary mapping a stream id to a thread
        self.client_threads = {}
        # A dictionary mapping a stream id to a list of messages to be sent to the specific client
        self.clients_queue = {}
        # A dictionary mapping the current iteration of received values.
        self.iteration_values = {}
        # The total number of parties
        self.num_parties = num_parties
        self.lock = threading.Lock()
        self.learning_in_progress = False
        self.learning_complete = threading.Event()
        self.batch_size = int(MAX_SECRET_BATCH_SIZE / self.num_parties)

        self.nillion_server = FedAvgNillionNetworkServer(num_parties)
        self.nillion_server.compile_program(self.batch_size, self.num_parties)
        self.program_id = asyncio.run(self.nillion_server.store_program())

    def __del__(self):
        self.end_learning_for_all_clients()

    def __str__(self):
        return f"FederatedLearningServicer(\n num_parties={self.num_parties},\n clients={self.clients},\n active_streams={self.active_streams},\n client_threads={self.client_threads},\n clients_queue={self.clients_queue}, \n iteration_values={self.iteration_values},\n learning_in_progress={self.learning_in_progress}\n)"

    def __repr__(self):
        return self.__str__()

    def is_valid_token(self, token):
        return token in self.clients

    def is_initial_request(self, request):
        return (
            len(request.store_ids) == 0
            and request.party_id == ""
            and request.token != ""
        )

    def RegisterClient(self, request, context):
        with self.lock:
            if len(self.clients) >= self.num_parties:
                context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
                context.set_details("The maximum number of clients has been reached.")
                return fl_pb2.ClientInfo(client_id=-1, token="")

            client_id = len(self.clients)
            token = str(uuid.uuid4())
            self.clients[token] = client_id
            logger.info(
                f"[SERVER] Registering client [{client_id}] with token: {uuid_str(token)}"
            )
            return fl_pb2.ClientInfo(client_id=client_id, token=token)

    def schedule_learning_iteration_messages(self):
        logger.debug("Scheduling learning iteration for all connected clients")
        with self.lock:
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
        logger.info("Scheduled learning iteration for all connected clients")

    def InitialState(self, stream_id, request):
        logger.info(
            f"[SERVER][{uuid_str(stream_id)}] Received initial request: {request}"
        )
        if self.is_initial_request(request):
            active_streams = 0
            with self.lock:
                self.active_streams[stream_id] = request.token
                active_streams = len(self.active_streams)

            if active_streams == self.num_parties:
                print(len(self.clients_queue), self.num_parties)
                self.schedule_learning_iteration_messages()
        else:
            logger.warning(
                f"[SERVER][{uuid_str(stream_id)}] Invalid initial request: {request}"
            )

    def LearningState(self, stream_id, request):
        logger.info(
            f"[SERVER][{uuid_str(stream_id)}] Received store ids: {request.store_ids}"
        )
        with self.lock:
            if self.learning_in_progress:
                if stream_id not in self.iteration_values:
                    self.iteration_values[stream_id] = request
                else:
                    raise ValueError("Received store ids before scheduling learning")
            else:
                return ClientState.END

        if len(self.iteration_values) == self.num_parties:
            store_ids = []
            with self.lock:
                store_ids = [request.store_ids[0] for request in self.iteration_values.values()]
            asyncio.run(self.nillion_server.compute(store_ids))
            self.schedule_learning_iteration_messages()
            # Signal that this client has completed its part
            self.learning_complete.set()
            self.iteration_values = {}
        return ClientState.READY

    def ScheduleLearningIteration(self, request_iterator, context):
        stream_id = str(uuid.uuid4())
        self.clients_queue[stream_id] = Queue()

        def client_request_handler():
            client_state = ClientState.INITIAL
            logger.info(
                f"[SERVER][{uuid_str(stream_id)}] Handling client request thread started"
            )
            try:
                print("A;,", request_iterator)
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
                logger.error(f"[SERVER][{uuid_str(stream_id)}] Error: {e}")
            finally:
                logger.error(
                    f"[SERVER][{uuid_str(stream_id)}] Handling client request thread stopped"
                )
                self.handle_client_disconnect(stream_id)

        logger.info(
            f"[SERVER][{uuid_str(stream_id)}] Starting client request handler thread"
        )
        self.client_threads[stream_id] = threading.Thread(
            target=client_request_handler, daemon=True
        )
        self.client_threads[stream_id].start()

        logger.info(f"[SERVER][{uuid_str(stream_id)}] Starting server loop")
        try:
            while stream_id in self.clients_queue:
                if not self.clients_queue[stream_id].empty():
                    message = self.clients_queue[stream_id].get(timeout=5)
                    logger.info(
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
        with self.lock:
            if stream_id in self.active_streams:
                token = self.active_streams.pop(stream_id)
                self.clients.pop(token, None)
            if stream_id in self.client_threads:
                self.client_threads.pop(stream_id)
            if stream_id in self.clients_queue:
                self.clients_queue.pop(stream_id)

            logger.debug(f"[SERVER] {self} \n")

        logger.info(
            f"[SERVER][{uuid_str(stream_id)}] Client disconnected and cleaned up"
        )
        # If a client disconnects during learning, end the learning for all clients
        self.end_learning_for_all_clients()

    def end_learning_for_all_clients(self):
        logger.info("Ending learning iteration for all connected clients")
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
        logger.info("Ended learning iteration for all connected clients")

    def stop(self):
        threads = []
        with self.lock:
            self.learning_in_progress = False
            threads = list(self.client_threads.values())
        for thread in threads:
            thread.join()

    def serve(self):
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        fl_pb2_grpc.add_FederatedLearningServiceServicer_to_server(self, server)
        server.add_insecure_port("[::]:50051")

        server.start()
        logger.info("Server started. Listening on port 50051.")
        try:
            while True:
                time.sleep(86400)
        except KeyboardInterrupt:
            logger.info("Server stopping...")
            self.stop()
            logger.info("All client threads stopped")
            server.stop(1)


if __name__ == "__main__":
    FederatedLearningServicer(num_parties=2).serve()
