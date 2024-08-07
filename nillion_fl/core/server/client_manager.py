import threading
import uuid
from queue import Queue

import grpc

import nillion_fl.network.fl_service_pb2 as fl_pb2
from nillion_fl.logs import logger, uuid_str


class ClientManager:
    def __init__(self, num_parties):
        self.num_parties = num_parties
        self.clients = {}
        self.active_streams = {}
        self.client_threads = {}
        self.clients_queue = {}
        self.lock = threading.Lock()

    def register_client(self, request, context):
        with self.lock:
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
            return (
                request.model_size,
                fl_pb2.ClientInfo(
                    client_id=client_id, token=token, num_parties=self.num_parties
                ),
            )

    @property
    def stream_ids(self):
        return list(self.clients_queue.keys())

    def new_client(self, stream_id):
        self.clients_queue[stream_id] = Queue()

    def get_client_id(self, token):
        with self.lock:
            return self.clients[token]

    def new_stream(self, stream_id, token):
        with self.lock:
            self.active_streams[stream_id] = token

    def is_client(self, stream_id):
        with self.lock:
            return stream_id in self.clients_queue

    def has_messages(self, stream_id):
        with self.lock:
            return not self.clients_queue[stream_id].empty()

    def get_last_message(self, stream_id):
        with self.lock:
            return self.clients_queue[stream_id].get(timeout=5)

    def is_valid_token(self, token):
        return token in self.clients

    def end_stream(self, stream_id):
        with self.lock:
            thread = self.client_manager.client_threads[stream_id]
        thread.join()

    def end_all_streams(self):
        with self.lock:
            threads = list(self.client_threads.values())
        for thread in threads:
            thread.join()

    def send(self, stream_id, message):
        with self.lock:
            self.clients_queue[stream_id].put(message)

    def handle_client_disconnect(self, stream_id):
        with self.lock:
            if stream_id in self.active_streams:
                token = self.active_streams.pop(stream_id)
                self.clients.pop(token, None)
            if stream_id in self.client_threads:
                self.client_threads.pop(stream_id)
            if stream_id in self.clients_queue:
                self.clients_queue.pop(stream_id)
        logger.debug(
            f"[SERVER][{uuid_str(stream_id)}] Client disconnected and cleaned up"
        )

    # Add other client management methods as needed
