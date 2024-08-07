""" Manages client connections and streams for the server. """

import threading
import uuid
from queue import Queue

import grpc

import nillion_fl.network.fl_service_pb2 as fl_pb2
from nillion_fl.logs import logger, uuid_str


class ClientManager:
    """
    Manages client connections and streams for the server.

    Attributes:
        num_parties (int): The maximum number of clients that can be registered.
        clients (dict): A dictionary mapping client tokens to client IDs.
        active_streams (dict): A dictionary mapping stream IDs to client tokens.
        client_threads (dict): A dictionary mapping stream IDs to client threads.
        clients_queue (dict): A dictionary mapping stream IDs to message queues.
        lock (threading.Lock): A lock for synchronizing access to shared resources.
    """

    def __init__(self, num_parties):
        """
        Initializes the ClientManager with the specified number of parties.

        Args:
            num_parties (int): The maximum number of clients that can be registered.
        """
        self.num_parties = num_parties
        self.clients = {}
        self.active_streams = {}
        self.client_threads = {}
        self.clients_queue = {}
        self.lock = threading.Lock()

    # pylint: disable=unused-argument
    def register_client(self, request, context):
        """
        Registers a new client if the maximum number of clients has not been reached.

        Args:
            request: The client registration request.
            context: The gRPC context for setting response codes and details.

        Returns:
            fl_pb2.ClientInfo: Information about the registered client or an error message.
        """

        with self.lock:
            if len(self.clients) >= self.num_parties:
                context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
                context.set_details("The maximum number of clients has been reached.")
                return fl_pb2.ClientInfo(  # fmt: off # pylint: disable=no-member
                    client_id=-1, token="", num_parties=self.num_parties
                )

            client_id = len(self.clients)
            token = str(uuid.uuid4())
            self.clients[token] = client_id
            logger.debug(
                "[SERVER] Registering client [%d] with token: %s",
                client_id,
                uuid_str(token),
            )
            return fl_pb2.ClientInfo(  # fmt: off # pylint: disable=no-member
                client_id=client_id, token=token, num_parties=self.num_parties
            )

    @property
    def stream_ids(self):
        """
        Returns:
            list: List of stream IDs that are currently active.
        """
        return list(self.clients_queue.keys())

    def new_client(self, stream_id):
        """
        Creates a new message queue for a client.

        Args:
            stream_id (str): The ID of the stream for the new client.
        """
        self.clients_queue[stream_id] = Queue()

    def get_client_id(self, token):
        """
        Retrieves the client ID for a given token.

        Args:
            token (str): The token of the client.

        Returns:
            int: The client ID associated with the token.
        """
        with self.lock:
            return self.clients[token]

    def new_stream(self, stream_id, token):
        """
        Registers a new stream with an associated token.

        Args:
            stream_id (str): The ID of the new stream.
            token (str): The token of the client associated with the stream.
        """
        with self.lock:
            self.active_streams[stream_id] = token

    def is_client(self, stream_id):
        """
        Checks if a stream ID is associated with a registered client.

        Args:
            stream_id (str): The ID of the stream.

        Returns:
            bool: True if the stream ID is associated with a client, False otherwise.
        """
        with self.lock:
            return stream_id in self.clients_queue

    def has_messages(self, stream_id):
        """
        Checks if there are any messages in the queue for a given stream ID.

        Args:
            stream_id (str): The ID of the stream.

        Returns:
            bool: True if there are messages in the queue, False otherwise.
        """
        with self.lock:
            return not self.clients_queue[stream_id].empty()

    def get_last_message(self, stream_id):
        """
        Retrieves the last message from the queue for a given stream ID.

        Args:
            stream_id (str): The ID of the stream.

        Returns:
            The last message in the queue.

        Raises:
            queue.Empty: If the queue is empty or if no message is found within the timeout period.
        """
        with self.lock:
            return self.clients_queue[stream_id].get(timeout=5)

    def is_valid_token(self, token):
        """
        Checks if a token is valid (i.e., registered).

        Args:
            token (str): The token to check.

        Returns:
            bool: True if the token is valid, False otherwise.
        """
        return token in self.clients

    def end_stream(self, stream_id):
        """
        Ends a stream and waits for the associated client thread to finish.

        Args:
            stream_id (str): The ID of the stream to end.
        """
        with self.lock:
            thread = self.client_threads[stream_id]
        if thread:
            thread.join()

    def end_all_streams(self):
        """
        Ends all active streams and waits for all associated client threads to finish.
        """
        with self.lock:
            threads = list(self.client_threads.values())
        for thread in threads:
            thread.join()

    def send(self, stream_id, message):
        """
        Sends a message to the queue of a given stream ID.

        Args:
            stream_id (str): The ID of the stream.
            message: The message to send.
        """
        with self.lock:
            self.clients_queue[stream_id].put(message)

    def handle_client_disconnect(self, stream_id):
        """
        Handles a client disconnection by cleaning up associated resources.

        Args:
            stream_id (str): The ID of the disconnected stream.
        """
        with self.lock:
            if stream_id in self.active_streams:
                token = self.active_streams.pop(stream_id)
                self.clients.pop(token, None)
            if stream_id in self.client_threads:
                self.client_threads.pop(stream_id)
            if stream_id in self.clients_queue:
                self.clients_queue.pop(stream_id)
        logger.debug(
            "[SERVER][%s] Client disconnected and cleaned up", uuid_str(stream_id)
        )
