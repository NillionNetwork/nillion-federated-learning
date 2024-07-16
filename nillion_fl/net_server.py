import grpc
from concurrent import futures
import time
import uuid
import threading
import logging
from queue import Queue

import nillion_fl.fl_net.fl_service_pb2 as fl_pb2
import nillion_fl.fl_net.fl_service_pb2_grpc as fl_pb2_grpc

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
        # The total number of parties
        self.num_parties = num_parties

    def is_valid_token(self, token):
        return token in self.clients

    def is_initial_request(self, request):
        return (
            len(request.store_ids) == 0
            and request.party_id == ""
            and request.token != ""
        )

    def RegisterClient(self, request, context):
        if len(self.clients) >= self.num_parties:
            context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
            context.set_details("The maximum number of clients has been reached.")
            return fl_pb2.ClientInfo(client_id=-1, token="")

        client_id = len(self.clients) + 1
        token = str(uuid.uuid4())
        self.clients[token] = client_id
        return fl_pb2.ClientInfo(client_id=client_id, token=token)

    def ScheduleLearningIteration(self, request_iterator, context):
        stream_id = str(uuid.uuid4())
        self.clients_queue[stream_id] = Queue()

        def client_request_handler():
            for request in request_iterator:
                if not self.is_valid_token(request.token):
                    logger.warning(
                        f"[SERVER][{stream_id}] Invalid token: {request.token}"
                    )
                    continue

                if self.is_initial_request(request):
                    self.active_streams[stream_id] = request.token
                    logger.info(
                        f"[SERVER][{stream_id}] Received initial request: {request}"
                    )
                    if len(self.clients_queue) == self.num_parties:
                        for client_stream_id in self.clients_queue.keys():
                            self.clients_queue[client_stream_id].put(
                                fl_pb2.ScheduleRequest(
                                    program_id=stream_id,
                                    user_id="user_456",
                                    batch_size=32,
                                    num_parties=5,
                                )
                            )
                else:
                    logger.info(
                        f"[SERVER][{stream_id}] Received store ids: {request.store_ids}"
                    )
                    self.clients_queue[stream_id].put(
                        fl_pb2.ScheduleRequest(
                            program_id=stream_id,
                            user_id="user_456",
                            batch_size=32,
                            num_parties=5,
                        )
                    )

        logger.info("[SERVER] Starting client request handler thread")
        # Start a new thread to listen for incoming requests
        self.client_threads[stream_id] = threading.Thread(
            target=client_request_handler, daemon=True
        )
        self.client_threads[stream_id].start()

        try:
            while stream_id in self.clients_queue:
                if not self.clients_queue[stream_id].empty():
                    message = self.clients_queue[stream_id].get()
                    logger.info(f"[SERVER][{stream_id}] Sending message: {message}")
                    yield message
                time.sleep(
                    5
                )  # Wait for 5 seconds before trying again with the next batch
        finally:
            self.active_streams.pop(stream_id)
            self.client_threads[stream_id].join()
            self.client_threads.pop(stream_id)
            self.clients_queue.pop(stream_id)
            self.clients.pop(stream_id)

    def serve(self):
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        fl_pb2_grpc.add_FederatedLearningServiceServicer_to_server(
            FederatedLearningServicer(num_parties=2), server
        )
        server.add_insecure_port("[::]:50051")

        server.start()
        logger.info("Server started. Listening on port 50051.")
        try:
            while True:
                time.sleep(86400)
        except KeyboardInterrupt:
            logger.info("Server stopping...")
            server.stop(0)


if __name__ == "__main__":
    FederatedLearningServicer(num_parties=5).serve()
