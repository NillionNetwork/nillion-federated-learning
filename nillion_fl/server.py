import logging
import threading
import time
import uuid
from concurrent import futures
from queue import Queue

import grpc

import nillion_fl.fl_net.fl_service_pb2 as fl_pb2
import nillion_fl.fl_net.fl_service_pb2_grpc as fl_pb2_grpc

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="[%(levelname)s] [%(asctime)s] - %(name)s - %(message)s"
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
        self.lock = threading.Lock()
        self.learning_in_progress = False
        self.learning_complete = threading.Event()

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

            client_id = len(self.clients) + 1
            token = str(uuid.uuid4())
            self.clients[token] = client_id
            return fl_pb2.ClientInfo(client_id=client_id, token=token)

    def ScheduleLearningIteration(self, request_iterator, context):
        stream_id = str(uuid.uuid4())
        self.clients_queue[stream_id] = Queue()

        def client_request_handler():
            try:
                for request in request_iterator:
                    if not self.is_valid_token(request.token):
                        logger.warning(
                            f"[SERVER][{stream_id}] Invalid token: {request.token}"
                        )
                        continue

                    if self.is_initial_request(request):
                        with self.lock:
                            self.active_streams[stream_id] = request.token
                        logger.info(
                            f"[SERVER][{stream_id}] Received initial request: {request}"
                        )
                        if len(self.clients_queue) == self.num_parties:
                            self.schedule_learning_iteration()
                    else:
                        logger.info(
                            f"[SERVER][{stream_id}] Received store ids: {request.store_ids}"
                        )
                        with self.lock:
                            if self.learning_in_progress:
                                self.clients_queue[stream_id].put(
                                    fl_pb2.ScheduleRequest(
                                        program_id=stream_id,
                                        user_id="user_456",
                                        batch_size=int(5000 / self.num_parties),
                                        num_parties=self.num_parties,
                                    )
                                )
                                # Signal that this client has completed its part
                                self.learning_complete.set()
                            else:
                                logger.warning(
                                    f"[SERVER][{stream_id}] Received store ids when learning is not in progress"
                                )
            except grpc.RpcError as e:
                logger.error(f"[SERVER][{stream_id}] RPC Error: {e}")
            finally:
                logger.error(
                    f"[SERVER][{stream_id}] Handling client request thread stopped"
                )
                self.handle_client_disconnect(stream_id)

        logger.info(f"[SERVER][{stream_id}] Starting client request handler thread")
        self.client_threads[stream_id] = threading.Thread(
            target=client_request_handler, daemon=True
        )
        self.client_threads[stream_id].start()

        try:
            while stream_id in self.clients_queue:
                if not self.clients_queue[stream_id].empty():
                    message = self.clients_queue[stream_id].get(timeout=5)
                    logger.info(f"[SERVER][{stream_id}] Sending message: {message}")
                    yield message

                    # Wait for learning to complete or timeout
                    if not self.learning_complete.wait():  # 60 seconds timeout
                        logger.warning(
                            f"[SERVER][{stream_id}] Learning iteration timed out"
                        )
                        self.client_threads[stream_id].join()
                        self.end_learning_for_all_clients()

                    # Reset the event for the next iteration
                    self.learning_complete.clear()

        except grpc.RpcError as e:
            logger.error(f"[SERVER][{stream_id}] RPC Error in message sending: {e}")
        finally:
            self.handle_client_disconnect(stream_id)

    def schedule_learning_iteration(self):
        with self.lock:
            self.learning_in_progress = True
            for client_stream_id in self.clients_queue.keys():
                self.clients_queue[client_stream_id].put(
                    fl_pb2.ScheduleRequest(
                        program_id=str(uuid.uuid4()),
                        user_id="user_456",
                        batch_size=32,
                        num_parties=5,
                    )
                )
        logger.info("Scheduled learning iteration for all connected clients")

    def handle_client_disconnect(self, stream_id):
        with self.lock:
            if stream_id in self.active_streams:
                token = self.active_streams.pop(stream_id)
                self.clients.pop(token, None)
            if stream_id in self.client_threads:
                self.client_threads.pop(stream_id)
            if stream_id in self.clients_queue:
                self.clients_queue.pop(stream_id)

            print(f"Active streams: {self.active_streams}")
            print(f"Clients: {self.clients}")
            print(f"Client threads: {self.client_threads}")
            print(f"Clients queue: {self.clients_queue}")

        # If a client disconnects during learning, end the learning for all clients
        self.end_learning_for_all_clients()

        logger.info(f"[SERVER][{stream_id}] Client disconnected and cleaned up")

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
            for _, thread in self.client_threads.items():
                thread.join()
            logger.info("All client threads stopped")
            server.stop(1)


if __name__ == "__main__":
    FederatedLearningServicer(num_parties=2).serve()
