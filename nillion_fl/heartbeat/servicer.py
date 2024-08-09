# heartbeat_server.py
import time
from typing import Dict
from concurrent import futures
import grpc
import nillion_fl.heartbeat.network.heartbeat_pb2 as hb_pb2
import nillion_fl.heartbeat.network.heartbeat_pb2_grpc as hb_grpc
import threading
import uuid
from nillion_fl.logs import logger, uuid_str
from queue import Queue

from nillion_fl.logs import logger


def handle_200(
    stream_id: str, client_queues: Dict[str, Queue], event: threading.Event
) -> None:
    #logger.debug("[%s] Heartbeat received from client", uuid_str(stream_id))
    client_queues[stream_id].put(
        hb_pb2.Heartbeat(status_code=200, msg="Heartbeat received")
    )


def handle_400(
    stream_id: str, client_queues: Dict[str, Queue], event: threading.Event
) -> None:
    #logger.debug("[%s] Heartbeat received from client", uuid_str(stream_id))
    event.set()
    for client_stream_id, client_queue in client_queues.items():
        if stream_id != client_stream_id:
            client_queue.put(
                hb_pb2.Heartbeat(status_code=400, msg="Client Disconnected")
            )


def handle_500(
    stream_id: str, client_queues: Dict[str, Queue], event: threading.Event
) -> None:
    #logger.debug("[%s] Heartbeat received from client", uuid_str(stream_id))
    event.set()
    for client_queue in client_queues.values():
        client_queue.put(
            hb_pb2.Heartbeat(status_code=500, msg="Server Disconnecting")
        )


STATUS_CODES = {
    200: handle_200,  # Ok
    400: handle_400,  # Client disconnected
    500: handle_500,  # Server disconnected
}


class HeartbeatServicer(hb_grpc.HeartbeatServiceServicer):
    def __init__(
        self, host="localhost", port=50052, heartbeat_interval=5, heartbeat_timeout=10
    ):

        self.host = host
        self.port = port

        self.client_queues = {}
        self.threads = {}
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.heartbeat_monitor_thread = None

    def client_handler(self, stream_id, request_iterator, context):
        logger.debug("Client handler: %s", uuid_str(stream_id))
        for request in request_iterator:
            status_code = request.status_code
            logger.info("[%s] Heartbeat received: %s", uuid_str(stream_id), status_code)
            handle = STATUS_CODES.get(status_code)
            with self.lock:
                handle(stream_id, self.client_queues, self.stop_event)

            if self.stop_event.is_set():
                break

    def SubscribeToHeartbeat(self, request_iterator, context):
        stream_id = str(uuid.uuid4())

        logger.debug("Client connected: %s", uuid_str(stream_id))
        stream_queue = Queue()
        with self.lock:
            self.client_queues[stream_id] = stream_queue
            self.threads[stream_id] = threading.Thread(
                target=self.client_handler, args=(stream_id, request_iterator, context)
            )
        logger.debug("Starting client thread: %s", uuid_str(stream_id))
        self.threads[stream_id].start()

        while not self.stop_event.is_set() or not stream_queue.empty():
            if not stream_queue.empty():
                logger.error("[%s] Heartbeat response sent", uuid_str(stream_id))
                yield stream_queue.get()
        self.threads[stream_id].join()

    def terminate(self):
        with self.lock:
            handle_500(None, self.client_queues, self.stop_event)

    def run_heartbeat_monitor(self):
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        hb_grpc.add_HeartbeatServiceServicer_to_server(self, server)
        server.add_insecure_port(f"{self.host}:{self.port}")
        server.start()
        logger.info(
            "Heartbeat server started. Listening on host: %s port: %s.",
            self.host,
            self.port,
        )

        logger.debug("Heartbeat server running...")
        
        while not self.stop_event.is_set():
            time.sleep(1)
        logger.info("Heartbeat server stopped")
        server.stop(grace=1)  # Stop the server

    def serve(self):
        # Run the heartbeat monitor in a separate thread
        self.heartbeat_monitor_thread = threading.Thread(
            target=self.run_heartbeat_monitor, daemon=True
        )
        self.heartbeat_monitor_thread.start()

        self.heartbeat_monitor_thread.join()
        logger.debug("All client threads stopped")

def main():
    heartbeat_servicer = HeartbeatServicer()
    heartbeat_servicer.serve()

if __name__ == "__main__":
    main()