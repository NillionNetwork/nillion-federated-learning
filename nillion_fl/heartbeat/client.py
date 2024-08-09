# heartbeat_client.py
import grpc
import nillion_fl.heartbeat.network.heartbeat_pb2 as hb_pb2
import nillion_fl.heartbeat.network.heartbeat_pb2_grpc as hb_grpc
import threading
import time
from queue import Queue
from nillion_fl.logs import logger

class HeartbeatClient:
    def __init__(self, host: str = "localhost", port=50052, timeout=5):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.channel = grpc.insecure_channel(f"{self.host}:{self.port}")
        self.stub = hb_grpc.HeartbeatServiceStub(self.channel)
        self.stop_event = threading.Event()
        self.heartbeat_thread = None
        self.message_queue = Queue()

    def handle_status(self, status_code: int, event: threading.Event) -> None:
        if status_code == 200:
            logger.info("Heartbeat [200] received from server")
        elif status_code == 400:
            logger.info("Heartbeat [400] received from server")
            event.set()
        elif status_code == 500:
            logger.info("Heartbeat [500] received from server")
            event.set()

    def client_heartbeat_sender(self):
        while not self.stop_event.is_set() or not self.message_queue.empty():
            try:
                if not self.message_queue.empty():
                    yield self.message_queue.get()
                else:
                    yield hb_pb2.Heartbeat(status_code=200 if self.message_queue.empty() else self.message_queue.get(), msg="")
            except grpc.RpcError as e:
                logger.error(f"GRPC error: {e}")
            time.sleep(self.timeout)

    def terminate(self):
        logger.warning("Sending a termination heartbeat client")
        self.message_queue.put(hb_pb2.Heartbeat(status_code=400, msg=""))
        self.stop_event.set()
        self.heartbeat_thread.join()
        self.channel.close()

    def send_heartbeats(self):
        responses = self.stub.SubscribeToHeartbeat(self.client_heartbeat_sender())
        for response in responses:
            self.handle_status(response.status_code, self.stop_event)
            if self.stop_event.is_set():
                break

    def start(self):
        self.heartbeat_thread = threading.Thread(target=self.send_heartbeats, daemon=True)
        self.heartbeat_thread.start()

def main():
    heartbeat_client = HeartbeatClient()
    heartbeat_client.start()
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        logger.warning("Keyboard interrupt detected")
        heartbeat_client.terminate()

if __name__ == "__main__":
    main()