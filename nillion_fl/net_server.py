import grpc
from concurrent import futures
import time
import uuid
import threading
import logging

import nillion_fl.fl_net.fl_service_pb2 as fl_pb2
import nillion_fl.fl_net.fl_service_pb2_grpc as fl_pb2_grpc

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FederatedLearningServicer(fl_pb2_grpc.FederatedLearningServiceServicer):
    def __init__(self, num_parties):
        self.clients = {}
        self.active_streams = {}
        self.num_parties = num_parties

    def is_valid_token(self, token):
        return token in self.clients

    def is_initial_request(self, request):
        return len(request.store_ids) == 0 and request.party_id == "" and request.token != ""
    
    def RegisterClient(self, request, context):
        client_id = len(self.clients) + 1
        token = str(uuid.uuid4())
        self.clients[token] = client_id
        return fl_pb2.ClientInfo(client_id=client_id, token=token)

    def ScheduleLearningIteration(self, request_iterator, context):
        stream_id = str(uuid.uuid4())
        def client_handler():
            for request in request_iterator:
                
                if not self.is_valid_token(request.token):
                    logger.warning(f"[SERVER][{stream_id}] Invalid token: {request.token}")
                    continue

                if self.is_initial_request(request):
                    self.ready.add(request.token)
                    logger.info(f"[SERVER][{stream_id}] Received initial request: {request}")
                    if self.ready

                if self.is_valid_token(request.token):
                    pass
                    
                logger.info(f"[SERVER][{stream_id}] Received store_ids: {request.store_ids}")
                time.sleep(60)
                yield fl_pb2.ScheduleRequest(
                    program_id=str(uuid.uuid4()),
                    user_id="user_456",
                    batch_size=32,
                    num_parties=5
                )
        logger.info("[SERVER] CLIENT DISCONNECTED")
        return None
        
    def listen_for_requests(self, request_iterator, stream_id):
        try:
            for request in request_iterator:
                logger.info(f"[SERVER][{stream_id}] Received request: {request}")
        except grpc.RpcError:
            logger.info("[SERVER] CLIENT DISCONNECTED")

    def serve(self):
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        fl_pb2_grpc.add_FederatedLearningServiceServicer_to_server(
            FederatedLearningServicer(num_parties=5), server)
        server.add_insecure_port('[::]:50051')

        server.start()
        logger.info("Server started. Listening on port 50051.")
        try:
            while True:
                time.sleep(86400)
        except KeyboardInterrupt:
            logger.info("Server stopping...")
            server.stop(0)

if __name__ == '__main__':
    FederatedLearningServicer(num_parties=5).serve()