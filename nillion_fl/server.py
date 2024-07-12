import grpc
from concurrent import futures
import time
import uuid

import nillion_fl.fl_net.fl_service_pb2 as fl_pb2
import nillion_fl.fl_net.fl_service_pb2_grpc as fl_pb2_grpc


class FederatedLearningServicer(fl_pb2_grpc.FederatedLearningServiceServicer):
    def __init__(self):
        self.clients = {}

    def RegisterClient(self, request, context):
        client_id = len(self.clients) + 1
        token = str(uuid.uuid4())
        self.clients[client_id] = token
        return fl_pb2.ClientInfo(client_id=client_id, token=token)

    def ScheduleLearningIteration(self, request_iterator, context):
        for request in request_iterator:
            # Process the incoming request
            print(f"Received request: {request}")
            
            requests = [
                fl_pb2.ScheduleRequest(program_id="program1", user_id="user1", batch_size=32, num_parties=3),
                fl_pb2.ScheduleRequest(program_id="program2", user_id="user2", batch_size=64, num_parties=4),
                fl_pb2.ScheduleRequest(program_id="program3", user_id="user3", batch_size=128, num_parties=5)
            ]
            

            yield requests

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    fl_pb2_grpc.add_FederatedLearningServiceServicer_to_server(
        FederatedLearningServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    serve()