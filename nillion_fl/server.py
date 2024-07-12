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
        print("Registered client with client_id: {}, token: {}".format(client_id, token))
        return fl_pb2.ClientInfo(client_id=client_id, token=token)

    def ScheduleLearningIteration(self, request, context):
        store_ids = [str(uuid.uuid4()) for _ in range(request.num_parties)]
        party_id = str(uuid.uuid4())
        print("Scheduled learning iteration with store_ids: {}, party_id: {}".format(store_ids, party_id))
        return fl_pb2.ScheduleResponse(store_ids=store_ids, party_id=party_id)

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