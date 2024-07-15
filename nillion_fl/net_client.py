import grpc
import time
import uuid

import nillion_fl.fl_net.fl_service_pb2 as fl_pb2
import nillion_fl.fl_net.fl_service_pb2_grpc as fl_pb2_grpc

class FederatedLearningClient:
    def __init__(self, host='localhost', port=50051):
        #self.channel = grpc.insecure_channel(f'{host}:{port}')
        channel_creds = grpc.alts_channel_credentials()
        self.channel = grpc.secure_channel(f'{host}:{port}', channel_creds)
        self.stub = fl_pb2_grpc.FederatedLearningServiceStub(self.channel)
        self.client_info = None

    def register_client(self):
        request = fl_pb2.RegisterRequest()
        self.client_info = self.stub.RegisterClient(request)
        print(f"Registered with client_id: {self.client_info.client_id}, token: {self.client_info.token}")

    def schedule_learning_iteration(self):
        def generate_responses():
            responses = [
                fl_pb2.StoreIDs(store_ids=[], party_id=""),
                fl_pb2.StoreIDs(store_ids=[str(uuid.uuid4()) for _ in range(1)], party_id=str(uuid.uuid4())),
                fl_pb2.StoreIDs(store_ids=[str(uuid.uuid4()) for _ in range(2)], party_id=str(uuid.uuid4())),
                fl_pb2.StoreIDs(store_ids=[str(uuid.uuid4()) for _ in range(3)], party_id=str(uuid.uuid4()))
            ]
            i = 0
            while True:
                i %= len(responses)
                print("[CLIENT] SENDING MESSAGE")
                yield responses[i]
                time.sleep(0.5)
                i += 1

        learning_requests = self.stub.ScheduleLearningIteration(generate_responses())
        for learning_request in learning_requests:
            print("[CLIENT] RECEIVED REQUEST")
            #print(f"Received LEARNING REQUEST: {learning_request.program_id}, USER ID: {learning_request.user_id}")

def run():
    client = FederatedLearningClient()
    
    # Register client
    client.register_client()
    
    # Schedule learning iterations
    client.schedule_learning_iteration()

if __name__ == '__main__':
    run()