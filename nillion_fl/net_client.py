import grpc
import time
import uuid

import nillion_fl.fl_net.fl_service_pb2 as fl_pb2
import nillion_fl.fl_net.fl_service_pb2_grpc as fl_pb2_grpc

class FederatedLearningClient:
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

        self.responses = []


    def register_client(self):
        request = fl_pb2.RegisterRequest()
        self.client_info = self.stub.RegisterClient(request)
        print(f"Registered with client_id: {self.client_info.client_id}, token: {self.client_info.token}")

    def schedule_learning_iteration(self):
        def generate_responses():
            print("[CLIENT] SENDING MESSAGE")
            yield fl_pb2.StoreIDs(store_ids=[], party_id="", token=self.client_info.token) # Empty first message
            
            while True:
                if len(self.responses) > 0:
                    response = self.responses.pop(0)
                    print("[CLIENT] SENDING MESSAGE")
                    yield response
                    time.sleep(0.5)
            
            print("[CLIENT][SEND] STOP")
            

        learning_requests = self.stub.ScheduleLearningIteration(generate_responses())
        for learning_request in learning_requests:
            print("[CLIENT] RECEIVED REQUEST")
            if learning_request.program_id[0] == "-1":
                print("Received STOP REQUEST")
                learning_requests.cancel()
                self.channel.close()
                break
            
            batch_size = learning_request.batch_size
            num_parties = learning_request.num_parties
            program_id = learning_request.program_id
            user_id = learning_request.user_id

            print("LEARNING REQUEST: ", learning_request)

            self.fit()

            store_ids = self.store_secrets(program_id, user_id, batch_size, num_parties)

            self.responses.append(fl_pb2.StoreIDs(store_ids=store_ids, party_id="abc", token=self.client_info.token))

    def start_client(self, host='localhost', port=50051):
        self.channel = grpc.insecure_channel(f'{host}:{port}')
        self.stub = fl_pb2_grpc.FederatedLearningServiceStub(self.channel)
        self.client_info = None
        self.register_client() # Creates the client_id and token
        self.schedule_learning_iteration()

    def fit(self):
        print("[CLIENT] FITTING")
        pass

    def store_secrets(sefl, program_id, user_id, batch_size, num_parties):
        print(f"[CLIENT] STORING SECRET: {program_id}, {user_id}, {batch_size}, {num_parties}")
        return [str(uuid.uuid4()) for _ in range(1)]



def run():
    client = FederatedLearningClient(None, None, None)
    client.start_client()

if __name__ == '__main__':
    run()