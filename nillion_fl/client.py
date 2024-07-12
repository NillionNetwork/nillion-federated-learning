import grpc

import nillion_fl.fl_net.fl_service_pb2 as fl_pb2
import nillion_fl.fl_net.fl_service_pb2_grpc as fl_pb2_grpc

class FederatedLearningClient:
    def __init__(self, host='localhost', port=50051):
        self.channel = grpc.insecure_channel(f'{host}:{port}')
        self.stub = fl_pb2_grpc.FederatedLearningServiceStub(self.channel)
        self.client_info = None

    def register_client(self):
        request = fl_pb2.RegisterRequest()
        self.client_info = self.stub.RegisterClient(request)
        print(f"Registered with client_id: {self.client_info.client_id}, token: {self.client_info.token}")

    def schedule_learning_iteration(self, program_id, user_id, batch_size, num_parties):
        request = fl_pb2.ScheduleRequest(
            program_id=program_id,
            user_id=user_id,
            batch_size=batch_size,
            num_parties=num_parties
        )
        response = self.stub.ScheduleLearningIteration(request)
        print(f"Scheduled learning iteration with store_ids: {response.store_ids}, party_id: {response.party_id}")

def run():
    client = FederatedLearningClient()
    
    # Register client
    client.register_client()
    
    # Schedule learning iteration
    client.schedule_learning_iteration(
        program_id="example_program",
        user_id="example_user",
        batch_size=32,
        num_parties=3
    )

if __name__ == '__main__':
    run()