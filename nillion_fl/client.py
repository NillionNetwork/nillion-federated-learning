import logging
import time
import uuid

import grpc

import nillion_fl.fl_net.fl_service_pb2 as fl_pb2
import nillion_fl.fl_net.fl_service_pb2_grpc as fl_pb2_grpc

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="[%(levelname)s] [%(asctime)s] - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FederatedLearningClient:
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.responses = []

    def register_client(self):
        request = fl_pb2.RegisterRequest()
        self.client_info = self.stub.RegisterClient(request)
        logger.info(
            f"Registered with client_id: {self.client_info.client_id}, token: {self.client_info.token}"
        )

    def schedule_learning_iteration(self):
        def generate_responses():
            logger.info("[CLIENT] Sending initial message")
            yield fl_pb2.StoreIDs(
                store_ids=[], party_id="", token=self.client_info.token
            )  # Empty first message

            while True:
                if len(self.responses) > 0:
                    response = self.responses.pop(0)
                    logger.info("[CLIENT] Sending store id response")
                    yield response
                    time.sleep(0.5)

            logger.info("[CLIENT][SEND] STOP")

        learning_requests = self.stub.ScheduleLearningIteration(generate_responses())
        for learning_request in learning_requests:
            logger.info("[CLIENT] Received learning request")
            if learning_request.program_id == "-1":
                logger.info("Received STOP training request")
                learning_requests.cancel()
                self.channel.close()
                break

            logger.info(f"Learning Request: {learning_request}")

            (store_ids, party_id) = self.learning_iteration(learning_request)

            self.responses.append(
                fl_pb2.StoreIDs(
                    store_ids=store_ids, party_id=party_id, token=self.client_info.token
                )
            )
        return None

    def learning_iteration(self, learning_request):
        parameters = self.fit()
        store_ids = self.store_secrets(
            learning_request.program_id,
            learning_request.user_id,
            learning_request.batch_size,
            learning_request.num_parties,
        )

        return (store_ids, "abc")

    def start_client(self, host="localhost", port=50051):
        try:
            self.channel = grpc.insecure_channel(f"{host}:{port}")
            self.stub = fl_pb2_grpc.FederatedLearningServiceStub(self.channel)
            self.client_info = None
            self.register_client()  # Creates the client_id and token
            self.schedule_learning_iteration()
        except KeyboardInterrupt:
            logger.info("Client stopping...")

    def fit(self):
        logger.info("[CLIENT] FITTING")
        pass

    def store_secrets(self, parameters, program_id, user_id, batch_size, num_parties):
        logger.info(
            f"[CLIENT] STORING SECRET: {program_id}, {user_id}, {batch_size}, {num_parties}"
        )
        return [str(uuid.uuid4()) for _ in range(1)]


def run():
    client = FederatedLearningClient(None, None, None)
    client.start_client()


if __name__ == "__main__":
    run()
