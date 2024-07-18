import asyncio
import time
import uuid

import grpc
import numpy as np

import nillion_fl.fl_net.fl_service_pb2 as fl_pb2
import nillion_fl.fl_net.fl_service_pb2_grpc as fl_pb2_grpc
from nillion_fl.logs import logger, uuid_str
from nillion_fl.nillion_network.client import NillionNetworkClient


class FederatedLearningClient:
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.responses = []
        self.client_info = None
        self.nillion_client = None
        self.parameters = None

    def register_client(self):
        request = fl_pb2.RegisterRequest()
        self.client_info = self.stub.RegisterClient(request)
        logger.info(
            f"""
            Registered with client_id: {self.client_info.client_id}, 
            token: {uuid_str(self.client_info.token)}, 
            num_parties: {self.client_info.num_parties}
            """
        )
        self.nillion_client = NillionNetworkClient(
            self.client_info.client_id, self.client_info.num_parties
        )

    def schedule_learning_iteration(self):
        def client_request_sender():
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

        learning_requests = self.stub.ScheduleLearningIteration(client_request_sender())

        for learning_request in learning_requests:
            logger.info("[CLIENT] Received learning request")
            if learning_request.program_id == "-1":
                logger.warning("Received STOP training request")
                learning_requests.cancel()
                self.channel.close()
                break

            logger.info(f"Learning Request: {learning_request}")

            self.learning_iteration(learning_request)

        return None

    def learning_iteration(self, learning_request):
        parameters = self.fit(self.parameters)
        store_ids = self.store_secrets(
            parameters,
            learning_request.program_id,
            learning_request.user_id,
            learning_request.batch_size,
            learning_request.num_parties,
        )

        self.responses.append(
            fl_pb2.StoreIDs(
                store_ids=store_ids,
                party_id=self.nillion_client.party_id,
                token=self.client_info.token,
            )
        )

        self.parameters = asyncio.run(self.nillion_client.get_compute_result())
        logger.debug(f"New Parameters: {self.parameters}")

    def fit(self, parameters):
        logger.debug("Fitting...")
        # Dummy parameters and function
        if parameters is None:
            return np.ones(10)
        else:
            return parameters + 0.5

    def store_secrets(self, parameters, program_id, user_id, batch_size, num_parties):

        # Create a batch of maximum batch size of the parameters vector
        store_ids = []
        remainder = len(parameters) % batch_size
        if remainder > 0:
            parameters = np.pad(parameters, (0, batch_size - remainder))

        print(f"Paramters: {divmod(len(parameters), batch_size)}")
        for i in range(0, len(parameters), batch_size):
            batch = parameters[i : i + batch_size]
            secret_name = chr(ord("A") + self.client_info.client_id)
            logger.debug(f"Storing secret {secret_name} with batch {batch}")
            store_id = asyncio.run(
                self.nillion_client.store_array(batch, secret_name, program_id, user_id)
            )
            store_ids.append(store_id)

        return store_ids

    def start_client(self, host="localhost", port=50051):
        try:
            self.channel = grpc.insecure_channel(f"{host}:{port}")
            self.stub = fl_pb2_grpc.FederatedLearningServiceStub(self.channel)
            self.client_info = None
            self.register_client()  # Creates the client_id and token
            self.schedule_learning_iteration()
        except KeyboardInterrupt:
            logger.info("Client stopping...")


def main():
    client = FederatedLearningClient(None, None, None)
    client.start_client()


if __name__ == "__main__":
    main()
