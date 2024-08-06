import threading
import numpy as np
import asyncio
from nillion_fl.logs import logger, uuid_str
from nillion_fl.core.client import FLClientCore
from nillion_fl.nilvm.client import NillionNetworkClient


class FederatedLearningClient(object):

    def __init__(self, host="localhost", port=50051, *, num_parameters):
        self.__nillion_client = None
        self.fl_client = FLClientCore(host, port)
        self.parameters = None
        self.num_parameters = num_parameters

        # Locks for critical sections
        self.stop_event = threading.Event()

    @property
    def nillion_client(self):
        if self.__nillion_client is None:
            self.__nillion_client = NillionNetworkClient(
                self.fl_client.client_info.client_id, self.fl_client.client_info.num_parties
            )
        return self.__nillion_client


    def learning_iteration(self, learning_request):
        """
        Perform a single learning iteration.

        Args:
            learning_request: The learning request from the server.
        """
        parameters = self.fit(self.parameters)
        expected_results = len(parameters) // learning_request.batch_size

        def store_secrets_thread():
            self.store_secrets(
                parameters,
                learning_request.program_id,
                learning_request.user_id,
                learning_request.batch_size,
            )

        self.store_secret_thread = threading.Thread(target=store_secrets_thread)
        self.store_secret_thread.start()

        new_parameters = asyncio.run(
            self.nillion_client.get_compute_result(expected_results)
        )

        new_parameters = sorted(new_parameters, key=lambda x: x[0])
        new_parameters = np.concatenate([x[1] for x in new_parameters])
        self.parameters = new_parameters

        logger.debug(f"Waiting for thread to join()...")
        self.store_secret_thread.join()
        self.store_secret_thread = None
        logger.info(f"New Parameters: {len(self.parameters)}")

    def fit(self, parameters):
        """
        Fit the model using the current parameters.

        Args:
            parameters: Current model parameters.

        Returns:
            numpy.ndarray: Updated model parameters.
        """
        pass

    def store_secrets(self, parameters, program_id, user_id, batch_size):
        """
        Store the parameters as secrets in the Nillion Network.

        Args:
            parameters (numpy.ndarray): Model parameters to store.
            program_id (str): ID of the program.
            user_id (str): ID of the user.
            batch_size (int): Size of each batch.
        """
        # Create a batch of maximum batch size of the parameters vector
        remainder = len(parameters) % batch_size
        if remainder > 0:
            parameters = np.pad(parameters, (0, batch_size - remainder))

        secret_name = chr(ord("A") + self.fl_client.client_info.client_id)

        logger.debug(f"Parameters: {divmod(len(parameters), batch_size)}")
        for batch_start in range(0, len(parameters), batch_size):
            if self.stop_event.is_set():
                logger.warning("Stopping store_secrets operation...")
                break
            batch_id = batch_start // batch_size
            batch = parameters[batch_start : batch_start + batch_size]

            store_id = asyncio.run(
                self.nillion_client.store_array(batch, secret_name, program_id, user_id)
            )
            # Prepare a response
            logger.debug(
                f"Storing secret {secret_name} with batch {np.concatenate([batch[:3], batch[-3:]])} {batch.shape}, store_id {store_id}, party_id {self.nillion_client.party_id}, batch_id {batch_id}, token {self.fl_client.client_info.token}"
            )
            self.fl_client.send_store_id(
                store_id, self.nillion_client.party_id, self.fl_client.client_info.token, batch_id
            )
    
    def start_client(self,):
        self.fl_client.register_client(num_parameters=self.num_parameters)
        self.nillion_client.init()
        self.fl_client.start_learning(callback=self.learning_iteration)
        self.stop_event.set()
        if self.store_secret_thread and self.store_secret_thread.is_alive():
            self.store_secret_thread.join()
    
def main():
    """
    Main function to create and start a FederatedLearningClient.
    """
    client = FederatedLearningClient(None, None, None)
    client.start_client()


if __name__ == "__main__":
    main()


