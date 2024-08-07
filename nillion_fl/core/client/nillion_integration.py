""" 
Handles integration with the Nillion network client
for performing compute operations and storing data. 

"""

import asyncio

from nillion_fl.nilvm.client import NillionNetworkClient


class NillionClientIntegration:
    """
    Handles integration with the Nillion network client for performing
    compute operations and storing data.

    Attributes:
        nillion_client (NillionNetworkClient): Instance of NillionNetworkClient.
        party_id (str): Party identifier from the Nillion network client.
    """

    def __init__(self, client_id, num_parties):
        """
        Initializes the NillionClientIntegration with the specified client ID and number of parties.

        Args:
            client_id (int): The identifier for the Nillion client.
            num_parties (int): The number of parties involved in the network.
        """
        self.nillion_client = NillionNetworkClient(client_id, num_parties)
        self.party_id = self.nillion_client.party_id

    def get_compute_result(self, expected_results):
        """
        Retrieves the compute result from the Nillion network client.

        Args:
            expected_results (list): The expected results to match.

        Returns:
            The result of the compute operation.
        """
        return asyncio.run(self.nillion_client.get_compute_result(expected_results))

    def store_array(self, batch, secret_name, program_id, user_id):
        """
        Stores an array in the Nillion network.

        Args:
            batch (list): The data batch to be stored.
            secret_name (str): The name of the secret.
            program_id (str): The identifier of the program.
            user_id (str): The user identifier.

        Returns:
            The result of the store operation.
        """
        return asyncio.run(
            self.nillion_client.store_array(batch, secret_name, program_id, user_id)
        )

    def init(self):
        """
        Initializes the Nillion network client.
        """
        self.nillion_client.init()
