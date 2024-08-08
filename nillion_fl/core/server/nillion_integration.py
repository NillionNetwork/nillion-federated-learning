""" Manages integration with the FedAvg Nillion Network server for federated averaging tasks. """

import asyncio

from nillion_fl.nilvm.server import FedAvgNillionNetworkServer


class NillionServerIntegration:
    """
    Manages integration with the FedAvg Nillion Network server for federated averaging tasks.

    Attributes:
        nillion_server (FedAvgNillionNetworkServer): Instance of FedAvgNillionNetworkServer.
        program_id (str): Program identifier assigned after storing the program.
        user_id (str): User identifier from the Nillion network server.
    """

    def __init__(self, num_parties, batch_size):
        """
        Initializes the NillionServerIntegration with the specified parameters.

        Args:
            num_parties (int): The number of parties involved in the federated network.
            batch_size (int): Size of the data batch to be compiled for the program.
        """
        self.nillion_server = FedAvgNillionNetworkServer(num_parties)
        self.nillion_server.compile_program(batch_size)
        self.program_id = asyncio.run(self.nillion_server.store_program())
        self.user_id = self.nillion_server.user_id

    def compute(self, store_ids, party_ids, batch_id):
        """
        Computes the result using the federated averaging server.

        Args:
            store_ids (list): Identifiers for the stored data.
            party_ids (list): Identifiers for the parties involved in the computation.
            batch_id (str): Identifier for the batch of data being processed.

        Returns:
            The result of the federated computation.
        """
        return asyncio.run(self.nillion_server.compute(store_ids, party_ids, batch_id))

    def get_program_id(self):
        """
        Retrieves the program identifier.

        Returns:
            str: The identifier of the stored program.
        """
        return self.program_id

    def get_user_id(self):
        """
        Retrieves the user identifier.

        Returns:
            str: The user identifier from the Nillion network server.
        """
        return self.user_id
