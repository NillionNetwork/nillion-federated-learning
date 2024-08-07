import asyncio

from nillion_fl.nilvm.client import NillionNetworkClient


class NillionClientIntegration:
    def __init__(self, client_id, num_parties):
        self.nillion_client = NillionNetworkClient(client_id, num_parties)
        self.party_id = self.nillion_client.party_id

    def get_compute_result(self, expected_results):
        return asyncio.run(self.nillion_client.get_compute_result(expected_results))

    def store_array(self, batch, secret_name, program_id, user_id):
        return asyncio.run(
            self.nillion_client.store_array(batch, secret_name, program_id, user_id)
        )

    def init(self):
        self.nillion_client.init()
