import asyncio

from nillion_fl.nilvm.server import FedAvgNillionNetworkServer


class NillionServerIntegration:
    def __init__(self, num_parties, program_number, batch_size):
        self.nillion_server = FedAvgNillionNetworkServer(num_parties, program_number)
        self.nillion_server.compile_program(batch_size)
        self.program_id = asyncio.run(self.nillion_server.store_program())
        self.user_id = self.nillion_server.user_id

    def compute(self, store_ids, party_ids, batch_id):
        return asyncio.run(self.nillion_server.compute(store_ids, party_ids, batch_id))
