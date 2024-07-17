"""Dot Product example script"""

import os

import nada_numpy.client as na_client
import numpy as np
import py_nillion_client as nillion
from cosmpy.aerial.client import LedgerClient
from cosmpy.aerial.wallet import LocalWallet
from cosmpy.crypto.keypairs import PrivateKey
from dotenv import load_dotenv
from nillion_fl.nillion_network.utils import store_secret_array
from nillion_fl.nillion_network.component import NillionNetworkComponent

from nillion_python_helpers import (
    create_nillion_client,
    create_payments_config
)

from py_nillion_client import NodeKey, UserKey

home = os.getenv("HOME")
# load_dotenv(f"{home}/.config/nillion/nillion-devnet.env")
# load_dotenv(f"/workspaces/ai/.nillion-testnet.env")

class NillionNetworkClient(NillionNetworkComponent):

    def __init__(self, client_id, num_parties, filename=f"{home}/.config/nillion/nillion-devnet.env"):
        super().__init__(client_id, num_parties, filename)


    async def store_array(self, array, secret_name, program_id, server_user_id):
        """
        Stores the given parameters as a secret array on the Nillion network.

        Args:
            parameters (np.ndarray): The parameters to store as a secret array
            program_id (str): The program ID to associate with the secret array
            user_id (str): The user ID to associate with the secret array
            party_id (str): The party ID to associate with the secret array
        """ 
        # Create a permissions object to attach to the stored secret
        permissions = nillion.Permissions.default_for_user(self.client.user_id)
        permissions.add_compute_permissions({server_user_id: {program_id}})

        # Create a secret
        store_id = await store_secret_array(
            self.client,
            self.payments_wallet,
            self.payments_client,
            self.cluster_id,
            program_id,
            array,
            secret_name,
            nillion.SecretInteger,
            1,
            permissions,
        )

        return store_id

    async def get_compute_result(self):

        print("⌛ Waiting for result...")
        while True:
            compute_event = await self.client.next_compute_event()
            if isinstance(compute_event, nillion.ComputeFinishedEvent):
                print(f"✅  Compute complete for compute_id {compute_event.uuid}")
                print(f"🖥️  The result is {compute_event.result.value}")
                return compute_event.result.value
