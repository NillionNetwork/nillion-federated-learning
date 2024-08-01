"""Dot Product example script"""

import os

import nada_numpy.client as na_client
import numpy as np
import py_nillion_client as nillion
from cosmpy.aerial.client import LedgerClient
from cosmpy.aerial.wallet import LocalWallet
from cosmpy.crypto.keypairs import PrivateKey
from dotenv import load_dotenv
from nillion_python_helpers import create_nillion_client, create_payments_config
from py_nillion_client import NodeKey, UserKey

from nillion_fl.nillion_network.utils import store_secret_array

home = os.getenv("HOME")
# load_dotenv(f"{home}/.config/nillion/nillion-devnet.env")
# load_dotenv(f"/workspaces/ai/.nillion-testnet.env")


class NillionNetworkComponent(object):

    def __init__(
        self,
        client_id,
        num_parties,
        filename=f"{home}/.config/nillion/nillion-devnet.env",
    ):
        """Main nada program"""
        load_dotenv(filename)
        self.num_parties = num_parties
        self.cluster_id = os.getenv("NILLION_CLUSTER_ID")
        self.grpc_endpoint = os.getenv("NILLION_NILCHAIN_GRPC")
        self.chain_id = os.getenv("NILLION_NILCHAIN_CHAIN_ID")
        self.seed = "my_seed_" + str(client_id)
        userkey = UserKey.from_seed((self.seed))
        nodekey = NodeKey.from_seed((self.seed))
        self.client = create_nillion_client(userkey, nodekey)
        self.party_id = self.client.party_id
        self.user_id = self.client.user_id

        self.party_names = na_client.parties(
            num_parties + 1
        )  # The +1 party is the program coordinator

        # Create payments config and set up Nillion wallet with a private key to pay for operations
        self.payments_config = create_payments_config(self.chain_id, self.grpc_endpoint)
        self.payments_client = LedgerClient(self.payments_config)
        self.payments_wallet = LocalWallet(
            PrivateKey(
                bytes.fromhex(os.getenv(f"NILLION_NILCHAIN_PRIVATE_KEY_{client_id}"))
            ),
            prefix="nillion",
        )
