""" Base class for Nillion Network components. """

import os
from abc import ABC, abstractmethod

import nada_numpy.client as na_client
from cosmpy.aerial.client import LedgerClient
from cosmpy.aerial.wallet import LocalWallet
from cosmpy.crypto.keypairs import PrivateKey
from dotenv import load_dotenv
from nillion_python_helpers import (create_nillion_client,
                                    create_payments_config)
# pylint: disable=no-name-in-module
from py_nillion_client import NodeKey  # Ensure the correct import paths
from py_nillion_client import UserKey

home = os.getenv("HOME")
# load_dotenv(f"{home}/.config/nillion/nillion-devnet.env")
# load_dotenv(f"/workspaces/ai/.nillion-testnet.env")


class NillionNetworkComponent(ABC):
    """
    Abstract base class for Nillion network components.

    Attributes:
        num_parties (int): Number of parties in the network.
        cluster_id (str): Cluster identifier.
        grpc_endpoint (str): gRPC endpoint.
        chain_id (str): Chain identifier.
        seed (str): Seed for generating keys.
        client: Nillion client instance.
        party_id (str): Party identifier.
        user_id (str): User identifier.
        party_names: List of party names.
        payments_config: Payments configuration.
        payments_client: Ledger client instance.
        payments_wallet: Wallet for payments.
    """

    # pylint: disable=too-many-instance-attributes disable=too-few-public-methods

    @abstractmethod
    def __init__(
        self,
        client_id,
        num_parties,
        filename=f"{home}/.config/nillion/nillion-devnet.env",
    ):
        """
        Initializes the NillionNetworkComponent.

        Args:
            client_id (int): The client identifier.
            num_parties (int): Number of parties involved in the network.
            filename (str): Path to the environment configuration file.
                Defaults to the devnet file.
        """
        load_dotenv(filename)
        self.num_parties = num_parties
        self.cluster_id = os.getenv("NILLION_CLUSTER_ID")
        self.grpc_endpoint = os.getenv("NILLION_NILCHAIN_GRPC")
        self.chain_id = os.getenv("NILLION_NILCHAIN_CHAIN_ID")
        self.seed = "my_seed_" + str(client_id)
        self.client = self._create_client()
        self.party_id = self.client.party_id
        self.user_id = self.client.user_id
        self.party_names = na_client.parties(
            num_parties + 1
        )  # The +1 party is the program coordinator
        self.payments_config = self._create_payments_config()
        self.payments_client = LedgerClient(self.payments_config)
        self.payments_wallet = self._create_payments_wallet(client_id)

    def _create_client(self):
        """
        Creates and returns a Nillion client instance.

        Returns:
            Nillion client instance.
        """
        userkey = UserKey.from_seed(self.seed)
        nodekey = NodeKey.from_seed(self.seed)
        return create_nillion_client(userkey, nodekey)

    def _create_payments_config(self):
        """
        Creates and returns the payments configuration.

        Returns:
            Payments configuration instance.
        """
        return create_payments_config(self.chain_id, self.grpc_endpoint)

    def _create_payments_wallet(self, client_id):
        """
        Creates and returns a LocalWallet instance.

        Args:
            client_id (int): The client identifier.

        Returns:
            LocalWallet instance.
        """
        private_key = os.getenv(f"NILLION_NILCHAIN_PRIVATE_KEY_{client_id}")
        return LocalWallet(
            PrivateKey(bytes.fromhex(private_key)),
            prefix="nillion",
        )
