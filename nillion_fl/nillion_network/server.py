"""Dot Product example script"""

import argparse
import asyncio
import os
import subprocess

import nada_numpy.client as na_client
import numpy as np
import py_nillion_client as nillion
from cosmpy.aerial.client import LedgerClient
from cosmpy.aerial.wallet import LocalWallet
from cosmpy.crypto.keypairs import PrivateKey
from dotenv import load_dotenv
from nillion_fl.nillion_network.utils import store_program
from nillion_fl.nillion_network.component import NillionNetworkComponent
from nillion_fl.logs import logger
from nillion_python_helpers import (
    create_nillion_client,
    create_payments_config,
    get_quote,
    get_quote_and_pay,
    pay_with_quote,
)
from py_nillion_client import NodeKey, UserKey

home = os.getenv("HOME")
# load_dotenv(f"{home}/.config/nillion/nillion-devnet.env")
# load_dotenv(f"/workspaces/ai/.nillion-testnet.env")
MAX_SECRET_BATCH_SIZE = 5000


class NillionNetworkServer(NillionNetworkComponent):

    def __init__(self, program_name, num_parties, filename=f"{home}/.config/nillion/nillion-devnet.env"):
        super().__init__(9, num_parties, filename)
        self.program_name = program_name


    @staticmethod
    def create_config_py(directory, parameters):
        """
        Creates a config.py file in the specified directory with given parameters.

        :param directory: The directory where the config.py file will be created.
        :param parameters: A dictionary of parameters to include in the config.py file.
        """
        config_path = os.path.join(directory, 'config.py')
        
        with open(config_path, 'w') as file:
            for key, value in parameters.items():
                file.write(f"{key} = {repr(value)}\n")
        print(f"config.py created in {directory}")

    @staticmethod
    def execute_nada_build(directory):
        """
        Executes the 'nada build' command in the specified directory.

        :param directory: The directory where the 'nada build' command will be executed.
        """
        try:
            subprocess.run(['nada', 'build'], cwd=directory, check=True)
            print("nada build executed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while running 'nada build': {e}")

    def compile_program(self, batch_size, num_parties):
        # Compile the program
        program_path = os.path.abspath(os.path.join(os.path.dirname(__file__), self.program_name))
        config_file_path = os.path.join(program_path, "src/")
        self.program_mir_path = os.path.join( program_path, f"target/{self.program_name}.nada.bin")
        NillionNetworkServer.create_config_py(config_file_path, {"DIM": batch_size, "NUM_PARTIES": num_parties})
        NillionNetworkServer.execute_nada_build(program_path)
        logger.debug(f"Compiled program {self.program_name}")
        logger.debug(f"Program MIR path: {self.program_mir_path}")
        return self.program_mir_path

    async def store_program(self):
        
        self.program_id = await store_program(
            self.client,
            self.payments_wallet,
            self.payments_client,
            self.user_id,
            self.cluster_id,
            self.program_name,
            self.program_mir_path,
        )
        logger.info(f"Stored program {self.program_id}")
        return self.program_id
    
    async def compute(self, secret_ids):
        pass

class FedAvgNillionNetworkServer(NillionNetworkServer):

    def __init__(self, num_parties, filename=f"{home}/.config/nillion/nillion-devnet.env"):
        super().__init__("fed_avg", num_parties, filename)

    
    async def compute(self, secret_ids):
        # Bind the parties in the computation to the client to set input and output parties
        compute_bindings = nillion.ProgramBindings(self.program_id)

        store_ids = []
        for i, secret_id in enumerate(secret_ids):
            compute_bindings.add_input_party(self.party_names[i], secret_id)
            compute_bindings.add_output_party(self.party_names[i], secret_id)

        # Create a computation time secret to use
        computation_time_secrets = nillion.NadaValues({})

        # Get cost quote, then pay for operation to compute
        receipt_compute = get_quote_and_pay(
            self.client,
            nillion.Operation.compute(self.program_id, computation_time_secrets),
            self.payments_wallet,
            self.payments_client,
            self.cluster_id,
        )

        # Compute, passing all params including the receipt that shows proof of payment
        uuid = await self.client.compute(
            self.cluster_id,
            compute_bindings,
            store_ids,
            computation_time_secrets,
            receipt_compute,
        )

        return uuid
