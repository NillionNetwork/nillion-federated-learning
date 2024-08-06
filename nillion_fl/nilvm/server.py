"""Dot Product example script"""

import argparse
import asyncio
import concurrent.futures
import os
import subprocess
import threading

import nada_numpy.client as na_client
import numpy as np
import py_nillion_client as nillion
from cosmpy.aerial.client import LedgerClient
from cosmpy.aerial.wallet import LocalWallet
from cosmpy.crypto.keypairs import PrivateKey
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from nillion_python_helpers import (
    create_nillion_client,
    create_payments_config,
    get_quote,
    get_quote_and_pay,
    pay_with_quote,
)
from py_nillion_client import NodeKey, UserKey

from nillion_fl.logs import logger
from nillion_fl.nilvm.component import NillionNetworkComponent
from nillion_fl.nilvm.utils import JsonDict, store_program

home = os.getenv("HOME")

MAX_SECRET_BATCH_SIZE = 5000


class NillionNetworkServer(NillionNetworkComponent):

    def __init__(
        self,
        program_name: str,
        num_parties: int,
        num_threads: int = 1,
        filename: str = f"{home}/.config/nillion/nillion-devnet.env",
    ):
        super().__init__(9, num_parties, filename)
        self.program_name = program_name
        self.num_threads = num_threads

    def create_src_file(self, src_directory, batch_size):
        """
        Creates the source files for the Nada program in the specified directory based on the Jinja templates.

        Args:
            directory: The directory where the source files will be created.
            parameters: A dictionary of parameters to include in the source files.
        """
        # template_path = os.path.join(src_directory, f"{self.program_name}.py.jinja")

        # Set up the Jinja2 environment
        env = Environment(loader=FileSystemLoader(src_directory))

        # Load the template
        src_template = env.get_template(f"{self.program_name}.py.jinja")

        # Define your data
        data = {
            "num_parties": self.num_parties,
            "dim": batch_size,
        }

        # Render the template
        output = src_template.render(data)
        src_file_name = f"custom_{self.program_name}"
        src_file_path = os.path.join(src_directory, f"{src_file_name}.py")

        # Write to a Python file
        with open(src_file_path, "w") as f:
            f.write(output)

        return src_file_path, src_file_name

    def update_toml_file(self, program_directory: os.PathLike, src_file_name: str):
        """
        Creates a TOML file for the Nada program in the specified directory.

        :param program_directory: The directory where the TOML file will be created.
        :param src_file_name: The name of the source file.
        """
        # Set up the Jinja2 environment
        env = Environment(loader=FileSystemLoader(program_directory))

        # Load the template
        toml_template = env.get_template("nada-project.toml.jinja")

        if not src_file_name.endswith(".py"):
            src_file_name += ".py"

        # Define your data
        data = {
            "programs": [src_file_name],
        }

        # Render the template
        output = toml_template.render(data)
        toml_file_path = os.path.join(program_directory, "nada-project.toml")

        # Write to a TOML file
        with open(toml_file_path, "w") as f:
            f.write(output)

    @staticmethod
    def execute_nada_build(directory, filename: str = None):
        """
        Executes the 'nada build' command in the specified directory.

        :param directory: The directory where the 'nada build' command will be executed.
        """
        try:
            cmd = ["nada", "build"]
            if filename is not None:
                cmd.append(filename)
            subprocess.run(cmd, cwd=directory, check=True)
            logger.debug("nada build executed successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error occurred while running 'nada build': {e}")

    def compile_program(self, batch_size: int):
        # Compile the program
        self.batch_size = batch_size

        program_directory = os.path.abspath(
            os.path.join(os.path.dirname(__file__), f"{self.program_name}/")
        )
        src_path = os.path.join(program_directory, "src/")

        src_file_path, src_file_name = self.create_src_file(src_path, batch_size)

        self.program_mir_path = os.path.join(
            program_directory, f"target/{src_file_name}.nada.bin"
        )

        self.update_toml_file(program_directory, src_file_name)
        NillionNetworkServer.execute_nada_build(program_directory, src_file_name)
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
        logger.debug(f"Stored program {self.program_id}")
        return self.program_id

    async def compute(self, secret_ids):
        pass


class FedAvgNillionNetworkServer(NillionNetworkServer):

    def __init__(
        self,
        num_parties: int,
        num_threads: int = 1,
        filename: str = f"{home}/.config/nillion/nillion-devnet.env",
    ):
        super().__init__("fed_avg", num_parties, num_threads, filename)

    async def compute(
        self,
        store_ids: list,
        party_ids: dict,
        program_order: int,
        *,
        __lock: threading.Lock = threading.Lock(),
    ):
        # Bind the parties in the computation to the client to set input and output parties
        compute_bindings = nillion.ProgramBindings(self.program_id)
        compute_bindings.add_input_party(self.party_names[-1], self.party_id)
        for client_id, party_id in party_ids.items():  # tuple (client_id(0..n), user_id)
            compute_bindings.add_input_party(self.party_names[client_id], party_id)
            compute_bindings.add_output_party(self.party_names[client_id], party_id)

        # Create a computation time secret to use
        computation_time_secrets = nillion.NadaValues(
            {"program_order": nillion.Integer(program_order)}
        )
        with __lock:
            # Get cost quote, then pay for operation to compute
            receipt_compute = await get_quote_and_pay(
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

    async def compute_multithread(
        self, num_threads: int, store_ids: list, party_ids: dict, program_order: int
    ):
        # Bind the parties in the computation to the client to set input and output parties
        compute_bindings = nillion.ProgramBindings(self.program_id)

        compute_bindings.add_input_party(self.party_names[-1], self.user_id)
        for client_id, user_id in party_ids.items():  # tuple (client_id(0..n), user_id)
            compute_bindings.add_input_party(self.party_names[client_id], user_id)
            compute_bindings.add_output_party(self.party_names[client_id], user_id)

        thread_store_ids = [
            [party_store_ids[thread] for party_store_ids in store_ids]
            for thread in range(num_threads)
        ]

        # Create a computation time secret to use
        computation_time_secrets_list = [
            nillion.NadaValues({"program_order": nillion.Integer(thread)})
            for thread in range(num_threads)
        ]

        async def get_quote_and_pay_wrapper(computation_time_secrets):

            # Get cost quote, then pay for operation to compute
            receipt_compute = await get_quote_and_pay(
                self.client,
                nillion.Operation.compute(self.program_id, computation_time_secrets),
                self.payments_wallet,
                self.payments_client,
                self.cluster_id,
            )

            return receipt_compute

        receipts_compute = await asyncio.gather(
            *[
                get_quote_and_pay_wrapper(computation_time_secret)
                for computation_time_secret in computation_time_secrets_list
            ]
        )

        async def compute_wrapper(
            thread_store_ids, computation_time_secrets, receipt_compute
        ):
            return await self.client.compute(
                self.cluster_id,
                compute_bindings,
                thread_store_ids,
                computation_time_secrets,
                receipt_compute,
            )

            # Use ThreadPoolExecutor to parallelize the operation

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                asyncio.wrap_future(
                    executor.submit(
                        asyncio.run,
                        compute_wrapper(
                            thread_store_id, computation_time_secrets, receipt_compute
                        ),
                    )
                )
                for thread_store_id, computation_time_secrets, receipt_compute in zip(
                    thread_store_ids, computation_time_secrets_list, receipts_compute
                )
            ]

            # Wait for all futures to complete
            uuids = await asyncio.gather(*futures)

        return uuids


async def main(batch_size, num_threads, client_ids):
    num_parties = len(client_ids)
    # FedAvgNillionNetworkServer instance
    nillion_server = FedAvgNillionNetworkServer(num_parties, num_threads=num_threads)
    nillion_server.compile_program(batch_size)
    program_id = await nillion_server.store_program()

    JsonDict(
        {"program_id": program_id, "server_user_id": nillion_server.user_id}
    ).to_json_file("/tmp/fed_avg.json")

    print(
        "Now users can proceed to run the client program with the following client IDs: "
    )
    for client_id in client_ids:
        print(
            f"\t poetry run python3 client.py {batch_size} {num_threads} {num_parties} {client_id} {chr(ord('A') + client_id)}"
        )
    input("Press ENTER to continue to compute ...")
    store_ids = []
    party_ids = {}
    for i, client_id in enumerate(client_ids):
        config = JsonDict.from_json_file(f"/tmp/client_{client_id}.json")
        store_ids.append(config["store_id"])
        party_ids[client_id] = config["party_id"]

    logger.debug(f"Using store_ids: {store_ids}")
    logger.debug(f"Using party_ids: {party_ids}")

    # Compute

    uuids = await nillion_server.compute(
        num_threads, store_ids, party_ids, program_order=1234567890
    )
    logger.debug(f"Compute complete with UUIDs: {uuids}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the client program with a client ID",
        epilog="Example: python server.py 10 5 [0 1 2]",
    )
    parser.add_argument(
        "batch_size",
        type=int,
        help="An integer as client argument (max. 1000)",
    )
    parser.add_argument(
        "num_threads",
        type=int,
        help="The number of concurrent computations",
    )
    parser.add_argument(
        "client_ids",
        type=int,
        nargs="+",
        help="A list of integers as client arguments (at least 2)",
    )

    args = parser.parse_args()

    asyncio.run(main(args.batch_size, args.num_threads, args.client_ids))
