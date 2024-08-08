"""
This module implements a Nillion Network server for federated averaging.
It handles program compilation, storage, and computation across multiple parties.
"""

import argparse
import asyncio
import concurrent.futures
import os
import subprocess
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import py_nillion_client as nillion
from jinja2 import Environment, FileSystemLoader
from nillion_python_helpers import get_quote_and_pay

from nillion_fl.logs import logger
from nillion_fl.nilvm.component import NillionNetworkComponent
from nillion_fl.nilvm.utils import JsonDict, store_program

home = os.getenv("HOME")

MAX_SECRET_BATCH_SIZE = 5000


class NillionNetworkServer(NillionNetworkComponent):
    """
    Base class for Nillion Network server components.
    Handles program compilation, storage, and basic computation setup.
    """

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
        self.batch_size: Optional[int] = None
        self.program_mir_path: Optional[str] = None
        self.program_id: Optional[str] = None

    def create_src_file(self, src_directory: str, batch_size: int) -> Tuple[str, str]:
        """
        Creates the source files for the Nada program
        in the specified directory based on the Jinja templates.

        Args:
            src_directory: The directory where the source files will be created.
            batch_size: The batch size for the program.

        Returns:
            Tuple of the source file path and name.
        """
        env = Environment(loader=FileSystemLoader(src_directory))
        src_template = env.get_template(f"{self.program_name}.py.jinja")
        data = {
            "num_parties": self.num_parties,
            "dim": batch_size,
        }
        output = src_template.render(data)
        src_file_name = f"custom_{self.program_name}"
        src_file_path = os.path.join(src_directory, f"{src_file_name}.py")

        with open(src_file_path, "w", encoding="utf-8") as f:
            f.write(output)

        return src_file_path, src_file_name

    def update_toml_file(self, program_directory: os.PathLike[str], src_file_name: str):
        """
        Creates a TOML file for the Nada program in the specified directory.

        Args:
            program_directory: The directory where the TOML file will be created.
            src_file_name: The name of the source file.
        """
        env = Environment(loader=FileSystemLoader(program_directory))
        toml_template = env.get_template("nada-project.toml.jinja")

        if not src_file_name.endswith(".py"):
            src_file_name += ".py"

        data = {
            "programs": [src_file_name],
        }

        output = toml_template.render(data)
        toml_file_path = os.path.join(program_directory, "nada-project.toml")

        with open(toml_file_path, "w", encoding="utf-8") as f:
            f.write(output)

    @staticmethod
    def execute_nada_build(directory: str, filename: Optional[str] = None):
        """
        Executes the 'nada build' command in the specified directory.

        Args:
            directory: The directory where the 'nada build' command will be executed.
            filename: The filename to build (optional).
        """
        try:
            cmd = ["nada", "build"]
            if filename is not None:
                cmd.append(filename)
            subprocess.run(cmd, cwd=directory, check=True)
            logger.debug("nada build executed successfully.")
        except subprocess.CalledProcessError as e:
            logger.error("Error occurred while running 'nada build': %s", e)

    def compile_program(self, batch_size: int) -> str:
        """
        Compiles the Nada program with the given batch size.

        Args:
            batch_size: The batch size for the program.

        Returns:
            str: The path to the compiled program MIR file.
        """
        self.batch_size = batch_size

        program_directory = os.path.abspath(
            os.path.join(os.path.dirname(__file__), f"{self.program_name}/")
        )
        src_path = os.path.join(program_directory, "src/")

        _, src_file_name = self.create_src_file(src_path, batch_size)

        self.program_mir_path = os.path.join(
            program_directory, f"target/{src_file_name}.nada.bin"
        )

        self.update_toml_file(Path(program_directory), src_file_name)
        NillionNetworkServer.execute_nada_build(program_directory, src_file_name)
        logger.debug("Compiled program %s", self.program_name)
        logger.debug("Program MIR path: %s", self.program_mir_path)
        return self.program_mir_path

    async def store_program(self) -> str:
        """
        Stores the compiled program on the Nillion network.

        Returns:
            str: The program ID of the stored program.
        """
        if self.program_mir_path is None:
            raise ValueError("Program MIR path is not set.")

        self.program_id = await store_program(
            self.client,
            self.payments_wallet,
            self.payments_client,
            self.user_id,
            self.cluster_id,
            self.program_name,
            self.program_mir_path,
        )
        logger.debug("Stored program %s", self.program_id)
        return self.program_id

    async def compute(
        self,
        store_ids: List[str],
        party_ids: Dict[int, str],
        program_order: int,
        __lock: threading.Lock = threading.Lock(),
    ) -> str:
        """
        Placeholder method for computation. To be implemented by subclasses.

        Args:
            store_ids: The store IDs to be used in the computation.
            party_ids: The party IDs to be used in the computation.
            program_order: The order of the program in the computation.
            __lock: Threading lock for synchronization.

        Returns:
            str: The UUID of the computation.
        """
        raise NotImplementedError


class FedAvgNillionNetworkServer(NillionNetworkServer):
    """
    Nillion Network server implementation for federated averaging.
    """

    def __init__(
        self,
        num_parties: int,
        num_threads: int = 1,
        filename: str = f"{home}/.config/nillion/nillion-devnet.env",
    ):
        super().__init__("fed_avg", num_parties, num_threads, filename)

    async def compute(
        self,
        store_ids: List[str],
        party_ids: Dict[int, str],
        program_order: int,
        __lock: threading.Lock = threading.Lock(),
    ) -> str:
        """
        Performs the federated averaging computation.

        Args:
            store_ids: List of store IDs for the computation.
            party_ids: Dictionary of party IDs.
            program_order: The program order for the computation.
            __lock: Threading lock for synchronization.

        Returns:
            str: The UUID of the computation.
        """
        # fmt: off
        # pylint: disable=no-member
        if self.program_id is None:
            raise ValueError("Program ID is None and cannot be None")

        compute_bindings = nillion.ProgramBindings(self.program_id)
        # fmt: on
        compute_bindings.add_input_party(self.party_names[-1], self.party_id)
        for client_id, party_id in party_ids.items():
            compute_bindings.add_input_party(self.party_names[client_id], party_id)
            compute_bindings.add_output_party(self.party_names[client_id], party_id)

        # fmt: off
        # pylint: disable=no-member
        computation_time_secrets = nillion.NadaValues(
            {"program_order": nillion.Integer(program_order)} # type: ignore[dict-item] # fmt: off # pylint: disable=line-too-long
        )
        # fmt: on
        with __lock:
            receipt_compute = await get_quote_and_pay(
                self.client,
                nillion.Operation.compute(self.program_id, computation_time_secrets),  # type: ignore[attr-defined] # fmt: off # pylint: disable=line-too-long
                self.payments_wallet,
                self.payments_client,
                self.cluster_id,
            )

        uuid = await self.client.compute(
            self.cluster_id,
            compute_bindings,
            store_ids,
            computation_time_secrets,
            receipt_compute,
        )

        return uuid

    async def compute_multithread(
        self, num_threads: int, store_ids: List[List[str]], party_ids: Dict[int, str]
    ) -> List[str]:
        """
        Performs multi-threaded federated averaging computation.

        Args:
            num_threads: Number of threads to use for computation.
            store_ids: List of lists of store IDs for the computation.
            party_ids: Dictionary of party IDs.

        Returns:
            list: List of UUIDs for the computations.
        """
        # fmt: off
        # pylint: disable=no-member
        if self.program_id is None:
            raise ValueError("Program ID is None and cannot be None")
        compute_bindings = nillion.ProgramBindings(self.program_id)
        # fmt: on
        compute_bindings.add_input_party(self.party_names[-1], self.user_id)
        for client_id, user_id in party_ids.items():
            compute_bindings.add_input_party(self.party_names[client_id], user_id)
            compute_bindings.add_output_party(self.party_names[client_id], user_id)

        thread_store_ids = [
            [party_store_ids[thread] for party_store_ids in store_ids]
            for thread in range(num_threads)
        ]

        # fmt: off
        # pylint: disable=no-member
        computation_time_secrets_list = [
            nillion.NadaValues({"program_order": nillion.Integer(thread)}) # type: ignore[dict-item] # fmt: off # pylint: disable=line-too-long
            for thread in range(num_threads)
        ]
        # fmt: on

        async def get_quote_and_pay_wrapper(
            computation_time_secrets: nillion.NadaValues,
        ) -> nillion.PaymentReceipt:
            return await get_quote_and_pay(
                self.client,
                nillion.Operation.compute(self.program_id, computation_time_secrets),  # type: ignore[attr-defined] # fmt: off # pylint: disable=line-too-long
                self.payments_wallet,
                self.payments_client,
                self.cluster_id,
            )

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

            uuids = await asyncio.gather(*futures)

        return uuids


async def main(batch_size, num_threads, client_ids):
    """
    Main function to run the Nillion Network server for federated averaging.

    Args:
        batch_size: The batch size for the computation.
        num_threads: Number of threads to use for computation.
        client_ids: List of client IDs participating in the computation.
    """
    num_parties = len(client_ids)
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
            f"\t poetry run python3 client.py {batch_size} {num_threads} {num_parties} {client_id} {chr(ord('A') + client_id)}"  # pylint: disable=line-too-long
        )
    input("Press ENTER to continue to compute ...")
    store_ids = []
    party_ids = {}
    for client_id in client_ids:
        config = JsonDict.from_json_file(f"/tmp/client_{client_id}.json")
        store_ids.append(config["store_id"])
        party_ids[client_id] = config["party_id"]

    logger.debug("Using store_ids: %s", store_ids)
    logger.debug("Using party_ids: %s", party_ids)

    uuids = await nillion_server.compute_multithread(num_threads, store_ids, party_ids)
    logger.debug("Compute complete with UUIDs: %s", uuids)


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
