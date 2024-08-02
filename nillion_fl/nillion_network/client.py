"""Dot Product example script"""

import argparse
import asyncio
import concurrent.futures
import os
from functools import partial

import nada_numpy.client as na_client
import numpy as np
import py_nillion_client as nillion
from cosmpy.aerial.client import LedgerClient
from cosmpy.aerial.wallet import LocalWallet
from cosmpy.crypto.keypairs import PrivateKey
from dotenv import load_dotenv
from nillion_fl.logs import logger
from nillion_fl.nillion_network.component import NillionNetworkComponent
from nillion_fl.nillion_network.utils import JsonDict, store_secret_array
from nillion_python_helpers import create_nillion_client, create_payments_config
from py_nillion_client import NodeKey, UserKey
from nillion_python_helpers import get_quote, get_quote_and_pay, pay_with_quote

home = os.getenv("HOME")
# load_dotenv(f"{home}/.config/nillion/nillion-devnet.env")
# load_dotenv(f"/workspaces/ai/.nillion-testnet.env")


class NillionNetworkClient(NillionNetworkComponent):

    def __init__(
        self,
        client_id,
        num_parties,
        num_threads,
        filename=f"{home}/.config/nillion/nillion-devnet.env",
    ):
        super().__init__(client_id, num_parties, filename)
        self.num_threads = num_threads

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
            na_client.SecretRational,
            1,
            permissions,
        )

        return store_id

    async def store_arrays(self, arrays, secret_name, program_id, server_user_id):
        """
        Stores multiple arrays using thread parallelism, while maintaining async compatibility.
        """
        # Create a permissions object to attach to the stored secret
        permissions = nillion.Permissions.default_for_user(self.client.user_id)
        permissions.add_compute_permissions({server_user_id: {program_id}})

        # Create secrets
        stored_secrets = [
            nillion.NadaValues(
                na_client.array(array, secret_name, na_client.SecretRational)
            )
            for array in arrays
        ]

        async def get_quote_wrapper(stored_secret):
            return await get_quote(
                self.client,
                nillion.Operation.store_values(stored_secret, ttl_days=1),
                self.cluster_id,
            )

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                asyncio.wrap_future(
                    executor.submit(asyncio.run, get_quote_wrapper(stored_secret))
                )
                for stored_secret in stored_secrets
            ]

            # Wait for all futures to complete
            quotes = await asyncio.gather(*futures)

        async def pay_with_quote_wrapper(quote):
            return await pay_with_quote(
                quote,
                self.payments_wallet,
                self.payments_client,
            )

        receipts_store = await asyncio.gather(
            *[pay_with_quote_wrapper(quote) for quote in quotes]
        )

        # # Get quotes and pay
        # async def get_quote_and_pay_wrapper(stored_secret):
        #     return await get_quote_and_pay(
        #         self.client,
        #         nillion.Operation.store_values(stored_secret, ttl_days=1),
        #         self.payments_wallet,
        #         self.payments_client,
        #         self.cluster_id,
        #     )

        # receipts_store = await asyncio.gather(
        #     *[get_quote_and_pay_wrapper(secret) for secret in stored_secrets]
        # )

        # Prepare store_values operations
        async def store_values_wrapper(stored_secret, receipt_store):
            return await self.client.store_values(
                self.cluster_id, stored_secret, permissions, receipt_store
            )

        # Use ThreadPoolExecutor to parallelize the operation
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                asyncio.wrap_future(
                    executor.submit(
                        asyncio.run, store_values_wrapper(stored_secret, receipt_store)
                    )
                )
                for stored_secret, receipt_store in zip(stored_secrets, receipts_store)
            ]

            # Wait for all futures to complete
            store_ids = await asyncio.gather(*futures)

        return store_ids

    async def get_compute_results_from_nillion(self):
        compute_results = []
        logger.info("⌛ Waiting for result...")
        while len(compute_results) < self.num_threads:
            logger.debug(
                f"🔍  Current compute results: {len(compute_results)} / {self.num_threads}"
            )
            compute_event = await self.client.next_compute_event()
            if isinstance(compute_event, nillion.ComputeFinishedEvent):
                logger.debug(
                    f"✅  Compute complete for compute_id {compute_event.uuid}"
                )
                logger.debug(f"🖥️  The result is {compute_event.result.value}")
                compute_results.append(compute_event.result.value)
        return compute_results

    async def get_compute_result(self):
        results = await self.get_compute_results_from_nillion()
        output_tuples = []
        for result in results:
            batch_size = len(result) - 1
            result_array = np.zeros((batch_size,))
            for i in range(batch_size):
                result_array[i] = na_client.float_from_rational(
                    result[f"my_output_{i}"]
                )  # Format specific for fed_avg
            logger.debug(
                "\n"
                f"📊  Result array: {result_array}"
                f"\n📊  Program order: {result['program_order']}"
            )
            output_tuples.append(
                (result["program_order"], result_array / self.num_parties)
            )
        return output_tuples


async def main(batch_size, num_parties, client_id, secret_name, num_threads):
    program_config = JsonDict.from_json_file("/tmp/fed_avg.json")
    program_id = program_config["program_id"]
    server_user_id = program_config["server_user_id"]

    nillion_client = NillionNetworkClient(client_id, num_parties, num_threads)

    store_ids = await nillion_client.store_arrays(
        [np.ones(batch_size) * i for i in range(1, num_threads + 1)],
        secret_name,
        program_id,
        server_user_id,
    )

    JsonDict({"store_id": store_ids, "party_id": nillion_client.party_id}).to_json_file(
        f"/tmp/client_{client_id}.json"
    )

    results = await nillion_client.get_compute_result()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the client program with a client ID",
        epilog="Example: python client.py 10 5 2 0 'A'",
    )
    parser.add_argument(
        "batch_size",
        type=int,
        help="The number of secrets in the computation",
    )
    parser.add_argument(
        "num_threads",
        type=int,
        help="The number of concurrent computations",
    )
    parser.add_argument(
        "num_parties",
        type=int,
        help="The number of parties in the computation",
    )

    parser.add_argument(
        "client_id",
        type=int,
        help="The client ID number (a value from 0 to 9 representing the client_id)",
    )
    parser.add_argument(
        "secret_name",
        type=str,
        help="The name of the secret being stored",
    )
    args = parser.parse_args()

    asyncio.run(
        main(
            args.batch_size,
            args.num_parties,
            args.client_id,
            args.secret_name,
            args.num_threads,
        )
    )
