""" The NillionNetworkClient class is a client"""

import argparse
import asyncio
import concurrent.futures
import os

import nada_numpy.client as na_client
import numpy as np
import py_nillion_client as nillion
from nillion_python_helpers import get_quote, pay_with_quote

from nillion_fl.logs import logger
from nillion_fl.nilvm.component import NillionNetworkComponent
from nillion_fl.nilvm.utils import JsonDict, store_secret_array

home = os.getenv("HOME")


class NillionNetworkClient(NillionNetworkComponent):
    """
    A client for interacting with the Nillion Network.
    Handles array storage, computation, and result retrieval.
    """

    def __init__(
        self,
        client_id,
        num_parties,
        filename=f"{home}/.config/nillion/nillion-devnet.env",
    ):
        logger.info(
            "🚀  Initializing NillionNetworkClient with client_id %s", client_id
        )
        super().__init__(client_id, num_parties, filename)

    def init(self):
        """Initialize the client. Currently a placeholder method."""
        return None

    async def store_array(self, array, secret_name, program_id, server_user_id):
        """
        Stores the given parameters as a secret array on the Nillion network.

        Args:
            array (np.ndarray): The array to store as a secret
            secret_name (str): The name of the secret
            program_id (str): The program ID to associate with the secret array
            server_user_id (str): The server user ID to associate with the secret array

        Returns:
            str: The store ID of the secret array
        """

        permissions = nillion.Permissions.default_for_user(  # fmt: off # pylint: disable=no-member
            self.client.user_id
        )
        permissions.add_compute_permissions({server_user_id: {program_id}})

        store_id = await store_secret_array(
            self.client,
            self.payments_wallet,
            self.payments_client,
            self.cluster_id,
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

        Args:
            arrays (List[np.ndarray]): The arrays to store
            secret_name (str): The name of the secrets
            program_id (str): The program ID to associate with the secret arrays
            server_user_id (str): The server user ID to associate with the secret arrays

        Returns:
            List[str]: The store IDs of the secret arrays
        """
        permissions = nillion.Permissions.default_for_user(  # fmt: off # pylint: disable=no-member
            self.client.user_id
        )
        permissions.add_compute_permissions({server_user_id: {program_id}})

        stored_secrets = [
            nillion.NadaValues(  # fmt: off # pylint: disable=no-member
                na_client.array(array, secret_name, na_client.SecretRational)
            )
            for array in arrays
        ]

        async def get_quote_wrapper(stored_secret):
            return await get_quote(
                self.client,
                nillion.Operation.store_values(  # pylint: disable=no-member
                    stored_secret, ttl_days=1
                ),
                self.cluster_id,
            )

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                asyncio.wrap_future(
                    executor.submit(asyncio.run, get_quote_wrapper(stored_secret))
                )
                for stored_secret in stored_secrets
            ]

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

    async def get_compute_results_from_nillion(self, num_results):
        """
        Retrieves compute results from the Nillion network.

        Args:
            num_results (int): The number of results to retrieve

        Returns:
            List: A list of compute results
        """
        compute_results = []
        logger.info("⌛ Waiting for result...")
        logger.info(
            "🔍  Current compute results: %d / %d", len(compute_results), num_results
        )
        while len(compute_results) < num_results:
            compute_event = await self.client.next_compute_event()
            if isinstance(
                compute_event,
                nillion.ComputeFinishedEvent,  # fmt: off # pylint: disable=no-member
            ):
                compute_results.append(compute_event.result.value)
                logger.info(
                    "🔍  Current compute results: %d / %d",
                    len(compute_results),
                    num_results,
                )
                logger.debug(
                    "✅  Compute complete for compute_id %s", compute_event.uuid
                )
                # logger.debug("🖥️  The result is %s", compute_event.result.value)
        return compute_results

    async def get_compute_result(self, num_results=1):
        """
        Retrieves and processes compute results from the Nillion network.

        Args:
            num_results (int): The number of results to retrieve

        Returns:
            List[Tuple]: A list of tuples containing program order and result arrays
        """
        results = await self.get_compute_results_from_nillion(num_results)
        output_tuples = []
        for result in results:
            batch_size = len(result) - 1
            result_array = np.zeros((batch_size,))
            for i in range(batch_size):
                result_array[i] = na_client.float_from_rational(
                    result[f"my_output_{i}"]
                )  # Format specific for fed_avg
            logger.debug(
                "📊  Result array (shape): %s \n📊  Program order: %s",
                result_array.shape,
                result["program_order"],
            )
            output_tuples.append(
                (result["program_order"], result_array / self.num_parties)
            )
        return output_tuples


async def main(batch_size, num_parties, client_id, secret_name, num_batches):
    """
    Main function to run the Nillion Network client.

    Args:
        batch_size (int): The number of secrets in the computation
        num_parties (int): The number of parties in the computation
        client_id (int): The client ID number
        secret_name (str): The name of the secret being stored
        num_batches (int): The number of concurrent batches being sent

    Returns:
        List[Tuple]: A list of tuples containing program order and result arrays
    """
    program_config = JsonDict.from_json_file("/tmp/fed_avg.json")
    program_id = program_config["program_id"]
    server_user_id = program_config["server_user_id"]

    nillion_client = NillionNetworkClient(client_id, num_parties)

    store_ids = await nillion_client.store_arrays(
        [np.ones(batch_size) * i for i in range(1, num_batches + 1)],
        secret_name,
        program_id,
        server_user_id,
    )

    JsonDict({"store_id": store_ids, "party_id": nillion_client.party_id}).to_json_file(
        f"/tmp/client_{client_id}.json"
    )

    results = await nillion_client.get_compute_result(num_results=num_batches)
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
        "num_batches",
        type=int,
        help="The number of concurrent batches being sent",
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
            args.num_batches,
        )
    )
