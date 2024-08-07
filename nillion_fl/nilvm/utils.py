"""General utils functions"""

import json
import os
import time
from typing import Any, Callable, Dict, List

import nada_numpy as na
import nada_numpy.client as na_client
import numpy as np
import py_nillion_client as nillion
from cosmpy.aerial.client import LedgerClient
from cosmpy.aerial.wallet import LocalWallet
from nillion_python_helpers import get_quote, get_quote_and_pay, pay_with_quote

from nillion_fl.logs import logger


def async_timer(file_path: os.PathLike) -> Callable:
    """
    Decorator function to measure and log the execution time of asynchronous functions.

    Args:
        file_path (os.PathLike): File to write performance metrics to.

    Returns:
        Callable: Wrapped function with timer.
    """

    def decorator(func: Callable) -> Callable:
        """
        Decorator function.

        Args:
            func (Callable): Function to decorate.

        Returns:
            Callable: Decorated function.
        """

        async def wrapper(*args, **kwargs) -> Any:
            """
            Returns function result and writes execution time to file.

            Returns:
                Any: Function result.
            """
            start_time = time.time()
            result = await func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            with open(file_path, "a", encoding="utf-8") as file:
                file.write(f"{elapsed_time:.6f},\n")
            return result

        return wrapper

    return decorator


# pylint: disable=too-many-arguments
async def store_program(
    client: "nillion.NillionClient",
    payments_wallet: LocalWallet,
    payments_client: LedgerClient,
    user_id: str,
    cluster_id: str,
    program_name: str,
    program_mir_path: str,
    verbose: bool = True,
) -> str:
    """
    Asynchronous function to store a program on the nillion client.

    Args:
        client (nillion.NillionClient): Nillion client.
        payments_wallet (LocalWallet): Payments wallet.
        payments_client (LedgerClient): Payments client.
        user_id (str): User ID.
        cluster_id (str): Cluster ID.
        program_name (str): Program name.
        program_mir_path (str): Path to program MIR.
        verbose (bool, optional): Verbosity level. Defaults to True.

    Returns:
        str: Program ID.
    """
    # fmt: off
    # pylint: disable=no-member
    quote_store_program = await get_quote(
        client, nillion.Operation.store_program(program_mir_path), cluster_id # type: ignore[attr-defined] # fmt: off # pylint: disable=line-too-long
    )
    # fmt: on

    receipt_store_program = await pay_with_quote(
        quote_store_program, payments_wallet, payments_client
    )

    # Store program, passing in the receipt that shows proof of payment
    action_id = await client.store_program(
        cluster_id, program_name, program_mir_path, receipt_store_program
    )

    program_id = f"{user_id}/{program_name}"
    if verbose:
        logger.debug("Stored program. action_id: %s", action_id)
        logger.debug("Stored program_id: %s", program_id)
    return program_id


# pylint: disable=too-many-arguments
async def store_secret_array(
    client: "nillion.NillionClient",
    payments_wallet: LocalWallet,
    payments_client: LedgerClient,
    cluster_id: str,
    secret_array: np.ndarray,
    secret_name: str,
    nada_type: Any,
    ttl_days: int = 1,
    permissions: "nillion.Permissions" = None,
):
    """
    Asynchronous function to store secret arrays on the nillion client.

    Args:
        client (nillion.NillionClient): Nillion client.
        payments_wallet (LocalWallet): Payments wallet.
        payments_client (LedgerClient): Payments client.
        cluster_id (str): Cluster ID.
        secret_array (np.ndarray): Secret array.
        secret_name (str): Secrets name.
        nada_type (Any): Nada type.
        ttl_days (int): Time to live in days.
        permissions (nillion.Permissions): Optional Permissions.

    Returns:
        str: Store ID.
    """
    # fmt: off
    # pylint: disable=no-member
    # Create a secret
    stored_secret = nillion.NadaValues(
        na_client.array(secret_array, secret_name, nada_type)
    )

    # Get cost quote, then pay for operation to store the secret
    receipt_store = await get_quote_and_pay(
        client,
        nillion.Operation.store_values(stored_secret, ttl_days=ttl_days), # type: ignore[attr-defined] # fmt: off # pylint: disable=line-too-long
        payments_wallet,
        payments_client,
        cluster_id,
    )
    # fmt: on

    # Store a secret, passing in the receipt that shows proof of payment
    store_id = await client.store_values(
        cluster_id, stored_secret, permissions, receipt_store
    )
    return store_id


# pylint: disable=too-many-arguments
async def store_secret_value(
    client: "nillion.NillionClient",
    payments_wallet: LocalWallet,
    payments_client: LedgerClient,
    cluster_id: str,
    secret_value: Any,
    secret_name: str,
    nada_type: Any,
    ttl_days: int = 1,
    permissions: "nillion.Permissions" = None,
):
    """
    Asynchronous function to store secret values on the nillion client.

    Args:
        client (nillion.NillionClient): Nillion client.
        payments_wallet (LocalWallet): Payments wallet.
        payments_client (LedgerClient): Payments client.
        cluster_id (str): Cluster ID.
        secret_value (Any): Secret single value.
        secret_name (str): Secrets name.
        nada_type (Any): Nada type.
        ttl_days (int): Time to live in days.
        permissions (nillion.Permissions): Optional Permissions.

    Returns:
        str: Store ID.
    """
    # fmt: off
    # pylint: disable=no-member
    if nada_type == na.Rational:
        secret_value = round(secret_value * 2 ** na.get_log_scale())
        nada_type = nillion.Integer
    elif nada_type == na.SecretRational:
        secret_value = round(secret_value * 2 ** na.get_log_scale())
        nada_type = nillion.SecretInteger

    # Create a secret
    stored_secret = nillion.NadaValues(
        {
            secret_name: nada_type(secret_value),
        }
    )

    # Get cost quote, then pay for operation to store the secret
    receipt_store = await get_quote_and_pay(
        client,
        nillion.Operation.store_values(stored_secret, ttl_days=ttl_days), # type: ignore[attr-defined] # fmt: off # pylint: disable=line-too-long
        payments_wallet,
        payments_client,
        cluster_id,
    )
    # fmt: on

    # Store a secret, passing in the receipt that shows proof of payment
    store_id = await client.store_values(
        cluster_id, stored_secret, permissions, receipt_store
    )
    return store_id


# pylint: disable=too-many-arguments
async def compute(
    client: "nillion.NillionClient",
    payments_wallet: LocalWallet,
    payments_client: LedgerClient,
    program_id: str,
    cluster_id: str,
    compute_bindings: "nillion.ProgramBindings",
    store_ids: List[str],
    computation_time_secrets: "nillion.NadaValues",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Asynchronous function to perform computation on the nillion client.

    Args:
        client (nillion.NillionClient): Nillion client.
        payments_wallet (LocalWallet): Payments wallet.
        payments_client (LedgerClient): Payments client.
        program_id (str): Program ID.
        cluster_id (str): Cluster ID.
        compute_bindings (nillion.ProgramBindings): Compute bindings.
        store_ids (List[str]): List of data store IDs.
        computation_time_secrets (nillion.Secrets): Computation time secrets.
        verbose (bool, optional): Verbosity level. Defaults to True.

    Returns:
        Dict[str, Any]: Result of computation.
    """
    # fmt: off
    # pylint: disable=no-member
    receipt_compute = await get_quote_and_pay(
        client,
        nillion.Operation.compute(program_id, computation_time_secrets), # type: ignore[attr-defined] # fmt: off # pylint: disable=line-too-long
        payments_wallet,
        payments_client,
        cluster_id,
    )

    # Compute, passing all params including the receipt that shows proof of payment
    await client.compute(
        cluster_id,
        compute_bindings,
        store_ids,
        computation_time_secrets,
        receipt_compute,
    )
    while True:
        compute_event = await client.next_compute_event()
        if isinstance(compute_event, nillion.ComputeFinishedEvent):
            if verbose:
                logger.debug(
                    "✅  Compute complete for compute_id %s",
                    compute_event.uuid
                )
                logger.debug("🖥️  The result is %s", compute_event.result.value)
            return compute_event.result.value
    # fmt: on


class JsonDict(dict):
    """A dictionary class with JSON serialization and deserialization capabilities."""

    @staticmethod
    def from_json(json_str: str) -> "JsonDict":
        """
        Create a JsonDict from a JSON string.

        Args:
            json_str (str): JSON string.

        Returns:
            JsonDict: JsonDict object.
        """
        return JsonDict(json.loads(json_str))

    def to_json(self) -> str:
        """
        Convert JsonDict to a JSON string.

        Returns:
            str: JSON string.
        """
        return json.dumps(self)

    @staticmethod
    def from_json_file(file_path: os.PathLike) -> "JsonDict":
        """
        Create a JsonDict from a JSON file.

        Args:
            file_path (os.PathLike): File path.

        Returns:
            JsonDict: JsonDict object.
        """
        logger.debug("Loading JsonDict from file: %s", file_path)

        with open(file_path, "r", encoding="utf-8") as file:
            return JsonDict(json.load(file))

    def to_json_file(self, file_path: os.PathLike) -> None:
        """
        Write JsonDict to a JSON file.

        Args:
            file_path (os.PathLike): File path.
        """
        logger.debug("Storing JsonDict to file: %s", file_path)
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(self, file)
