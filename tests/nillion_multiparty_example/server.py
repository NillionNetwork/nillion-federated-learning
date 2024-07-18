"""Dot Product example script"""

import argparse
import asyncio
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
print(
    "Appended to path: ", os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
print(
    "Appended to path: ",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")),
)


import nada_numpy.client as na_client
import numpy as np
import py_nillion_client as nillion
from cosmpy.aerial.client import LedgerClient
from cosmpy.aerial.wallet import LocalWallet
from cosmpy.crypto.keypairs import PrivateKey
from dotenv import load_dotenv
from nillion_python_helpers import (create_nillion_client,
                                    create_payments_config, get_quote,
                                    get_quote_and_pay, pay_with_quote)
from py_nillion_client import NodeKey, UserKey

from nillion_fl.nillion_network.fed_avg.src.config import DIM, NUM_PARTIES
from nillion_fl.nillion_network.utils import JsonDict, store_program

home = os.getenv("HOME")
load_dotenv(f"{home}/.config/nillion/nillion-devnet.env")
# load_dotenv(f"/workspaces/ai/.nillion-testnet.env")


# Main asynchronous function to coordinate the process
async def main(client_ids: list) -> None:
    """Main nada program"""

    cluster_id = os.getenv("NILLION_CLUSTER_ID")
    grpc_endpoint = os.getenv("NILLION_NILCHAIN_GRPC")
    chain_id = os.getenv("NILLION_NILCHAIN_CHAIN_ID")
    seed = "my_server_seed"
    userkey = UserKey.from_seed((seed))
    nodekey = NodeKey.from_seed((seed))

    client = create_nillion_client(userkey, nodekey)
    user_id = client.user_id
    program_name = "fed_avg"
    program_mir_path = (
        f"nillion_fl/nillion_network/{program_name}/target/{program_name}.nada.bin"
    )

    # Create payments config and set up Nillion wallet with a private key to pay for operations
    payments_config = create_payments_config(chain_id, grpc_endpoint)
    payments_client = LedgerClient(payments_config)
    payments_wallet = LocalWallet(
        PrivateKey(bytes.fromhex(os.getenv("NILLION_NILCHAIN_PRIVATE_KEY_9"))),
        prefix="nillion",
    )

    party_names = na_client.parties(NUM_PARTIES * 2)
    input_party_names = party_names[:NUM_PARTIES]
    output_party_names = party_names[NUM_PARTIES:]

    ##### STORE PROGRAM
    print("-----STORE PROGRAM")

    program_id = await store_program(
        client,
        payments_wallet,
        payments_client,
        user_id,
        cluster_id,
        program_name,
        program_mir_path,
    )

    JsonDict({"program_id": program_id, "server_user_id": user_id}).to_json_file(
        "/tmp/fed_avg.json"
    )

    input("Press ENTER to continue to compute ...")

    ##### COMPUTE
    print("-----COMPUTE")

    # Bind the parties in the computation to the client to set input and output parties
    compute_bindings = nillion.ProgramBindings(program_id)

    store_ids = []
    for i, client_id in enumerate(client_ids):
        config = JsonDict.from_json_file(f"/tmp/client_{client_id}.json")
        store_ids.append(config["store_id"])
        compute_bindings.add_input_party(input_party_names[i], config["party_id"])
        compute_bindings.add_output_party(input_party_names[i], config["party_id"])

    print("Using store_ids: ", store_ids)

    # Create a computation time secret to use
    computation_time_secrets = nillion.NadaValues({})

    # Get cost quote, then pay for operation to compute
    receipt_compute = await get_quote_and_pay(
        client,
        nillion.Operation.compute(program_id, computation_time_secrets),
        payments_wallet,
        payments_client,
        cluster_id,
    )

    # Compute, passing all params including the receipt that shows proof of payment
    uuid = await client.compute(
        cluster_id,
        compute_bindings,
        store_ids,
        computation_time_secrets,
        receipt_compute,
    )

    JsonDict({"uuid": uuid}).to_json_file("/tmp/compute.json")

    # input("Press ENTER to FINISH SERVER EXECUTION...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the client program with a client ID",
        epilog="Example: python client.py 1",
    )
    parser.add_argument(
        "client_ids",
        type=int,
        nargs="+",
        help="A list of integers as client arguments (at least 2)",
    )

    args = parser.parse_args()

    asyncio.run(main(args.client_ids))
