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
from nillion_python_helpers import (
    create_nillion_client,
    create_payments_config,
    get_quote,
    get_quote_and_pay,
    pay_with_quote,
)
from py_nillion_client import NodeKey, UserKey

from nillion_fl.nillion_network.fed_avg.src.config import DIM, NUM_PARTIES
from nillion_fl.nillion_network.utils import (
    JsonDict,
    compute,
    store_program,
    store_secret_array,
)

home = os.getenv("HOME")
load_dotenv(f"{home}/.config/nillion/nillion-devnet.env")
# load_dotenv(f"/workspaces/ai/.nillion-testnet.env")


# Main asynchronous function to coordinate the process
async def main(client_id: int, secret_name: str) -> None:
    """Main nada program"""

    cluster_id = os.getenv("NILLION_CLUSTER_ID")
    grpc_endpoint = os.getenv("NILLION_NILCHAIN_GRPC")
    chain_id = os.getenv("NILLION_NILCHAIN_CHAIN_ID")
    seed = "my_seed_" + str(client_id)
    userkey = UserKey.from_seed((seed))
    nodekey = NodeKey.from_seed((seed))
    client = create_nillion_client(userkey, nodekey)
    party_id = client.party_id
    user_id = client.user_id

    party_names = na_client.parties(NUM_PARTIES)
    program_name = "dot_product"
    program_mir_path = f"target/{program_name}.nada.bin"

    # Create payments config and set up Nillion wallet with a private key to pay for operations
    payments_config = create_payments_config(chain_id, grpc_endpoint)
    payments_client = LedgerClient(payments_config)
    payments_wallet = LocalWallet(
        PrivateKey(
            bytes.fromhex(os.getenv(f"NILLION_NILCHAIN_PRIVATE_KEY_{client_id}"))
        ),
        prefix="nillion",
    )

    program_config = JsonDict.from_json_file("/tmp/fed_avg.json")
    program_id = program_config["program_id"]
    server_user_id = program_config["server_user_id"]

    ##### STORE SECRETS
    print("-----STORE SECRETS")
    A = np.ones([DIM])

    # Create a permissions object to attach to the stored secret
    permissions = nillion.Permissions.default_for_user(client.user_id)
    permissions.add_compute_permissions({server_user_id: {program_id}})

    # Create a secret
    store_id = await store_secret_array(
        client,
        payments_wallet,
        payments_client,
        cluster_id,
        program_id,
        A,
        secret_name,
        nillion.SecretInteger,
        1,
        permissions,
    )

    print("Stored secret array. store_id:", store_id)
    JsonDict({"store_id": store_id, "party_id": party_id}).to_json_file(
        f"/tmp/client_{client_id}.json"
    )

    # input("Press ENTER to continue to compute ...")

    print("⌛ Waiting for result...")
    while True:
        compute_event = await client.next_compute_event()
        if isinstance(compute_event, nillion.ComputeFinishedEvent):
            print(f"✅  Compute complete for compute_id {compute_event.uuid}")
            print(f"🖥️  The result is {compute_event.result.value}")
            return compute_event.result.value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the client program with a client ID",
        epilog="Example: python client.py 1",
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

    asyncio.run(main(args.client_id, args.secret_name))
