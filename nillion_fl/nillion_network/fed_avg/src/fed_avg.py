"""Main Nada program"""

import functools
import inspect
import time
from typing import List

# Step 0: Nada Numpy is imported with this line
import nada_numpy as na
from nada_dsl import Integer, Output, SecretInteger

from nillion_fl.nillion_network.fed_avg.src.config import DIM, NUM_PARTIES


def nada_main() -> List[Output]:
    """
    Main dot product Nada program.

    Returns:
        List[Output]: List of program outputs.
    """
    # Step 1: We use Nada Numpy wrapper to create "Party0", "Party1" and "Party2"
    input_parties = na.parties(NUM_PARTIES)
    # output_parties = na.parties(NUM_PARTIES)

    input_arrays = [
        na.array([DIM], input_parties[i], chr(ord("A") + i), SecretInteger)
        for i in range(NUM_PARTIES)
    ]

    result = input_arrays[0]
    for array in input_arrays[1:]:
        result += array

    # result = result / Integer(NUM_PARTIES)

    # Result is output to all parties
    outputs = []
    for i in range(NUM_PARTIES):
        outputs += result.output(input_parties[i], "my_output")

    return outputs
