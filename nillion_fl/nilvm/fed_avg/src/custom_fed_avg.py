"""This is an example file to showcase how the resulting output would look like"""

from typing import List

# Step 0: Nada Numpy is imported with this line
import nada_numpy as na
from nada_dsl import Output, PublicInteger, Input, SecretInteger

NUM_PARTIES = 2
DIM = 2500

def nada_main() -> List[Output]:
    """
    Main dot product Nada program.

    Returns:
        List[Output]: List of program outputs.
    """
    # Step 1: We use Nada Numpy wrapper to create "Party0", "Party1" and "Party2"
    input_parties = na.parties(NUM_PARTIES + 1)

    coordinator = input_parties[-1]
    input_parties = input_parties[:-1]

    program_order = PublicInteger(
        Input(name="program_order", party=coordinator)
    )  # This is the program in execution order

    input_arrays = [
        na.array([DIM], party, chr(ord("A") + i), SecretInteger)
        for i, party in enumerate(input_parties)
    ]

    result = input_arrays[0]
    for array in input_arrays[1:]:
        result += array

    # Result is output to all parties
    outputs = []
    for party in input_parties:
        outputs += result.output(
            party, "my_output"
        )  # The output is given to all parties
        outputs += [
            Output(program_order, "program_order", party)
        ]  # The program order is also publically given to all parties

    return outputs
