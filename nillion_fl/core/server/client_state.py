""" Enum class for client state """

from enum import Enum


class ClientState(Enum):
    """Enum class for client state"""

    INITIAL = 0
    READY = 1
    END = 2
