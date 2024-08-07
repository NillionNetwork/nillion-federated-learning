"""
This module contains server-side components for the Nillion Federated Learning (FL) framework.

The server-side components handle various aspects of federated learning, including:

- Managing client connections and communication.
- Scheduling and coordinating learning iterations.
- Aggregating client updates and integrating with the Nillion network.

Key classes in this module:
- `LearningManager`: Manages the learning process, 
    including scheduling iterations and handling client interactions.
- `ClientManager`: Handles client connections, token validation, 
    and message passing between clients and the server.
- `FLClientCore`: Core client class for handling federated learning operations.

This module works in conjunction with other components of the Nillion FL 
    framework to facilitate distributed learning across multiple parties.
"""

from nillion_fl.client import FederatedLearningClient
from nillion_fl.server import FederatedLearningServer
