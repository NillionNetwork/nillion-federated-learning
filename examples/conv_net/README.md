# Convolutional Neural Network Example - Nillion Federated Learning

This example demonstrates how to run a convolutional neural network model using Nillion Federated Learning for CIFAR 10 dataset.

## Running the Convolutional Neural Network Example

1. Start the Nillion devnet:
```bash
# Terminal 0
nillion-devnet
```

2. Start the server:
```bash
# Terminal 1
poetry run python3 examples/conv_net/server.py
```

3. Start the first client:
```bash
# Terminal 2
poetry run python3 examples/conv_net/client.py 0
```

4. Start the second client:
```bash
# Terminal 3
poetry run python3 examples/conv_net/client.py 1
```