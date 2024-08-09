# Neural Network Example with 5 parties - Nillion Federated Learning

This example demonstrates how to run a neural network model using Nillion Federated Learning.

## Running the 5-party Neural Network Example

1. Start the Nillion devnet:
```bash
# Terminal 0
nillion-devnet
```

2. Start the server:
```bash
# Terminal 1
poetry run python3 examples/neural_net_5/server.py
```

3. Start the first client:
```bash
# Terminal 2
poetry run python3 examples/neural_net_5/client.py 0
```

4. Start the second client:
```bash
# Terminal 3
poetry run python3 examples/neural_net_5/client.py 1
```

5. Start the first client:
```bash
# Terminal 4
poetry run python3 examples/neural_net_5/client.py 2
```

6. Start the second client:
```bash
# Terminal 5
poetry run python3 examples/neural_net_5/client.py 3
```

7. Start the first client:
```bash
# Terminal 6
poetry run python3 examples/neural_net_5/client.py 4
```
