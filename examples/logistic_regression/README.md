# Logistic Regression Example - Nillion Federated Learning

This example demonstrates how to run a logistic regression model using Nillion Federated Learning.

## Running the Logistic Regression Example

1. Start the Nillion devnet:
```bash
# Terminal 0
nillion-devnet
```

2. Start the server:
```bash
# Terminal 1
poetry run python3 examples/logistic_regression/server.py
```

3. Start the first client:
```bash
# Terminal 2
poetry run python3 examples/logistic_regression/client.py 0
```

4. Start the second client:
```bash
# Terminal 3
poetry run python3 examples/logistic_regression/client.py 1
```