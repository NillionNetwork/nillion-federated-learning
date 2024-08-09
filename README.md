# Nillion Federated Learning

This repository contains the Proof of Concept (PoC) for running Federated Learning on top of Nillion. It consists of two main components: a Server and a Client.

## Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Setup](#setup)
4. [Running the System](#running-the-system)
5. [Regenerating gRPC](#regenerating-grpc)
6. [Project Structure](#project-structure)
7. [License](#license)

## Introduction

Federated Learning is a machine learning technique that trains algorithms across multiple decentralized edge devices or servers holding local data samples, without exchanging them. This project demonstrates how to implement Federated Learning using Nillion's secure infrastructure.

## Prerequisites

- Python 3.10 or higher
- pip
- poetry

## Setup

1. Install poetry:

```bash
pip install poetry
```

2. Install project dependencies with examples extras to run the examples:

```bash
poetry install -E examples
```

## Running the System

To run the Federated Learning system, you need to start the Nillion devnet, the server, and at least two clients.

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

You can start additional clients by incrementing the client number on the `server.py` file.

## Regenerating gRPC

If you make changes to the `fl_service.proto` file, you need to regenerate the gRPC code:

```bash
poetry run python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. nillion_fl/network/fl_service.proto
```

## Project Structure

- `examples/`: Includes various example implementations (convolutional networks, logistic regression, neural networks).
- `nillion_fl/`: The main package for Nillion Federated Learning (contains base client, and server implementations)
  - `core/`: Core components for client and server implementations.
  - `network/`: Contains protocol buffer definitions for network communication.
  - `nilvm/`: Nillion Network related components, including federated averaging implementation.
  - `pytorch/`: PyTorch-specific implementations for client and server.

## License

This project is licensed under the Apache2 License. See the LICENSE file for details.