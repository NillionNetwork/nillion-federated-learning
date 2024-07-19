# Nillion Federated Learning

This repository contains the PoC for running Federated Learning on top of Nillion. It consists of two pieces of code, a Server and a Client. 


## Setup

```bash
pip install poetry
poetry install -E examples
```

On a separate terminal:

```bash
# Terminal 0
nillion-devnet
```

```bash
# Terminal 1
poetry run lr_server
```


```bash
# Terminal 2
poetry run lr_client 0
```

```bash
# Terminal 3
poetry run lr_client 1
```