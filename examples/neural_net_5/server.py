from nillion_fl.pytorch import PytorchFLServer


def main():
    PytorchFLServer(config={"num_parties": 5, "batch_size": 1000}).serve()


if __name__ == "__main__":
    main()
