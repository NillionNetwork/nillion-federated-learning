from nillion_fl.pytorch import PytorchFLServer


def main():
    PytorchFLServer(config={"num_parties": 2, "batch_size": 2500}).serve()


if __name__ == "__main__":
    main()
