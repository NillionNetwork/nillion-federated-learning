from nillion_fl.server import FederatedLearningServer


def main():
    FederatedLearningServer(config={"num_parties": 2, "batch_size": 1000}).serve()


if __name__ == "__main__":
    main
