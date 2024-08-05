from nillion_fl.server import FederatedLearningServicer


def main():
    FederatedLearningServicer(num_parties=2, batch_size=100).serve()


if __name__ == "__main__":
    main
