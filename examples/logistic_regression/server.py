from nillion_fl.server import FederatedLearningServer


def main():
    FederatedLearningServer(num_parties=2, batch_size=100).serve()


if __name__ == "__main__":
    main
