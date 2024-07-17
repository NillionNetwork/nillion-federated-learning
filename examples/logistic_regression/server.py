from nillion_fl.server import FederatedLearningServicer

if __name__ == "__main__":
    FederatedLearningServicer(num_parties=2).serve()
