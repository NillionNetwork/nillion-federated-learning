import torch
from torch import nn, optim


class NeuralNet(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()
        self.linear_1 = nn.Linear(input_dim, 10, bias=False)
        self.linear_2 = nn.Linear(10, 1, bias=False)

    def forward(self, x):
        return torch.sigmoid(self.linear_2(self.linear_1(x)))


if __name__ == "__main__":
    from examples.neural_net.dataset import load_datasets

    # Training parameters
    num_epochs = 20
    learning_rate = 0.01

    # Instantiate the model, loss function, and optimizer
    input_dim = 750  # Number of features in our dataset
    model = NeuralNet(input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Generate data
    trainloaders, valloaders = load_datasets(
        1, batch_size=0, num_features=input_dim
    )  # We're using only one client for this example

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_inputs, batch_labels in trainloaders[0]:
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(trainloaders[0])
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in valloaders[0]:
                predicted = model(inputs)
                labels = (labels > 0.5).float()
                predicted = (predicted > 0.5).float()
                total += predicted.size(0)
                correct += (predicted == labels).sum().item()
                #if predicted != labels:
                #    print(f"Predicted: {predicted}, Actual: {labels}")

        accuracy = 100 * correct / total
        print(f"Validation Accuracy: {accuracy:.2f}%")

    print("Finished Training")
    print(model.state_dict())
