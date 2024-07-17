import torch
from torch import nn, optim


class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.fc = nn.Linear(10, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


if __name__ == "__main__":
    from examples.logistic_regression.dataset import load_datasets

    # Training parameters
    num_epochs = 20
    batch_size = 32
    learning_rate = 0.001

    # Instantiate the model, loss function, and optimizer
    model = Net()
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Generate data
    trainloaders, valloaders = load_datasets(1)

    for epoch in range(num_epochs):
        for batch_inputs, batch_labels in trainloaders[0]:
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    print("Finished Training")
    print(model.state_dict())
