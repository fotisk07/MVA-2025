from itertools import product

import pandas as pd
import torch
import torchvision
from models import CNN, MLP
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms


def train(
    num_epochs,
    batch_size,
    criterion,
    optimizer,
    model,
    dataset,
    model_type="mlp",  # "mlp" or "cnn"
):
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.train()

    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for images, labels in train_loader:
            if model_type == "mlp":
                x = images.view(images.size(0), -1)
            else:
                x = images  # CNN keeps [N, C, H, W]

            # Forward
            y_pred = model(x)

            # One-hot labels
            labels_one_hot = torch.zeros(labels.size(0), 10)
            labels_one_hot.scatter_(1, labels.view(-1, 1), 1)

            loss = criterion(y_pred, labels_one_hot)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * labels.size(0) / len(dataset)

        losses.append(epoch_loss)

    return losses


results = []

# Sweep space
model_types = ["mlp", "cnn"]
n_layers_list = [1, 2, 3]
hidden_sizes = [5, 10, 20]
activations = ["relu", "tanh", "sigmoid"]

num_epochs = 10
batch_size = 10

dataset = torchvision.datasets.USPS(
    root="USPS/",
    train=True,
    transform=transforms.ToTensor(),
    download=True,
)

criterion = nn.MSELoss()

for model_type, n_layers, hidden_size, activation in product(
    model_types,
    n_layers_list,
    hidden_sizes,
    activations,
):
    print(f"\nRunning {model_type} | layers={n_layers} | hidden={hidden_size} | act={activation}")

    if model_type == "mlp":
        hidden_dims = [hidden_size] * n_layers
        model = MLP(
            input_dim=16 * 16,
            hidden_dims=hidden_dims,
            output_dim=10,
            activation=activation,
            use_softmax=True,
        )

    else:  # CNN
        conv_channels = [hidden_size] * n_layers
        model = CNN(
            in_channels=1,
            num_classes=10,
            conv_channels=conv_channels,
            activation=activation,
            use_softmax=True,
        )

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    losses = train(
        num_epochs=num_epochs,
        batch_size=batch_size,
        criterion=criterion,
        optimizer=optimizer,
        model=model,
        dataset=dataset,
        model_type=model_type,
    )

    results.append(
        {
            "model_type": model_type,
            "n_layers": n_layers,
            "hidden_size": hidden_size,
            "activation": activation,
            "final_loss": losses[-1],
            "min_loss": min(losses),
        },
    )

df = pd.DataFrame(results)
df = df.sort_values("final_loss")
df.to_csv("results_ex1.csv")
print(df.head())
