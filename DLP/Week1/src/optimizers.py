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


def make_optimizer(name, params, lr):
    name = name.lower()
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr)
    raise ValueError(f"Unknown optimizer {name}")


batch_sizes = [10, 50, 200, 1000]
learning_rates = [0.001, 0.01, 0.1, 1.0, 10.0]
optimizers = ["sgd", "adam", "rmsprop"]
num_epochs_list = [10, 30]
import pandas as pd

dataset = torchvision.datasets.USPS(
    root="USPS/",
    train=True,
    transform=transforms.ToTensor(),
    download=True,
)

criterion = nn.MSELoss()

results = []

batch_sizes = [10, 50, 200, 1000]
learning_rates = [0.001, 0.01, 0.1, 1.0]
optimizers = ["sgd", "adam", "rmsprop"]
num_epochs_list = [10, 30]
architectures = ["mlp", "cnn"]


criterion = nn.MSELoss()

for arch in architectures:
    for optimizer_name in optimizers:
        for batch_size in batch_sizes:
            for lr in learning_rates:
                for num_epochs in num_epochs_list:
                    print(
                        f"{arch.upper()} | opt={optimizer_name} "
                        f"| bs={batch_size} | lr={lr} | epochs={num_epochs}"
                    )

                    # ---- Model ----
                    if arch == "mlp":
                        model = MLP(
                            input_dim=16 * 16,
                            hidden_dims=[20, 20],
                            output_dim=10,
                            activation="relu",
                            use_softmax=True,
                        )
                        model_type = "mlp"

                    else:  # CNN
                        model = CNN(
                            in_channels=1,
                            num_classes=10,
                            conv_channels=[16, 32],
                            activation="relu",
                            use_softmax=True,
                        )
                        model_type = "cnn"

                    # ---- Optimizer ----
                    if optimizer_name == "sgd":
                        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
                    elif optimizer_name == "adam":
                        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                    elif optimizer_name == "rmsprop":
                        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

                    # ---- Train ----
                    losses = train(
                        num_epochs=num_epochs,
                        batch_size=batch_size,
                        criterion=criterion,
                        optimizer=optimizer,
                        model=model,
                        dataset=dataset,
                        model_type=model_type,
                    )

                    # ---- Log ----
                    results.append(
                        {
                            "architecture": arch,
                            "optimizer": optimizer_name,
                            "batch_size": batch_size,
                            "learning_rate": lr,
                            "epochs": num_epochs,
                            "final_loss": losses[-1],
                            "min_loss": min(losses),
                        }
                    )

df = pd.DataFrame(results)
df.sort_values("final_loss").head()


df.to_csv("results_ex2.csv")
