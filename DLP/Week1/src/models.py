import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(name: str):
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU()
    if name == "leaky_relu":
        return nn.LeakyReLU()
    if name == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"Unknown activation: {name}")


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        activation: str = "relu",
        use_softmax: bool = True,
    ):
        super().__init__()

        dims = [input_dim] + hidden_dims
        self.layers = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1))

        self.out = nn.Linear(dims[-1], output_dim)
        self.act = get_activation(activation)
        self.use_softmax = use_softmax

    def forward(self, x):
        for layer in self.layers:
            x = self.act(layer(x))

        x = self.out(x)

        if self.use_softmax:
            x = F.softmax(x, dim=1)

        return x


class CNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        conv_channels: list[int] = [16, 32],
        kernel_size: int = 3,
        activation: str = "relu",
        use_softmax: bool = True,
    ):
        super().__init__()

        self.act = get_activation(activation)
        self.use_softmax = use_softmax

        # --- Convolutional backbone ---
        convs = []
        c_in = in_channels
        for c_out in conv_channels:
            convs.append(nn.Conv2d(c_in, c_out, kernel_size, padding=1))
            convs.append(self.act)
            convs.append(nn.MaxPool2d(2))
            c_in = c_out

        self.conv = nn.Sequential(*convs)

        # For 16x16 input, each MaxPool(2) divides spatial size by 2
        # Example: 16x16 → 8x8 → 4x4 (if 2 conv blocks)
        final_spatial = 16 // (2 ** len(conv_channels))
        fc_input_dim = c_in * final_spatial * final_spatial

        # --- Classifier ---
        self.fc = nn.Linear(fc_input_dim, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)

        if self.use_softmax:
            x = F.softmax(x, dim=1)

        return x
