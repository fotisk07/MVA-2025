import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.integrate import solve_ivp
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def duffing(array_x: np.ndarray) -> np.ndarray:
    array_dx = np.zeros(array_x.shape)
    array_dx[0] = array_x[1]
    array_dx[1] = array_x[0] - array_x[0] ** 3
    return array_dx


class Encoder(nn.Module):
    def __init__(self, list_layer_dim: list[int]):
        super().__init__()
        self.list_layer_dim = list_layer_dim
        self.list_FC = nn.ModuleList()
        for i in range(len(self.list_layer_dim) - 1):
            self.list_FC.append(nn.Linear(self.list_layer_dim[i], self.list_layer_dim[i + 1]))

    def forward(self, tensor2d_x: torch.Tensor) -> torch.Tensor:
        for i in range(len(self.list_layer_dim) - 2):
            tensor2d_x = F.elu(self.list_FC[i](tensor2d_x))
        return self.list_FC[-1](tensor2d_x)


class Decoder(nn.Module):
    def __init__(self, list_layer_dim: list[int]):
        super().__init__()
        self.list_layer_dim = list_layer_dim
        self.list_FC = nn.ModuleList()
        for i in range(len(self.list_layer_dim) - 1, 0, -1):
            self.list_FC.append(nn.Linear(self.list_layer_dim[i], self.list_layer_dim[i - 1]))

    def forward(self, tensor2d_x: torch.Tensor) -> torch.Tensor:
        for i in range(len(self.list_layer_dim) - 2):
            tensor2d_x = F.elu(self.list_FC[i](tensor2d_x))
        return self.list_FC[-1](tensor2d_x)


class Autoencoder(nn.Module):
    def __init__(self, feature_dim: int, hidden_layer: int, output_dim: int):
        super().__init__()
        list_layer_dim = [
            output_dim if i == hidden_layer else feature_dim + i * (output_dim - feature_dim) // hidden_layer
            for i in range(hidden_layer + 1)
        ]
        self.encoder = Encoder(list_layer_dim)
        self.decoder = Decoder(list_layer_dim)


class KoopmanOperator(nn.Module):
    def __init__(self, koopman_operator_dim: int):
        super().__init__()
        self.koopman_operator_dim = koopman_operator_dim
        self.raw_weight = nn.Parameter(torch.randn(koopman_operator_dim, koopman_operator_dim) * 0.02)

    def koopman_matrix(self) -> torch.Tensor:
        return F.softmax(self.raw_weight, dim=1)

    def forward(self, tensor2d_x: torch.Tensor) -> torch.Tensor:
        if tensor2d_x.shape[1] != self.koopman_operator_dim:
            sys.exit(
                f"Wrong Input Features. Please use tensor with {self.koopman_operator_dim} Input Features"
            )
        return tensor2d_x @ self.koopman_matrix().T


def build_dataset(
    t_max: int = 500,
    n_iter: int = 5000,
    n_initial_conditions: int = 60,
    batch_size: int = 2000,
):
    dim_system = 2
    matrix_x0 = (np.random.rand(n_initial_conditions, dim_system) - 0.5) * 4
    array_t = np.linspace(0, t_max, n_iter)
    array3d_xt = np.zeros((matrix_x0.shape[0], matrix_x0.shape[1], n_iter))

    for i in tqdm(range(matrix_x0.shape[0]), desc="Generating trajectories"):
        ode_result = solve_ivp(
            lambda _t, array_x: duffing(array_x),
            [0, t_max],
            matrix_x0[i],
            method="RK45",
            t_eval=array_t,
        )
        array3d_xt[i, :] = ode_result.y

    matrix_x_data = array3d_xt[:, :, :-1].swapaxes(0, 1).reshape(2, -1).T
    matrix_x_next_data = array3d_xt[:, :, 1:].swapaxes(0, 1).reshape(2, -1).T

    (matrix_x_data_train, matrix_x_data_test, matrix_x_next_data_train, matrix_x_next_data_test) = (
        train_test_split(matrix_x_data, matrix_x_next_data, test_size=0.2, random_state=0)
    )

    matrix_x_data_train = matrix_x_data_train.astype(np.float32)
    matrix_x_data_test = matrix_x_data_test.astype(np.float32)
    matrix_x_next_data_train = matrix_x_next_data_train.astype(np.float32)
    matrix_x_next_data_test = matrix_x_next_data_test.astype(np.float32)

    phase_mean = matrix_x_data_train.mean(axis=0, keepdims=True)
    phase_std = matrix_x_data_train.std(axis=0, keepdims=True) + 1e-6

    matrix_x_data_train = (matrix_x_data_train - phase_mean) / phase_std
    matrix_x_data_test = (matrix_x_data_test - phase_mean) / phase_std
    matrix_x_next_data_train = (matrix_x_next_data_train - phase_mean) / phase_std
    matrix_x_next_data_test = (matrix_x_next_data_test - phase_mean) / phase_std

    train_dataset = TensorDataset(
        torch.from_numpy(matrix_x_data_train), torch.from_numpy(matrix_x_next_data_train)
    )
    test_dataset = TensorDataset(
        torch.from_numpy(matrix_x_data_test), torch.from_numpy(matrix_x_next_data_test)
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader


def loss_koopman(
    tensor2d_x: torch.Tensor,
    tensor2d_x_next: torch.Tensor,
    tensor2d_observable: torch.Tensor,
    tensor2d_decoded_x: torch.Tensor,
    tensor2d_observable_next: torch.Tensor,
    tensor2d_koopman_observable: torch.Tensor,
    tensor2d_predict_x: torch.Tensor,
):
    criterion = nn.MSELoss()

    loss_reconstruction = criterion(tensor2d_decoded_x, tensor2d_x)
    loss_koopman_space = criterion(tensor2d_koopman_observable, tensor2d_observable_next)
    loss_phase_space = criterion(tensor2d_predict_x, tensor2d_x_next)

    latent_mean_penalty = tensor2d_observable.mean(dim=0).pow(2).mean()
    latent_std_penalty = (tensor2d_observable.std(dim=0) + 1e-6).sub(1.0).pow(2).mean()
    loss_latent = latent_mean_penalty + latent_std_penalty

    lambda_recon = 1.0
    lambda_koop = 5.0
    lambda_phase = 2.0
    lambda_latent = 0.1

    total = (
        lambda_recon * loss_reconstruction
        + lambda_koop * loss_koopman_space
        + lambda_phase * loss_phase_space
        + lambda_latent * loss_latent
    )
    return total, loss_reconstruction, loss_koopman_space, loss_phase_space, loss_latent


def plot_losses(history: dict[str, list[float]]) -> None:
    epochs = np.arange(len(history["train_total"]))
    fig, axes = plt.subplots(1, 5, figsize=(28, 6))

    plots = [
        ("recon", "Reconstruction Loss", "steelblue"),
        ("koop", "Koopman Space Loss", "tomato"),
        ("phase", "Phase Space Loss", "seagreen"),
        ("latent", "Latent Regularisation", "darkorange"),
        ("total", "Total Loss", "purple"),
    ]

    for ax, (key, title, color) in zip(axes, plots):
        ax.plot(epochs, history[f"train_{key}"], label="Train", marker="o", color=color)
        ax.plot(
            epochs,
            history[f"test_{key}"],
            label="Test",
            marker="o",
            linestyle="--",
            color=color,
            alpha=0.5,
        )
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.show()


def train_koopman(n_epoch: int = 30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    np.random.seed(0)

    train_dataloader, test_dataloader = build_dataset()

    feature_dim = 2
    hidden_layer = 5
    output_dim = 30

    autoencoder = Autoencoder(feature_dim, hidden_layer, output_dim).to(device)
    koopman_operator = KoopmanOperator(output_dim).to(device)

    optimiser_autoencoder = torch.optim.Adam(autoencoder.parameters(), lr=3e-4, weight_decay=1e-6)
    optimiser_koopman = torch.optim.Adam(koopman_operator.parameters(), lr=1e-4)

    history = {
        "train_total": [],
        "test_total": [],
        "train_recon": [],
        "test_recon": [],
        "train_koop": [],
        "test_koop": [],
        "train_phase": [],
        "test_phase": [],
        "train_latent": [],
        "test_latent": [],
    }

    n_batch = len(train_dataloader)
    n_test = len(test_dataloader)

    for epoch in range(n_epoch):
        autoencoder.train()
        koopman_operator.train()
        total_train_loss = 0.0
        total_train_recon = 0.0
        total_train_koop = 0.0
        total_train_phase = 0.0
        total_train_latent = 0.0

        for tensor2d_batch_x, tensor2d_batch_x_next in train_dataloader:
            tensor2d_batch_x = tensor2d_batch_x.to(device)
            tensor2d_batch_x_next = tensor2d_batch_x_next.to(device)
            optimiser_autoencoder.zero_grad()
            optimiser_koopman.zero_grad()

            tensor2d_observable = autoencoder.encoder(tensor2d_batch_x)
            tensor2d_observable_next = autoencoder.encoder(tensor2d_batch_x_next)
            tensor2d_decoded_x = autoencoder.decoder(tensor2d_observable)
            tensor2d_koopman_observable = koopman_operator(tensor2d_observable)
            tensor2d_predict_x = autoencoder.decoder(tensor2d_koopman_observable)

            tensor_loss, l_recon, l_koop, l_phase, l_latent = loss_koopman(
                tensor2d_batch_x,
                tensor2d_batch_x_next,
                tensor2d_observable,
                tensor2d_decoded_x,
                tensor2d_observable_next,
                tensor2d_koopman_observable,
                tensor2d_predict_x,
            )

            main_loss = 1.0 * l_recon + 5.0 * l_koop + 0.1 * l_latent
            main_loss.backward(retain_graph=True)

            for p in autoencoder.encoder.parameters():
                p.requires_grad = False

            phase_only_loss = 2.0 * l_phase
            phase_only_loss.backward()

            for p in autoencoder.encoder.parameters():
                p.requires_grad = True

            nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=1.0)
            nn.utils.clip_grad_norm_(koopman_operator.parameters(), max_norm=1.0)
            optimiser_autoencoder.step()
            optimiser_koopman.step()

            total_train_loss += tensor_loss.item()
            total_train_recon += l_recon.item()
            total_train_koop += l_koop.item()
            total_train_phase += l_phase.item()
            total_train_latent += l_latent.item()

        autoencoder.eval()
        koopman_operator.eval()
        total_test_loss = 0.0
        total_test_recon = 0.0
        total_test_koop = 0.0
        total_test_phase = 0.0
        total_test_latent = 0.0

        with torch.no_grad():
            for tensor2d_batch_x, tensor2d_batch_x_next in test_dataloader:
                tensor2d_batch_x = tensor2d_batch_x.to(device)
                tensor2d_batch_x_next = tensor2d_batch_x_next.to(device)

                tensor2d_observable = autoencoder.encoder(tensor2d_batch_x)
                tensor2d_observable_next = autoencoder.encoder(tensor2d_batch_x_next)
                tensor2d_decoded_x = autoencoder.decoder(tensor2d_observable)
                tensor2d_koopman_observable = koopman_operator(tensor2d_observable)
                tensor2d_predict_x = autoencoder.decoder(tensor2d_koopman_observable)

                tensor_loss, l_recon, l_koop, l_phase, l_latent = loss_koopman(
                    tensor2d_batch_x,
                    tensor2d_batch_x_next,
                    tensor2d_observable,
                    tensor2d_decoded_x,
                    tensor2d_observable_next,
                    tensor2d_koopman_observable,
                    tensor2d_predict_x,
                )

                total_test_loss += tensor_loss.item()
                total_test_recon += l_recon.item()
                total_test_koop += l_koop.item()
                total_test_phase += l_phase.item()
                total_test_latent += l_latent.item()

        print(
            f"Epoch {epoch:02d} | "
            f"Train: total={total_train_loss / n_batch:.4f} recon={total_train_recon / n_batch:.4f} "
            f"koop={total_train_koop / n_batch:.4f} phase={total_train_phase / n_batch:.4f} "
            f"latent={total_train_latent / n_batch:.4f} | "
            f"Test: total={total_test_loss / n_test:.4f} recon={total_test_recon / n_test:.4f} "
            f"koop={total_test_koop / n_test:.4f} phase={total_test_phase / n_test:.4f} "
            f"latent={total_test_latent / n_test:.4f}"
        )

        history["train_total"].append(total_train_loss / n_batch)
        history["test_total"].append(total_test_loss / n_test)
        history["train_recon"].append(total_train_recon / n_batch)
        history["test_recon"].append(total_test_recon / n_test)
        history["train_koop"].append(total_train_koop / n_batch)
        history["test_koop"].append(total_test_koop / n_test)
        history["train_phase"].append(total_train_phase / n_batch)
        history["test_phase"].append(total_test_phase / n_test)
        history["train_latent"].append(total_train_latent / n_batch)
        history["test_latent"].append(total_test_latent / n_test)

    eigvals = torch.linalg.eigvals(koopman_operator.koopman_matrix()).detach().cpu().numpy()
    print("Spectral radius:", np.abs(eigvals).max())

    return autoencoder, koopman_operator, history


if __name__ == "__main__":
    autoencoder, koopman_operator, history = train_koopman(n_epoch=30)
    plot_losses(history)
