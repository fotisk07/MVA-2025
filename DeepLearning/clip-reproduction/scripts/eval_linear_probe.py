import csv
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPImageProcessor

from clip_reproduction import utils
from clip_reproduction.datasets import get_classification_train_test_datasets
from clip_reproduction.models.factory import create_model


def l2_normalize(features: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return features / norms


def _get_image_features(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "encode_image_penultimate"):
        return model.encode_image_penultimate(images)
    return model.encode_image(images)


def _build_openai_clip_transform(model_name: str):
    processor = CLIPImageProcessor.from_pretrained(model_name)
    return transforms.Compose(
        [
            transforms.Lambda(
                lambda img: processor(
                    images=img.convert("RGB"),
                    return_tensors="pt",
                )["pixel_values"].squeeze(0),
            ),
        ],
    )


@torch.no_grad()
def compute_features(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    all_features = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Extracting features"):
        images = images.to(device, non_blocking=True)
        features = _get_image_features(model=model, images=images)
        all_features.append(features.cpu().numpy().astype(np.float32))
        all_labels.append(labels.numpy())

    features_np = np.concatenate(all_features, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    return features_np, labels_np


def cache_path(cache_dir: Path, feature_id: str, split: str) -> Path:
    return cache_dir / f"{feature_id}_{split}_features.npz"


def load_or_compute_features(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    cache_dir: Path,
    feature_id: str,
    split: str,
) -> tuple[np.ndarray, np.ndarray]:
    path = cache_path(cache_dir=cache_dir, feature_id=feature_id, split=split)
    if path.exists():
        data = np.load(path)
        return data["features"], data["labels"]

    features, labels = compute_features(model=model, loader=loader, device=device)
    np.savez_compressed(path, features=features, labels=labels)
    return features, labels


@hydra.main(version_base=None, config_path="../conf", config_name="eval_linear_probe_conf")
def main(cfg) -> None:
    utils.seed_everything(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.use_cuda else "cpu")
    print(f"Using device: {device}")

    if cfg.model not in {"resnet50", "openai_clip"}:
        raise ValueError("`model` must be one of ['resnet50', 'openai_clip'] for linear probing.")

    cache_dir = Path(cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    openai_model_tag = cfg.openai_model_name.replace("/", "-")
    image_tag = cfg.image_size if cfg.model != "openai_clip" else "hfproc"
    feature_id = f"{cfg.dataset}_{cfg.model}_{openai_model_tag}_{image_tag}"

    train_tf = None
    eval_tf = None
    if cfg.model == "openai_clip":
        clip_tf = _build_openai_clip_transform(cfg.openai_model_name)
        train_tf = clip_tf
        eval_tf = clip_tf

    train_ds, test_ds, num_classes, _ = get_classification_train_test_datasets(
        name=cfg.dataset,
        root=cfg.data_root,
        image_size=cfg.image_size,
        download=True,
        train_transform=train_tf,
        eval_transform=eval_tf,
    )

    model_kwargs = {"num_classes": num_classes}
    if cfg.model == "openai_clip":
        model_kwargs = {"openai_model_name": cfg.openai_model_name}

    model = create_model(name=cfg.model, **model_kwargs)
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )

    x_train, y_train = load_or_compute_features(
        model=model,
        loader=train_loader,
        device=device,
        cache_dir=cache_dir,
        feature_id=feature_id,
        split="train",
    )
    x_test, y_test = load_or_compute_features(
        model=model,
        loader=test_loader,
        device=device,
        cache_dir=cache_dir,
        feature_id=feature_id,
        split="test",
    )

    x_train = l2_normalize(x_train)
    x_test = l2_normalize(x_test)

    clf = LogisticRegression(
        solver="lbfgs",
        penalty="l2",
        C=cfg.logreg_c,
        max_iter=1000,
        n_jobs=None,
        random_state=cfg.seed,
    )
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    test_acc = accuracy_score(y_test, y_pred)

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    result_path = output_dir / "result.csv"
    with result_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "model", "test_accuracy", "device"])
        writer.writeheader()
        writer.writerow(
            {
                "dataset": cfg.dataset,
                "model": cfg.model,
                "test_accuracy": f"{test_acc:.4f}",
            },
        )

    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Saved result to {result_path}")


if __name__ == "__main__":
    main()
