import random
from collections.abc import Iterable
from pathlib import Path

import hydra
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from clip_reproduction import utils
from clip_reproduction.datasets import get_classification_datasets, get_clip_datasets
from clip_reproduction.models.factory import create_model, is_clip_model
from clip_reproduction.models.text import ByteTokenizer


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    predictions = logits.argmax(dim=1)
    return (predictions == labels).float().mean()


def _clip_accuracy(logits_per_image: torch.Tensor, logits_per_text: torch.Tensor) -> torch.Tensor:
    batch_size = logits_per_image.size(0)
    targets = torch.arange(batch_size, device=logits_per_image.device)
    image_to_text_acc = _accuracy(logits_per_image, targets)
    text_to_image_acc = _accuracy(logits_per_text, targets)
    return 0.5 * (image_to_text_acc + text_to_image_acc)


def train_one_epoch_classifier(
    model: nn.Module,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    log_every: int,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    step = 0

    for step, (images, labels) in tqdm(enumerate(loader, start=1), total=len(loader)):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += _accuracy(logits.detach(), labels).item()

        if step % log_every == 0:
            print(f"[train][classifier] step={step} loss={total_loss / step:.4f} acc={total_acc / step:.4f}")

    num_steps = max(step, 1)
    return {"loss": total_loss / num_steps, "acc": total_acc / num_steps}


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module,
    loader: Iterable,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    step = 0

    for step, (images, labels) in enumerate(loader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        total_acc += _accuracy(logits, labels).item()

    num_steps = max(step, 1)
    return {"loss": total_loss / num_steps, "acc": total_acc / num_steps}


def train_one_epoch_clip(
    model: nn.Module,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    log_every: int,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    step = 0

    for step, (images, token_ids) in tqdm(enumerate(loader, start=1), total=len(loader)):
        images = images.to(device)
        token_ids = token_ids.to(device)

        outputs = model(images, token_ids)
        loss = outputs["loss"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if step % log_every == 0:
            print(f"[train][clip] step={step} loss={total_loss / step:.4f}")

    num_steps = max(step, 1)
    return {"loss": total_loss / num_steps}


@torch.no_grad()
def evaluate_clip(
    model: nn.Module,
    loader: Iterable,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    step = 0

    for step, (images, token_ids) in enumerate(loader, start=1):
        images = images.to(device)
        token_ids = token_ids.to(device)

        outputs = model(images, token_ids)
        total_loss += outputs["loss"].item()
        total_acc += _clip_accuracy(
            outputs["logits_per_image"],
            outputs["logits_per_text"],
        ).item()

    num_steps = max(step, 1)
    return {"loss": total_loss / num_steps, "acc": total_acc / num_steps}


@hydra.main(version_base=None, config_path="../conf", config_name="train_conf")
def main(cfg) -> None:
    utils.seed_everything(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device", device)
    print(f"Starting training: model={cfg.model} dataset={cfg.dataset}")
    device = torch.device(device)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.model == "clip":
        tokenizer = ByteTokenizer(context_length=cfg.context_length)
        train_ds, val_ds, _, _ = get_clip_datasets(
            name=cfg.dataset,
            root=cfg.data_root,
            image_size=cfg.image_size,
            val_ratio=cfg.val_ratio,
            seed=cfg.seed,
            tokenizer=tokenizer,
        )

        model = create_model(
            "clip",
            image_size=cfg.image_size,
            context_length=cfg.context_length,
            embed_dim=cfg.embed_dim,
            vision_width=cfg.vision_width,
            vision_layers=cfg.vision_layers,
            vision_heads=cfg.vision_heads,
            vision_patch_size=cfg.vision_patch_size,
            text_width=cfg.text_width,
            text_layers=cfg.text_layers,
            text_heads=cfg.text_heads,
            dropout=cfg.dropout,
            encoder=cfg.encoder,
        )
    else:
        if cfg.model == "resnet50":
            raise ValueError(
                "`resnet50` is the frozen feature-extractor variant (for linear probing). "
                "Use `resnet50_finetuning` for training a trainable last layer."
            )
        train_ds, val_ds, num_classes, _ = get_classification_datasets(
            name=cfg.dataset,
            root=cfg.data_root,
            image_size=cfg.image_size,
            val_ratio=cfg.val_ratio,
            seed=cfg.seed,
        )
        model = create_model(cfg.model, num_classes=num_classes)

    model = model.to(device)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = float("inf")
    for epoch in range(1, cfg.epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.epochs}")

        if is_clip_model(model):
            train_metrics = train_one_epoch_clip(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                log_every=cfg.log_every,
            )
        else:
            train_metrics = train_one_epoch_classifier(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=nn.CrossEntropyLoss(),
                device=device,
                log_every=cfg.log_every,
            )

        print(f"[train] {train_metrics}")

        if epoch % cfg.eval_every == 0:
            if is_clip_model(model):
                val_metrics = evaluate_clip(model=model, loader=val_loader, device=device)
            else:
                val_metrics = evaluate_classifier(
                    model=model,
                    loader=val_loader,
                    criterion=nn.CrossEntropyLoss(),
                    device=device,
                )
            print(f"[val] {val_metrics}")

            val_loss = val_metrics["loss"]
            if val_loss < best_val:
                best_val = val_loss
                ckpt_path = output_dir / f"best_{cfg.model}_{cfg.dataset}.pt"
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "val_metrics": val_metrics,
                        "cfg": vars(cfg),
                    },
                    ckpt_path,
                )
                print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
