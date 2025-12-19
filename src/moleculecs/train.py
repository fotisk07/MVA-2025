import os
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from moleculecs.data_utils import (
    PreprocessedGraphDataset,
    collate_fn,
    load_id2emb,
)


@hydra.main(
    version_base=None,
    config_path="conf/",
    config_name="train",
)
def main(cfg: DictConfig):
    HYDRA_DIR = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    # --------------------------------------------------
    # Device
    # --------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --------------------------------------------------
    # Instantiate task (Hydra)
    # --------------------------------------------------
    task = hydra.utils.instantiate(cfg.task)

    # --------------------------------------------------
    # Load text embeddings
    # --------------------------------------------------
    train_emb = load_id2emb(cfg.train_emb_csv)

    if os.path.exists(cfg.val_emb_csv):
        val_emb = load_id2emb(cfg.val_emb_csv)
    else:
        val_emb = None

    emb_dim = len(next(iter(train_emb.values())))

    # --------------------------------------------------
    # Datasets & loaders
    # --------------------------------------------------
    train_ds = PreprocessedGraphDataset(cfg.train_graphs, train_emb)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    if val_emb is not None and os.path.exists(cfg.val_graphs):
        val_ds = PreprocessedGraphDataset(cfg.val_graphs, val_emb)
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
    else:
        val_loader = None

    # --------------------------------------------------
    # Instantiate model (Hydra)
    # --------------------------------------------------
    model = hydra.utils.instantiate(cfg.model, out_dim=emb_dim).to(device)

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------
    for epoch in range(cfg.epochs):
        train_loss = task.train_epoch(
            model=model, loader=train_loader, optimizer=optimizer, device=device
        )

        if val_loader is not None:
            val_metrics = task.evaluate(model=model, loader=val_loader, device=device)
        else:
            val_metrics = {}

        print(
            f"Epoch {epoch + 1}/{cfg.epochs} | "
            f"loss={train_loss:.4f} | val={val_metrics}"
        )

    # --------------------------------------------------
    # Save model
    # --------------------------------------------------
    ckpt_path = HYDRA_DIR / "model.pt"
    torch.save(model.state_dict(), ckpt_path)

    print(f"Model saved to {ckpt_path}")


if __name__ == "__main__":
    main()
