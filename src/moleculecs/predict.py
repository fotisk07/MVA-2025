from pathlib import Path

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from moleculecs.data_utils import (
    PreprocessedGraphDataset,
    collate_fn,
    load_descriptions_from_graphs,
    load_id2emb,
)


@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="predict",  # reuse the same top-level config
)
def main(cfg: DictConfig):
    HYDRA_DIR = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    GLOBAL_DIR = HYDRA_DIR.parent
    # --------------------------------------------------
    # Device
    # --------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --------------------------------------------------
    # Instantiate task and model (Hydra)
    # --------------------------------------------------
    task = hydra.utils.instantiate(cfg.task)
    model = hydra.utils.instantiate(cfg.model)

    # --------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------
    ckpt_path = GLOBAL_DIR / "train" / "model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    print(f"Loaded model from {ckpt_path}")

    # --------------------------------------------------
    # Load training text embeddings + descriptions
    # --------------------------------------------------
    train_emb_dict = load_id2emb(cfg.train_emb_csv)
    train_ids = list(train_emb_dict.keys())
    train_text_embeddings = torch.stack([train_emb_dict[i] for i in train_ids])

    train_id2desc = load_descriptions_from_graphs(cfg.train_graphs)

    # --------------------------------------------------
    # Test dataset & loader
    # --------------------------------------------------
    test_ds = PreprocessedGraphDataset(cfg.test_graphs)
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    print(f"Test set size: {len(test_ds)}")

    # --------------------------------------------------
    # Run prediction via task
    # --------------------------------------------------
    predictions = task.predict(
        model=model,
        test_loader=test_loader,
        train_text_embeddings=train_text_embeddings,
        train_ids=train_ids,
        device=device,
        train_id2desc=train_id2desc,
    )
    df = pd.DataFrame(predictions)

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------
    output_csv = HYDRA_DIR / "predictions.csv"
    df.to_csv(output_csv, index=False)

    print(f"Saved {len(df)} predictions to {output_csv}")

    # Optional sanity print
    print("\nSample predictions:")
    print(df.head())


if __name__ == "__main__":
    main()
