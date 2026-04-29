from dataclasses import dataclass
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from tqdm import tqdm

from clip_reproduction.datasets import PreprocessedGraphDataset


# -----------------------
# Config
# -----------------------
@dataclass
class TextEmbeddingConfig:
    data_path: Path
    output_dir: Path
    batch_size: int = 64
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------
# Text preprocessing
# -----------------------
def clean_description(text: str) -> str:
    text = text.strip().lower()
    prefixes = [
        "the molecule is ",
        "the molecule is a ",
        "the molecule is an ",
    ]
    for p in prefixes:
        text = text.removeprefix(p)
    return text


# -----------------------
# Collate
# -----------------------
def collate_text(batch):
    ids = [g.id for g in batch]
    texts = [clean_description(g.description) for g in batch]
    return ids, texts


# -----------------------
# Main embedding routine
# -----------------------
def generate_embeddings(dataset, split: str, cfg: TextEmbeddingConfig):
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_text,
        pin_memory=True,
    )

    model = SentenceTransformer(cfg.model_name, device=cfg.device)

    emb_dict: dict[str, torch.Tensor] = {}

    for ids, texts in tqdm(loader, desc=f"[{split}] embedding"):
        emb = model.encode(
            texts,
            batch_size=len(texts),
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        for id_, e in zip(ids, emb.cpu()):
            emb_dict[id_] = e

    out_path = cfg.output_dir / f"{split}_sentencetransformer_embeddings.pt"
    torch.save(emb_dict, out_path)
    print(f"[{split}] saved {len(emb_dict)} embeddings → {out_path}")


def main():
    cfg = TextEmbeddingConfig(
        data_path=Path("data/molecule/raw"),
        output_dir=Path("data/molecule/processed/st_text_embeddings"),
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = PreprocessedGraphDataset(cfg.data_path / "train_graphs.pkl")
    val_ds = PreprocessedGraphDataset(cfg.data_path / "validation_graphs.pkl")

    generate_embeddings(train_ds, "train", cfg)
    generate_embeddings(val_ds, "validation", cfg)


main()
