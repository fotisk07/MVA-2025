from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from train_llm import ClipConfig, ProjectionHead

from moleculecs.data_utils import (
    PreprocessedGraphDataset,
    collate_fn,
    load_descriptions_from_graphs,
)
from moleculecs.models.gcn_graph_encoder import GINEGlobalGraphEncoderEnhanced


@torch.no_grad()
def encode_texts(text_emb_dict, text_proj, device):
    """
    Returns:
        text_ids: List[str]
        T: (N, D) normalized
    """
    print("Encoding texts")
    text_ids = list(text_emb_dict.keys())
    E = torch.stack([text_emb_dict[i] for i in text_ids]).to(device)
    T = text_proj(E)
    return text_ids, T


@torch.no_grad()
def encode_graphs(graph_encoder, loader, device):
    """
    Returns:
        graph_ids: List[str]
        G: (N, D) normalized
    """
    graph_ids = []
    all_g = []

    for graphs in tqdm(loader, desc="Encoding graphs"):
        graphs = graphs.to(device)
        g = graph_encoder(graphs)
        all_g.append(g)
        graph_ids.extend(graphs.id)  # assumes Batch exposes ids

    G = torch.cat(all_g, dim=0)
    return graph_ids, G


def predict_topk(
    G: torch.Tensor,
    T: torch.Tensor,
    text_ids,
    k: int = 10,
):
    """
    Returns:
        List[List[str]]: top-k text ids per graph
    """
    sim = G @ T.T  # (Ng, Nt)
    topk = sim.topk(k, dim=1).indices
    return [[text_ids[j] for j in row.tolist()] for row in topk]


@dataclass
class PredictConfig(ClipConfig):
    K: int = 10  # top-k
    output_path: str = "outputs/predictions.csv"
    ckpt_path: str = "models/clip/clip_best.pt"
    test_graphs: str = "data/raw/test_graphs.pkl"


def main():
    # -----------------------
    # Config
    # -----------------------
    cfg = PredictConfig()
    device = cfg.device

    ckpt_path = Path(cfg.ckpt_path)
    text_emb_path = Path(cfg.train_text_emb)
    graph_path = Path(cfg.test_graphs)

    out_path = Path(cfg.output_path)

    # -----------------------
    # Load checkpoint
    # -----------------------
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    graph_encoder = GINEGlobalGraphEncoderEnhanced(
        hidden=cfg.graph_hidden,
        out_dim=cfg.embed_dim,
        layers=cfg.graph_layers,
        dropout=cfg.graph_dropout,
        global_feat_dim=cfg.global_feat_dim,
    ).to(device)
    graph_encoder.load_state_dict(ckpt["graph_encoder"])
    graph_encoder.eval()

    text_proj = ProjectionHead(cfg.text_embed_dim, cfg.embed_dim).to(device)
    text_proj.load_state_dict(ckpt["text_proj"])
    text_proj.eval()

    # -----------------------
    # Load data
    # -----------------------
    text_emb_dict = torch.load(text_emb_path, map_location="cpu")

    graph_ds = PreprocessedGraphDataset(str(graph_path))
    graph_loader = DataLoader(
        graph_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # -----------------------
    # Encode
    # -----------------------
    text_ids, T = encode_texts(text_emb_dict, text_proj, device)
    graph_ids, G = encode_graphs(graph_encoder, graph_loader, device)

    # -----------------------
    # Predict
    # -----------------------
    # train_ids must correspond to text_emb_dict order
    train_id2desc = load_descriptions_from_graphs(cfg.train_graphs)
    train_ids = list(text_emb_dict.keys())

    predictions = []

    sim = G @ T.T
    nn_idx = sim.argmax(dim=1)

    for gid, idx in zip(graph_ids, nn_idx.tolist()):
        predictions.append(
            {
                "ID": gid,
                "description": train_id2desc[train_ids[idx]],
            }
        )
    df = pd.DataFrame(predictions)
    df.to_csv(out_path, index=False)

    print(f"Saved {len(df)} predictions to {out_path}")

    # Optional sanity print
    print("\nSample predictions:")
    print(df.head())


if __name__ == "__main__":
    main()
