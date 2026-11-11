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
from moleculecs.train_reranker import Reranker


# -------------------------------------------------
# Config
# -------------------------------------------------
@dataclass
class PredictConfig(ClipConfig):
    ckpt_path: str = "models/clip/clip_best.pt"
    reranker_path: str = "models/clip/reranker.pt"
    test_graphs: str = "data/raw/test_graphs.pkl"
    output_path: str = "outputs/clip/predictions_reranked.csv"
    K: int = 5


# -------------------------------------------------
# Main
# -------------------------------------------------
@torch.no_grad()
def main():
    cfg = PredictConfig()
    device = cfg.device

    # -----------------------
    # Load CLIP checkpoint
    # -----------------------
    ckpt = torch.load(cfg.ckpt_path, map_location=device, weights_only=False)

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
    # Load reranker
    # -----------------------
    reranker_ckpt = torch.load(
        cfg.reranker_path, map_location=device, weights_only=False
    )
    reranker = Reranker(cfg.embed_dim).to(device)
    reranker.load_state_dict(reranker_ckpt["state_dict"])
    reranker.eval()

    # -----------------------
    # Load TRAIN text side
    # -----------------------
    train_text_emb = torch.load(cfg.train_text_emb, map_location="cpu")
    train_ids = list(train_text_emb.keys())
    train_id2desc = load_descriptions_from_graphs(cfg.train_graphs)

    E = torch.stack([train_text_emb[i] for i in train_ids]).to(device)
    T = text_proj(E)  # (N_text, D), normalized

    # -----------------------
    # Load TEST graphs
    # -----------------------
    test_ds = PreprocessedGraphDataset(cfg.test_graphs)
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # -----------------------
    # Predict with reranking
    # -----------------------
    predictions = []

    for graphs in tqdm(test_loader, desc="Predict (reranked)"):
        graphs = graphs.to(device)
        graphs.x = graphs.x.to(device)  # IMPORTANT (same fix as training)

        zg = graph_encoder(graphs)  # (B, D)

        sim = zg @ T.T
        topk_scores, topk_idx = sim.topk(cfg.K, dim=1)

        for i, gid in enumerate(graphs.id):
            idxs = topk_idx[i].tolist()
            zt = T[idxs]  # (K, D)

            # reranker scores
            scores = reranker(
                zg[i].expand(cfg.K, -1),
                zt,
            )

            best = scores.argmax().item()
            best_train_id = train_ids[idxs[best]]

            predictions.append(
                {
                    "ID": gid,
                    "description": train_id2desc[best_train_id],
                }
            )

    # -----------------------
    # Save
    # -----------------------
    df = pd.DataFrame(predictions)
    Path(cfg.output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.output_path, index=False)

    print(f"Saved {len(df)} predictions to {cfg.output_path}")
    print(df.head())


if __name__ == "__main__":
    main()
