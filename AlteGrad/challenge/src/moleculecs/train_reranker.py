from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from train_llm import ClipConfig, ProjectionHead

from moleculecs.data_utils import (
    PreprocessedGraphDataset,
    collate_fn,
    load_descriptions_from_graphs,
)
from moleculecs.models.gcn_graph_encoder import GINEGlobalGraphEncoderEnhanced


# -------------------------------------------------
# Config
# -------------------------------------------------
@dataclass
class RerankerConfig(ClipConfig):
    ckpt_path: str = "outputs/clip/clip_best.pt"
    K: int = 5
    epochs: int = 10
    lr: float = 1e-3
    output_path: str = "outputs/clip/reranker.pt"


# -------------------------------------------------
# Reranker model
# -------------------------------------------------
class Reranker(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4 * dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
        )

    def forward(self, zg, zt):
        x = torch.cat(
            [zg, zt, torch.abs(zg - zt), zg * zt],
            dim=-1,
        )
        return self.net(x).squeeze(-1)  # (K,)


def main():
    cfg = RerankerConfig()
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
    # Load TRAIN text embeddings
    # -----------------------
    train_text_emb = torch.load(cfg.train_text_emb, map_location="cpu")
    train_ids = list(train_text_emb.keys())

    E = torch.stack([train_text_emb[i] for i in train_ids]).to(device)
    with torch.no_grad():
        T = text_proj(E)  # (N_text, D), normalized

    train_id2desc = load_descriptions_from_graphs(cfg.train_graphs)

    # -----------------------
    # Load VAL graphs
    # -----------------------
    val_ds = PreprocessedGraphDataset(cfg.val_graphs)
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_id2desc = load_descriptions_from_graphs(cfg.val_graphs)

    # -----------------------
    # Build training data
    # -----------------------
    reranker = Reranker(cfg.embed_dim).to(device)
    opt = torch.optim.AdamW(reranker.parameters(), lr=cfg.lr)

    print("Training reranker")

    for epoch in range(cfg.epochs):
        reranker.train()
        total_loss = 0.0
        seen = 0

        for graphs in tqdm(val_loader, desc=f"Epoch {epoch:02d}"):
            graphs = graphs.to(device)

            # Get graph embeddings (no grad needed for encoder)
            with torch.no_grad():
                zg = graph_encoder(graphs)  # (B, D)
                sim = zg @ T.T  # (B, N_text)
                topk_scores, topk_idx = sim.topk(cfg.K, dim=1)  # (B, K)

            # Collect valid training samples for this batch
            batch_zg = []
            batch_zt = []
            batch_labels = []

            for i, gid in enumerate(graphs.id):
                true_desc = val_id2desc[gid]
                idxs = topk_idx[i].tolist()
                candidates = [train_id2desc[train_ids[j]] for j in idxs]

                if true_desc not in candidates:
                    continue  # skip if GT not in top-K

                label = candidates.index(true_desc)

                # Collect this sample
                batch_zg.append(zg[i].unsqueeze(0).expand(cfg.K, -1))  # (K, D)
                batch_zt.append(T[idxs])  # (K, D)
                batch_labels.append(label)

            if len(batch_labels) == 0:
                continue

            # Stack into batched tensors
            # Shape: (num_valid_samples, K, D)
            batch_zg = torch.stack(batch_zg, dim=0)  # (N, K, D)
            batch_zt = torch.stack(batch_zt, dim=0)  # (N, K, D)
            batch_labels = torch.tensor(batch_labels, device=device)  # (N,)

            # Flatten for reranker: (N*K, D)
            N = batch_zg.size(0)
            zg_flat = batch_zg.reshape(-1, cfg.embed_dim)
            zt_flat = batch_zt.reshape(-1, cfg.embed_dim)

            # Get scores and reshape: (N*K,) -> (N, K)
            scores = reranker(zg_flat, zt_flat).reshape(N, cfg.K)

            # Compute loss
            loss = F.cross_entropy(scores, batch_labels)

            # Backprop
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * N
            seen += N

        if seen > 0:
            print(
                f"[Epoch {epoch:02d}] loss={total_loss / seen:.4f} "
                f"(used {seen} examples)"
            )

    # -----------------------
    # Save reranker
    # -----------------------
    Path(cfg.output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": reranker.state_dict(),
            "cfg": cfg.__dict__,
        },
        cfg.output_path,
    )

    print(f"Saved reranker → {cfg.output_path}")


if __name__ == "__main__":
    main()
