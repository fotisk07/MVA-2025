from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from clip_reproduction.datasets import PreprocessedGraphDataset, collate_graph_text_batch
from clip_reproduction.models.gcn_graph_encoder import GINEGlobalGraphEncoderEnhanced


@dataclass
class ClipConfig:
    # data
    train_graphs: str = "data/molecule/raw/train_graphs.pkl"
    val_graphs: str = "data/molecule/raw/validation_graphs.pkl"
    train_text_emb: str = "data/molecule/processed/st_text_embeddings/train_sentencetransformer_embeddings.pt"
    val_text_emb: str = (
        "data/molecule/processed/st_text_embeddings/validation_sentencetransformer_embeddings.pt"
    )

    # runtime
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 2

    # training
    batch_size: int = 128
    epochs: int = 50
    lr: float = 3e-4
    weight_decay: float = 1e-4
    temperature: float = 0.07
    use_amp: bool = True
    grad_clip: float = 1.0

    # dims
    text_embed_dim: int = 384  # MiniLM embeddings
    embed_dim: int = 256  # shared space D

    # graph encoder
    graph_hidden: int = 256
    graph_layers: int = 4
    graph_dropout: float = 0.1
    global_feat_dim: int = 10

    out_dir: Path = Path("outputs/molecule/")


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln(self.proj(x))
        return F.normalize(x, dim=-1)


def clip_loss(g: torch.Tensor, t: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = (g @ t.T) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


@torch.no_grad()
def compute_recall(g: torch.Tensor, t: torch.Tensor, ks=(1, 5, 10)):
    sim = g @ t.T
    ranks_g2t = sim.argsort(dim=1, descending=True)
    ranks_t2g = sim.argsort(dim=0, descending=True)
    targets = torch.arange(sim.size(0), device=sim.device)

    out = {}
    for k in ks:
        out[f"R@{k}_g2t"] = (ranks_g2t[:, :k] == targets[:, None]).any(dim=1).float().mean().item()
        out[f"R@{k}_t2g"] = (ranks_t2g[:k, :] == targets[None, :]).any(dim=0).float().mean().item()
    return out


@torch.no_grad()
def validate(graph_encoder, text_proj, loader, cfg: ClipConfig):
    graph_encoder.eval()
    text_proj.eval()

    total = 0.0
    all_g, all_t = [], []

    for graphs, text_emb in loader:
        graphs = graphs.to(cfg.device)
        text_emb = text_emb.to(cfg.device)

        g = graph_encoder(graphs)  # already normalized (B, D)
        t = text_proj(text_emb)  # normalized (B, D)

        loss = clip_loss(g, t, cfg.temperature)
        total += float(loss.item())

        all_g.append(g)
        all_t.append(t)

    G = torch.cat(all_g, dim=0)
    T = torch.cat(all_t, dim=0)
    return total / max(1, len(loader)), compute_recall(G, T)


def main():
    cfg = ClipConfig()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    # Load embedding dicts
    train_emb = torch.load(cfg.train_text_emb, map_location="cpu")
    val_emb = torch.load(cfg.val_text_emb, map_location="cpu")

    train_ds = PreprocessedGraphDataset(cfg.train_graphs, emb_dict=train_emb)
    val_ds = PreprocessedGraphDataset(cfg.val_graphs, emb_dict=val_emb)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_graph_text_batch,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_graph_text_batch,
        drop_last=False,
    )

    # Graph encoder outputs embed_dim directly
    graph_encoder = GINEGlobalGraphEncoderEnhanced(
        hidden=cfg.graph_hidden,
        out_dim=cfg.embed_dim,
        layers=cfg.graph_layers,
        dropout=cfg.graph_dropout,
        global_feat_dim=cfg.global_feat_dim,
    ).to(cfg.device)

    # Text projector: 384 -> embed_dim
    text_proj = ProjectionHead(cfg.text_embed_dim, cfg.embed_dim).to(cfg.device)

    params = list(graph_encoder.parameters()) + list(text_proj.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler(enabled=(cfg.use_amp and cfg.device == "cuda"))

    best_r10 = -1.0

    for epoch in range(cfg.epochs):
        graph_encoder.train()
        text_proj.train()

        total = 0.0
        for graphs, text_emb in tqdm(train_loader, desc=f"Train {epoch:03d}"):
            graphs = graphs.to(cfg.device)
            text_emb = text_emb.to(cfg.device)

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=cfg.device):
                g = graph_encoder(graphs)  # (B, D) normalized
                t = text_proj(text_emb)  # (B, D) normalized
                loss = clip_loss(g, t, cfg.temperature)

            scaler.scale(loss).backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            scaler.step(opt)
            scaler.update()

            total += float(loss.item())

        train_loss = total / max(1, len(train_loader))
        val_loss, recalls = validate(graph_encoder, text_proj, val_loader, cfg)

        line = f"[{epoch:03d}] train={train_loss:.4f} val={val_loss:.4f} " + " ".join(
            f"{k}={v:.3f}" for k, v in sorted(recalls.items())
        )
        print(line)

        r10 = recalls.get("R@10_g2t", 0.0)
        if r10 > best_r10:
            best_r10 = r10
            torch.save(
                {
                    "epoch": epoch,
                    "graph_encoder": graph_encoder.state_dict(),
                    "text_proj": text_proj.state_dict(),
                    "opt": opt.state_dict(),
                    "recalls": recalls,
                    "cfg": cfg.__dict__,
                },
                cfg.out_dir / "clip_best.pt",
            )


if __name__ == "__main__":
    main()
