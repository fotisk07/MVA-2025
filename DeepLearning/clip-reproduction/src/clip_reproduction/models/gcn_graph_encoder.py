import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv, GINEConv, GlobalAttention, global_add_pool
from torch_geometric.utils import to_dense_batch


class GCNGraphEncoder(nn.Module):
    def __init__(self, hidden=128, out_dim=768, layers=3):
        super().__init__()

        # Use a single learnable embedding for all nodes (no node features)
        self.node_init = nn.Parameter(torch.randn(hidden))

        self.convs = nn.ModuleList()
        for _ in range(layers):
            self.convs.append(GCNConv(hidden, hidden))

        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, batch: Batch):
        # Initialize all nodes with the same learnable embedding
        num_nodes = batch.x.size(0)
        h = self.node_init.unsqueeze(0).expand(num_nodes, -1)

        for conv in self.convs:
            h = conv(h, batch.edge_index)
            h = F.relu(h)
        g = global_add_pool(h, batch.batch)
        g = self.proj(g)
        g = F.normalize(g, dim=-1)
        return g


class GINEGraphEncoder(nn.Module):
    def __init__(
        self,
        hidden: int = 256,
        out_dim: int = 768,
        layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        node_cardinalities = [119, 9, 11, 12, 9, 5, 8, 2, 2]
        edge_cardinalities = [21, 6, 2]

        self.hidden = hidden
        self.out_dim = out_dim

        self.node_embs = nn.ModuleList([nn.Embedding(c, hidden) for c in node_cardinalities])
        self.edge_embs = nn.ModuleList([nn.Embedding(c, hidden) for c in edge_cardinalities])

        self.dropout = nn.Dropout(dropout)

        self.convs = nn.ModuleList()
        for _ in range(layers):
            mlp = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            self.convs.append(GINEConv(nn=mlp, edge_dim=hidden))

        self.proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def embed_nodes(self, x: torch.Tensor) -> torch.Tensor:
        h = 0
        for j, emb in enumerate(self.node_embs):
            h = h + emb(x[:, j])
        return h

    def embed_edges(self, edge_attr: torch.Tensor) -> torch.Tensor:
        e = 0
        for j, emb in enumerate(self.edge_embs):
            e = e + emb(edge_attr[:, j])
        return e

    def message_passing(self, batch: Batch) -> torch.Tensor:
        h = self.embed_nodes(batch.x)
        e = self.embed_edges(batch.edge_attr)

        h = self.dropout(h)
        for conv in self.convs:
            h = conv(h, batch.edge_index, e)
            h = F.relu(h)
            h = self.dropout(h)

        return h

    def pool(self, h: torch.Tensor, batch: Batch) -> torch.Tensor:
        return global_add_pool(h, batch.batch)

    def forward(self, batch: Batch) -> torch.Tensor:
        h = self.message_passing(batch)
        g = self.pool(h, batch)
        g = self.proj(g)
        return F.normalize(g, dim=-1)


class GINEGlobalGraphEncoder(GINEGraphEncoder):
    def __init__(
        self,
        hidden: int = 256,
        out_dim: int = 768,
        layers: int = 4,
        dropout: float = 0.1,
        global_feat_dim: int = 12,
    ):
        super().__init__(
            hidden=hidden,
            out_dim=out_dim,
            layers=layers,
            dropout=dropout,
        )

        self.global_proj = nn.Sequential(
            nn.Linear(global_feat_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # override projection head
        self.proj = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def global_features(self, batch: Batch) -> torch.Tensor:
        device = batch.x.device
        B = int(batch.batch.max().item()) + 1

        ones = torch.ones(batch.num_nodes, device=device)
        num_nodes = torch.zeros(B, device=device).scatter_add_(0, batch.batch, ones)

        is_aromatic = batch.x[:, 7].float()
        is_in_ring = batch.x[:, 8].float()

        aromatic_count = torch.zeros(B, device=device).scatter_add_(0, batch.batch, is_aromatic)
        ring_count = torch.zeros(B, device=device).scatter_add_(0, batch.batch, is_in_ring)

        aromatic_frac = aromatic_count / (num_nodes + 1e-6)
        ring_frac = ring_count / (num_nodes + 1e-6)

        atomic_num = batch.x[:, 0]

        def count(z):
            return torch.zeros(B, device=device).scatter_add_(0, batch.batch, (atomic_num == z).float())

        count_C = count(6)
        count_N = count(7)
        count_O = count(8)
        count_S = count(16)

        hetero = num_nodes - count_C

        feats = torch.stack(
            [
                num_nodes,
                ring_count,
                aromatic_count,
                ring_frac,
                aromatic_frac,
                count_C,
                count_N,
                count_O,
                count_S,
                hetero,
            ],
            dim=1,
        )

        if feats.size(1) < self.global_proj[0].in_features:
            pad = torch.zeros(
                B,
                self.global_proj[0].in_features - feats.size(1),
                device=device,
            )
            feats = torch.cat([feats, pad], dim=1)

        return feats

    def forward(self, batch: Batch) -> torch.Tensor:
        h = self.message_passing(batch)
        g_local = self.pool(h, batch)
        g_global = self.global_proj(self.global_features(batch))

        g = torch.cat([g_local, g_global], dim=-1)
        g = self.proj(g)
        return F.normalize(g, dim=-1)


class GINEGlobalGraphEncoderEnhanced(GINEGraphEncoder):
    def __init__(
        self,
        hidden: int = 256,
        out_dim: int = 256,
        layers: int = 4,
        dropout: float = 0.1,
        global_feat_dim: int = 10,
        num_attention_layers: int = 2,
        num_attention_heads: int = 8,
    ):
        super().__init__(
            hidden=hidden,
            out_dim=hidden,  # IMPORTANT: project later
            layers=layers,
            dropout=dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=num_attention_heads,
            dim_feedforward=4 * hidden,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.attn_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_attention_layers,
        )

        self.attn_pool = GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )
        )

        self.global_proj = nn.Sequential(
            nn.Linear(global_feat_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.proj = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def global_features(self, batch: Batch) -> torch.Tensor:
        device = batch.x.device
        B = int(batch.batch.max().item()) + 1

        ones = torch.ones(batch.num_nodes, device=device)
        num_nodes = torch.zeros(B, device=device).scatter_add_(0, batch.batch, ones)

        is_aromatic = batch.x[:, 7].float()
        is_in_ring = batch.x[:, 8].float()

        aromatic_count = torch.zeros(B, device=device).scatter_add_(0, batch.batch, is_aromatic)
        ring_count = torch.zeros(B, device=device).scatter_add_(0, batch.batch, is_in_ring)

        aromatic_frac = aromatic_count / (num_nodes + 1e-6)
        ring_frac = ring_count / (num_nodes + 1e-6)

        atomic_num = batch.x[:, 0]

        def count(z):
            return torch.zeros(B, device=device).scatter_add_(0, batch.batch, (atomic_num == z).float())

        count_C = count(6)
        count_N = count(7)
        count_O = count(8)
        count_S = count(16)

        hetero = num_nodes - count_C

        feats = torch.stack(
            [
                torch.log1p(num_nodes),
                torch.log1p(ring_count),
                torch.log1p(aromatic_count),
                ring_frac,
                aromatic_frac,
                torch.log1p(count_C),
                torch.log1p(count_N),
                torch.log1p(count_O),
                torch.log1p(count_S),
                torch.log1p(hetero),
            ],
            dim=1,
        )

        return feats

    def forward(self, batch: Batch) -> torch.Tensor:
        h = self.message_passing(batch)  # (num_nodes, hidden)
        h_dense, mask = to_dense_batch(h, batch.batch)  # (B, Nmax, hidden)
        h_dense = self.attn_encoder(h_dense, src_key_padding_mask=~mask)
        h = h_dense[mask]  # back to (num_nodes, hidden)
        g_local = self.attn_pool(h, batch.batch)
        g_global = self.global_proj(self.global_features(batch))
        g = torch.cat([g_local, g_global], dim=-1)
        g = self.proj(g)

        return F.normalize(g, dim=-1)
