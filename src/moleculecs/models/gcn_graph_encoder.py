import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv, GINEConv, global_add_pool


#! =========================================================
# MODEL: GNN to encode graphs (simple GCN, no edge features)
# =========================================================
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
        self.node_embs = nn.ModuleList(
            [nn.Embedding(c, hidden) for c in node_cardinalities]
        )
        self.edge_embs = nn.ModuleList(
            [nn.Embedding(c, hidden) for c in edge_cardinalities]
        )

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
        # x: [num_nodes, 9] integer categorical indices
        # Sum embeddings across the 9 categorical fields (standard in molecular GNNs)
        h = 0
        for j, emb in enumerate(self.node_embs):
            h = h + emb(x[:, j])
        return h

    def embed_edges(self, edge_attr: torch.Tensor) -> torch.Tensor:
        # edge_attr: [num_edges, 3]
        e = 0
        for j, emb in enumerate(self.edge_embs):
            e = e + emb(edge_attr[:, j])
        return e

    def forward(self, batch: Batch) -> torch.Tensor:
        # Node & edge embeddings
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr

        h = self.embed_nodes(x)
        e = self.embed_edges(edge_attr)

        h = self.dropout(h)

        for conv in self.convs:
            h = conv(h, edge_index, e)
            h = F.relu(h)
            h = self.dropout(h)

        g = global_add_pool(h, batch.batch)
        g = self.proj(g)
        g = F.normalize(g, dim=-1)
        return g
