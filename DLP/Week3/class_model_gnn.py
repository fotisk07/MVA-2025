# Define model ( in your class_model_gnn.py)
import torch
import torch.nn.functional as F
import torch_geometric.nn as graphnn
from torch import nn


class StudentModel(nn.Module):
    def __init__(self):
        super().__init__()

        in_dim = 50
        out_dim = 121

        hidden = 64
        heads1 = 8
        heads2 = 8
        p = 0.2

        self.p = p

        # GAT blocks
        self.gat1 = graphnn.GATConv(in_dim, hidden, heads=heads1, concat=True, dropout=p)
        self.norm1 = nn.LayerNorm(hidden * heads1)
        self.res1 = nn.Linear(in_dim, hidden * heads1, bias=False)

        self.gat2 = graphnn.GATConv(hidden * heads1, hidden, heads=heads2, concat=True, dropout=p)
        self.norm2 = nn.LayerNorm(hidden * heads2)
        self.res2 = nn.Linear(hidden * heads1, hidden * heads2, bias=False)

        # Output layer (no concat => directly out_dim)
        self.gat3 = graphnn.GATConv(hidden * heads2, out_dim, heads=1, concat=False, dropout=p)
        self.res3 = nn.Linear(hidden * heads2, out_dim, bias=False)

    def forward(self, x, edge_index):
        # Block 1
        x_in = x
        x = F.dropout(x, p=self.p, training=self.training)
        x = self.gat1(x, edge_index)
        x = x + self.res1(x_in)
        x = self.norm1(x)
        x = F.elu(x)

        # Block 2
        x_in = x
        x = F.dropout(x, p=self.p, training=self.training)
        x = self.gat2(x, edge_index)
        x = x + self.res2(x_in)
        x = self.norm2(x)
        x = F.elu(x)

        # Output
        x_in = x
        x = F.dropout(x, p=self.p, training=self.training)
        x = self.gat3(x, edge_index)
        x = x + self.res3(x_in)

        return x


# Initialize model
model = StudentModel()

## Save the model
torch.save(model.state_dict(), "model.pth")


### This is the part we will run in the inference to grade your model
## Load the model
model = StudentModel()  # !  Important : No argument
model.load_state_dict(torch.load("model.pth", weights_only=True))
model.eval()
print("Model loaded successfully")
