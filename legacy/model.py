"""
model.py — Path 1 GNN for per-face B-rep classification.

Architecture (small, fast, debuggable):
  input MLP  ->  [GINEConv + BatchNorm + ReLU + residual] x num_layers  ->  head MLP
GINEConv is used because it natively consumes edge_attr (our convexity/angle signal),
which is exactly the information that separates a cut face from an outer stock face.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv


class BRepGNN(nn.Module):
    def __init__(self, node_in, edge_in, hidden, num_classes,
                 num_layers=4, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(node_in, hidden)
        # GINEConv requires edge_attr projected to node hidden dim:
        self.edge_proj = nn.Linear(edge_in, hidden)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            self.convs.append(GINEConv(mlp, edge_dim=hidden))
            self.norms.append(nn.BatchNorm1d(hidden))

        self.dropout = dropout
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x, edge_index, edge_attr):
        h = self.input_proj(x)
        e = self.edge_proj(edge_attr)
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, edge_index, e)
            h_new = norm(h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h + h_new            # residual: stabilizes deeper stacks
        return self.head(h)          # logits per node [N, num_classes]
