"""
Graph Neural Network Architecture for Drug-Target Interaction Prediction
Bond-aware GINEConv graph encoder + 1D CNN protein encoder + classical prediction head.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool
from torch_geometric.data import Batch

from common.protein_encoder import Protein1DCNNEncoder


class GNNEncoder(nn.Module):
    """
    Bond-aware GINE graph encoder for small molecules.
    Aggregates atom and bond features across configurable layers.
    """
    def __init__(
        self,
        atom_feature_dim: int = 41,
        bond_feature_dim: int = 16,
        hidden_dim: int = 64,
        num_layers: int = 3,
        out_dim: int = 64,
        dropout: float = 0.2
    ):
        super().__init__()
        self.atom_proj = nn.Linear(atom_feature_dim, hidden_dim)
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            conv = GINEConv(mlp, edge_dim=bond_feature_dim)
            norm = nn.BatchNorm1d(hidden_dim)
            self.convs.append(conv)
            self.norms.append(norm)

        self.readout_proj = nn.Sequential(
            nn.Linear(2 * hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        h = self.atom_proj(x)
        for i in range(self.num_layers):
            h_new = self.convs[i](h, edge_index, edge_attr)
            h_new = self.norms[i](h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h + h_new  # Residual skip connection

        # Global pooling (mean + max)
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        h_pooled = torch.cat([h_mean, h_max], dim=-1)

        drug_embed = self.readout_proj(h_pooled)
        return drug_embed


class DTIGNNModel(nn.Module):
    """
    Full DTI Graph Neural Network Model:
    Drug Branch: Molecular Graph -> GNNEncoder -> Drug Embedding (64)
    Protein Branch: Sequence Tokens -> Protein1DCNNEncoder -> Target Embedding (64)
    Fusion: Concatenation (128)
    Prediction Head: Classical MLP -> Continuous pKi
    """
    def __init__(
        self,
        atom_feature_dim: int = 41,
        bond_feature_dim: int = 16,
        gnn_hidden_dim: int = 64,
        gnn_num_layers: int = 3,
        gnn_out_dim: int = 64,
        prot_out_dim: int = 64,
        dropout: float = 0.2
    ):
        super().__init__()
        self.gnn_encoder = GNNEncoder(
            atom_feature_dim=atom_feature_dim,
            bond_feature_dim=bond_feature_dim,
            hidden_dim=gnn_hidden_dim,
            num_layers=gnn_num_layers,
            out_dim=gnn_out_dim,
            dropout=dropout
        )
        self.protein_encoder = Protein1DCNNEncoder(
            out_dim=prot_out_dim,
            dropout=dropout
        )
        fusion_dim = gnn_out_dim + prot_out_dim
        self.classical_head = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        # Initialize final bias to mean pKi (~7.32) for fast convergence
        nn.init.constant_(self.classical_head[-1].bias, 7.32)

    def forward(self, batch_data: Batch) -> torch.Tensor:
        # Drug branch
        drug_embed = self.gnn_encoder(
            batch_data.x,
            batch_data.edge_index,
            batch_data.edge_attr,
            batch_data.batch
        )
        # Protein branch
        # batch_data.target_tokens shape: (batch_size, seq_len)
        target_tokens = batch_data.target_tokens
        if target_tokens.dim() == 1:
            # Reshape if flattened during batching
            batch_size = drug_embed.size(0)
            target_tokens = target_tokens.view(batch_size, -1)

        target_embed = self.protein_encoder(target_tokens)

        # Concatenation fusion
        joint_embed = torch.cat([drug_embed, target_embed], dim=-1)

        # Regression output
        pred_pki = self.classical_head(joint_embed)
        return pred_pki
