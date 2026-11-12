"""
Proposed Hybrid GNN + QNN Architecture for Drug-Target Interaction Prediction
Combines bond-aware molecular GNN with 1D CNN protein encoder and a parameterized quantum prediction head.
Enables end-to-end backpropagation across graph, protein, and quantum layers.
"""

import math
import sys
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch

from models.GNN.model import GNNEncoder
from common.protein_encoder import Protein1DCNNEncoder
from models.GNN_QNN.quantum_circuit import HybridQuantumCircuit


class DTIGraphQuantumHybridModel(nn.Module):
    """
    Proposed Hybrid Architecture:
    1. Drug Branch: Molecular Graph -> GNNEncoder (Bond-Aware GINE) -> Drug Embedding (64)
    2. Protein Branch: Amino Acid Sequence -> Protein1DCNNEncoder -> Target Embedding (64)
    3. Fusion & Reduction: Concatenation (128) -> Linear(128->32) -> Linear(32->n_qubits) -> Tanh * pi
    4. Quantum Branch: AngleEmbedding -> StronglyEntanglingLayers -> Pauli-Z Expectations (n_qubits)
    5. Regression Readout: Linear(n_qubits->16) -> ReLU -> Linear(16->1) -> Predicted pKi
    """
    def __init__(
        self,
        atom_feature_dim: int = 41,
        bond_feature_dim: int = 16,
        gnn_hidden_dim: int = 64,
        gnn_num_layers: int = 3,
        gnn_out_dim: int = 64,
        prot_out_dim: int = 64,
        reduction_hidden_dim: int = 32,
        n_qubits: int = 4,
        circuit_depth: int = 2,
        quantum_device: str = "default.qubit",
        dropout: float = 0.2
    ):
        super().__init__()
        self.n_qubits = n_qubits

        # Shared GNN Drug Encoder
        self.gnn_encoder = GNNEncoder(
            atom_feature_dim=atom_feature_dim,
            bond_feature_dim=bond_feature_dim,
            hidden_dim=gnn_hidden_dim,
            num_layers=gnn_num_layers,
            out_dim=gnn_out_dim,
            dropout=dropout
        )

        # Shared Protein Encoder
        self.protein_encoder = Protein1DCNNEncoder(
            out_dim=prot_out_dim,
            dropout=dropout
        )

        # Fusion & Dimensionality Reduction to Qubits
        fusion_dim = gnn_out_dim + prot_out_dim
        self.reduction_mlp = nn.Sequential(
            nn.Linear(fusion_dim, reduction_hidden_dim),
            nn.ReLU(),
            nn.Linear(reduction_hidden_dim, n_qubits),
            nn.Tanh() # Maps to [-1, 1], then multiplied by pi
        )

        # Quantum Prediction Component
        self.quantum_circuit = HybridQuantumCircuit(
            n_qubits=n_qubits,
            n_layers=circuit_depth,
            device_name=quantum_device
        )

        # Classical Readout from Quantum Expectation Values
        self.readout_head = nn.Sequential(
            nn.Linear(n_qubits, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        # Initialize final bias to mean pKi (~7.32) for fast convergence
        nn.init.constant_(self.readout_head[-1].bias, 7.32)

    def forward(self, batch_data: Batch) -> torch.Tensor:
        # Drug branch
        drug_embed = self.gnn_encoder(
            batch_data.x,
            batch_data.edge_index,
            batch_data.edge_attr,
            batch_data.batch
        )

        # Protein branch
        target_tokens = batch_data.target_tokens
        if target_tokens.dim() == 1:
            batch_size = drug_embed.size(0)
            target_tokens = target_tokens.view(batch_size, -1)

        target_embed = self.protein_encoder(target_tokens)

        # Concatenation & Reduction to quantum angle space
        joint_embed = torch.cat([drug_embed, target_embed], dim=-1)
        quantum_angles = self.reduction_mlp(joint_embed) * math.pi

        # Quantum circuit forward pass
        q_expvals = self.quantum_circuit(quantum_angles)

        # Regression prediction
        pred_pki = self.readout_head(q_expvals)
        return pred_pki
