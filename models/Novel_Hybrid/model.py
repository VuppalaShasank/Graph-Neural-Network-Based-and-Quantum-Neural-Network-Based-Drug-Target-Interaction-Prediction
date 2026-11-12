"""
Model 4: Proposed Novel Bi-Partite Quantum Docking Hybrid Model (BP-QDA)
Combines:
1. Bond-Aware GINE molecular graph encoder (64-d)
2. 1D CNN protein sequence encoder (64-d)
3. Domain-specific angle projections (2 drug angles, 2 target angles)
4. Bi-Partite Quantum Docking Circuit with cross-entity entanglement (8 observables)
5. Classical-Quantum Residual Skip Connection (128 classical + 8 quantum = 136-d)
6. Calibrated regression readout head
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
from models.Novel_Hybrid.quantum_circuit import BiPartiteQuantumDockingCircuit


class DTIBiPartiteQuantumHybridModel(nn.Module):
    """
    Model 4: Novel Bi-Partite Quantum Docking Hybrid Architecture (BP-QDA)
    """
    def __init__(
        self,
        atom_feature_dim: int = 41,
        bond_feature_dim: int = 16,
        gnn_hidden_dim: int = 64,
        gnn_num_layers: int = 3,
        gnn_out_dim: int = 64,
        prot_out_dim: int = 64,
        n_qubits: int = 4,
        quantum_device: str = "default.qubit",
        enable_reuploading: bool = True,
        enable_residual: bool = True,
        dropout: float = 0.2
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.enable_residual = enable_residual

        # 1. Drug Graph Encoder (Bond-Aware GINE)
        self.gnn_encoder = GNNEncoder(
            atom_feature_dim=atom_feature_dim,
            bond_feature_dim=bond_feature_dim,
            hidden_dim=gnn_hidden_dim,
            num_layers=gnn_num_layers,
            out_dim=gnn_out_dim,
            dropout=dropout
        )

        # 2. Shared Protein Sequence Encoder (1D CNN)
        self.protein_encoder = Protein1DCNNEncoder(
            out_dim=prot_out_dim,
            dropout=dropout
        )

        # 3. Domain-Specific Independent Angle Projections
        # Drug projection: 64 -> 32 -> 2 angles
        self.drug_angle_mlp = nn.Sequential(
            nn.Linear(gnn_out_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Tanh()
        )

        # Target projection: 64 -> 32 -> 2 angles
        self.target_angle_mlp = nn.Sequential(
            nn.Linear(prot_out_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Tanh()
        )

        # 4. Bi-Partite Quantum Docking Circuit
        self.quantum_circuit = BiPartiteQuantumDockingCircuit(
            n_qubits=n_qubits,
            device_name=quantum_device,
            enable_reuploading=enable_reuploading
        )
        n_quantum_observables = 8

        # 5. Residual Fusion & Classical Readout Head
        # If residual enabled: concatenates 128 classical + 8 quantum = 136-d
        in_head_dim = (gnn_out_dim + prot_out_dim + n_quantum_observables) if enable_residual else n_quantum_observables
        self.readout_head = nn.Sequential(
            nn.Linear(in_head_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        # Initialize output bias to training mean (7.32) for fast, stable convergence
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

        # Independent Angle Projection (mapped to [-pi, pi])
        theta_drug = self.drug_angle_mlp(drug_embed) * math.pi
        theta_target = self.target_angle_mlp(target_embed) * math.pi

        # Combine into (batch, 4): wires 0,1 for drug, wires 2,3 for target
        angles_4d = torch.cat([theta_drug, theta_target], dim=-1)

        # Bi-Partite Quantum Docking execution
        q_expvals = self.quantum_circuit(angles_4d)

        # Residual Fusion
        if self.enable_residual:
            joint_features = torch.cat([drug_embed, target_embed, q_expvals], dim=-1)
        else:
            joint_features = q_expvals

        # Continuous pKi Prediction
        pred_pki = self.readout_head(joint_features)
        return pred_pki
