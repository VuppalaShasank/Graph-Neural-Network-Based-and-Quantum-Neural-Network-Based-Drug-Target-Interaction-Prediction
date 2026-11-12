"""
Quantum Neural Network Architecture for Drug-Target Interaction Prediction
Combines Morgan Fingerprints + RDKit descriptors with 1D CNN protein encoder,
projects to 4-qubit Hilbert space, and predicts pKi via variational quantum circuit measurements.
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

from common.protein_encoder import Protein1DCNNEncoder
from models.QNN.quantum_circuit import ParameterizedQuantumCircuit


class DTIQNNModel(nn.Module):
    """
    Full DTI Quantum Neural Network Model:
    Drug Branch: ECFP4 (256) + Descriptors (8) -> MLP -> Drug Embedding (64)
    Protein Branch: Sequence Tokens -> Shared Protein1DCNNEncoder -> Target Embedding (64)
    Fusion: Concatenation (128) -> Trainable Projection -> Quantum State Preparation (n_qubits)
    Quantum Circuit: Parameterized Quantum Circuit with Entanglement -> Pauli-Z Expectations
    Readout Head: Classical Regression Readout -> Continuous pKi
    """
    def __init__(
        self,
        drug_feature_dim: int = 264,
        drug_hidden_dim: int = 64,
        prot_out_dim: int = 64,
        joint_hidden_dim: int = 32,
        n_qubits: int = 4,
        circuit_depth: int = 2,
        quantum_device: str = "default.qubit",
        dropout: float = 0.2
    ):
        super().__init__()
        self.n_qubits = n_qubits

        # Drug representation branch
        self.drug_mlp = nn.Sequential(
            nn.Linear(drug_feature_dim, drug_hidden_dim),
            nn.LayerNorm(drug_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Protein representation branch
        self.protein_encoder = Protein1DCNNEncoder(
            out_dim=prot_out_dim,
            dropout=dropout
        )

        # Joint projection to quantum dimensionality
        fusion_dim = drug_hidden_dim + prot_out_dim
        self.joint_projection = nn.Sequential(
            nn.Linear(fusion_dim, joint_hidden_dim),
            nn.ReLU(),
            nn.Linear(joint_hidden_dim, n_qubits),
            nn.Tanh() # Maps to [-1, 1], then scaled by pi for rotation angles
        )

        # Parameterized Quantum Circuit
        self.qnn_circuit = ParameterizedQuantumCircuit(
            n_qubits=n_qubits,
            n_layers=circuit_depth,
            device_name=quantum_device
        )

        # Classical readout head from Pauli-Z expectations
        self.readout_head = nn.Sequential(
            nn.Linear(n_qubits, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        # Initialize final bias to mean pKi (~7.32) for fast convergence
        nn.init.constant_(self.readout_head[-1].bias, 7.32)

    def forward(self, drug_features: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            drug_features: (batch_size, 264)
            target_tokens: (batch_size, seq_len)
        Returns:
            pred_pki: (batch_size, 1)
        """
        # Drug embedding
        drug_embed = self.drug_mlp(drug_features)

        # Target embedding
        target_embed = self.protein_encoder(target_tokens)

        # Joint concatenation & projection to qubit angles
        joint_embed = torch.cat([drug_embed, target_embed], dim=-1)
        quantum_angles = self.joint_projection(joint_embed) * math.pi

        # Variational quantum circuit execution
        q_expvals = self.qnn_circuit(quantum_angles)

        # Classical readout to continuous pKi
        pred_pki = self.readout_head(q_expvals)
        return pred_pki
