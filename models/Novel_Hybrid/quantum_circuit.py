"""
Bi-Partite Quantum Docking Circuit (BP-QDA) for Model 4 (Novel Hybrid)
Implements:
1. Subsystem partitioning: Qubits 0 & 1 (Drug Subspace), Qubits 2 & 3 (Target Subspace)
2. Domain-specific angle embedding (wires 0,1: Drug; wires 2,3: Target)
3. Intra-molecular transformations (drug conformation & protein backbone)
4. Inter-molecular Quantum Docking gate strictly across the interface (Q1 <-> Q2)
5. Quantum Data Re-Uploading for universal non-linear approximation
6. Multi-basis and cross-entity correlator measurements (<Z1 @ Z2>, <X1 @ X2>)
"""

import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import pennylane as qml


class BiPartiteQuantumDockingCircuit(nn.Module):
    """
    Novel Bi-Partite Quantum Docking Ansatz:
    Inputs: (batch_size, 4) tensor where:
        Index 0, 1: Drug subspace angles (from GNN)
        Index 2, 3: Target subspace angles (from 1D CNN)
    Wires:
        0: Drug global scaffold
        1: Drug binding interface pharmacophore
        2: Target binding pocket residue contact
        3: Target global backbone context
    """
    def __init__(
        self,
        n_qubits: int = 4,
        device_name: str = "default.qubit",
        enable_reuploading: bool = True
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.enable_reuploading = enable_reuploading
        self.device_name = device_name

        self.dev = qml.device(device_name, wires=n_qubits)

        # Trainable parameters:
        # Layer 1: intra_drug (2, 3), intra_target (2, 3), docking (2, 3) = 6 * 3 = 18 params
        # Layer 2 (re-uploading): same structure = 18 params
        n_layers = 2 if enable_reuploading else 1
        self.weights = nn.Parameter(torch.randn(n_layers, 3, 2, 3) * 0.1)

        @qml.qnode(self.dev, interface='torch', diff_method='backprop')
        def _circuit(inputs, weights):
            # ==========================================
            # LAYER 1: INITIAL STATE PREPARATION & DOCKING
            # ==========================================
            # 1. Independent Subsystem Encodings (Ry rotations)
            # Wires 0, 1: Drug | Wires 2, 3: Target
            qml.AngleEmbedding(inputs, wires=range(4), rotation='Y')

            # 2. Intra-Drug Conformational Evolution
            qml.Rot(*weights[0, 0, 0], wires=0)
            qml.Rot(*weights[0, 0, 1], wires=1)
            qml.CNOT(wires=[0, 1])

            # 3. Intra-Target Backbone Evolution
            qml.Rot(*weights[0, 1, 0], wires=2)
            qml.Rot(*weights[0, 1, 1], wires=3)
            qml.CNOT(wires=[2, 3])

            # 4. Inter-Entity Quantum Docking Interface (Q1 <-> Q2)
            qml.CNOT(wires=[1, 2])
            qml.Rot(*weights[0, 2, 0], wires=1)
            qml.Rot(*weights[0, 2, 1], wires=2)
            qml.CNOT(wires=[2, 1])

            # ==========================================
            # LAYER 2: QUANTUM DATA RE-UPLOADING (Rx)
            # ==========================================
            if enable_reuploading:
                # Re-upload classical features into the evolved entangled state
                qml.AngleEmbedding(inputs, wires=range(4), rotation='X')

                # Intra-Drug Evolution 2
                qml.Rot(*weights[1, 0, 0], wires=0)
                qml.Rot(*weights[1, 0, 1], wires=1)
                qml.CNOT(wires=[1, 0])

                # Intra-Target Evolution 2
                qml.Rot(*weights[1, 1, 0], wires=2)
                qml.Rot(*weights[1, 1, 1], wires=3)
                qml.CNOT(wires=[3, 2])

                # Inter-Entity Docking Evolution 2
                qml.CNOT(wires=[1, 2])
                qml.Rot(*weights[1, 2, 0], wires=1)
                qml.Rot(*weights[1, 2, 1], wires=2)
                qml.CNOT(wires=[1, 2])

            # ==========================================
            # 5. OBSERVABLES: SINGLE & CROSS-CORRELATORS
            # ==========================================
            return [
                # Single-qubit longitudinal expectation values (4 values)
                qml.expval(qml.PauliZ(0)),
                qml.expval(qml.PauliZ(1)),
                qml.expval(qml.PauliZ(2)),
                qml.expval(qml.PauliZ(3)),
                # Transverse coherence at the binding interface (2 values)
                qml.expval(qml.PauliX(1)),
                qml.expval(qml.PauliX(2)),
                # Novel Cross-Entity Interaction Correlators (2 values)
                qml.expval(qml.PauliZ(1) @ qml.PauliZ(2)),  # Longitudinal binding energy correlator
                qml.expval(qml.PauliX(1) @ qml.PauliX(2)),  # Transverse coherence correlator
            ]

        self.qnode = _circuit

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (batch_size, 4) - [theta_drug_0, theta_drug_1, theta_target_0, theta_target_1]
        Returns:
            q_out: (batch_size, 8) - Measured expectation values
        """
        expvals = self.qnode(inputs, self.weights)
        q_out = torch.stack(expvals, dim=1).float()
        return q_out
