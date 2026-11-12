"""
Parameterized Quantum Circuit for Standalone QNN
Implements AngleEmbedding, StronglyEntanglingLayers, and Pauli-Z expectation measurements in PennyLane.
Supports PyTorch automatic differentiation via backprop.
"""

import sys
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import pennylane as qml


class ParameterizedQuantumCircuit(nn.Module):
    """
    Variational Quantum Circuit using PennyLane PyTorch interface.
    
    Architecture:
    1. Angle Embedding: maps n_qubits classical features to qubit rotation angles (RY rotations).
    2. Strongly Entangling Layers: parameterized single-qubit rotations (RX, RY, RZ) + CNOT entanglement.
    3. Expectation Value Measurements: Pauli-Z expectation on each wire -> (batch_size, n_qubits).
    """
    def __init__(self, n_qubits: int = 4, n_layers: int = 2, device_name: str = "default.qubit"):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device_name = device_name

        self.dev = qml.device(device_name, wires=n_qubits)

        # Define QNode
        @qml.qnode(self.dev, interface='torch', diff_method='backprop')
        def _circuit(inputs, weights):
            # inputs shape: (batch_size, n_qubits)
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.qnode = _circuit

        # Initialize trainable variational weights: shape (n_layers, n_qubits, 3)
        weight_shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
        self.weights = nn.Parameter(torch.randn(weight_shape) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, n_qubits) - Scaled input features (typically [-pi, pi])
        Returns:
            out: (batch_size, n_qubits) - Measured Pauli-Z expectation values in [-1, 1]
        """
        # Execute batched QNode
        expvals = self.qnode(x, self.weights)
        # expvals is a list of tensors of shape (batch_size,)
        out = torch.stack(expvals, dim=1).float()
        return out
