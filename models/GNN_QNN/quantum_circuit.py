"""
Parameterized Quantum Circuit for Proposed Hybrid GNN + QNN
Executes angle-encoded variational quantum circuit using PennyLane with PyTorch backpropagation.
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


class HybridQuantumCircuit(nn.Module):
    """
    Variational Quantum Circuit for Hybrid GNN + QNN.
    Receives joint drug-target representations from the GNN and protein encoders,
    encodes them into quantum states via angle embedding, processes via entangling variational layers,
    and returns expectation values for classical regression readout.
    """
    def __init__(self, n_qubits: int = 4, n_layers: int = 2, device_name: str = "default.qubit"):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device_name = device_name

        self.dev = qml.device(device_name, wires=n_qubits)

        @qml.qnode(self.dev, interface='torch', diff_method='backprop')
        def _circuit(inputs, weights):
            # Angle embedding along Y axis
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
            # Strongly entangling layers (rotations + CNOT entanglers)
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            # Measure Pauli-Z expectation on all qubits
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.qnode = _circuit

        weight_shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
        self.weights = nn.Parameter(torch.randn(weight_shape) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, n_qubits)
        Returns:
            out: (batch_size, n_qubits)
        """
        expvals = self.qnode(x, self.weights)
        out = torch.stack(expvals, dim=1).float()
        return out
