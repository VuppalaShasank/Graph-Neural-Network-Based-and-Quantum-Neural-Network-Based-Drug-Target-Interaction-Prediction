"""
4-Model Comparative Evaluation Script
Benchmarks all four implemented architectures on the held-out test partition:
  - Model 1: Classical Bond-Aware GNN (GINEConv + 1D CNN)
  - Model 2: Standalone Variational QNN (4-Qubit VQC via PennyLane)
  - Model 3: Baseline Hybrid GNN + QNN (32-fold Linear Bottleneck)
  - Model 4: Proposed Novel Bi-Partite Quantum Docking Hybrid (BP-QDA)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.compare_4models import run_4model_comparison


if __name__ == "__main__":
    print("=" * 70)
    print("Running Comparative Evaluation across all 4 models on held-out test set...")
    print("=" * 70)
    run_4model_comparison()
