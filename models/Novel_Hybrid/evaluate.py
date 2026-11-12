"""
Evaluation Script for Model 4 (Novel Bi-Partite Quantum Docking Hybrid)
Computes RMSE, MAE, R2, Pearson r, Spearman rho on held-out test split.
Saves test predictions, metrics, and actual vs. predicted plot.
"""

import os
import sys
from pathlib import Path
import json
from typing import Optional, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import pandas as pd

from common.config import SPLITS_DIR, RESULTS_DIR, CHECKPOINTS_DIR
from common.dataset_loader import DTIGraphDataset
from torch_geometric.loader import DataLoader as PyGDataLoader
from common.evaluation import calculate_metrics, plot_actual_vs_predicted
from common.reproducibility import get_device
from models.Novel_Hybrid.model import DTIBiPartiteQuantumHybridModel


def evaluate_novel_hybrid(
    checkpoint_path: Optional[str] = None,
    test_parquet_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    batch_size: int = 64,
    device_name: str = "auto"
) -> Dict[str, Any]:
    device = get_device(device_name)
    print(f"[Model 4 Evaluation] Using device: {device}")

    if checkpoint_path is None:
        checkpoint_path = str(CHECKPOINTS_DIR / "Novel_Hybrid" / "best_novel_hybrid_model.pt")
    if test_parquet_path is None:
        test_parquet_path = str(SPLITS_DIR / "test.parquet")
    if output_dir is None:
        output_dir = str(RESULTS_DIR / "Novel_Hybrid")

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
    if not os.path.exists(test_parquet_path):
        raise FileNotFoundError(f"Test split not found at: {test_parquet_path}")

    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})

    model = DTIBiPartiteQuantumHybridModel(
        atom_feature_dim=config.get('atom_feature_dim', 41),
        bond_feature_dim=config.get('bond_feature_dim', 16),
        gnn_hidden_dim=config.get('gnn_hidden_dim', 64),
        gnn_num_layers=config.get('gnn_num_layers', 3),
        gnn_out_dim=config.get('gnn_out_dim', 64),
        prot_out_dim=config.get('prot_out_dim', 64),
        n_qubits=config.get('n_qubits', 4),
        quantum_device=config.get('quantum_device', 'default.qubit'),
        enable_reuploading=config.get('enable_data_reuploading', True),
        enable_residual=config.get('enable_residual_connection', True),
        dropout=config.get('gnn_dropout', 0.2)
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load test dataset
    test_df = pd.read_parquet(test_parquet_path)
    test_ds = DTIGraphDataset(test_df)
    test_loader = PyGDataLoader(test_ds, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_targets = []
    all_pair_ids = []

    print(f"[Model 4 Evaluation] Running inference on {len(test_df)} test samples...")
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            preds = model(batch)
            all_preds.extend(preds.view(-1).cpu().numpy())
            all_targets.extend(batch.y.view(-1).cpu().numpy())
            if hasattr(batch, 'pair_id'):
                all_pair_ids.extend(batch.pair_id)

    y_true = np.array(all_targets, dtype=np.float64)
    y_pred = np.array(all_preds, dtype=np.float64)

    metrics = calculate_metrics(y_true, y_pred)
    metrics['model'] = 'Novel_Hybrid_BP_QDA'

    print("\n=======================================================")
    print(" MODEL 4 (NOVEL BP-QDA HYBRID) TEST METRICS")
    print("=======================================================")
    print(f"  RMSE:        {metrics['rmse']:.4f}")
    print(f"  MAE:         {metrics['mae']:.4f}")
    print(f"  R²:          {metrics['r2']:.4f}")
    print(f"  Pearson r:   {metrics['pearson_r']:.4f} (p={metrics['pearson_p']:.2e})")
    print(f"  Spearman ρ:  {metrics['spearman_rho']:.4f} (p={metrics['spearman_p']:.2e})")
    print(f"  Test N:      {metrics['n_samples']}")
    print("=======================================================\n")

    # Save metrics
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save predictions dataframe
    preds_df = pd.DataFrame({
        'pair_id': all_pair_ids if all_pair_ids else [f"TEST_{i}" for i in range(len(y_true))],
        'pKi': y_true,
        'predicted_pKi': y_pred,
        'residual': y_true - y_pred,
        'canonical_smiles': test_df['canonical_smiles'].values,
        'target_id': test_df['target_id'].values if 'target_id' in test_df.columns else test_df['uniprot_id'].values
    })
    preds_csv = os.path.join(output_dir, "test_predictions.csv")
    preds_df.to_csv(preds_csv, index=False)
    print(f"Saved test predictions to: {preds_csv}")

    # Plot actual vs predicted
    plot_path = os.path.join(output_dir, "actual_vs_predicted.png")
    plot_actual_vs_predicted(
        y_true, y_pred,
        "Model 4: Novel Bi-Partite Quantum Docking Hybrid (BP-QDA)",
        plot_path,
        metrics=metrics
    )
    print(f"Saved scatter plot to: {plot_path}")

    return metrics


if __name__ == '__main__':
    evaluate_novel_hybrid()
