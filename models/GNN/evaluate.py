"""
Evaluation Script for Model 1 (GNN)
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
from models.GNN.model import DTIGNNModel


def evaluate_gnn(
    checkpoint_path: Optional[str] = None,
    test_parquet_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    batch_size: int = 64,
    device_name: str = "auto"
) -> Dict[str, Any]:
    device = get_device(device_name)
    print(f"[GNN Evaluation] Using device: {device}")

    if checkpoint_path is None:
        checkpoint_path = str(CHECKPOINTS_DIR / "GNN" / "best_gnn_model.pt")
    if test_parquet_path is None:
        test_parquet_path = str(SPLITS_DIR / "test.parquet")
    if output_dir is None:
        output_dir = str(RESULTS_DIR / "GNN")

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
    if not os.path.exists(test_parquet_path):
        raise FileNotFoundError(f"Test split not found at: {test_parquet_path}")

    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    
    model = DTIGNNModel(
        atom_feature_dim=config.get('atom_feature_dim', 41),
        bond_feature_dim=config.get('bond_feature_dim', 16),
        gnn_hidden_dim=config.get('hidden_dim', 64),
        gnn_num_layers=config.get('gnn_num_layers', 3),
        gnn_out_dim=config.get('gnn_out_dim', 64),
        prot_out_dim=config.get('prot_out_dim', 64),
        dropout=config.get('dropout', 0.2)
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

    print(f"[GNN Evaluation] Running inference on {len(test_df)} test samples...")
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
    metrics['model'] = 'GNN'

    print("\n[GNN Test Metrics]")
    print(f"  RMSE:        {metrics['rmse']:.4f}")
    print(f"  MAE:         {metrics['mae']:.4f}")
    print(f"  R²:          {metrics['r2']:.4f}")
    print(f"  Pearson r:   {metrics['pearson_r']:.4f} (p={metrics['pearson_p']:.2e})")
    print(f"  Spearman ρ:  {metrics['spearman_rho']:.4f} (p={metrics['spearman_p']:.2e})")

    # Save predictions
    pred_df = test_df.copy()
    pred_df['predicted_pKi'] = np.round(y_pred, 4)
    pred_df['residual'] = np.round(y_true - y_pred, 4)
    pred_csv_path = os.path.join(output_dir, "test_predictions.csv")
    pred_df.to_csv(pred_csv_path, index=False)
    print(f"Saved test predictions to: {pred_csv_path}")

    # Save metrics JSON
    metrics_json_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    # Plot actual vs predicted
    plot_path = os.path.join(output_dir, "actual_vs_predicted.png")
    plot_actual_vs_predicted(y_true, y_pred, "GNN (Bond-Aware GINE)", plot_path, metrics)
    print(f"Saved scatter plot to: {plot_path}")

    return metrics


if __name__ == '__main__':
    evaluate_gnn()
