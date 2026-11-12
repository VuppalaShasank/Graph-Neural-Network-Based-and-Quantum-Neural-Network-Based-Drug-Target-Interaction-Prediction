"""
Evaluation Script for Model 2 (Standalone QNN)
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
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from common.config import SPLITS_DIR, RESULTS_DIR, CHECKPOINTS_DIR
from common.dataset_loader import DTIQNNFeatureDataset
from common.evaluation import calculate_metrics, plot_actual_vs_predicted
from common.reproducibility import get_device
from models.QNN.model import DTIQNNModel


def evaluate_qnn(
    checkpoint_path: Optional[str] = None,
    test_parquet_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    batch_size: int = 64,
    device_name: str = "auto"
) -> Dict[str, Any]:
    device = get_device(device_name)
    print(f"[QNN Evaluation] Using device: {device}")

    if checkpoint_path is None:
        checkpoint_path = str(CHECKPOINTS_DIR / "QNN" / "best_qnn_model.pt")
    if test_parquet_path is None:
        test_parquet_path = str(SPLITS_DIR / "test.parquet")
    if output_dir is None:
        output_dir = str(RESULTS_DIR / "QNN")

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
    if not os.path.exists(test_parquet_path):
        raise FileNotFoundError(f"Test split not found at: {test_parquet_path}")

    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})

    model = DTIQNNModel(
        drug_feature_dim=config.get('drug_feature_dim', 264),
        drug_hidden_dim=config.get('drug_hidden_dim', 64),
        prot_out_dim=config.get('prot_out_dim', 64),
        joint_hidden_dim=config.get('joint_hidden_dim', 32),
        n_qubits=config.get('n_qubits', 4),
        circuit_depth=config.get('qnn_circuit_depth', 2),
        quantum_device=config.get('qnn_device', 'default.qubit'),
        dropout=config.get('dropout', 0.2)
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load test dataset
    test_df = pd.read_parquet(test_parquet_path)
    test_ds = DTIQNNFeatureDataset(test_df)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_targets = []

    print(f"[QNN Evaluation] Running inference on {len(test_df)} test samples...")
    with torch.no_grad():
        for drug_feats, target_tokens, pki, pair_id in test_loader:
            drug_feats = drug_feats.to(device)
            target_tokens = target_tokens.to(device)
            preds = model(drug_feats, target_tokens)
            all_preds.extend(preds.view(-1).cpu().numpy())
            all_targets.extend(pki.view(-1).cpu().numpy())

    y_true = np.array(all_targets, dtype=np.float64)
    y_pred = np.array(all_preds, dtype=np.float64)

    metrics = calculate_metrics(y_true, y_pred)
    metrics['model'] = 'QNN'

    print("\n[QNN Test Metrics]")
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
    plot_actual_vs_predicted(y_true, y_pred, "QNN (Variational Quantum Circuit)", plot_path, metrics)
    print(f"Saved scatter plot to: {plot_path}")

    return metrics


if __name__ == '__main__':
    evaluate_qnn()

