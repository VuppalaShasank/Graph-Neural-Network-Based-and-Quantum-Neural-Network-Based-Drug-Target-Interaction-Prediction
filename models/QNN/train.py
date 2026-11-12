"""
Training Script for Model 2 (Standalone QNN)
Handles training loop, AdamW optimizer, early stopping, validation monitoring, and checkpointing.
"""

import os
import sys
import argparse
import time
from pathlib import Path
import json
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
from tqdm import tqdm

from common.config import (
    SPLITS_DIR, RESULTS_DIR, CHECKPOINTS_DIR, LOGS_DIR,
    RANDOM_SEED, DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE, DEFAULT_EPOCHS, PATIENCE
)
from common.dataset_loader import get_feature_dataloaders
from common.evaluation import calculate_metrics, plot_training_curves
from common.reproducibility import set_seed, get_device
from models.QNN.model import DTIQNNModel
from models.QNN.evaluate import evaluate_qnn


def train_qnn(
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    smoke_test: bool = False,
    resume: bool = False,
    seed: int = RANDOM_SEED,
    device_name: str = "auto"
):
    set_seed(seed)
    device = get_device(device_name)
    print(f"\n=======================================================")
    print(f" TRAINING MODEL 2: QNN (Variational Quantum Circuit)")
    print(f" Device: {device} | Epochs: {epochs} | Batch Size: {batch_size}")
    print(f"=======================================================\n")

    # Paths
    train_parquet = SPLITS_DIR / "train.parquet"
    val_parquet = SPLITS_DIR / "val.parquet"
    test_parquet = SPLITS_DIR / "test.parquet"

    checkpoints_dir = CHECKPOINTS_DIR / "QNN"
    results_dir = RESULTS_DIR / "QNN"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    log_file = LOGS_DIR / "qnn_train.log"

    # Load data
    train_df = pd.read_parquet(train_parquet)
    val_df = pd.read_parquet(val_parquet)
    test_df = pd.read_parquet(test_parquet)

    if smoke_test:
        print("[SMOKE TEST MODE] Restricting datasets to small subset...")
        train_df = train_df.head(64)
        val_df = val_df.head(32)
        test_df = test_df.head(32)
        epochs = 2

    # Load config
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Build DataLoaders
    train_loader, val_loader, test_loader = get_feature_dataloaders(
        train_df, val_df, test_df, batch_size=batch_size
    )

    # Model
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

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=config.get('weight_decay', 1e-4))
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_epoch = 0
    start_epoch = 1
    train_losses = []
    val_losses = []

    best_ckpt_path = checkpoints_dir / "best_qnn_model.pt"
    latest_ckpt_path = checkpoints_dir / "latest_qnn_model.pt"

    if resume and latest_ckpt_path.exists():
        print(f"Resuming training from: {latest_ckpt_path}")
        checkpoint = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        print(f"Resumed from epoch {start_epoch-1}, Best Val Loss so far: {best_val_loss:.4f}")

    start_train_time = time.time()
    no_improve_epochs = 0

    with open(log_file, 'a', encoding='utf-8') as log_f:
        log_f.write(f"\n--- QNN Training Session Started at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        log_f.write(f"Config: {config}\n")

        for epoch in range(start_epoch, epochs + 1):
            epoch_start = time.time()
            model.train()
            total_train_loss = 0.0
            n_train_samples = 0

            for drug_feats, target_tokens, pki, _ in train_loader:
                drug_feats = drug_feats.to(device)
                target_tokens = target_tokens.to(device)
                pki = pki.to(device)

                optimizer.zero_grad()
                preds = model(drug_feats, target_tokens)
                loss = criterion(preds.view(-1), pki.view(-1))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                batch_len = drug_feats.size(0)
                total_train_loss += loss.item() * batch_len
                n_train_samples += batch_len

            train_loss = total_train_loss / max(1, n_train_samples)
            train_losses.append(train_loss)

            # Validation
            model.eval()
            total_val_loss = 0.0
            n_val_samples = 0
            val_preds_list = []
            val_targets_list = []

            with torch.no_grad():
                for drug_feats, target_tokens, pki, _ in val_loader:
                    drug_feats = drug_feats.to(device)
                    target_tokens = target_tokens.to(device)
                    pki = pki.to(device)

                    preds = model(drug_feats, target_tokens)
                    loss = criterion(preds.view(-1), pki.view(-1))

                    batch_len = drug_feats.size(0)
                    total_val_loss += loss.item() * batch_len
                    n_val_samples += batch_len

                    val_preds_list.extend(preds.view(-1).cpu().numpy())
                    val_targets_list.extend(pki.view(-1).cpu().numpy())

            val_loss = total_val_loss / max(1, n_val_samples)
            val_losses.append(val_loss)
            scheduler.step(val_loss)

            val_metrics = calculate_metrics(np.array(val_targets_list), np.array(val_preds_list))
            epoch_time = time.time() - epoch_start

            log_line = (f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | "
                        f"Val Loss: {val_loss:.4f} | Val RMSE: {val_metrics['rmse']:.4f} | "
                        f"Val R²: {val_metrics['r2']:.4f} | Val r: {val_metrics['pearson_r']:.4f} | "
                        f"Time: {epoch_time:.1f}s")
            print(log_line)
            log_f.write(log_line + "\n")
            log_f.flush()

            # Save latest checkpoint
            checkpoint_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'config': config,
            }
            torch.save(checkpoint_state, latest_ckpt_path)

            # Checkpoint best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                no_improve_epochs = 0
                torch.save(checkpoint_state, best_ckpt_path)
                print(f"  --> Saved new best model checkpoint (Val Loss: {best_val_loss:.4f})")
            else:
                no_improve_epochs += 1

            # Early stopping check
            if not smoke_test and no_improve_epochs >= PATIENCE:
                print(f"Early stopping triggered after {PATIENCE} epochs without improvement.")
                break

    total_training_time = time.time() - start_train_time
    print(f"\nQNN Training finished in {total_training_time/60:.2f} minutes.")
    print(f"Best Epoch: {best_epoch} with Val Loss: {best_val_loss:.4f}")

    # Plot training curves
    plot_curves_path = results_dir / "training_curves.png"
    plot_training_curves(train_losses, val_losses, "QNN (Variational Quantum Circuit)", str(plot_curves_path))
    print(f"Saved learning curves to: {plot_curves_path}")

    # Evaluate on held-out test split
    print("\nRunning final evaluation on held-out test split...")
    test_metrics = evaluate_qnn(
        checkpoint_path=str(best_ckpt_path),
        test_parquet_path=str(test_parquet),
        output_dir=str(results_dir),
        batch_size=batch_size,
        device_name=device_name
    )
    test_metrics['training_time_seconds'] = round(total_training_time, 2)
    test_metrics['best_epoch'] = best_epoch

    # Save training summary
    summary_path = results_dir / "training_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(test_metrics, f, indent=2)

    return test_metrics


def main():
    parser = argparse.ArgumentParser(description="Train QNN Model")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE, help="Learning rate")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick smoke test")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (cpu/cuda/auto)")

    args = parser.parse_args()
    train_qnn(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        smoke_test=args.smoke_test,
        resume=args.resume,
        seed=args.seed,
        device_name=args.device
    )


if __name__ == '__main__':
    main()
