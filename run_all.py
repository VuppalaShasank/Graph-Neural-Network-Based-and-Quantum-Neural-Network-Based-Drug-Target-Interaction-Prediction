"""
Master End-to-End Execution Script
Executes all project stages sequentially:
1. Environment and dataset inspection
2. Curated dataset construction (10,000 observations)
3. Common pair-group-aware data splitting (70/15/15)
4. Smoke test validation
5. Training Model 1: GNN
6. Training Model 2: QNN
7. Training Model 3: Proposed Hybrid GNN+QNN
8. Fair model evaluation and comparison
9. Report generation and result artifact compilation
"""

import os
import sys
import time
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.config import DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE, DEFAULT_EPOCHS, RANDOM_SEED


def run_entire_pipeline(
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    smoke_test: bool = False,
    resume: bool = False,
    seed: int = RANDOM_SEED,
    device_name: str = "auto"
):
    master_start_time = time.time()
    print("\n" + "="*80)
    print(" STARTING MASTER END-TO-END DTI PREDICTION EXPERIMENT PIPELINE")
    print("="*80)
    print(f"Epochs: {epochs} | Batch Size: {batch_size} | Smoke Test: {smoke_test} | Seed: {seed}\n")

    # Step 1: Inspection
    print("\n>>> STEP 1: Running Dataset & Environment Inspection...")
    from common.dataset_inspection import main as inspect_main
    inspect_main()

    # Step 2: Curation
    print("\n>>> STEP 2: Running Curated Dataset Construction (10,000 observations)...")
    from common.preprocess import run_curation_pipeline
    run_curation_pipeline(random_seed=seed)

    # Step 3: Splitting
    print("\n>>> STEP 3: Running Common Data Splitting (70% / 15% / 15%)...")
    from common.data_splitting import create_and_save_splits
    create_and_save_splits(random_seed=seed)

    # Step 4: Smoke Tests
    print("\n>>> STEP 4: Executing Smoke Tests...")
    import unittest
    from tests.test_smoke import TestSmokePipeline
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSmokePipeline)
    runner = unittest.TextTestRunner(verbosity=1)
    res = runner.run(suite)
    if not res.wasSuccessful():
        raise RuntimeError("Smoke tests failed! Aborting training.")

    # Step 5: Model 1 (GNN)
    print("\n>>> STEP 5: Training Model 1: GNN (Bond-Aware GINE)...")
    from models.GNN.train import train_gnn
    train_gnn(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        smoke_test=smoke_test,
        resume=resume,
        seed=seed,
        device_name=device_name
    )

    # Step 6: Model 2 (QNN)
    print("\n>>> STEP 6: Training Model 2: Standalone QNN (Variational Quantum Circuit)...")
    from models.QNN.train import train_qnn
    train_qnn(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        smoke_test=smoke_test,
        resume=resume,
        seed=seed,
        device_name=device_name
    )

    # Step 7: Model 3 (Hybrid GNN+QNN)
    print("\n>>> STEP 7: Training Model 3: Proposed Hybrid (GNN + QNN)...")
    from models.GNN_QNN.train import train_gnn_qnn
    train_gnn_qnn(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        smoke_test=smoke_test,
        resume=resume,
        seed=seed,
        device_name=device_name
    )

    # Step 8: Fair Comparison
    print("\n>>> STEP 8: Generating Fair Model Comparison Table & Visualizations...")
    from common.compare_models import run_model_comparison
    run_model_comparison()

    total_time = time.time() - master_start_time
    print("\n" + "="*80)
    print(f" ALL EXPERIMENTAL PIPELINE STAGES COMPLETED SUCCESSFULLY IN {total_time/60:.2f} MINUTES")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run Entire DTI Experiment Pipeline")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE, help="Learning rate")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick smoke test")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Compute device")

    args = parser.parse_args()
    run_entire_pipeline(
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

