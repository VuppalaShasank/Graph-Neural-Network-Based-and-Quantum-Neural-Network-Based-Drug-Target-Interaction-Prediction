"""
Unified Pipeline Execution CLI
Supports executing stages independently or together:
- inspect
- preprocess
- split
- smoke-test
- train-gnn
- train-qnn
- train-gnn-qnn
- evaluate-gnn
- evaluate-qnn
- evaluate-gnn-qnn
- compare
"""

import os
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.config import DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE, DEFAULT_EPOCHS, RANDOM_SEED


def main():
    parser = argparse.ArgumentParser(
        description="Unified Pipeline for Drug-Target Interaction Prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=[
            "inspect", "preprocess", "split", "smoke-test",
            "train-gnn", "train-qnn", "train-gnn-qnn", "train-novel-hybrid",
            "evaluate-gnn", "evaluate-qnn", "evaluate-gnn-qnn", "evaluate-novel-hybrid",
            "compare", "compare-4models", "all"
        ],
        help="Pipeline stage to execute"
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE, help="Learning rate")
    parser.add_argument("--smoke-test", action="store_true", help="Run in quick smoke test mode")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (cpu/cuda/auto)")

    args = parser.parse_args()

    if args.stage == "inspect":
        from common.dataset_inspection import main as inspect_main
        inspect_main()

    elif args.stage == "preprocess":
        from common.preprocess import run_curation_pipeline
        run_curation_pipeline()

    elif args.stage == "split":
        from common.data_splitting import create_and_save_splits
        create_and_save_splits()

    elif args.stage == "smoke-test":
        import unittest
        from tests.test_smoke import TestSmokePipeline
        suite = unittest.TestLoader().loadTestsFromTestCase(TestSmokePipeline)
        runner = unittest.TextTestRunner(verbosity=2)
        runner.run(suite)

    elif args.stage == "train-gnn":
        from models.GNN.train import train_gnn
        train_gnn(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            smoke_test=args.smoke_test,
            resume=args.resume,
            seed=args.seed,
            device_name=args.device
        )

    elif args.stage == "train-qnn":
        from models.QNN.train import train_qnn
        train_qnn(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            smoke_test=args.smoke_test,
            resume=args.resume,
            seed=args.seed,
            device_name=args.device
        )

    elif args.stage == "train-gnn-qnn":
        from models.GNN_QNN.train import train_gnn_qnn
        train_gnn_qnn(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            smoke_test=args.smoke_test,
            resume=args.resume,
            seed=args.seed,
            device_name=args.device
        )

    elif args.stage == "train-novel-hybrid":
        from models.Novel_Hybrid.train import train_novel_hybrid
        train_novel_hybrid(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            smoke_test=args.smoke_test,
            resume=args.resume,
            seed=args.seed,
            device_name=args.device
        )

    elif args.stage == "evaluate-gnn":
        from models.GNN.evaluate import evaluate_gnn
        evaluate_gnn(batch_size=args.batch_size, device_name=args.device)

    elif args.stage == "evaluate-qnn":
        from models.QNN.evaluate import evaluate_qnn
        evaluate_qnn(batch_size=args.batch_size, device_name=args.device)

    elif args.stage == "evaluate-gnn-qnn":
        from models.GNN_QNN.evaluate import evaluate_gnn_qnn
        evaluate_gnn_qnn(batch_size=args.batch_size, device_name=args.device)

    elif args.stage == "evaluate-novel-hybrid":
        from models.Novel_Hybrid.evaluate import evaluate_novel_hybrid
        evaluate_novel_hybrid(batch_size=args.batch_size, device_name=args.device)

    elif args.stage == "compare":
        from common.compare_models import run_model_comparison
        run_model_comparison()

    elif args.stage == "compare-4models":
        from common.compare_4models import run_4model_comparison
        run_4model_comparison()

    elif args.stage == "all":
        from run_all import run_entire_pipeline
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

