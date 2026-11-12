"""
Common Data Splitting Pipeline
Implements Phase 3:
- Pair-group-aware 70% / 15% / 15% train/val/test partitioning
- Deterministic seed 42
- Ensures zero leakage of drug-target pairs between splits
- Generates data/splits/train.parquet, val.parquet, test.parquet and CSVs
- Generates split statistics and reports/data_splitting_report.md
"""

import os
import sys
from pathlib import Path
import json
from typing import Tuple

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from common.config import (
    CURATED_PARQUET, SPLITS_DIR, REPORTS_DIR, RANDOM_SEED,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO
)
from common.reproducibility import set_seed


def create_and_save_splits(
    parquet_path: str = str(CURATED_PARQUET),
    splits_dir: str = str(SPLITS_DIR),
    random_seed: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generates identical, reproducible train, validation, and test splits."""
    set_seed(random_seed)
    
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Curated parquet not found at {parquet_path}. Run preprocessing first.")
        
    df = pd.read_parquet(parquet_path)
    total_len = len(df)
    print(f"Loading curated dataset: {total_len} rows from {parquet_path}")
    
    # Construct unique pair key: (canonical_smiles, target_sequence)
    # In our curated dataset, each row is already a unique pair!
    # To be strictly safe and pair-group-aware:
    df['pair_signature'] = df['canonical_smiles'] + "::" + df['target_sequence']
    unique_pairs = df['pair_signature'].unique()
    
    # Shuffle unique pair signatures deterministically with seed 42
    rng = np.random.RandomState(random_seed)
    shuffled_pairs = rng.permutation(unique_pairs)
    
    n_total_pairs = len(shuffled_pairs)
    n_train = int(round(TRAIN_RATIO * n_total_pairs))
    n_val = int(round(VAL_RATIO * n_total_pairs))
    n_test = n_total_pairs - n_train - n_val
    
    train_pair_set = set(shuffled_pairs[:n_train])
    val_pair_set = set(shuffled_pairs[n_train:n_train + n_val])
    test_pair_set = set(shuffled_pairs[n_train + n_val:])
    
    train_df = df[df['pair_signature'].isin(train_pair_set)].copy().reset_index(drop=True)
    val_df = df[df['pair_signature'].isin(val_pair_set)].copy().reset_index(drop=True)
    test_df = df[df['pair_signature'].isin(test_pair_set)].copy().reset_index(drop=True)
    
    # Verification checks
    train_pairs = set(train_df['pair_signature'])
    val_pairs = set(val_df['pair_signature'])
    test_pairs = set(test_df['pair_signature'])
    
    assert len(train_pairs.intersection(val_pairs)) == 0, "Train and Val contain overlapping pairs!"
    assert len(train_pairs.intersection(test_pairs)) == 0, "Train and Test contain overlapping pairs!"
    assert len(val_pairs.intersection(test_pairs)) == 0, "Val and Test contain overlapping pairs!"
    assert len(train_df) + len(val_df) + len(test_df) == total_len, "Split row count mismatch!"
    
    print("\nSplit statistics:")
    print(f"  Train: {len(train_df):,} rows ({len(train_df)/total_len*100:.1f}%) | "
          f"Unique drugs: {train_df['canonical_smiles'].nunique():,} | Unique targets: {train_df['target_id'].nunique():,}")
    print(f"  Val:   {len(val_df):,} rows ({len(val_df)/total_len*100:.1f}%) | "
          f"Unique drugs: {val_df['canonical_smiles'].nunique():,} | Unique targets: {val_df['target_id'].nunique():,}")
    print(f"  Test:  {len(test_df):,} rows ({len(test_df)/total_len*100:.1f}%) | "
          f"Unique drugs: {test_df['canonical_smiles'].nunique():,} | Unique targets: {test_df['target_id'].nunique():,}")
    
    # Save splits
    os.makedirs(splits_dir, exist_ok=True)
    train_parquet = os.path.join(splits_dir, "train.parquet")
    train_csv = os.path.join(splits_dir, "train.csv")
    val_parquet = os.path.join(splits_dir, "val.parquet")
    val_csv = os.path.join(splits_dir, "val.csv")
    test_parquet = os.path.join(splits_dir, "test.parquet")
    test_csv = os.path.join(splits_dir, "test.csv")
    
    train_df.to_parquet(train_parquet, index=False)
    train_df.to_csv(train_csv, index=False)
    val_df.to_parquet(val_parquet, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_parquet(test_parquet, index=False)
    test_df.to_csv(test_csv, index=False)
    
    # Save metadata json
    split_info = {
        'total_observations': total_len,
        'random_seed': random_seed,
        'train_ratio': TRAIN_RATIO,
        'val_ratio': VAL_RATIO,
        'test_ratio': TEST_RATIO,
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'train_unique_drugs': int(train_df['canonical_smiles'].nunique()),
        'train_unique_targets': int(train_df['target_id'].nunique()),
        'val_unique_drugs': int(val_df['canonical_smiles'].nunique()),
        'val_unique_targets': int(val_df['target_id'].nunique()),
        'test_unique_drugs': int(test_df['canonical_smiles'].nunique()),
        'test_unique_targets': int(test_df['target_id'].nunique()),
        'pKi_train_mean': round(float(train_df['pKi'].mean()), 4),
        'pKi_train_std': round(float(train_df['pKi'].std()), 4),
        'pKi_val_mean': round(float(val_df['pKi'].mean()), 4),
        'pKi_val_std': round(float(val_df['pKi'].std()), 4),
        'pKi_test_mean': round(float(test_df['pKi'].mean()), 4),
        'pKi_test_std': round(float(test_df['pKi'].std()), 4),
    }
    
    split_json_path = os.path.join(splits_dir, "split_info.json")
    with open(split_json_path, 'w', encoding='utf-8') as f:
        json.dump(split_info, f, indent=2)
        
    # Generate splitting report
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(REPORTS_DIR, "data_splitting_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"""# Phase 3: Common Data Splitting Report

## 1. Splitting Protocol
- **Total Curated Observations**: {total_len:,}
- **Partition Ratio**: 70% Train / 15% Validation / 15% Test
- **Partitioning Strategy**: Pair-group-aware random partitioning
- **Leakage Prevention**: Guaranteed 0 overlapping `(drug, target)` pairs between any two splits.
- **Evaluation Type Note**: This is a standard pair-held-out evaluation partition; it is not a cold-drug or cold-target split.
- **Random Seed**: {random_seed}

## 2. Partition Summary
| Split | Samples | Percentage | Unique Compounds | Unique Targets | Mean pKi ± Std |
|---|---|---|---|---|---|
| **Training** | {len(train_df):,} | {len(train_df)/total_len*100:.1f}% | {train_df['canonical_smiles'].nunique():,} | {train_df['target_id'].nunique():,} | {train_df['pKi'].mean():.2f} ± {train_df['pKi'].std():.2f} |
| **Validation** | {len(val_df):,} | {len(val_df)/total_len*100:.1f}% | {val_df['canonical_smiles'].nunique():,} | {val_df['target_id'].nunique():,} | {val_df['pKi'].mean():.2f} ± {val_df['pKi'].std():.2f} |
| **Testing** | {len(test_df):,} | {len(test_df)/total_len*100:.1f}% | {test_df['canonical_smiles'].nunique():,} | {test_df['target_id'].nunique():,} | {test_df['pKi'].mean():.2f} ± {test_df['pKi'].std():.2f} |

## 3. Storage Locations
- Train: `data/splits/train.parquet` ({len(train_df):,} rows)
- Validation: `data/splits/val.parquet` ({len(val_df):,} rows)
- Test: `data/splits/test.parquet` ({len(test_df):,} rows)
- Split Metadata: `data/splits/split_info.json`
""")
    print(f"Data splitting report saved to {report_path}")
    return train_df, val_df, test_df


if __name__ == '__main__':
    create_and_save_splits()
