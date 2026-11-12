# Phase 3: Common Data Splitting Report

## 1. Splitting Protocol
- **Total Curated Observations**: 10,000
- **Partition Ratio**: 70% Train / 15% Validation / 15% Test
- **Partitioning Strategy**: Pair-group-aware random partitioning
- **Leakage Prevention**: Guaranteed 0 overlapping `(drug, target)` pairs between any two splits.
- **Evaluation Type Note**: This is a standard pair-held-out evaluation partition; it is not a cold-drug or cold-target split.
- **Random Seed**: 42

## 2. Partition Summary
| Split | Samples | Percentage | Unique Compounds | Unique Targets | Mean pKi ± Std |
|---|---|---|---|---|---|
| **Training** | 7,000 | 70.0% | 6,186 | 776 | 7.33 ± 1.48 |
| **Validation** | 1,500 | 15.0% | 1,458 | 417 | 7.28 ± 1.48 |
| **Testing** | 1,500 | 15.0% | 1,443 | 435 | 7.31 ± 1.51 |

## 3. Storage Locations
- Train: `data/splits/train.parquet` (7,000 rows)
- Validation: `data/splits/val.parquet` (1,500 rows)
- Test: `data/splits/test.parquet` (1,500 rows)
- Split Metadata: `data/splits/split_info.json`
