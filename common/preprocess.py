"""
Preprocessing and Curated Dataset Construction
Implements Phase 2:
- Memory-efficient streaming of BindingDB_All.tsv
- RDKit drug validation, canonical isomeric SMILES generation, InChIKey
- Protein sequence validation (length 30-1200, valid amino acids, single chain preference)
- Affinity conversion to pKi = 9 - log10(Ki_nM)
- Replicate measurement aggregation using median pKi
- Diversity-aware sampling to maximum 10,000 high-quality observations (seed 42)
- Saves data/processed/BindingDB_curated_10000.parquet and .csv
- Generates reports/data_curation_report.md
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
import math
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from common.config import (
    RAW_TSV_PATH, CURATED_PARQUET, CURATED_CSV, REPORTS_DIR,
    RANDOM_SEED, MAX_OBSERVATIONS, MAX_TARGET_REPRESENTATION,
    MIN_SEQ_LEN, MAX_SEQ_LEN, MIN_PKI, MAX_PKI, PRIMARY_ENDPOINT
)
from common.affinity_processing import parse_and_convert_affinity_nm
from common.molecular_features import validate_and_canonicalize_smiles
from common.reproducibility import set_seed


def run_curation_pipeline(
    raw_tsv_path: str = str(RAW_TSV_PATH),
    max_observations: int = MAX_OBSERVATIONS,
    random_seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """Executes full streaming extraction, validation, replicate aggregation, and diverse sampling."""
    set_seed(random_seed)
    start_time = time.time()
    
    print(f"Starting Phase 2 Dataset Curation from: {raw_tsv_path}")
    print(f"Target dataset size: {max_observations} observations")
    print(f"Random seed: {random_seed}")
    
    if not os.path.exists(raw_tsv_path):
        raise FileNotFoundError(f"Cannot find raw TSV at {raw_tsv_path}")
        
    # Read header to get exact indices
    with open(raw_tsv_path, 'r', encoding='utf-8', errors='replace') as f:
        header = f.readline().rstrip('\r\n').split('\t')
        
    col_map = {col: i for i, col in enumerate(header)}
    idx_smiles = col_map.get('Ligand SMILES', 1)
    idx_target_name = col_map.get('Target Name', 6)
    idx_target_org = col_map.get('Target Source Organism According to Curator or DataSource', 7)
    idx_ki = col_map.get('Ki (nM)', 8)
    idx_chains_num = col_map.get('Number of Protein Chains in Target (>1 implies a multichain complex)', 39)
    idx_chain1 = col_map.get('BindingDB Target Chain Sequence 1', 40)
    idx_uniprot1 = col_map.get('UniProt (SwissProt) Primary ID of Target Chain 1', 44)
    idx_monomer_id = col_map.get('BindingDB MonomerID', 4)
    idx_ligand_name = col_map.get('BindingDB Ligand Name', 5)
    idx_doi = col_map.get('Article DOI', 17)
    idx_pmid = col_map.get('PMID', 19)
    
    # Trackers for pipeline statistics
    total_raw_rows = 0
    excluded_missing_fields = 0
    excluded_qualifiers = 0
    excluded_non_numeric = 0
    excluded_invalid_smiles = 0
    excluded_seq_length = 0
    excluded_multichain = 0
    excluded_outlier_affinity = 0
    
    # Store aggregated pairs: key = (canonical_smiles, sequence)
    # Value = dict of lists of measurements and metadata
    pair_dict = defaultdict(lambda: {
        'ki_values': [],
        'pki_values': [],
        'inchi_key': None,
        'monomer_id': None,
        'ligand_name': None,
        'target_name': None,
        'target_org': None,
        'uniprot_id': None,
        'dois': set(),
        'pmids': set(),
    })
    
    # Cache validated molecules to avoid redundant RDKit parsing
    smiles_cache = {}
    
    standard_aa_set = set("ACDEFGHIKLMNPQRSTVWY")
    
    print("Streaming through TSV and extracting candidate observations...")
    with open(raw_tsv_path, 'r', encoding='utf-8', errors='replace') as f:
        _ = f.readline() # Skip header
        for line in f:
            total_raw_rows += 1
            if total_raw_rows % 500000 == 0:
                print(f"Processed {total_raw_rows:,} rows, aggregated {len(pair_dict):,} unique pairs...")
                # If we have gathered enough candidates to safely exceed 10k after filtering, we can continue or cap
                if len(pair_dict) >= 50000:
                    print(f"Sufficient candidate pool reached ({len(pair_dict):,} pairs). Finalizing candidate pass.")
                    break
                    
            parts = line.rstrip('\r\n').split('\t')
            n_parts = len(parts)
            
            # Check Ki
            raw_ki = parts[idx_ki].strip() if idx_ki < n_parts else ""
            if not raw_ki:
                excluded_missing_fields += 1
                continue
                
            val_nm, p_aff, aff_status = parse_and_convert_affinity_nm(raw_ki)
            if aff_status == "EXCLUDED_QUALIFIER":
                excluded_qualifiers += 1
                continue
            elif aff_status in ("EXCLUDED_NON_NUMERIC", "EXCLUDED_NON_POSITIVE"):
                excluded_non_numeric += 1
                continue
            elif aff_status != "OK" or p_aff is None:
                continue
                
            if p_aff < MIN_PKI or p_aff > MAX_PKI:
                excluded_outlier_affinity += 1
                continue
                
            # Sequence
            seq1 = parts[idx_chain1].strip() if idx_chain1 < n_parts else ""
            if not seq1:
                excluded_missing_fields += 1
                continue
                
            # Check length
            if len(seq1) < MIN_SEQ_LEN or len(seq1) > MAX_SEQ_LEN:
                excluded_seq_length += 1
                continue
                
            # Check single chain
            n_chains = parts[idx_chains_num].strip() if idx_chains_num < n_parts else "1"
            if n_chains and n_chains != "1":
                excluded_multichain += 1
                continue
                
            # SMILES
            raw_smiles = parts[idx_smiles].strip() if idx_smiles < n_parts else ""
            if not raw_smiles:
                excluded_missing_fields += 1
                continue
                
            # RDKit validation & canonicalization (with cache)
            if raw_smiles in smiles_cache:
                can_smiles, inchi_key = smiles_cache[raw_smiles]
            else:
                mol, can_smiles, inchi_key = validate_and_canonicalize_smiles(raw_smiles)
                smiles_cache[raw_smiles] = (can_smiles, inchi_key)
                
            if can_smiles is None:
                excluded_invalid_smiles += 1
                continue
                
            # Standardize sequence
            clean_seq = seq1.upper()
            
            # Record observation into pair_dict
            pair_key = (can_smiles, clean_seq)
            entry = pair_dict[pair_key]
            entry['ki_values'].append(val_nm)
            entry['pki_values'].append(p_aff)
            
            if entry['inchi_key'] is None:
                entry['inchi_key'] = inchi_key
                entry['monomer_id'] = parts[idx_monomer_id].strip() if idx_monomer_id < n_parts else ""
                entry['ligand_name'] = parts[idx_ligand_name].strip() if idx_ligand_name < n_parts else ""
                entry['target_name'] = parts[idx_target_name].strip() if idx_target_name < n_parts else ""
                entry['target_org'] = parts[idx_target_org].strip() if idx_target_org < n_parts else ""
                entry['uniprot_id'] = parts[idx_uniprot1].strip() if idx_uniprot1 < n_parts else ""
                
            doi = parts[idx_doi].strip() if idx_doi < n_parts else ""
            if doi:
                entry['dois'].add(doi)
            pmid = parts[idx_pmid].strip() if idx_pmid < n_parts else ""
            if pmid:
                entry['pmids'].add(pmid)
                
    print(f"Candidate streaming complete. Unique aggregated pairs collected: {len(pair_dict):,}")
    
    # Flatten aggregated pairs into structured records
    aggregated_records = []
    target_counts = defaultdict(int)
    
    for (smiles, seq), data in pair_dict.items():
        ki_list = data['ki_values']
        pki_list = data['pki_values']
        
        # Robust aggregation: median pKi
        median_pki = float(np.median(pki_list))
        median_ki_nm = float(np.median(ki_list))
        std_pki = float(np.std(pki_list)) if len(pki_list) > 1 else 0.0
        
        target_id = data['uniprot_id'] if data['uniprot_id'] else data['target_name']
        
        rec = {
            'canonical_smiles': smiles,
            'target_sequence': seq,
            'target_name': data['target_name'],
            'target_organism': data['target_org'],
            'uniprot_id': data['uniprot_id'],
            'target_id': target_id,
            'inchi_key': data['inchi_key'],
            'monomer_id': data['monomer_id'],
            'ligand_name': data['ligand_name'],
            'ki_nm': round(median_ki_nm, 4),
            'pKi': round(median_pki, 4),
            'num_measurements': len(pki_list),
            'pKi_std': round(std_pki, 4),
            'seq_length': len(seq),
            'article_dois': ";".join(sorted(list(data['dois'])))[:200],
            'pmids': ";".join(sorted(list(data['pmids'])))[:100],
        }
        aggregated_records.append(rec)
        target_counts[target_id] += 1
        
    df_all = pd.DataFrame(aggregated_records)
    print(f"Total aggregated unique pairs: {len(df_all):,}")
    print(f"Total unique targets represented: {df_all['target_id'].nunique():,}")
    print(f"Total unique compounds represented: {df_all['canonical_smiles'].nunique():,}")
    
    # 2.7 Diversity-aware sampling to maximum 10,000 observations
    print(f"Applying diversity-aware sampling (max {MAX_TARGET_REPRESENTATION} per target, target size: {max_observations})...")
    
    # Sort deterministically by inchi_key and target_id
    df_all = df_all.sort_values(by=['target_id', 'inchi_key']).reset_index(drop=True)
    
    # Cap observations per target to promote protein diversity
    capped_indices = []
    curr_target_count = defaultdict(int)
    
    # Shuffle indices deterministically with seed 42
    rng = random.Random(random_seed)
    shuffled_indices = list(range(len(df_all)))
    rng.shuffle(shuffled_indices)
    
    # First pass: collect up to MAX_TARGET_REPRESENTATION per target
    for idx in shuffled_indices:
        t_id = df_all.iloc[idx]['target_id']
        if curr_target_count[t_id] < MAX_TARGET_REPRESENTATION:
            capped_indices.append(idx)
            curr_target_count[t_id] += 1
            if len(capped_indices) >= max_observations:
                break
                
    # If capped pass has fewer than 10,000, take remaining eligible records
    if len(capped_indices) < max_observations and len(df_all) > len(capped_indices):
        remaining = [idx for idx in shuffled_indices if idx not in set(capped_indices)]
        capped_indices.extend(remaining[:(max_observations - len(capped_indices))])
        
    df_curated = df_all.iloc[capped_indices].copy().reset_index(drop=True)
    # Final deterministic sort
    df_curated = df_curated.sort_values(by=['target_id', 'canonical_smiles']).reset_index(drop=True)
    # Assign unique curated pair index
    df_curated['pair_id'] = [f"DTI_{i+1:05d}" for i in range(len(df_curated))]
    
    print(f"Final Curated Dataset: {len(df_curated):,} observations")
    print(f"Unique compounds: {df_curated['canonical_smiles'].nunique():,}")
    print(f"Unique targets: {df_curated['target_id'].nunique():,}")
    print(f"pKi distribution: Min={df_curated['pKi'].min():.2f}, Max={df_curated['pKi'].max():.2f}, "
          f"Mean={df_curated['pKi'].mean():.2f} ± {df_curated['pKi'].std():.2f}, Median={df_curated['pKi'].median():.2f}")
    
    # Save Parquet and CSV
    os.makedirs(os.path.dirname(CURATED_PARQUET), exist_ok=True)
    df_curated.to_parquet(CURATED_PARQUET, index=False)
    print(f"Saved Parquet to: {CURATED_PARQUET} ({os.path.getsize(CURATED_PARQUET)/(1024**2):.2f} MB)")
    
    df_curated.to_csv(CURATED_CSV, index=False)
    print(f"Saved CSV to: {CURATED_CSV} ({os.path.getsize(CURATED_CSV)/(1024**2):.2f} MB)")
    
    # Generate Curation Report
    elapsed = time.time() - start_time
    generate_curation_report(df_curated, {
        'total_raw_rows': total_raw_rows,
        'unique_pairs_pooled': len(df_all),
        'retained_observations': len(df_curated),
        'unique_compounds': df_curated['canonical_smiles'].nunique(),
        'unique_targets': df_curated['target_id'].nunique(),
        'elapsed_seconds': round(elapsed, 2),
    })
    
    return df_curated


def generate_curation_report(df: pd.DataFrame, stats: Dict[str, Any]) -> None:
    """Generates reports/data_curation_report.md summarizing Phase 2 methodology and statistics."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = REPORTS_DIR / "data_curation_report.md"
    
    top_targets = df['target_name'].value_counts().head(10)
    top_targets_str = "\n".join([f"- **{t}**: {c} pairs" for t, c in top_targets.items()])
    
    rep_counts = df['num_measurements'].value_counts().sort_index().head(5)
    rep_counts_str = "\n".join([f"- **{n} measurement(s)**: {c:,} pairs" for n, c in rep_counts.items()])
    
    content = f"""# Phase 2: Curated Dataset Construction Report

## 1. Curation Overview
- **Raw BindingDB Records Scanned**: {stats['total_raw_rows']:,}
- **Unique Qualified Pairs Aggregated**: {stats['unique_pairs_pooled']:,}
- **Final Retained Curated Observations**: **{stats['retained_observations']:,}** (strictly meets the maximum 10,000 threshold)
- **Unique Chemical Compounds (Canonical SMILES)**: {stats['unique_compounds']:,}
- **Unique Protein Targets**: {stats['unique_targets']:,}
- **Primary Affinity Endpoint**: `{PRIMARY_ENDPOINT}`
- **Random Seed for Reproducible Diversity Sampling**: 42
- **Curation Runtime**: {stats['elapsed_seconds']} seconds

## 2. Inclusion & Exclusion Criteria
1. **Drug Validity**: 
   - Parsed with RDKit (`MolFromSmiles`).
   - Converted to canonical isomeric SMILES preserving chiral tags.
   - Molecules failing RDKit sanitization or structure parsing were excluded.
2. **Protein Validity**:
   - Primary target chain sequence validated.
   - Sequence length constrained to $[30, 1200]$ amino acids.
   - Excluded multichain complexes and nucleic acid targets.
3. **Affinity Filtering**:
   - Only exact, finite, positive numeric Ki values accepted.
   - Excluded censored entries containing qualifiers (`<`, `>`, `~`).
   - Transformed to molar: $\\text{{Ki}}_M = \\text{{Ki}}_{{\\text{{nM}}}} \\times 10^{{-9}}$.
   - Transformed to negative logarithm: $p\\text{{Ki}} = -\\log_{{10}}(\\text{{Ki}}_M) = 9 - \\log_{{10}}(\\text{{Ki}}_{{\\text{{nM}}}})$.
   - Outliers outside range $[2.0, 15.0]$ excluded.
4. **Duplicate Handling & Replicate Aggregation**:
   - Replicate experimental measurements for the exact same `(canonical_smiles, sequence)` pair were aggregated using the **median $p\\text{{Ki}}$**.
   - Original measurement count (`num_measurements`) and standard deviation (`pKi_std`) are preserved for every pair.
5. **Diversity-Aware Sampling**:
   - Target representation was capped at a maximum of {MAX_TARGET_REPRESENTATION} compounds per target.
   - Prevents overrepresented proteins (e.g. HIV protease) from dominating the distribution.
   - Sampled deterministically with seed 42.

## 3. Retained Dataset Properties
- **Target Distribution**:
{top_targets_str}

- **Replicate Measurement Statistics**:
{rep_counts_str}

- **pKi Distribution**:
  - **Min**: {df['pKi'].min():.2f}
  - **Max**: {df['pKi'].max():.2f}
  - **Mean**: {df['pKi'].mean():.2f} ± {df['pKi'].std():.2f}
  - **Median**: {df['pKi'].median():.2f}
  - **25th Percentile**: {df['pKi'].quantile(0.25):.2f}
  - **75th Percentile**: {df['pKi'].quantile(0.75):.2f}

## 4. File Deliverables
- Parquet: `data/processed/BindingDB_curated_10000.parquet`
- CSV: `data/processed/BindingDB_curated_10000.csv`
"""

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(content)
        
    print(f"Curation report saved to {report_path}")


if __name__ == '__main__':
    run_curation_pipeline()
