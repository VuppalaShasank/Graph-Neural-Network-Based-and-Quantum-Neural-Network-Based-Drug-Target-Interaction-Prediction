# Phase 2: Curated Dataset Construction Report

## 1. Curation Overview
- **Raw BindingDB Records Scanned**: 500,000
- **Unique Qualified Pairs Aggregated**: 67,591
- **Final Retained Curated Observations**: **10,000** (strictly meets the maximum 10,000 threshold)
- **Unique Chemical Compounds (Canonical SMILES)**: 8,508
- **Unique Protein Targets**: 853
- **Primary Affinity Endpoint**: `Ki (nM)`
- **Random Seed for Reproducible Diversity Sampling**: 42
- **Curation Runtime**: 38.85 seconds

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
   - Transformed to molar: $\text{Ki}_M = \text{Ki}_{\text{nM}} \times 10^{-9}$.
   - Transformed to negative logarithm: $p\text{Ki} = -\log_{10}(\text{Ki}_M) = 9 - \log_{10}(\text{Ki}_{\text{nM}})$.
   - Outliers outside range $[2.0, 15.0]$ excluded.
4. **Duplicate Handling & Replicate Aggregation**:
   - Replicate experimental measurements for the exact same `(canonical_smiles, sequence)` pair were aggregated using the **median $p\text{Ki}$**.
   - Original measurement count (`num_measurements`) and standard deviation (`pKi_std`) are preserved for every pair.
5. **Diversity-Aware Sampling**:
   - Target representation was capped at a maximum of 150 compounds per target.
   - Prevents overrepresented proteins (e.g. HIV protease) from dominating the distribution.
   - Sampled deterministically with seed 42.

## 3. Retained Dataset Properties
- **Target Distribution**:
- **Trace amine-associated receptor 1**: 271 pairs
- **D(2) dopamine receptor**: 202 pairs
- **Orexin/Hypocretin receptor type 1**: 200 pairs
- **5-hydroxytryptamine receptor 2A**: 186 pairs
- **Mu-type opioid receptor**: 181 pairs
- **Arachidonate 5-lipoxygenase-activating protein**: 174 pairs
- **Apoptosis regulator Bcl-2**: 156 pairs
- **Orexin receptor type 2**: 152 pairs
- **Serine/threonine-protein kinase pim-1**: 150 pairs
- **Coagulation factor X**: 149 pairs

- **Replicate Measurement Statistics**:
- **1 measurement(s)**: 8,658 pairs
- **2 measurement(s)**: 883 pairs
- **3 measurement(s)**: 261 pairs
- **4 measurement(s)**: 88 pairs
- **5 measurement(s)**: 34 pairs

- **pKi Distribution**:
  - **Min**: 2.01
  - **Max**: 13.82
  - **Mean**: 7.32 ± 1.49
  - **Median**: 7.36
  - **25th Percentile**: 6.27
  - **75th Percentile**: 8.37

## 4. File Deliverables
- Parquet: `data/processed/BindingDB_curated_10000.parquet`
- CSV: `data/processed/BindingDB_curated_10000.csv`
