# Phase 0: BindingDB Dataset and Environment Inspection Report

## 1. Actual TSV File Information
- **Filename**: `BindingDB_All.tsv`
- **Absolute Path**: `C:\Users\vuppa\OneDrive\Desktop\AIDD_Proj_Sem7\BindingDB_All.tsv`
- **File Size**: `8.368 GB` (8,984,698,089 bytes)
- **Total Header Columns**: 640
- **Target Chain Sequence Columns Available**: 50 (Chain 1 to Chain 50)
- **Profiling Time**: 44.41 seconds across all 3,237,052 rows.

## 2. Actual Column Names and Key Schema Mapping
| Semantic Entity | Detected TSV Column Name | Column Index |
|---|---|---|
| **Drug ID** | `BindingDB Reactant_set_id` | 0 |
| **Ligand SMILES** | `Ligand SMILES` | 1 |
| **Ligand InChI** | `Ligand InChI` | 2 |
| **Ligand InChI Key** | `Ligand InChI Key` | 3 |
| **Monomer ID** | `BindingDB MonomerID` | 4 |
| **Ligand Name** | `BindingDB Ligand Name` | 5 |
| **Target Name** | `Target Name` | 6 |
| **Target Organism** | `Target Source Organism According to Curator or DataSource` | 7 |
| **Affinity: Ki** | `Ki (nM)` | 8 |
| **Affinity: IC50** | `IC50 (nM)` | 9 |
| **Affinity: Kd** | `Kd (nM)` | 10 |
| **Affinity: EC50** | `EC50 (nM)` | 11 |
| **pH** | `pH` | 14 |
| **Temperature** | `Temp (C)` | 15 |
| **Curation/DataSource** | `Curation/DataSource` | 16 |
| **Article DOI** | `Article DOI` | 17 |
| **PMID** | `PMID` | 19 |
| **Date of publication** | `Date of publication` | 23 |
| **Date in BindingDB** | `Date in BindingDB` | 24 |
| **Number of Chains** | `Number of Protein Chains in Target (>1 implies a multichain complex)` | 39 |
| **Target Sequence 1** | `BindingDB Target Chain Sequence 1` | 40 |
| **Target UniProt ID 1** | `UniProt (SwissProt) Primary ID of Target Chain 1` | 44 |
| **PDB ID(s)** | `PDB ID(s) for Ligand-Target Complex` | 30 |

## 3. Full Dataset Profiling Statistics
- **Total Raw Records**: 3,237,052
- **Records with SMILES**: 3,236,971 (100.00%)
- **Records with Target Chain 1 Sequence**: 3,237,046 (100.00%)
- **Records with Any Target Chain Sequence**: 3,237,046 (100.00%)
- **Single-Chain Targets**: 3,050,612
- **Multi-Chain Targets**: 186,434
- **Unique SMILES (Approx)**: 1,432,770
- **Unique Targets**: 8,927
- **Unique Drug-Target Pairs**: 2,438,225
- **Repeated / Duplicate Drug-Target Observations**: 798,746
- **Records Lacking All Affinities**: 2,949
- **Records with >=1 Valid Numeric Affinity**: 2,583,404

## 4. Affinity Availability, Qualifiers, and Distributions
| Endpoint | Total Present | Exact Numeric | `>` (Censored) | `<` (Censored) | `~` (Approx) | Invalid/Non-numeric |
|---|---|---|---|---|---|---|
| **Ki (nM)** | 622,916 | 541,869 | 62,209 | 18,811 | 0 | 27 |
| **Kd (nM)** | 131,524 | 82,471 | 40,286 | 8,734 | 0 | 33 |
| **IC50 (nM)** | 2,214,673 | 1,747,452 | 280,895 | 186,191 | 0 | 135 |
| **EC50 (nM)** | 270,817 | 216,812 | 42,936 | 11,042 | 0 | 27 |

### Distribution of Exact Numeric Measurements (Sampled 100k Values)

#### Ki (nM)
- **Min**: 6e-06 nM | **Max**: 1,000,000,000.0 nM | **Median**: 35.555 nM
- **pAffinity ($-\log_{10}(M)$)**:
  - Median: 7.45
  - Mean: 7.39 ± 1.49
  - IQR: [6.35, 8.42]
  - Min: 0.00 | Max: 14.22

#### Kd (nM)
- **Min**: 1e-06 nM | **Max**: 3,492,000,000.0 nM | **Median**: 219.0 nM
- **pAffinity ($-\log_{10}(M)$)**:
  - Median: 6.66
  - Mean: 6.75 ± 1.66
  - IQR: [5.60, 8.00]
  - Min: -0.54 | Max: 15.00

#### IC50 (nM)
- **Min**: 0.0012 nM | **Max**: 459,000,000.0 nM | **Median**: 2600.0 nM
- **pAffinity ($-\log_{10}(M)$)**:
  - Median: 5.59
  - Mean: 5.94 ± 1.40
  - IQR: [4.87, 6.96]
  - Min: 0.34 | Max: 11.92

#### EC50 (nM)
- **Min**: 1.4e-07 nM | **Max**: 11,320,000,000.0 nM | **Median**: 280.25 nM
- **pAffinity ($-\log_{10}(M)$)**:
  - Median: 6.55
  - Mean: 6.59 ± 1.58
  - IQR: [5.37, 7.76]
  - Min: -1.05 | Max: 15.85

## 5. Data Quality and Validation Findings
- **SMILES Validation (RDKit Sample N=10,000)**:
  - Valid: 10,000 (100.0%)
  - Invalid: 0
- **Protein Sequence Validation (Sample N=10,000)**:
  - Pure Standard Amino Acids (20 canonical): 9,989 (99.89%)
  - Sequences with Ambiguous/Non-standard Characters: 11
  - Character frequencies: `{'B': 11}` (e.g. 'X' for unknown, 'U' for Sec).
- **Measurement Units**: All header endpoints specify units explicitly as `(nM)`. Values are reported in nM. Formula $p\text{Affinity} = 9 - \log_{10}(\text{Value}_{\text{nM}})$ is verified.

## 6. Environment and Hardware Inspection
- **Operating System**: Windows 10 (10.0.26300)
- **CPU**: Intel64 Family 6 Model 198 Stepping 2, GenuineIntel
- **System RAM**: 15.38 GB total, 2.77 GB currently available.
- **SSD Free Space**: 387.15 GB free on drive C.
- **NVIDIA GPU**: NVIDIA GeForce RTX 5060 Laptop GPU
- **GPU VRAM**: 8151 MiB (8 GB)
- **NVIDIA Driver**: 591.91 (supports CUDA 13.1)
- **PyTorch Installed**: `2.9.0+cpu`
  - CUDA Available in PyTorch: `False` (installed build is CPU build; automatic CPU fallback enabled for QNN and GNN; CUDA can be utilized if PyTorch with CUDA is configured).
- **PyTorch Geometric**: `2.8.0.post1`
- **RDKit**: `True`
- **PennyLane**: `Not installed` (needs pip install for QNN phase)

## 7. Estimated Usable Observations
- Records with valid SMILES + Target Chain 1 Sequence + Exact Numeric Ki: **>541,869** candidates.
- The curated dataset target size of **10,000 observations** can be easily and strictly satisfied using only high-quality, exact numeric Ki measurements without relaxing quality criteria or relying on censored/approximate entries.

## 8. Recommended Primary Affinity Endpoint
- **Primary Endpoint**: `Ki (nM)`
- **Justification**: Ki offers 541,869 exact numeric observations (well exceeding the 10,000 threshold), representing the true thermodynamic equilibrium inhibition constant. It avoids the assay-dependent variability inherent in IC50 and provides a direct biophysical binding parameter.
- Ki is the gold-standard inhibition constant directly tied to binding free energy ($\Delta G = RT \ln K_i$), uncorrupted by assay substrate concentrations.

## 9. Recommended Preprocessing Strategy
1. **Streaming Filter**: Stream `BindingDB_All.tsv`, extracting records where `Ki (nM)` is exact, positive and finite, `Ligand SMILES` is non-empty, and `BindingDB Target Chain Sequence 1` is non-empty.
2. **Drug Standardization**: Canonicalize SMILES via RDKit, preserving stereochemistry, removing stereochemically impossible molecules, salts, and fragments, computing ECFP4 fingerprints and RDKit descriptors.
3. **Protein Validation**: Standardize sequence characters, filter out non-protein/nucleotide sequences, retain single-chain proteins with valid sequence lengths (e.g. 30 to 1,500 amino acids).
4. **Endpoint Transformation**: Convert Ki to molar ($K_i 	imes 10^-9$), compute $pKi = 9 - \log_10(K_i)$.
5. **Replicate Aggregation**: Group identical (Canonical SMILES, Target Sequence) pairs and aggregate replicate measurements using the median pKi, recording replicate count and variance.
6. **Diversity-Aware Selection**: Sample 10,000 observations deterministically (random seed 42) with compound and target diversity preservation (stratified target capping to avoid any single target dominating).
7. **Storage**: Save to `data/processed/BindingDB_curated_10000.parquet` and `.csv`.

## 10. Estimated Storage and Memory Requirements
- Curated dataset (10,000 records): Parquet ~5-10 MB, CSV ~8-15 MB.
- Molecular Graph Cache: ~20-30 MB on disk.
- Protein Embedding Cache: ~15-25 MB on disk.
- Model Checkpoints: GNN (~15 MB), QNN (~5 MB), Hybrid (~20 MB).
- Peak RAM usage during streaming: < 150 MB.
- Peak RAM usage during model training (batch size 32-64): < 1.5 GB.
- Execution is fully within the 15.38 GB RAM and 8 GB VRAM envelope.
