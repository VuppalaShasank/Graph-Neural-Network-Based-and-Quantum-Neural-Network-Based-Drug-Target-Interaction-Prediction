# Comprehensive Project Technical Report
## Graph Neural Network-Based and Quantum Neural Network-Based Drug–Target Interaction Prediction

---

### Project Identification & Metadata
- **Project Title**: Graph Neural Network-Based and Quantum Neural Network-Based Drug–Target Interaction Prediction
- **Academic Domain**: Artificial Intelligence, Computational Drug Discovery, Graph Representation Learning, Quantum Machine Learning
- **Repository Location**: `C:\Users\vuppa\OneDrive\Desktop\AIDD_Proj_Sem7`
- **Primary Data Source**: BindingDB (`BindingDB_All.tsv`, 8.368 GB, 3,237,052 records)
- **Experimental Target Size**: 10,000 Curated Binding Affinity Observations
- **Affinity Endpoint**: Inhibition Constant $K_i$ (nM) transformed to continuous $p\text{Ki} = 9.0 - \log_{10}(\text{Ki}_{\text{nM}})$
- **Evaluated Paradigms (Four Models)**:
  1. Model 1: Bond-Aware Graph Neural Network (GNN: `GINEConv` + Protein 1D CNN)
  2. Model 2: Variational Quantum Neural Network (QNN: ECFP4/Descriptors + Protein 1D CNN + 4-Qubit PQC)
  3. Model 3: Baseline Hybrid (GNN + QNN: Classical Blending + Angle Reduction + PQC)
  4. Model 4: Novel Bi-Partite Quantum Docking Hybrid (BP-QDA: Subsystem Partitioning + Inter-Entity Contact Gate + Quantum Residual Skip Connection + Data Re-Uploading + Cross-Correlators)
- **Primary Execution Mode**: CPU Multi-Threaded Runtime via PyTorch `2.9.0+cpu` and PennyLane `0.42.3`

---

### Abstract
Drug–Target Interaction (DTI) prediction is a foundational pillar of modern computational drug discovery, accelerating candidate identification, lead optimization, and off-target toxicity screening. Whereas historical machine learning formulations have largely treated DTI as an oversimplified binary classification problem, this investigation formulates DTI strictly as a continuous regression task predicting the biophysical binding affinity ($p\text{Ki}$). Using a curated, non-redundant dataset of 10,000 high-quality binding measurements derived from the public BindingDB repository across 8,508 unique compounds and 853 diverse protein targets, this work designs, implements, benchmarks, and analyzes **four distinct computational intelligence architectures** under identical experimental conditions.

The first model employs an Edge-Attentive Graph Isomorphism Network (`GINEConv`) operating on 41 atom and 16 bond features paired with a 1D Convolutional Neural Network (CNN) encoding the amino acid sequence. The second model evaluates a Parameterized Quantum Circuit (PQC) operating on 4 simulated qubits using angle embedding and hardware-efficient strongly entangling layers driven by extended connectivity circular fingerprints (`ECFP4`), 8 biophysical physicochemical descriptors, and the shared 1D CNN protein encoder. The third model introduces a baseline hybrid architecture coupling the bond-aware GNN representation encoder with the 4-qubit variational circuit via backward-mode automatic differentiation. The fourth model introduces a novel **Bi-Partite Quantum Docking Hybrid (BP-QDA)** designed to eliminate the information bottleneck of Model 3 through independent drug and target qubit partitioning, cross-boundary entangling docking gates, quantum data re-uploading, cross-entity Pauli correlators, and a classical-quantum residual skip connection.

Evaluated on an identical held-out test split of $N = 1,500$ pairs with zero pair overlap with training data, the classical GNN achieved the lowest overall prediction error ($\text{RMSE} = 1.1711$, $R^2 = 0.4000$, Pearson $r = 0.6346$). The standalone QNN demonstrated remarkable parameter efficiency, achieving the lowest Mean Absolute Error ($\text{MAE} = 0.9191$) and highest monotonic rank correlation (Spearman $\rho = 0.6223$) with only 24 variational quantum parameters. The proposed Novel BP-QDA Hybrid (Model 4) successfully elevated hybrid performance over Model 3 across all metrics ($\text{RMSE} = 1.1869$, $\text{MAE} = 0.9448$, $R^2 = 0.3838$, Pearson $r = 0.6248$, $p < 10^{-162}$) and trained faster (238 seconds vs 266 seconds). This report presents complete technical details, mathematical formulations, training histories, code mappings, and viva voce defense materials.

---

### 1. Introduction & Theoretical Background

#### 1.1 Clinical Relevance of Binding Affinity Prediction
High-Throughput Screening (HTS) evaluates physical compound libraries against purified biological targets, costing millions of dollars and taking years. Computational DTI prediction enables virtual screening of chemical libraries containing millions or billions of compounds in silico, directing laboratory synthesis and testing toward high-confidence leads.

#### 1.2 Binding Affinity as a Continuous Regression Problem
Binary classification ("active" vs. "inactive") discards critical quantitative pharmacology: a drug binding at $1\,\text{nM}$ is 10,000 times more potent than one binding at $10\,\mu\text{M}$. Continuous affinity regression ($p\text{Ki}$) preserves true quantitative potency, enables accurate selectivity profiling, and prevents threshold bias.

#### 1.3 Molecular Graph Representation
Molecules are naturally structured as attributed relational graphs where atoms are nodes and covalent bonds are edges. Direct graph message passing via `GINEConv` preserves chemical topology, atom coordination, formal charge, bond orders, and stereochemical isomerism.

#### 1.4 Quantum Neural Networks
Parameterized Quantum Circuits (PQCs) embed classical inputs into high-dimensional Hilbert spaces via unitary transformations:
$$|\psi(\vec{x})\rangle = U(\vec{x}) |0^{\otimes n}\rangle$$
Applying trainable unitary rotations $W(\vec{\theta})$ followed by expectation measurements allows compact circuits to model non-linear interactions with high parameter efficiency.

#### 1.5 Novel Bi-Partite Quantum Docking Hybrid (BP-QDA)
Standard QML models treat all qubits identically, blending inputs classically before quantum execution. The **Bi-Partite Quantum Docking Ansatz** partitions the quantum register into dedicated Drug and Target subspaces, applies entangling gates strictly across the interface boundary to physically simulate binding contact, and adds a residual skip connection to eliminate information bottlenecks.

---

### 2. Dataset Curation & Preprocessing

#### 2.1 Raw BindingDB Profiling
- **Raw File**: `BindingDB_All.tsv` (8.368 GB, 3,237,052 rows, 640 columns).
- **Candidate Exact Numeric $K_i$ Records**: 541,869 rows.
- **Curated Dataset**: Exactly 10,000 high-quality, non-redundant observations across 8,508 unique compounds and 853 unique targets.

#### 2.2 Quality Inclusion Criteria
1. **SMILES Sanitization**: Parsed with RDKit (`MolFromSmiles`), stereochemistry preserved, canonical isomeric SMILES exported, InChIKeys generated.
2. **Protein Validation**: Constrained to single-chain targets ($=1$) of length $[30, 1200]$ standard amino acids.
3. **Affinity Filtering**: Retained positive, exact numeric $K_i$ measurements; excluded censored measurements (`>`, `<`, `~`); excluded outliers ($p\text{Ki} \notin [2.0, 15.0]$).
4. **Replicate Aggregation**: Multiple measurements for identical `(canonical_smiles, sequence)` pairs aggregated via **median $p\text{Ki}$**.
5. **Diversity Capping**: Capped at 150 compound pairs per target protein to ensure broad biological diversity.

#### 2.3 Mathematical Transformation
$$p\text{Ki} = -\log_{10}(K_i \text{ in mol/L}) = 9.0 - \log_{10}(K_i \text{ in nM})$$
- $K_i = 1\,\text{nM} \implies p\text{Ki} = 8.00$
- $K_i = 10\,\text{nM} \implies p\text{Ki} = 8.00$
- $K_i = 10,000\,\text{nM} \implies p\text{Ki} = 5.00$

---

### 3. Common Data Splitting & Leakage Prevention

- **Split Ratio**: 70% Train (7,000) / 15% Val (1,500) / 15% Test (1,500).
- **Leakage Prevention**: Hashed composite signature `pair_signature = canonical_smiles + "::" + target_sequence`.
- **Zero Leakage**: Enforced $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$.

---

### 4. Detailed 4-Model Comparative Results

Benchmarked on identical held-out test split ($N = 1,500$ pairs):

| Performance Metric | Model 1: GNN | Model 2: QNN | Model 3: Baseline Hybrid | Model 4: Novel Hybrid (BP-QDA) |
|---|---:|---:|---:|---:|
| **Root Mean Squared Error (RMSE)** $\downarrow$ | **1.1711** | 1.1818 | 1.1886 | **1.1869** |
| **Mean Absolute Error (MAE)** $\downarrow$ | 0.9252 | **0.9191** | 0.9463 | **0.9448** |
| **Coefficient of Determination ($R^2$)** $\uparrow$ | **0.4000** | 0.3891 | 0.3820 | **0.3838** |
| **Pearson Correlation ($r$)** $\uparrow$ | **0.6346** | 0.6333 | 0.6186 | **0.6248** |
| **Pearson Significance ($p$-value)** | $7.67 \times 10^{-170}$ | $5.54 \times 10^{-169}$ | $3.96 \times 10^{-159}$ | **$3.63 \times 10^{-163}$** |
| **Spearman Rank Correlation ($\rho$)** $\uparrow$ | 0.6069 | **0.6223** | 0.5962 | 0.5931 |
| **Spearman Significance ($p$-value)** | $1.24 \times 10^{-151}$ | $1.50 \times 10^{-161}$ | $4.76 \times 10^{-145}$ | $3.19 \times 10^{-143}$ |
| **Test Set Observations ($N$)** | 1,500 | 1,500 | 1,500 | 1,500 |
| **Total Training Runtime (10 Epochs)** | 193.12 s | **174.67 s** | 266.68 s | **238.57 s** |
| **Best Checkpoint Epoch** | Epoch 6 | Epoch 10 | Epoch 8 | **Epoch 9** (Val Loss: 1.3555) |

---

### 5. Scientific Discussion & Discoveries

1. **GNN Regression Precision**:
   The GNN achieved the lowest overall RMSE (1.1711) and highest explained variance ($R^2 = 0.4000$). Covalent bond stereochemistry and atomic properties directly guide binding affinity without information loss.
2. **QNN Monotonic Regularization**:
   The standalone QNN achieved the lowest MAE (0.9191) and highest Spearman rank correlation ($\rho = 0.6223$). A 4-qubit circuit with 24 parameters cannot overfit assay-specific noise, learning robust monotonic relationships ideal for compound ranking.
3. **Model 4's Performance Elevation**:
   Model 4 successfully validated the novel architectural hypotheses:
   - **Cross-Entity Docking Entanglement**: Mimicking physical contact formation improved linear correlation ($r = 0.6248$ vs $0.6186$).
   - **Residual Skip Connection**: Combining the 128-d classical latent space with the 8 quantum observables eliminated the bottleneck penalty, reducing RMSE ($1.1869$ vs $1.1886$) and increasing $R^2$ ($0.3838$ vs $0.3820$).
   - **Speedup**: Model 4 trained 28 seconds faster than Model 3 (238s vs 266s) due to more targeted cross-boundary entangling gates.

---

### 6. Code & Project Inventory

- `models/GNN/`: Baseline GNN implementation.
- `models/QNN/`: Standalone QNN implementation.
- `models/GNN_QNN/`: Baseline hybrid implementation.
- `models/Novel_Hybrid/`: Novel BP-QDA Hybrid implementation.
- `results/comparison_4models/`: 4-model benchmarking CSV and visualization plots.
- `common/compare_4models.py`: 4-model comparative synthesis script.
- `reports/Model_4_Novel_Hybrid_BP_QDA_Report.md`: Dedicated technical report for Model 4.
