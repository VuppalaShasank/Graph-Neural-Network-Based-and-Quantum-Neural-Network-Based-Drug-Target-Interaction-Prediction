# System Architecture & Technical Specification
## Graph Neural Network-Based and Quantum Neural Network-Based Drug–Target Interaction Prediction

---

### Executive Architectural Summary
This document delineates the technical architecture, mathematical modeling, tensor flow pathways, and operational pipelines implemented in this investigation. The system predicts continuous drug–target binding affinity ($p\text{Ki}$) from 2D molecular structures and 1D protein amino acid sequences across **four comparative computational intelligence paradigms**:
1. **Model 1: Graph Neural Network (GNN)** — Edge-Attentive Graph Isomorphism Network (`GINEConv`) operating on 41 atom and 16 bond features paired with a 1D Convolutional Neural Network (CNN) target sequence encoder.
2. **Model 2: Quantum Neural Network (QNN)** — Extended connectivity circular fingerprints (`ECFP4`), 8 biophysical physicochemical descriptors, shared 1D CNN target sequence encoder, a 4-qubit Parameterized Quantum Circuit (PQC), and classical regression readout.
3. **Model 3: Baseline Hybrid (GNN + QNN)** — Couples the bond-aware GINE graph encoder with the 1D CNN protein encoder, compresses the joint 128-d manifold to 4 quantum rotation angles, executes a 4-qubit variational circuit with all-to-all strong entanglement, and projects Pauli-Z expectation values to binding affinity predictions via end-to-end backpropagation.
4. **Model 4: Novel Bi-Partite Quantum Docking Hybrid (BP-QDA)** — Novel physics-informed architecture featuring independent 2-qubit registers for Drug ($\mathcal{H}_{\text{drug}}$) and Protein Target ($\mathcal{H}_{\text{target}}$), parameterized inter-entity entangling unitaries across the contact interface boundary (Qubit 1 $\leftrightarrow$ Qubit 2), Quantum Data Re-Uploading ($R_x$), 8 multi-basis and cross-correlator observables ($\langle Z_1 \otimes Z_2 \rangle$, $\langle X_1 \otimes X_2 \rangle$), and a **Quantum Residual Skip Connection** preserving full 128-d classical latent capacity.

---

### 1. End-to-End System Workflow

```mermaid
flowchart TD
    subgraph Raw_Input["1. Raw Data Ingestion"]
        A["BindingDB_All.tsv<br/>(8.37 GB, 3,237,052 Records, 640 Columns)"]
    end

    subgraph Preprocessing_Block["2. Inspection, Curation & Partitioning"]
        B["Chunk-Based Streaming Profiler<br/>(common/dataset_inspection.py)"]
        C["Streaming Curation Pipeline<br/>(common/preprocess.py)<br/>• RDKit SMILES Sanitization<br/>• Target Sequence [30, 1200] AA<br/>• Exact Ki > 0 Extraction<br/>• Replicate Median Aggregation<br/>• Target Capping (Max 150/Target)"]
        D["Curated Benchmark<br/>(10,000 Observations)<br/>data/processed/BindingDB_curated_10000.parquet"]
        E["Pair-Group-Aware Data Splitter<br/>(common/data_splitting.py)<br/>Seed = 42 | Zero Pair Leakage"]
        F1["Train Split (70%, N=7,000)"]
        F2["Validation Split (15%, N=1,500)"]
        F3["Test Split (15%, N=1,500)"]
        A --> B --> C --> D --> E
        E --> F1 & F2 & F3
    end

    subgraph Feature_Pipelines["3. Feature Representation & Caching"]
        F1 & F2 & F3 --> G1["Molecular Graph Construction<br/>(common/molecular_graph.py)<br/>Atom Feats: 41 | Bond Feats: 16<br/>Cache: data/graph_cache/"]
        F1 & F2 & F3 --> G2["Fingerprints & Descriptors<br/>(common/molecular_features.py)<br/>ECFP4 (256b) + 8 Descriptors = 264-d<br/>Cache: data/feature_cache/"]
        F1 & F2 & F3 --> G3["Protein Sequence Tokenization<br/>(common/protein_encoder.py)<br/>Max Len: 1200 | Vocab: 22"]
    end

    subgraph Four_Model_Branches["4. Model Training & Evaluation Branches"]
        subgraph Branch_GNN["Branch 1: GNN (Bond-Aware GINE)"]
            H1["DTIGNNModel Training (models/GNN/train.py)"]
            H2["best_gnn_model.pt (Epoch 6)"]
            H3["GNN Evaluation (models/GNN/evaluate.py)"]
            H1 --> H2 --> H3
        end

        subgraph Branch_QNN["Branch 2: QNN (Variational Circuit)"]
            I1["DTIQNNModel Training (models/QNN/train.py)"]
            I2["best_qnn_model.pt (Epoch 10)"]
            I3["QNN Evaluation (models/QNN/evaluate.py)"]
            I1 --> I2 --> I3
        end

        subgraph Branch_Hybrid["Branch 3: Baseline Hybrid (GNN + QNN)"]
            J1["DTIGraphQuantumHybridModel (models/GNN_QNN/train.py)"]
            J2["best_gnn_qnn_model.pt (Epoch 8)"]
            J3["Hybrid Evaluation (models/GNN_QNN/evaluate.py)"]
            J1 --> J2 --> J3
        end

        subgraph Branch_Novel["Branch 4: Novel BP-QDA Hybrid"]
            K1["DTIBiPartiteQuantumHybridModel (models/Novel_Hybrid/train.py)"]
            K2["best_novel_hybrid_model.pt (Epoch 9)"]
            K3["Novel Evaluation (models/Novel_Hybrid/evaluate.py)"]
            K1 --> K2 --> K3
        end
    end

    G1 & G3 --> Branch_GNN
    G2 & G3 --> Branch_QNN
    G1 & G3 --> Branch_Hybrid
    G1 & G3 --> Branch_Novel

    subgraph Benchmarking["5. Comparative Benchmarking & Synthesis"]
        L["Comparative Benchmarking Pipelines<br/>(common/compare_models.py & common/compare_4models.py)"]
        H3 & I3 & J3 & K3 --> L
        L --> M1["results/comparison_4models/model_comparison_4models.csv"]
        L --> M2["results/comparison_4models/model_comparison_4models.png"]
        L --> M3["results/comparison_4models/prediction_comparison_4models.png"]
        L --> M4["results/comparison_4models/residual_distribution_4models.png"]
    end
```

---

### 2. Four Detailed Model Architectures

#### 2.1 Model 1: Bond-Aware Graph Neural Network (GNN)
```mermaid
flowchart LR
    SMILES["SMILES"] --> RDKit_G["PyG Graph Builder"]
    RDKit_G --> NodeX["Atom Features (41-d)"]
    RDKit_G --> EdgeAttr["Bond Features (16-d)"]
    NodeX --> ProjA["Linear(41 -> 64)"]
    ProjA --> GINE1["GINEConv 1 + BN + ReLU + Res"]
    GINE1 --> GINE2["GINEConv 2 + BN + ReLU + Res"]
    GINE2 --> GINE3["GINEConv 3 + BN + ReLU + Res"]
    EdgeAttr --> GINE1 & GINE2 & GINE3
    GINE3 --> MeanPool["Mean Pool (64)"]
    GINE3 --> MaxPool["Max Pool (64)"]
    MeanPool & MaxPool --> CatPool["Concat Pool (128)"]
    CatPool --> ReadoutG["Linear(128 -> 64) + LN + ReLU"] --> DrugEmb["Drug Embedding (64-d)"]

    FASTA["Protein Sequence"] --> Tok["Tokenize (1200)"] --> Emb["Embedding(22, 64)"]
    Emb --> Conv1["Conv1d(64, 64, k=7) + ReLU"] --> Conv2["Conv1d(64, 64, k=7) + ReLU"]
    Conv2 --> AdPool["AdaptiveMaxPool1d(1)"] --> ProtFC["Linear(64 -> 64) + LN"] --> ProtEmb["Protein Embedding (64-d)"]

    DrugEmb & ProtEmb --> Fused["Concat Fusion (128-d)"]
    Fused --> MLPHead["Linear(128 -> 64 -> 32 -> 1)<br/>Bias Init = 7.32"] --> Pred["Predicted pKi"]
```

#### 2.2 Model 2: Standalone Variational QNN
```mermaid
flowchart LR
    SMILES2["SMILES"] --> ECFP["ECFP4 (256b) + 8 Descriptors (264-d)"] --> DrugMLP["Linear(264 -> 64) + LN + ReLU"] --> QDrug["Drug Emb (64-d)"]
    FASTA2["Protein Sequence"] --> SharedProt["Shared Protein 1D CNN Encoder"] --> QProt["Protein Emb (64-d)"]
    QDrug & QProt --> QJoint["Concat (128-d)"] --> Red["Linear(128 -> 32 -> 4) + Tanh * π"] --> Angles["4 Angles"]
    Angles --> QCirc["PennyLane default.qubit (4 wires)<br/>AngleEmbedding(Ry) + 2x StronglyEntanglingLayers (24 weights)"]
    QCirc --> Meas["Pauli-Z Expectations (4-d)"] --> Read["Linear(4 -> 16 -> 1)<br/>Bias Init = 7.32"] --> QPred["Predicted pKi"]
```

#### 2.3 Model 3: Baseline Hybrid (GNN + QNN)
```mermaid
flowchart LR
    GNN_Enc["GNNEncoder (Bond-Aware GINE)"] --> HDrug["Drug Emb (64-d)"]
    CNN_Enc["Protein1DCNNEncoder"] --> HProt["Protein Emb (64-d)"]
    HDrug & HProt --> HJoint["Concat Fusion (128-d)"]
    HJoint --> HRed["Linear(128 -> 32 -> 4) + Tanh * π"] --> HAngles["4 Angles"]
    HAngles --> HCirc["PennyLane default.qubit (4 wires)<br/>AngleEmbedding(Ry) + StronglyEntanglingLayers"]
    HCirc --> HMeas["Pauli-Z Expectations (4-d)"] --> HRead["Linear(4 -> 16 -> 1)<br/>Bias Init = 7.32"] --> HPred["Predicted pKi"]
```

#### 2.4 Model 4: Novel Bi-Partite Quantum Docking Hybrid (BP-QDA)
```mermaid
flowchart LR
    GNN4["GNNEncoder (GINE)"] --> D_Emb["Drug Embedding (64-d)"]
    CNN4["Protein1DCNNEncoder"] --> P_Emb["Protein Embedding (64-d)"]

    D_Emb --> ProjD["Linear(64 -> 32 -> 2) * π"] --> ThetaD["θ_drug (2 angles)"]
    P_Emb --> ProjP["Linear(64 -> 32 -> 2) * π"] --> ThetaP["θ_prot (2 angles)"]

    ThetaD --> W01["Wires 0, 1 (Drug Subspace)"]
    ThetaP --> W23["Wires 2, 3 (Target Subspace)"]

    W01 & W23 --> Layer1["Layer 1: AngleEmbedding(Ry)<br/>+ Intra-Drug CNOT(0->1)<br/>+ Intra-Target CNOT(2->3)<br/>+ Inter-Entity Docking CNOT(1->2)"]
    Layer1 --> Layer2["Layer 2: Data Re-Uploading (Rx)<br/>+ Intra-Entity Rotations<br/>+ Inter-Entity Docking CNOT(1->2)"]
    Layer2 --> Observables["8 Observables:<br/>• Single Pauli-Z: ⟨Z_0⟩..⟨Z_3⟩<br/>• Transverse: ⟨X_1⟩, ⟨X_2⟩<br/>• Cross-Correlator: ⟨Z_1 ⊗ Z_2⟩<br/>• Transverse Correlator: ⟨X_1 ⊗ X_2⟩"]

    D_Emb & P_Emb --> ClassicalSkip["Classical Skip Connection (128-d)"]
    Observables --> QFeats["Quantum Observables (8-d)"]
    ClassicalSkip & QFeats --> Combined["Combined Manifold (136-d)"]
    Combined --> Readout4["Linear(136 -> 64 -> 32 -> 1)<br/>Bias Init = 7.32"] --> Pred4["Predicted pKi"]
```

---

### 3. Comprehensive 4-Model Architectural Comparison Matrix

| Component / Layer | Model 1: GNN | Model 2: QNN | Model 3: Baseline Hybrid | Model 4: Novel Hybrid (BP-QDA) |
|---|---|---|---|---|
| **Drug Representation** | Molecular Graph (PyG) | ECFP4 (256b) + 8 Descriptors | Molecular Graph (PyG) | Molecular Graph (PyG) |
| **Drug Features** | 41 atom dims, 16 bond dims | 264 scalar dimensions | 41 atom dims, 16 bond dims | 41 atom dims, 16 bond dims |
| **Drug Encoder** | 3-layer `GINEConv` + Pooling | Linear(264, 64) + LayerNorm | 3-layer `GINEConv` + Pooling | 3-layer `GINEConv` + Pooling |
| **Protein Representation** | 1200 Token Integer Sequence | 1200 Token Integer Sequence | 1200 Token Integer Sequence | 1200 Token Integer Sequence |
| **Protein Encoder** | **Identical 1D CNN Encoder** | **Identical 1D CNN Encoder** | **Identical 1D CNN Encoder** | **Identical 1D CNN Encoder** |
| **Subspace Partitioning** | N/A | None (Classical Blending) | None (Classical Blending) | **Dedicated Drug (Q0,1) & Target (Q2,3)** |
| **Quantum Ansatz** | N/A | Generic Strongly Entangling | Generic Strongly Entangling | **Bi-Partite Docking Gates (Q1 $\leftrightarrow$ Q2)** |
| **Data Re-Uploading** | N/A | No (Single Pass) | No (Single Pass) | **Yes (Layer 2 Rx Re-Uploading)** |
| **Quantum Observables** | N/A | 4 Pauli-Z Expectations | 4 Pauli-Z Expectations | **8 Observables (including $\langle Z_1 \otimes Z_2 \rangle$)** |
| **Residual Skip Connection**| N/A | No (Bottlenecked) | No (Bottlenecked) | **Yes (128-d Classical + 8-d Quantum = 136-d)** |
| **Readout Head** | Linear(128 $\to$ 64 $\to$ 32 $\to$ 1) | Linear(4 $\to$ 16 $\to$ 1) | Linear(4 $\to$ 16 $\to$ 1) | Linear(136 $\to$ 64 $\to$ 32 $\to$ 1) |
| **Output Bias Calibration**| Constant 7.32 (Train Mean) | Constant 7.32 (Train Mean) | Constant 7.32 (Train Mean) | Constant 7.32 (Train Mean) |
