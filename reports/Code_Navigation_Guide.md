# Code Navigation Guide & Implementation Directory
## Drug–Target Interaction Prediction: 4-Model Comparative Suite

---

### 1. Rapid Demonstration Guide: "If Mam asks me to show the code..."

| Question / Topic from Guide | Primary File to Open | Primary Function or Class | What to Point Out and Explain |
|---|---|---|---|
| **"Where does the project begin?"** | `run_pipeline.py` or `run_all.py` | `main()` / `run_entire_pipeline()` | Show the modular CLI supporting stage execution (`--stage train-gnn`, `train-novel-hybrid`, `compare-4models`, etc.). |
| **"Where do you read the 8 GB BindingDB TSV?"** | `common/preprocess.py` | `run_curation_pipeline()` (lines 45–205) | Line-by-line streaming without memory exhaustion, header index mapping, and `pair_dict` aggregation. |
| **"How are SMILES converted into graphs?"** | `common/molecular_graph.py` | `smiles_to_graph()` & `mol_to_graph_data()` | Atom features (41 dims) and bond features (16 dims) converted to PyG `Data(x, edge_index, edge_attr)`. |
| **"Where are protein sequences encoded?"** | `common/protein_encoder.py` | `tokenize_sequence()`, `Protein1DCNNEncoder` | Vocabulary 22, padding to 1200, 1D CNN with kernel 7, adaptive max pooling, and LayerNorm. |
| **"Where is the GNN model implemented?"** | `models/GNN/model.py` | `GNNEncoder`, `DTIGNNModel` | 3 `GINEConv` layers with bond attributes, residual skip connections (`h = h + h_new`), and mean+max dual pooling. |
| **"Where is the QNN circuit implemented?"** | `models/QNN/quantum_circuit.py` | `ParameterizedQuantumCircuit` | PennyLane QNode `@qml.qnode`, `AngleEmbedding`, `StronglyEntanglingLayers` (24 weights), and Pauli-Z measurements. |
| **"Where is the baseline Hybrid model?"** | `models/GNN_QNN/model.py` | `DTIGraphQuantumHybridModel` | Coupling `GNNEncoder` with quantum circuit via `reduction_mlp` (128 $\to$ 4) and end-to-end backprop. |
| **"Where is the novel 4th model implemented?"** | `models/Novel_Hybrid/model.py` | `DTIBiPartiteQuantumHybridModel` | Subsystem angle projections, Bi-Partite Quantum Docking, and the Classical-Quantum Residual Skip Connection (136-d). |
| **"Where is the novel quantum docking circuit?"**| `models/Novel_Hybrid/quantum_circuit.py`| `BiPartiteQuantumDockingCircuit` | Drug (wires 0,1), Target (wires 2,3), cross-entity CNOT docking gates, $R_x$ re-uploading, and cross-correlator $\langle Z_1 \otimes Z_2 \rangle$. |
| **"How do you ensure zero data leakage?"** | `common/data_splitting.py` | `create_and_save_splits()` | Unique pair hash signature creation (`smiles + "::" + seq`) and assertion checks guaranteeing 0 split overlap. |
| **"Where is model training implemented?"** | `models/Novel_Hybrid/train.py` | `train_novel_hybrid()` | AdamW optimizer, MSELoss, gradient clipping (`max_norm=5.0`), and checkpoint tracking. |
| **"Where are 4-model comparison results saved?"** | `common/compare_4models.py` | `run_4model_comparison()` | Benchmarking output saved to `results/comparison_4models/model_comparison_4models.csv` and comparative plots. |

---

### 2. Comprehensive File Inventory & Directory Structure

#### 2.1 Model 4: Novel Bi-Partite Quantum Docking Hybrid (`models/Novel_Hybrid/`)
- `models/Novel_Hybrid/config.yaml`: Model and training hyperparameters for Model 4.
- `models/Novel_Hybrid/quantum_circuit.py`: `BiPartiteQuantumDockingCircuit` class implementing domain-specific angle embedding on dedicated drug (0,1) and target (2,3) qubits, intra-molecular unitaries, cross-boundary quantum docking gates (1 $\leftrightarrow$ 2), data re-uploading, and 8 observables.
- `models/Novel_Hybrid/model.py`: `DTIBiPartiteQuantumHybridModel` class coupling GNN graph representations with the docking circuit and a 136-d residual skip connection into the readout head.
- `models/Novel_Hybrid/train.py`: Dedicated training loop saving checkpoints to `checkpoints/Novel_Hybrid/best_novel_hybrid_model.pt` and logging to `logs/novel_hybrid_train.log`.
- `models/Novel_Hybrid/evaluate.py`: Inference on the 1,500 test samples saving `results/Novel_Hybrid/metrics.json` and `test_predictions.csv`.

#### 2.2 Preserved Core & Model Directories (100% Intact)
- `common/`: `config.py`, `affinity_processing.py`, `molecular_features.py`, `molecular_graph.py`, `protein_encoder.py`, `preprocess.py`, `data_splitting.py`, `dataset_loader.py`, `evaluation.py`, `compare_models.py`, `reproducibility.py`.
- `models/GNN/`: Baseline GNN architecture, training loop, and evaluation.
- `models/QNN/`: Standalone QNN architecture, quantum circuit, training loop, and evaluation.
- `models/GNN_QNN/`: Baseline hybrid GNN+QNN architecture, quantum circuit, training loop, and evaluation.
- `results/comparison/`: Pristine original 3-model comparison outputs (`model_comparison.csv`, `.png`).
- `results/comparison_4models/`: Comprehensive 4-model comparison outputs (`model_comparison_4models.csv`, `.png`, `prediction_comparison_4models.png`, `residual_distribution_4models.png`).
