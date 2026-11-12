# Viva Preparation & Comprehensive Technical Defense
## 4-Model Comparative Suite: GNN, QNN, Baseline Hybrid & Novel BP-QDA Hybrid

---

### Section A: 5-Minute Verbal Project Explanation Script

*"Good morning respected guide and evaluators,*

*Today I present our project: **Graph Neural Network-Based and Quantum Neural Network-Based Drug–Target Interaction Prediction**.*

***1. Problem Statement & Motivation:***
*Identifying whether a small molecule binds to a therapeutic protein target and predicting its quantitative binding affinity is one of the most critical challenges in early drug discovery. Instead of framing DTI as a simplified binary classification problem, our work formulates DTI strictly as a **continuous binding affinity regression task**, predicting the thermodynamic inhibition constant $p\text{Ki} = -\log_{10}(\text{Ki in M})$.*

***2. Dataset & Quality Filtering:***
*We selected **BindingDB**, a premier open-access database containing over 3.2 million experimental binding records. We profiled the 8.37 GB dataset and curated a benchmark of **10,000 high-quality observations** across 8,508 unique compounds and 853 diverse protein targets, standardizing units, aggregating replicate assays with median $p\text{Ki}$, and enforcing zero pair leakage across a reproducible 70/15/15 partition ($N = 1,500$ held-out test samples).*

***3. Evaluated Paradigms (Four Architectures):***
*To systematically assess computational intelligence paradigms, we implemented and benchmarked four distinct neural architectures:*
- *First, a **Graph Neural Network (GNN)** using 3 `GINEConv` message-passing layers with 41 atom and 16 bond features paired with a 1D CNN protein encoder.*
- *Second, a **Quantum Neural Network (QNN)** using ECFP4 fingerprints and 8 physical descriptors mapped to a 4-qubit Variational Quantum Circuit in PennyLane with 24 parameters.*
- *Third, a **Baseline Hybrid (GNN + QNN)** architecture, coupling the GNN encoder with the quantum circuit via end-to-end backpropagation.*
- *Fourth, our **Proposed Novel Hybrid: Bi-Partite Quantum Docking Hybrid (BP-QDA)**. To eliminate the 32-fold information bottleneck of Model 3, we partitioned the 4 qubits into dedicated Drug and Target subspaces, applied parameterized entangling gates strictly across the contact boundary (Q1 $\leftrightarrow$ Q2), introduced Quantum Data Re-Uploading ($R_x$), measured cross-entity correlators $\langle Z_1 \otimes Z_2 \rangle$, and incorporated a Classical-Quantum Residual Skip Connection preserving full 128-d representation capacity.*

***4. Experimental Results & Discoveries:***
*On the held-out test set ($N=1,500$):*
- *The classical **GNN achieved the lowest overall error** ($\text{RMSE} = 1.1711$, $R^2 = 0.4000$, Pearson $r = 0.6346$).*
- *The **standalone QNN demonstrated remarkable parameter efficiency**, achieving the lowest MAE ($0.9191$) and highest rank correlation ($\rho = 0.6223$) with only 24 quantum parameters.*
- *Our **Novel Hybrid BP-QDA (Model 4) successfully elevated hybrid performance**, improving over Model 3 across all metrics ($\text{RMSE} = 1.1869$, $\text{MAE} = 0.9448$, $R^2 = 0.3838$, and Pearson $r = 0.6248$, $p < 10^{-162}$), while training faster (238 seconds vs 266 seconds).*

*This project confirms that while quantum advantage over classical GNNs remains an ongoing frontier, physics-informed bi-partite quantum architectures with residual skip connections effectively mitigate quantum bottlenecking and establish a strong foundation for future NISQ drug discovery pipelines. I am now ready for your questions and code demonstration. Thank you."*

---

### Section B: 10-Step Code Demonstration Sequence

1. Open `common/preprocess.py` $\to$ Show streaming TSV reading, qualifier rejection, and median replicate aggregation (`#L45-L205`).
2. Open `common/molecular_graph.py` $\to$ Show 41 atom features and 16 bond features converted to PyG `Data` (`#L59-L130`).
3. Open `common/protein_encoder.py` $\to$ Show vocabulary mapping, padding to 1200, and `Protein1DCNNEncoder` (`#L21-L102`).
4. Open `models/GNN/model.py` $\to$ Show 3 `GINEConv` layers, residual skip connections, and mean+max dual pooling (`#L23-L148`).
5. Open `models/QNN/quantum_circuit.py` $\to$ Show PennyLane `@qml.qnode`, `AngleEmbedding`, and `StronglyEntanglingLayers` (`#L20-L62`).
6. Open `models/GNN_QNN/model.py` $\to$ Show baseline hybrid coupling with `reduction_mlp` (`#L26-L119`).
7. Open `models/Novel_Hybrid/quantum_circuit.py` $\to$ Show **Novel BP-QDA Circuit**: Drug (wires 0,1), Target (wires 2,3), cross-entity docking gates (1 $\leftrightarrow$ 2), data re-uploading, and cross-correlator $\langle Z_1 \otimes Z_2 \rangle$.
8. Open `models/Novel_Hybrid/model.py` $\to$ Show **Quantum Residual Skip Connection** (136-d combined manifold: 128-d classical + 8-d quantum).
9. Open `common/data_splitting.py` $\to$ Show `pair_signature` hashing and assertion checks verifying 0 split overlap (`#L48-L78`).
10. Open `results/comparison_4models/model_comparison_4models.csv` and `model_comparison_4models.png` $\to$ Show 4-model performance comparison.

---

### Section C: Key Technical Viva Questions & Answers

**Q: Why was Model 4 (BP-QDA) developed?**
- **A**: Model 3 suffered from an information bottleneck: compressing a 128-d joint embedding into 4 rotation angles caused a minor performance drop. Model 4 was developed to overcome this by introducing domain-specific qubit partitioning, cross-entity docking gates, and a classical-quantum residual skip connection that preserves full 128-d representation capacity.

**Q: What is the physical significance of the cross-entity correlator $\langle \sigma_z^{(1)} \otimes \sigma_z^{(2)} \rangle$?**
- **A**: In quantum information, $\langle Z_1 \otimes Z_2 \rangle$ measures the joint correlation and entanglement across the interface between the drug and protein qubits. If the drug and target do not interact, $\langle Z_1 \otimes Z_2 \rangle = \langle Z_1 \rangle \langle Z_2 \rangle$. When binding occurs, cross-boundary entanglement alters this expectation value, directly reflecting the biophysical binding free energy.

**Q: Did Model 4 improve over Model 3?**
- **A**: **Yes, across all metrics.** Model 4 achieved lower RMSE ($1.1869$ vs $1.1886$), lower MAE ($0.9448$ vs $0.9463$), higher $R^2$ ($0.3838$ vs $0.3820$), and significantly higher Pearson correlation ($0.6248$ vs $0.6186$), while training faster ($238\,\text{s}$ vs $266\,\text{s}$).
