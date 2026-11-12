"""
Comprehensive Smoke Tests for Drug-Target Interaction Prediction Pipeline
Implements Phase 11 verification:
1. Small TSV chunk loading
2. SMILES parsing & canonicalization
3. Molecular graph generation (PyG Data)
4. Protein sequence encoding (Tokenization & 1D CNN)
5. Training batch creation
6. GNN forward & backward passes (finite loss)
7. QNN forward & backward passes (finite loss)
8. Hybrid GNN+QNN forward & backward passes (finite loss)
9. Checkpoint saving and reloading
10. Output prediction dimensions
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch_geometric.data import Batch

from common.config import RAW_TSV_PATH, CURATED_PARQUET
from common.molecular_features import validate_and_canonicalize_smiles, compute_molecular_feature_vector
from common.molecular_graph import smiles_to_graph
from common.protein_encoder import tokenize_sequence, Protein1DCNNEncoder
from common.affinity_processing import parse_and_convert_affinity_nm
from models.GNN.model import DTIGNNModel
from models.QNN.model import DTIQNNModel
from models.GNN_QNN.model import DTIGraphQuantumHybridModel


class TestSmokePipeline(unittest.TestCase):
    
    def test_01_tsv_chunk_readable(self):
        """Verify that BindingDB_All.tsv is present and first chunk is readable."""
        self.assertTrue(os.path.exists(RAW_TSV_PATH), f"TSV not found at {RAW_TSV_PATH}")
        with open(RAW_TSV_PATH, 'r', encoding='utf-8', errors='replace') as f:
            header = f.readline()
            self.assertIn("Ligand SMILES", header)
            self.assertIn("Ki (nM)", header)
            first_row = f.readline()
            self.assertGreater(len(first_row), 10)

    def test_02_smiles_validation_and_canonicalization(self):
        """Verify SMILES parsing, stereochemistry preservation and InChIKey."""
        test_smiles = "CC(=O)N[C@@H](Cc1ccccc1)C(=O)O"
        mol, can_smiles, inchi_key = validate_and_canonicalize_smiles(test_smiles)
        self.assertIsNotNone(mol)
        self.assertIsNotNone(can_smiles)
        self.assertIsNotNone(inchi_key)
        self.assertIn("@", can_smiles, "Stereochemical chiral center should be preserved")

    def test_03_molecular_graph_generation(self):
        """Verify molecular graph generation with atom and bond features."""
        test_smiles = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
        g = smiles_to_graph(test_smiles)
        self.assertIsNotNone(g)
        self.assertEqual(g.num_nodes, 13)
        self.assertEqual(g.x.shape[1], 41, "Atom feature dim should be 41")
        self.assertEqual(g.edge_attr.shape[1], 16, "Bond feature dim should be 16")

    def test_04_protein_encoding(self):
        """Verify protein tokenization and 1D CNN embedding."""
        seq = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR"
        tokens = tokenize_sequence(seq, max_length=1200)
        self.assertEqual(tokens.shape[0], 1200)
        
        encoder = Protein1DCNNEncoder(out_dim=64)
        embed = encoder(tokens.unsqueeze(0))
        self.assertEqual(embed.shape, (1, 64))
        self.assertTrue(torch.isfinite(embed).all())

    def test_05_affinity_conversion(self):
        """Verify Ki conversion from nM to molar and pKi."""
        val_nm, p_aff, status = parse_and_convert_affinity_nm(" 10.0")
        self.assertEqual(status, "OK")
        self.assertAlmostEqual(val_nm, 10.0, places=3)
        # 9 - log10(10) = 8.0
        self.assertAlmostEqual(p_aff, 8.0, places=3)
        
        # Test qualifiers
        _, _, q_status = parse_and_convert_affinity_nm(">1000")
        self.assertEqual(q_status, "EXCLUDED_QUALIFIER")

    def test_06_gnn_forward_backward(self):
        """Verify GNN forward, loss computation, and backprop."""
        model = DTIGNNModel()
        g1 = smiles_to_graph("c1ccccc1")
        g1.target_tokens = tokenize_sequence("ACDEFGHIKLMNPQRSTVWY")
        g1.y = torch.tensor([[7.5]])

        g2 = smiles_to_graph("CC(=O)O")
        g2.target_tokens = tokenize_sequence("WVTSRQPNMLKIHGFEDCA")
        g2.y = torch.tensor([[6.2]])

        batch = Batch.from_data_list([g1, g2])
        preds = model(batch)
        self.assertEqual(preds.shape, (2, 1))
        self.assertTrue(torch.isfinite(preds).all())

        criterion = nn.MSELoss()
        loss = criterion(preds, batch.y)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

        # Check gradients exist
        has_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        self.assertTrue(has_grad)

    def test_07_qnn_forward_backward(self):
        """Verify standalone QNN forward, loss, and backprop."""
        model = DTIQNNModel(n_qubits=4, circuit_depth=2)
        mol1, _, _ = validate_and_canonicalize_smiles("c1ccccc1")
        mol2, _, _ = validate_and_canonicalize_smiles("CC(=O)O")

        f1 = torch.tensor(compute_molecular_feature_vector(mol1)).unsqueeze(0)
        f2 = torch.tensor(compute_molecular_feature_vector(mol2)).unsqueeze(0)
        drug_feats = torch.cat([f1, f2], dim=0)

        t1 = tokenize_sequence("ACDEFGHIKLMNPQRSTVWY").unsqueeze(0)
        t2 = tokenize_sequence("WVTSRQPNMLKIHGFEDCA").unsqueeze(0)
        targets = torch.cat([t1, t2], dim=0)

        y = torch.tensor([[7.5], [6.2]])

        preds = model(drug_feats, targets)
        self.assertEqual(preds.shape, (2, 1))
        self.assertTrue(torch.isfinite(preds).all())

        criterion = nn.MSELoss()
        loss = criterion(preds, y)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

        has_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        self.assertTrue(has_grad)

    def test_08_hybrid_gnn_qnn_forward_backward(self):
        """Verify proposed Hybrid GNN+QNN forward, loss, and backprop."""
        model = DTIGraphQuantumHybridModel(n_qubits=4, circuit_depth=2)
        g1 = smiles_to_graph("c1ccccc1")
        g1.target_tokens = tokenize_sequence("ACDEFGHIKLMNPQRSTVWY")
        g1.y = torch.tensor([[7.5]])

        g2 = smiles_to_graph("CC(=O)O")
        g2.target_tokens = tokenize_sequence("WVTSRQPNMLKIHGFEDCA")
        g2.y = torch.tensor([[6.2]])

        batch = Batch.from_data_list([g1, g2])
        preds = model(batch)
        self.assertEqual(preds.shape, (2, 1))
        self.assertTrue(torch.isfinite(preds).all())

        criterion = nn.MSELoss()
        loss = criterion(preds, batch.y)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

        has_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        self.assertTrue(has_grad)

    def test_09_checkpoint_save_and_load(self):
        """Verify saving and loading model checkpoints."""
        model = DTIGNNModel()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            torch.save({'model_state_dict': model.state_dict(), 'epoch': 1}, tmp_path)
            loaded = torch.load(tmp_path, weights_only=False)
            self.assertEqual(loaded['epoch'], 1)
            model2 = DTIGNNModel()
            model2.load_state_dict(loaded['model_state_dict'])
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == '__main__':
    unittest.main(verbosity=2)

