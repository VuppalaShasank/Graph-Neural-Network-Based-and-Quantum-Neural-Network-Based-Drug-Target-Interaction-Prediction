"""
Dataset and DataLoader Utilities for GNN, QNN, and Hybrid GNN+QNN
Handles graph caching, molecular feature caching, protein tokenization, and memory-efficient batching.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as PyGDataLoader
import pandas as pd
import numpy as np

from common.config import (
    GRAPH_CACHE_DIR, FEATURE_CACHE_DIR,
    MAX_SEQ_LEN, DEFAULT_BATCH_SIZE
)
from common.molecular_graph import smiles_to_graph
from common.molecular_features import validate_and_canonicalize_smiles, compute_molecular_feature_vector
from common.protein_encoder import tokenize_sequence


class DTIGraphDataset(Dataset):
    """
    Dataset for GNN and Hybrid GNN+QNN.
    Returns:
        - graph: PyG Data object (x, edge_index, edge_attr)
        - target_tokens: LongTensor of shape (MAX_SEQ_LEN,)
        - pki: FloatTensor of shape (1,)
        - pair_id: str
    Uses disk and in-memory caching for molecular graphs.
    """
    def __init__(self, df: pd.DataFrame, cache_dir: Path = GRAPH_CACHE_DIR, pre_cache: bool = True):
        self.df = df.reset_index(drop=True)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache = {}
        
        # Pre-tokenize all protein sequences in memory (fast and lightweight: 10k x 1200 x 8 bytes ~ 96MB)
        self.tokenized_targets = []
        for seq in self.df['target_sequence']:
            self.tokenized_targets.append(tokenize_sequence(seq, max_length=MAX_SEQ_LEN))
            
        self.pki_values = torch.tensor(self.df['pKi'].values, dtype=torch.float32).unsqueeze(1)
        self.pair_ids = self.df['pair_id'].tolist()
        self.smiles_list = self.df['canonical_smiles'].tolist()
        
        if pre_cache:
            self._warm_cache()

    def _warm_cache(self):
        """Pre-computes and caches molecular graphs for all unique SMILES."""
        unique_smiles = set(self.smiles_list)
        for sm in unique_smiles:
            cache_file = self.cache_dir / f"{abs(hash(sm))}.pt"
            if cache_file.exists():
                try:
                    g = torch.load(cache_file, weights_only=False)
                    self.memory_cache[sm] = g
                    continue
                except Exception:
                    pass
            g = smiles_to_graph(sm)
            if g is not None:
                self.memory_cache[sm] = g
                try:
                    torch.save(g, cache_file)
                except Exception:
                    pass

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        smiles = self.smiles_list[idx]
        graph = self.memory_cache.get(smiles)
        if graph is None:
            cache_file = self.cache_dir / f"{abs(hash(smiles))}.pt"
            if cache_file.exists():
                graph = torch.load(cache_file, weights_only=False)
            else:
                graph = smiles_to_graph(smiles)
                if graph is not None:
                    torch.save(graph, cache_file)
            self.memory_cache[smiles] = graph

        # Clone graph to avoid in-place mutation during batching
        g_data = graph.clone()
        g_data.target_tokens = self.tokenized_targets[idx]
        g_data.y = self.pki_values[idx]
        g_data.pair_id = self.pair_ids[idx]
        return g_data


class DTIQNNFeatureDataset(Dataset):
    """
    Dataset for standalone QNN.
    Returns:
        - drug_features: FloatTensor of shape (264,) (ECFP4 256 bits + 8 RDKit descriptors)
        - target_tokens: LongTensor of shape (MAX_SEQ_LEN,)
        - pki: FloatTensor of shape (1,)
        - pair_id: str
    """
    def __init__(self, df: pd.DataFrame, cache_dir: Path = FEATURE_CACHE_DIR, pre_cache: bool = True):
        self.df = df.reset_index(drop=True)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache = {}
        
        # Tokenize proteins
        self.tokenized_targets = []
        for seq in self.df['target_sequence']:
            self.tokenized_targets.append(tokenize_sequence(seq, max_length=MAX_SEQ_LEN))
            
        self.pki_values = torch.tensor(self.df['pKi'].values, dtype=torch.float32).unsqueeze(1)
        self.pair_ids = self.df['pair_id'].tolist()
        self.smiles_list = self.df['canonical_smiles'].tolist()
        
        if pre_cache:
            self._warm_cache()

    def _warm_cache(self):
        """Pre-computes and caches molecular feature vectors for all unique SMILES."""
        unique_smiles = set(self.smiles_list)
        for sm in unique_smiles:
            cache_file = self.cache_dir / f"{abs(hash(sm))}_feats.pt"
            if cache_file.exists():
                try:
                    feat = torch.load(cache_file, weights_only=False)
                    self.memory_cache[sm] = feat
                    continue
                except Exception:
                    pass
            mol, can_smiles, _ = validate_and_canonicalize_smiles(sm)
            if mol is not None:
                feat_np = compute_molecular_feature_vector(mol)
                feat = torch.tensor(feat_np, dtype=torch.float32)
                self.memory_cache[sm] = feat
                try:
                    torch.save(feat, cache_file)
                except Exception:
                    pass

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        smiles = self.smiles_list[idx]
        feat = self.memory_cache.get(smiles)
        if feat is None:
            cache_file = self.cache_dir / f"{abs(hash(smiles))}_feats.pt"
            if cache_file.exists():
                feat = torch.load(cache_file, weights_only=False)
            else:
                mol, _, _ = validate_and_canonicalize_smiles(smiles)
                feat_np = compute_molecular_feature_vector(mol)
                feat = torch.tensor(feat_np, dtype=torch.float32)
                torch.save(feat, cache_file)
            self.memory_cache[smiles] = feat

        return feat, self.tokenized_targets[idx], self.pki_values[idx], self.pair_ids[idx]


def get_graph_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = 0
) -> Tuple[PyGDataLoader, PyGDataLoader, PyGDataLoader]:
    """Builds PyG DataLoaders for GNN and Hybrid models."""
    train_ds = DTIGraphDataset(train_df)
    val_ds = DTIGraphDataset(val_df)
    test_ds = DTIGraphDataset(test_df)
    
    train_loader = PyGDataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = PyGDataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = PyGDataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader


def get_feature_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Builds standard PyTorch DataLoaders for standalone QNN."""
    train_ds = DTIQNNFeatureDataset(train_df)
    val_ds = DTIQNNFeatureDataset(val_df)
    test_ds = DTIQNNFeatureDataset(test_df)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader

