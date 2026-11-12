"""
Protein Sequence Encoder
Lightweight, reproducible 1D Convolutional Neural Network with trainable amino acid embeddings.
Used identically across all three prediction architectures (GNN, QNN, GNN+QNN).
"""

from typing import List, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.config import (
    AMINO_ACIDS, PAD_TOKEN, UNK_TOKEN, VOCAB_SIZE,
    PROT_EMBED_DIM, PROT_NUM_FILTERS, PROT_KERNEL_SIZE, PROT_OUT_DIM, MAX_SEQ_LEN
)

# Amino acid vocabulary mapping
AA_TO_ID: Dict[str, int] = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}


def tokenize_sequence(seq: str, max_length: int = MAX_SEQ_LEN) -> torch.Tensor:
    """
    Tokenizes an amino acid sequence into integer tensor.
    Pads or truncates to max_length.
    Canonical amino acids -> 1..20
    Non-standard / ambiguous -> UNK_TOKEN (21)
    Padding -> PAD_TOKEN (0)
    """
    clean_seq = seq.strip().upper()
    tokens = []
    for aa in clean_seq[:max_length]:
        tokens.append(AA_TO_ID.get(aa, UNK_TOKEN))
        
    # Pad to max_length
    if len(tokens) < max_length:
        tokens.extend([PAD_TOKEN] * (max_length - len(tokens)))
        
    return torch.tensor(tokens, dtype=torch.long)


class Protein1DCNNEncoder(nn.Module):
    """
    Lightweight 1D CNN Protein Sequence Encoder.
    Architecture:
        Embedding(vocab_size=22, embed_dim=64, padding_idx=0)
        Conv1D(in_channels=64, out_channels=64, kernel_size=7, padding=3) + ReLU
        Conv1D(in_channels=64, out_channels=64, kernel_size=7, padding=3) + ReLU
        AdaptiveMaxPool1D(1)
        Linear(64 -> out_dim) + LayerNorm
    """
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim: int = PROT_EMBED_DIM,
        num_filters: int = PROT_NUM_FILTERS,
        kernel_size: int = PROT_KERNEL_SIZE,
        out_dim: int = PROT_OUT_DIM,
        dropout: float = 0.2
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_TOKEN)
        
        self.conv1 = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.conv2 = nn.Conv1d(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.fc = nn.Linear(num_filters, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_seqs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_seqs: (batch_size, seq_len)
        Returns:
            prot_embeddings: (batch_size, out_dim)
        """
        # Shape: (batch_size, seq_len, embed_dim)
        x = self.embedding(token_seqs)
        # Permute for Conv1d: (batch_size, embed_dim, seq_len)
        x = x.transpose(1, 2)
        
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        
        # Global max pooling over sequence length -> (batch_size, num_filters, 1)
        x = F.adaptive_max_pool1d(x, 1).squeeze(2)
        
        # Projection to fixed dimension
        out = self.fc(x)
        out = self.layer_norm(out)
        return out

