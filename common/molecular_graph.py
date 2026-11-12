"""
Molecular Graph Representation for GNN Models
Extracts rich atom and bond features from RDKit Mol objects and converts to PyTorch Geometric Data.
"""

from typing import Optional, List, Dict
import os
import torch
from rdkit import Chem
from torch_geometric.data import Data

# Allowed atom features
ALLOWED_ATOMIC_NUMBERS = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53] # Common drug atoms + other
ALLOWED_DEGREES = [0, 1, 2, 3, 4, 5, 6]
ALLOWED_FORMAL_CHARGES = [-2, -1, 0, 1, 2]
ALLOWED_HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
ALLOWED_CHIRAL_TAGS = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
]

# Allowed bond features
ALLOWED_BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
ALLOWED_BOND_DIRS = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.BEGINWEDGE,
    Chem.rdchem.BondDir.BEGINDASH,
]
ALLOWED_BOND_STEREO = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
]


def one_hot_encode(val, allowed_list) -> List[float]:
    """One-hot encode a value with an extra element for unknown."""
    encoding = [1.0 if val == allowed else 0.0 for allowed in allowed_list]
    if 1.0 not in encoding:
        encoding.append(1.0) # unknown
    else:
        encoding.append(0.0)
    return encoding


def get_atom_features(atom: Chem.Atom) -> List[float]:
    """Generates rich feature vector for a single atom."""
    feats = []
    # Atomic number (one-hot)
    feats.extend(one_hot_encode(atom.GetAtomicNum(), ALLOWED_ATOMIC_NUMBERS))
    # Degree (one-hot)
    feats.extend(one_hot_encode(atom.GetTotalDegree(), ALLOWED_DEGREES))
    # Formal charge (one-hot)
    feats.extend(one_hot_encode(atom.GetFormalCharge(), ALLOWED_FORMAL_CHARGES))
    # Hybridization (one-hot)
    feats.extend(one_hot_encode(atom.GetHybridization(), ALLOWED_HYBRIDIZATIONS))
    # Aromaticity (binary)
    feats.append(1.0 if atom.GetIsAromatic() else 0.0)
    # Ring membership (binary)
    feats.append(1.0 if atom.IsInRing() else 0.0)
    # Chirality tag (one-hot)
    feats.extend(one_hot_encode(atom.GetChiralTag(), ALLOWED_CHIRAL_TAGS))
    # Hydrogen count
    feats.append(float(atom.GetTotalNumHs()))
    return feats


def get_bond_features(bond: Chem.Bond) -> List[float]:
    """Generates feature vector for a single bond."""
    feats = []
    # Bond type (one-hot)
    feats.extend(one_hot_encode(bond.GetBondType(), ALLOWED_BOND_TYPES))
    # Bond direction (one-hot)
    feats.extend(one_hot_encode(bond.GetBondDir(), ALLOWED_BOND_DIRS))
    # Bond stereo (one-hot)
    feats.extend(one_hot_encode(bond.GetStereo(), ALLOWED_BOND_STEREO))
    # Conjugated (binary)
    feats.append(1.0 if bond.GetIsConjugated() else 0.0)
    # In ring (binary)
    feats.append(1.0 if bond.IsInRing() else 0.0)
    return feats


def mol_to_graph_data(mol: Chem.Mol) -> Optional[Data]:
    """Converts an RDKit Mol to a PyTorch Geometric Data object."""
    if mol is None or mol.GetNumAtoms() == 0:
        return None
        
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))
    x = torch.tensor(atom_features, dtype=torch.float)
    
    edge_indices = []
    edge_features = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        b_feat = get_bond_features(bond)
        
        # Add undirected edge (both directions)
        edge_indices.append([i, j])
        edge_features.append(b_feat)
        edge_indices.append([j, i])
        edge_features.append(b_feat)
        
    if len(edge_indices) > 0:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
    else:
        # Isolated single atom (no bonds)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        sample_bond_feat_len = len(get_bond_features(Chem.Bond())) if False else 16
        edge_attr = torch.empty((0, sample_bond_feat_len), dtype=torch.float)
        
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=x.size(0))


def smiles_to_graph(smiles: str) -> Optional[Data]:
    """Direct conversion from SMILES to PyG Data object."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return mol_to_graph_data(mol)

