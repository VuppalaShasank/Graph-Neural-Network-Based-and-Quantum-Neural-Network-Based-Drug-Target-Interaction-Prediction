"""
Molecular Feature Utilities for Standalone QNN
Computes canonical SMILES, ECFP4 fingerprints, and physical-chemical RDKit descriptors.
"""

from typing import Optional, Dict, Tuple, Any
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors


DESCRIPTOR_FUNCS = {
    'MolWt': Descriptors.MolWt,
    'MolLogP': Descriptors.MolLogP,
    'NumHDonors': Descriptors.NumHDonors,
    'NumHAcceptors': Descriptors.NumHAcceptors,
    'TPSA': Descriptors.TPSA,
    'NumRotatableBonds': Descriptors.NumRotatableBonds,
    'RingCount': Descriptors.RingCount,
    'FractionCSP3': Descriptors.FractionCSP3,
}


def validate_and_canonicalize_smiles(smiles: str) -> Tuple[Optional[Chem.Mol], Optional[str], Optional[str]]:
    """
    Validates a SMILES string and produces canonical isomeric SMILES and InChIKey.
    Preserves stereochemistry.
    Returns:
        (mol, canonical_isomeric_smiles, inchi_key)
    """
    if not smiles or not isinstance(smiles, str):
        return None, None, None
        
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None, None, None
        
    try:
        # Generate canonical isomeric SMILES preserving chiral tags
        can_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        # Re-parse to verify round-trip
        mol_can = Chem.MolFromSmiles(can_smiles)
        if mol_can is None:
            return None, None, None
        inchi_key = Chem.MolToInchiKey(mol_can)
        return mol_can, can_smiles, inchi_key
    except Exception:
        return None, None, None


def compute_ecfp4_fingerprint(mol: Chem.Mol, radius: int = 2, n_bits: int = 256) -> np.ndarray:
    """Computes fixed-length Morgan / ECFP4 fingerprint as float numpy array."""
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    # Convert ExplicitBitVect to numpy
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def compute_rdkit_descriptors(mol: Chem.Mol) -> np.ndarray:
    """Calculates standardized 8 selected physical-chemical descriptors."""
    vals = []
    for name, func in DESCRIPTOR_FUNCS.items():
        try:
            v = float(func(mol))
            vals.append(v)
        except Exception:
            vals.append(0.0)
    return np.array(vals, dtype=np.float32)


def compute_molecular_feature_vector(mol: Chem.Mol, radius: int = 2, n_bits: int = 256) -> np.ndarray:
    """Combines ECFP4 fingerprint and normalized molecular descriptors for QNN input."""
    fp = compute_ecfp4_fingerprint(mol, radius=radius, n_bits=n_bits)
    desc = compute_rdkit_descriptors(mol)
    return np.concatenate([fp, desc], axis=0)

