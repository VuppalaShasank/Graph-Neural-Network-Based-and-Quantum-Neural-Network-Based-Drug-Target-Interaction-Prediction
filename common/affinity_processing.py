"""
Affinity Processing Utilities
Handles parsing, conversion from nM to molar, pAffinity computation, and quality validation.
"""

import math
from typing import Optional, Tuple


def parse_and_convert_affinity_nm(val_str: str) -> Tuple[Optional[float], Optional[float], str]:
    """
    Parses a raw affinity string in nM.
    Returns:
        (affinity_nm, p_affinity, status)
        where status is 'OK', 'EXCLUDED_QUALIFIER', 'EXCLUDED_NON_NUMERIC', 'EXCLUDED_NON_POSITIVE'
    """
    if not val_str:
        return None, None, "MISSING"
        
    s = val_str.strip()
    if s.startswith('>') or s.startswith('<') or s.startswith('~'):
        return None, None, "EXCLUDED_QUALIFIER"
        
    clean = s.replace(' ', '').replace(',', '')
    try:
        val_nm = float(clean)
    except ValueError:
        return None, None, "EXCLUDED_NON_NUMERIC"
        
    if val_nm <= 0 or not math.isfinite(val_nm):
        return None, None, "EXCLUDED_NON_POSITIVE"
        
    # pAffinity = -log10(val_nm * 1e-9) = 9 - log10(val_nm)
    p_affinity = 9.0 - math.log10(val_nm)
    
    return val_nm, p_affinity, "OK"


def paffinity_to_affinity_nm(p_aff: float) -> float:
    """Invert pAffinity back to nM: Affinity_nM = 10^(9 - pAffinity)."""
    return 10.0 ** (9.0 - p_aff)

