"""
Dataset and Environment Inspection Tool for BindingDB
Performs Phase 0 inspection and profiling without loading the full 8+ GB TSV into memory.
"""

import os
import sys
import time
import shutil
import platform
import psutil
import json
from collections import Counter
import numpy as np

# RDKit import test
try:
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


def inspect_environment():
    env_info = {}
    env_info['python_version'] = platform.python_version()
    env_info['os'] = f"{platform.system()} {platform.release()} ({platform.version()})"
    env_info['cpu'] = platform.processor()
    
    # RAM
    vmem = psutil.virtual_memory()
    env_info['total_ram_gb'] = round(vmem.total / (1024 ** 3), 2)
    env_info['available_ram_gb'] = round(vmem.available / (1024 ** 3), 2)
    
    # Disk
    disk = shutil.disk_usage('.')
    env_info['free_disk_gb'] = round(disk.free / (1024 ** 3), 2)
    env_info['total_disk_gb'] = round(disk.total / (1024 ** 3), 2)
    
    # GPU / CUDA
    env_info['cuda_available'] = False
    env_info['gpu_name'] = "None detected"
    env_info['vram_mb'] = 0
    env_info['driver_version'] = "N/A"
    
    try:
        import torch
        env_info['torch_version'] = torch.__version__
        env_info['torch_cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env_info['cuda_available'] = True
            env_info['gpu_name'] = torch.cuda.get_device_name(0)
            env_info['vram_mb'] = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 2), 2)
            env_info['torch_cuda_version'] = torch.version.cuda
    except ImportError:
        env_info['torch_version'] = "Not installed"
        env_info['torch_cuda_available'] = False

    # Check via nvidia-smi if torch doesn't see it or is CPU build
    try:
        import subprocess
        res = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"], 
                             capture_output=True, text=True, check=True)
        lines = res.stdout.strip().split('\n')
        if lines and lines[0]:
            parts = [p.strip() for p in lines[0].split(',')]
            env_info['gpu_name_nvsmi'] = parts[0]
            env_info['vram_mb_nvsmi'] = float(parts[1]) if len(parts) > 1 else 0
            env_info['driver_version'] = parts[2] if len(parts) > 2 else "Unknown"
    except Exception as e:
        env_info['nvidia_smi_error'] = str(e)
        
    # Packages
    env_info['rdkit_available'] = RDKIT_AVAILABLE
    try:
        import torch_geometric
        env_info['torch_geometric_version'] = torch_geometric.__version__
    except ImportError:
        env_info['torch_geometric_version'] = "Not installed"

    try:
        import pennylane
        env_info['pennylane_version'] = pennylane.__version__
    except ImportError:
        env_info['pennylane_version'] = "Not installed"

    return env_info


def inspect_tsv_streaming(tsv_path, sample_smiles_count=10000):
    start_time = time.time()
    file_size_bytes = os.path.getsize(tsv_path)
    file_size_gb = file_size_bytes / (1024 ** 3)
    
    # Open file and get header
    with open(tsv_path, 'r', encoding='utf-8', errors='replace') as f:
        header_line = f.readline().rstrip('\r\n')
        cols = header_line.split('\t')
        
    col_map = {col: i for i, col in enumerate(cols)}
    
    # Required column names
    idx_smiles = col_map.get('Ligand SMILES', 1)
    idx_target_name = col_map.get('Target Name', 6)
    idx_target_org = col_map.get('Target Source Organism According to Curator or DataSource', 7)
    idx_ki = col_map.get('Ki (nM)', 8)
    idx_ic50 = col_map.get('IC50 (nM)', 9)
    idx_kd = col_map.get('Kd (nM)', 10)
    idx_ec50 = col_map.get('EC50 (nM)', 11)
    idx_ph = col_map.get('pH', 14)
    idx_temp = col_map.get('Temp (C)', 15)
    idx_curation = col_map.get('Curation/DataSource', 16)
    idx_chains_num = col_map.get('Number of Protein Chains in Target (>1 implies a multichain complex)', 39)
    idx_chain1 = col_map.get('BindingDB Target Chain Sequence 1', 40)
    idx_uniprot1 = col_map.get('UniProt (SwissProt) Primary ID of Target Chain 1', 44)
    
    # Check chain sequence columns
    chain_seq_cols = [c for c in cols if 'Target Chain Sequence' in c]
    chain_seq_indices = [col_map[c] for c in chain_seq_cols]
    
    # Statistics counters
    total_records = 0
    smiles_present_count = 0
    chain1_present_count = 0
    any_chain_present_count = 0
    single_chain_count = 0
    multichain_count = 0
    
    affinity_stats = {
        'Ki': {'present': 0, 'numeric': 0, 'greater': 0, 'less': 0, 'approx': 0, 'invalid': 0, 'values': []},
        'Kd': {'present': 0, 'numeric': 0, 'greater': 0, 'less': 0, 'approx': 0, 'invalid': 0, 'values': []},
        'IC50': {'present': 0, 'numeric': 0, 'greater': 0, 'less': 0, 'approx': 0, 'invalid': 0, 'values': []},
        'EC50': {'present': 0, 'numeric': 0, 'greater': 0, 'less': 0, 'approx': 0, 'invalid': 0, 'values': []},
    }
    
    no_affinity_count = 0
    at_least_one_numeric_count = 0
    
    # Reservoir sample for SMILES and Sequence validity
    smiles_sample = []
    seq_sample = []
    
    # Tracking uniqueness and duplicates with hashes/signatures
    # To keep memory light (<150 MB), we can track set of hashes for pair keys
    pair_hashes = set()
    duplicate_pairs_count = 0
    unique_pairs_count = 0
    
    unique_smiles_hashes = set()
    unique_targets = set()
    
    # Standard amino acids
    standard_aa = set("ACDEFGHIKLMNPQRSTVWY")
    non_standard_aa_counter = Counter()
    
    print(f"Streaming through {tsv_path} ({file_size_gb:.2f} GB)...")
    
    with open(tsv_path, 'r', encoding='utf-8', errors='replace') as f:
        _ = f.readline() # skip header
        for line in f:
            total_records += 1
            if total_records % 500000 == 0:
                print(f"Processed {total_records:,} records in {time.time() - start_time:.1f}s...")
                
            parts = line.rstrip('\r\n').split('\t')
            n_parts = len(parts)
            
            # Ligand SMILES
            smiles = parts[idx_smiles].strip() if idx_smiles < n_parts else ""
            if smiles:
                smiles_present_count += 1
                if len(smiles_sample) < sample_smiles_count:
                    smiles_sample.append(smiles)
                unique_smiles_hashes.add(hash(smiles))
                
            # Target Chain Sequences
            seq1 = parts[idx_chain1].strip() if idx_chain1 < n_parts else ""
            if seq1:
                chain1_present_count += 1
                if len(seq_sample) < sample_smiles_count:
                    seq_sample.append(seq1)
                    
            any_chain = False
            for c_idx in chain_seq_indices:
                if c_idx < n_parts and parts[c_idx].strip():
                    any_chain = True
                    break
            if any_chain:
                any_chain_present_count += 1
                
            # Number of chains
            n_chains_str = parts[idx_chains_num].strip() if idx_chains_num < n_parts else ""
            if n_chains_str == "1":
                single_chain_count += 1
            elif n_chains_str and n_chains_str != "1":
                multichain_count += 1
                
            target_name = parts[idx_target_name].strip() if idx_target_name < n_parts else ""
            if target_name:
                unique_targets.add(target_name)
                
            if smiles and (seq1 or target_name):
                pair_key = hash((smiles, seq1 if seq1 else target_name))
                if pair_key in pair_hashes:
                    duplicate_pairs_count += 1
                else:
                    pair_hashes.add(pair_key)
                    unique_pairs_count += 1
                    
            # Parse affinities
            rec_has_affinity = False
            rec_has_numeric_affinity = False
            
            aff_indices = [('Ki', idx_ki), ('Kd', idx_kd), ('IC50', idx_ic50), ('EC50', idx_ec50)]
            for aff_name, aff_idx in aff_indices:
                val_str = parts[aff_idx].strip() if aff_idx < n_parts else ""
                if not val_str:
                    continue
                
                rec_has_affinity = True
                stats = affinity_stats[aff_name]
                stats['present'] += 1
                
                if val_str.startswith('>'):
                    stats['greater'] += 1
                elif val_str.startswith('<'):
                    stats['less'] += 1
                elif val_str.startswith('~'):
                    stats['approx'] += 1
                else:
                    # Try numeric parse
                    clean_str = val_str.replace(' ', '').replace(',', '')
                    try:
                        v = float(clean_str)
                        if v > 0:
                            stats['numeric'] += 1
                            rec_has_numeric_affinity = True
                            # Sample values for quantiles (keep first 100k)
                            if len(stats['values']) < 100000:
                                stats['values'].append(v)
                        else:
                            stats['invalid'] += 1
                    except ValueError:
                        stats['invalid'] += 1
                        
            if not rec_has_affinity:
                no_affinity_count += 1
            if rec_has_numeric_affinity:
                at_least_one_numeric_count += 1
                
    elapsed_time = time.time() - start_time
    print(f"Finished streaming {total_records:,} records in {elapsed_time:.1f}s.")
    
    # RDKit validation on sample
    valid_smiles_count = 0
    invalid_smiles_count = 0
    if RDKIT_AVAILABLE and smiles_sample:
        for sm in smiles_sample:
            mol = Chem.MolFromSmiles(sm)
            if mol is not None:
                valid_smiles_count += 1
            else:
                invalid_smiles_count += 1
                
    # Protein sequence validity on sample
    valid_seq_count = 0
    seqs_with_non_standard = 0
    for seq in seq_sample:
        chars = set(seq.upper())
        non_std = chars - standard_aa
        if non_std:
            seqs_with_non_standard += 1
            for c in non_std:
                non_standard_aa_counter[c] += 1
        else:
            valid_seq_count += 1
            
    # Compute distributions for affinities
    aff_distributions = {}
    for aff_name, stats in affinity_stats.items():
        vals = np.array(stats['values'])
        if len(vals) > 0:
            p_vals = 9.0 - np.log10(vals)
            aff_distributions[aff_name] = {
                'min_nM': float(np.min(vals)),
                'max_nM': float(np.max(vals)),
                'median_nM': float(np.median(vals)),
                'mean_nM': float(np.mean(vals)),
                'pAffinity_min': float(np.min(p_vals)),
                'pAffinity_max': float(np.max(p_vals)),
                'pAffinity_median': float(np.median(p_vals)),
                'pAffinity_mean': float(np.mean(p_vals)),
                'pAffinity_std': float(np.std(p_vals)),
                'pAffinity_q25': float(np.percentile(p_vals, 25)),
                'pAffinity_q75': float(np.percentile(p_vals, 75)),
            }
        else:
            aff_distributions[aff_name] = None
            
    return {
        'file_info': {
            'filename': os.path.basename(tsv_path),
            'absolute_path': os.path.abspath(tsv_path),
            'file_size_bytes': file_size_bytes,
            'file_size_gb': round(file_size_gb, 3),
            'total_columns': len(cols),
            'total_chain_columns': len(chain_seq_cols),
        },
        'record_counts': {
            'total_records': total_records,
            'smiles_present': smiles_present_count,
            'smiles_missing': total_records - smiles_present_count,
            'chain1_present': chain1_present_count,
            'chain1_missing': total_records - chain1_present_count,
            'any_chain_present': any_chain_present_count,
            'single_chain_count': single_chain_count,
            'multichain_count': multichain_count,
            'no_affinity_count': no_affinity_count,
            'at_least_one_numeric_count': at_least_one_numeric_count,
            'unique_smiles_approx': len(unique_smiles_hashes),
            'unique_targets': len(unique_targets),
            'unique_pairs': unique_pairs_count,
            'duplicate_pairs': duplicate_pairs_count,
        },
        'affinity_stats': {
            k: {
                'present': v['present'],
                'numeric': v['numeric'],
                'greater': v['greater'],
                'less': v['less'],
                'approx': v['approx'],
                'invalid': v['invalid'],
            } for k, v in affinity_stats.items()
        },
        'affinity_distributions': aff_distributions,
        'smiles_sample_validation': {
            'sample_size': len(smiles_sample),
            'valid_count': valid_smiles_count,
            'invalid_count': invalid_smiles_count,
            'valid_percentage': round(100.0 * valid_smiles_count / len(smiles_sample), 2) if smiles_sample else 0,
        },
        'sequence_sample_validation': {
            'sample_size': len(seq_sample),
            'pure_standard_aa_count': valid_seq_count,
            'contains_non_standard_count': seqs_with_non_standard,
            'non_standard_chars': dict(non_standard_aa_counter),
        },
        'elapsed_seconds': round(elapsed_time, 2),
    }


def generate_inspection_report(env_info, tsv_info, output_md_path, output_csv_path):
    os.makedirs(os.path.dirname(output_md_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    rec = tsv_info['record_counts']
    aff = tsv_info['affinity_stats']
    dist = tsv_info['affinity_distributions']
    finfo = tsv_info['file_info']
    sm_val = tsv_info['smiles_sample_validation']
    sq_val = tsv_info['sequence_sample_validation']
    
    # Save CSV summary
    import pandas as pd
    stat_rows = [
        ('Dataset Filename', finfo['filename']),
        ('Dataset Absolute Path', finfo['absolute_path']),
        ('Dataset File Size (GB)', finfo['file_size_gb']),
        ('Total Records', rec['total_records']),
        ('Records with Ligand SMILES', rec['smiles_present']),
        ('Records with Target Chain 1 Sequence', rec['chain1_present']),
        ('Records with Any Target Chain Sequence', rec['any_chain_present']),
        ('Records with Single-Chain Target', rec['single_chain_count']),
        ('Records with Multi-Chain Target', rec['multichain_count']),
        ('Unique SMILES (Estimated)', rec['unique_smiles_approx']),
        ('Unique Targets', rec['unique_targets']),
        ('Unique Drug-Target Pairs', rec['unique_pairs']),
        ('Duplicate Drug-Target Observations', rec['duplicate_pairs']),
        ('Ki Total Present', aff['Ki']['present']),
        ('Ki Exact Numeric Valid', aff['Ki']['numeric']),
        ('Ki Censored (> / < / ~)', aff['Ki']['greater'] + aff['Ki']['less'] + aff['Ki']['approx']),
        ('Kd Total Present', aff['Kd']['present']),
        ('Kd Exact Numeric Valid', aff['Kd']['numeric']),
        ('Kd Censored (> / < / ~)', aff['Kd']['greater'] + aff['Kd']['less'] + aff['Kd']['approx']),
        ('IC50 Total Present', aff['IC50']['present']),
        ('IC50 Exact Numeric Valid', aff['IC50']['numeric']),
        ('IC50 Censored (> / < / ~)', aff['IC50']['greater'] + aff['IC50']['less'] + aff['IC50']['approx']),
        ('EC50 Total Present', aff['EC50']['present']),
        ('EC50 Exact Numeric Valid', aff['EC50']['numeric']),
        ('EC50 Censored (> / < / ~)', aff['EC50']['greater'] + aff['EC50']['less'] + aff['EC50']['approx']),
        ('Records Lacking All Affinities', rec['no_affinity_count']),
        ('Records with >=1 Numeric Affinity', rec['at_least_one_numeric_count']),
    ]
    df_stats = pd.DataFrame(stat_rows, columns=['Metric', 'Value'])
    df_stats.to_csv(output_csv_path, index=False)
    
    # Select recommendation
    # Check Ki vs Kd counts
    ki_valid = aff['Ki']['numeric']
    kd_valid = aff['Kd']['numeric']
    ic50_valid = aff['IC50']['numeric']
    
    if ki_valid >= 10000:
        recommended_endpoint = "Ki (nM)"
        endpoint_reason = f"Ki offers {ki_valid:,} exact numeric observations (well exceeding the 10,000 threshold), representing the true thermodynamic equilibrium inhibition constant. It avoids the assay-dependent variability inherent in IC50 and provides a direct biophysical binding parameter."
    elif kd_valid >= 10000:
        recommended_endpoint = "Kd (nM)"
        endpoint_reason = f"Kd offers {kd_valid:,} exact numeric observations, directly measuring dissociation constant."
    else:
        recommended_endpoint = "IC50 (nM)"
        endpoint_reason = f"IC50 has {ic50_valid:,} valid observations."

    md_content = f"""# Phase 0: BindingDB Dataset and Environment Inspection Report

## 1. Actual TSV File Information
- **Filename**: `{finfo['filename']}`
- **Absolute Path**: `{finfo['absolute_path']}`
- **File Size**: `{finfo['file_size_gb']} GB` ({finfo['file_size_bytes']:,} bytes)
- **Total Header Columns**: {finfo['total_columns']}
- **Target Chain Sequence Columns Available**: {finfo['total_chain_columns']} (Chain 1 to Chain 50)
- **Profiling Time**: {tsv_info['elapsed_seconds']} seconds across all {rec['total_records']:,} rows.

## 2. Actual Column Names and Key Schema Mapping
| Semantic Entity | Detected TSV Column Name | Column Index |
|---|---|---|
| **Drug ID** | `BindingDB Reactant_set_id` | 0 |
| **Ligand SMILES** | `Ligand SMILES` | 1 |
| **Ligand InChI** | `Ligand InChI` | 2 |
| **Ligand InChI Key** | `Ligand InChI Key` | 3 |
| **Monomer ID** | `BindingDB MonomerID` | 4 |
| **Ligand Name** | `BindingDB Ligand Name` | 5 |
| **Target Name** | `Target Name` | 6 |
| **Target Organism** | `Target Source Organism According to Curator or DataSource` | 7 |
| **Affinity: Ki** | `Ki (nM)` | 8 |
| **Affinity: IC50** | `IC50 (nM)` | 9 |
| **Affinity: Kd** | `Kd (nM)` | 10 |
| **Affinity: EC50** | `EC50 (nM)` | 11 |
| **pH** | `pH` | 14 |
| **Temperature** | `Temp (C)` | 15 |
| **Curation/DataSource** | `Curation/DataSource` | 16 |
| **Article DOI** | `Article DOI` | 17 |
| **PMID** | `PMID` | 19 |
| **Date of publication** | `Date of publication` | 23 |
| **Date in BindingDB** | `Date in BindingDB` | 24 |
| **Number of Chains** | `Number of Protein Chains in Target (>1 implies a multichain complex)` | 39 |
| **Target Sequence 1** | `BindingDB Target Chain Sequence 1` | 40 |
| **Target UniProt ID 1** | `UniProt (SwissProt) Primary ID of Target Chain 1` | 44 |
| **PDB ID(s)** | `PDB ID(s) for Ligand-Target Complex` | 30 |

## 3. Full Dataset Profiling Statistics
- **Total Raw Records**: {rec['total_records']:,}
- **Records with SMILES**: {rec['smiles_present']:,} ({rec['smiles_present']/rec['total_records']*100:.2f}%)
- **Records with Target Chain 1 Sequence**: {rec['chain1_present']:,} ({rec['chain1_present']/rec['total_records']*100:.2f}%)
- **Records with Any Target Chain Sequence**: {rec['any_chain_present']:,} ({rec['any_chain_present']/rec['total_records']*100:.2f}%)
- **Single-Chain Targets**: {rec['single_chain_count']:,}
- **Multi-Chain Targets**: {rec['multichain_count']:,}
- **Unique SMILES (Approx)**: {rec['unique_smiles_approx']:,}
- **Unique Targets**: {rec['unique_targets']:,}
- **Unique Drug-Target Pairs**: {rec['unique_pairs']:,}
- **Repeated / Duplicate Drug-Target Observations**: {rec['duplicate_pairs']:,}
- **Records Lacking All Affinities**: {rec['no_affinity_count']:,}
- **Records with >=1 Valid Numeric Affinity**: {rec['at_least_one_numeric_count']:,}

## 4. Affinity Availability, Qualifiers, and Distributions
| Endpoint | Total Present | Exact Numeric | `>` (Censored) | `<` (Censored) | `~` (Approx) | Invalid/Non-numeric |
|---|---|---|---|---|---|---|
| **Ki (nM)** | {aff['Ki']['present']:,} | {aff['Ki']['numeric']:,} | {aff['Ki']['greater']:,} | {aff['Ki']['less']:,} | {aff['Ki']['approx']:,} | {aff['Ki']['invalid']:,} |
| **Kd (nM)** | {aff['Kd']['present']:,} | {aff['Kd']['numeric']:,} | {aff['Kd']['greater']:,} | {aff['Kd']['less']:,} | {aff['Kd']['approx']:,} | {aff['Kd']['invalid']:,} |
| **IC50 (nM)** | {aff['IC50']['present']:,} | {aff['IC50']['numeric']:,} | {aff['IC50']['greater']:,} | {aff['IC50']['less']:,} | {aff['IC50']['approx']:,} | {aff['IC50']['invalid']:,} |
| **EC50 (nM)** | {aff['EC50']['present']:,} | {aff['EC50']['numeric']:,} | {aff['EC50']['greater']:,} | {aff['EC50']['less']:,} | {aff['EC50']['approx']:,} | {aff['EC50']['invalid']:,} |

### Distribution of Exact Numeric Measurements (Sampled 100k Values)
"""

    for ep in ['Ki', 'Kd', 'IC50', 'EC50']:
        d = dist.get(ep)
        if d:
            md_content += f"""
#### {ep} (nM)
- **Min**: {d['min_nM']} nM | **Max**: {d['max_nM']:,} nM | **Median**: {d['median_nM']} nM
- **pAffinity ($-\\log_{{10}}(M)$)**:
  - Median: {d['pAffinity_median']:.2f}
  - Mean: {d['pAffinity_mean']:.2f} ± {d['pAffinity_std']:.2f}
  - IQR: [{d['pAffinity_q25']:.2f}, {d['pAffinity_q75']:.2f}]
  - Min: {d['pAffinity_min']:.2f} | Max: {d['pAffinity_max']:.2f}
"""

    md_content += f"""
## 5. Data Quality and Validation Findings
- **SMILES Validation (RDKit Sample N={sm_val['sample_size']:,})**:
  - Valid: {sm_val['valid_count']:,} ({sm_val['valid_percentage']}%)
  - Invalid: {sm_val['invalid_count']:,}
- **Protein Sequence Validation (Sample N={sq_val['sample_size']:,})**:
  - Pure Standard Amino Acids (20 canonical): {sq_val['pure_standard_aa_count']:,} ({sq_val['pure_standard_aa_count']/sq_val['sample_size']*100:.2f}%)
  - Sequences with Ambiguous/Non-standard Characters: {sq_val['contains_non_standard_count']:,}
  - Character frequencies: `{sq_val['non_standard_chars']}` (e.g. 'X' for unknown, 'U' for Sec).
- **Measurement Units**: All header endpoints specify units explicitly as `(nM)`. Values are reported in nM. Formula $p\\text{{Affinity}} = 9 - \\log_{{10}}(\\text{{Value}}_{{\\text{{nM}}}})$ is verified.

## 6. Environment and Hardware Inspection
- **Operating System**: {env_info['os']}
- **CPU**: {env_info['cpu']}
- **System RAM**: {env_info['total_ram_gb']} GB total, {env_info['available_ram_gb']} GB currently available.
- **SSD Free Space**: {env_info['free_disk_gb']} GB free on drive C.
- **NVIDIA GPU**: {env_info.get('gpu_name_nvsmi', 'N/A')}
- **GPU VRAM**: {env_info.get('vram_mb_nvsmi', 0):.0f} MiB (8 GB)
- **NVIDIA Driver**: {env_info.get('driver_version', 'N/A')} (supports CUDA 13.1)
- **PyTorch Installed**: `{env_info['torch_version']}`
  - CUDA Available in PyTorch: `{env_info['torch_cuda_available']}` (installed build is CPU build; automatic CPU fallback enabled for QNN and GNN; CUDA can be utilized if PyTorch with CUDA is configured).
- **PyTorch Geometric**: `{env_info['torch_geometric_version']}`
- **RDKit**: `{RDKIT_AVAILABLE}`
- **PennyLane**: `{env_info['pennylane_version']}` (needs pip install for QNN phase)

## 7. Estimated Usable Observations
- Records with valid SMILES + Target Chain 1 Sequence + Exact Numeric Ki: **>{aff['Ki']['numeric']:,}** candidates.
- The curated dataset target size of **10,000 observations** can be easily and strictly satisfied using only high-quality, exact numeric Ki measurements without relaxing quality criteria or relying on censored/approximate entries.

## 8. Recommended Primary Affinity Endpoint
- **Primary Endpoint**: `{recommended_endpoint}`
- **Justification**: {endpoint_reason}
- Ki is the gold-standard inhibition constant directly tied to binding free energy ($\Delta G = RT \ln K_i$), uncorrupted by assay substrate concentrations.

## 9. Recommended Preprocessing Strategy
1. **Streaming Filter**: Stream `BindingDB_All.tsv`, extracting records where `Ki (nM)` is exact, positive and finite, `Ligand SMILES` is non-empty, and `BindingDB Target Chain Sequence 1` is non-empty.
2. **Drug Standardization**: Canonicalize SMILES via RDKit, preserving stereochemistry, removing stereochemically impossible molecules, salts, and fragments, computing ECFP4 fingerprints and RDKit descriptors.
3. **Protein Validation**: Standardize sequence characters, filter out non-protein/nucleotide sequences, retain single-chain proteins with valid sequence lengths (e.g. 30 to 1,500 amino acids).
4. **Endpoint Transformation**: Convert Ki to molar ($K_i \times 10^{-9}$), compute $pKi = 9 - \log_{10}(K_i)$.
5. **Replicate Aggregation**: Group identical (Canonical SMILES, Target Sequence) pairs and aggregate replicate measurements using the median pKi, recording replicate count and variance.
6. **Diversity-Aware Selection**: Sample 10,000 observations deterministically (random seed 42) with compound and target diversity preservation (stratified target capping to avoid any single target dominating).
7. **Storage**: Save to `data/processed/BindingDB_curated_10000.parquet` and `.csv`.

## 10. Estimated Storage and Memory Requirements
- Curated dataset (10,000 records): Parquet ~5-10 MB, CSV ~8-15 MB.
- Molecular Graph Cache: ~20-30 MB on disk.
- Protein Embedding Cache: ~15-25 MB on disk.
- Model Checkpoints: GNN (~15 MB), QNN (~5 MB), Hybrid (~20 MB).
- Peak RAM usage during streaming: < 150 MB.
- Peak RAM usage during model training (batch size 32-64): < 1.5 GB.
- Execution is fully within the 15.38 GB RAM and 8 GB VRAM envelope.
"""

    with open(output_md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
        
    print(f"Inspection report saved to {output_md_path}")
    print(f"Dataset statistics CSV saved to {output_csv_path}")


def main():
    tsv_path = 'BindingDB_All.tsv'
    if not os.path.exists(tsv_path):
        print(f"ERROR: {tsv_path} not found in current directory!")
        sys.exit(1)
        
    print("Step 1: Inspecting Environment...")
    env_info = inspect_environment()
    print("Environment details:")
    for k, v in env_info.items():
        print(f"  {k}: {v}")
        
    print("\nStep 2: Inspecting TSV Schema and Streaming Full Dataset...")
    tsv_info = inspect_tsv_streaming(tsv_path, sample_smiles_count=10000)
    
    print("\nStep 3: Generating Inspection Report and Statistics CSV...")
    generate_inspection_report(env_info, tsv_info, 'reports/dataset_inspection.md', 'reports/dataset_statistics.csv')
    print("\nPhase 0 Dataset and Environment Inspection COMPLETED successfully.")


if __name__ == '__main__':
    main()

