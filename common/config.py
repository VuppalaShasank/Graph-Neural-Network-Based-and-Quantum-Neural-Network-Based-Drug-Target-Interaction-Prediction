"""
Central Configuration for Drug-Target Interaction Prediction
Contains dataset paths, random seeds, hyperparameters, and directory settings.
"""

import os
from pathlib import Path
import tempfile

# Set non-standard matplotlib config directory to eliminate warnings
os.environ.setdefault('MPLCONFIGDIR', os.path.join(tempfile.gettempdir(), 'mpl_dti_config'))

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_TSV_PATH = BASE_DIR / "BindingDB_All.tsv"

PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
GRAPH_CACHE_DIR = DATA_DIR / "graph_cache"
FEATURE_CACHE_DIR = DATA_DIR / "feature_cache"

RESULTS_DIR = BASE_DIR / "results"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
LOGS_DIR = BASE_DIR / "logs"
REPORTS_DIR = BASE_DIR / "reports"

# Ensure runtime directories exist
for d in [DATA_DIR, PROCESSED_DATA_DIR, SPLITS_DIR, GRAPH_CACHE_DIR, FEATURE_CACHE_DIR,
          RESULTS_DIR, CHECKPOINTS_DIR, LOGS_DIR, REPORTS_DIR,
          RESULTS_DIR / "GNN", RESULTS_DIR / "QNN", RESULTS_DIR / "GNN_QNN", RESULTS_DIR / "comparison",
          CHECKPOINTS_DIR / "GNN", CHECKPOINTS_DIR / "QNN", CHECKPOINTS_DIR / "GNN_QNN"]:
    d.mkdir(parents=True, exist_ok=True)

# Curated dataset paths
CURATED_PARQUET = PROCESSED_DATA_DIR / "BindingDB_curated_10000.parquet"
CURATED_CSV = PROCESSED_DATA_DIR / "BindingDB_curated_10000.csv"

# Global reproducibility seed
RANDOM_SEED = 42

# Preprocessing & Curation parameters
MAX_OBSERVATIONS = 10000
PRIMARY_ENDPOINT = "Ki (nM)"
MAX_TARGET_REPRESENTATION = 150  # Cap per target to ensure diverse protein coverage
MIN_SEQ_LEN = 30
MAX_SEQ_LEN = 1200
MIN_PKI = 2.0   # 10 mM
MAX_PKI = 15.0  # 1 fM

# Data Splitting
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Drug Graph & Fingerprint parameters
FP_RADIUS = 2  # ECFP4
FP_NBITS = 256 # For standalone QNN input vector compression
SELECTED_DESCRIPTORS = [
    'MolWt', 'MolLogP', 'NumHDonors', 'NumHAcceptors',
    'TPSA', 'NumRotatableBonds', 'RingCount', 'FractionCSP3'
]

# Protein Sequence Encoder parameters
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"  # 20 canonical amino acids
PAD_TOKEN = 0
UNK_TOKEN = 21
VOCAB_SIZE = 22  # 0: PAD, 1..20: Canonical AA, 21: UNK / Non-standard
PROT_EMBED_DIM = 64
PROT_NUM_FILTERS = 64
PROT_KERNEL_SIZE = 7
PROT_OUT_DIM = 64

# GNN parameters
GNN_HIDDEN_DIM = 64
GNN_OUT_DIM = 64
GNN_NUM_LAYERS = 3
GNN_DROPOUT = 0.2

# QNN parameters
N_QUBITS = 4
QNN_CIRCUIT_DEPTH = 2
QNN_DEVICE = "default.qubit"

# Training defaults
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_EPOCHS = 12
PATIENCE = 5
