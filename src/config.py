from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_DIR / "train_dataset"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FEATURE_DIR = OUTPUT_DIR / "features"
OOF_DIR = OUTPUT_DIR / "oof"
MODEL_DIR = OUTPUT_DIR / "models"
SUBMISSION_DIR = OUTPUT_DIR / "submissions"

TRAIN_FILE = TRAIN_DIR / "Train.csv"
GENE_SEQUENCE_FILE = TRAIN_DIR / "gene_seq.csv"
MIRNA_SEQUENCE_FILE = TRAIN_DIR / "mirna_seq.csv"
TEST_FILE = DATA_DIR / "test_dataset.csv"
SUBMIT_EXAMPLE_FILE = DATA_DIR / "submit_example.csv"

TARGET_COLUMN = "results"
GENE_COLUMN = "gene"
MIRNA_COLUMN = "miRNA"
GENE_SEQUENCE_COLUMN = "gene_sequence"
MIRNA_SEQUENCE_COLUMN = "mirna_sequence"

SEED = 42
N_SPLITS = 5
