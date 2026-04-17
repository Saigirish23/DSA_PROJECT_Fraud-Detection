
import os
import logging

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

RAW_TRANSACTIONS_PATH = os.path.join(RAW_DATA_DIR, "transactions.csv")
ACCOUNT_GROUND_TRUTH_PATH = os.path.join(RAW_DATA_DIR, "account_ground_truth.csv")
NODE_FEATURES_PATH = os.path.join(PROCESSED_DATA_DIR, "node_features.csv")
LABELS_PATH = os.path.join(PROCESSED_DATA_DIR, "labels.csv")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_gcn.pt")

RAW_DIR = RAW_DATA_DIR
PROCESSED_DIR = PROCESSED_DATA_DIR

RANDOM_SEED = 42

NUM_ACCOUNTS = 500
NUM_TRANSACTIONS = 2000
FRAUD_RATIO = 0.10

MIN_TRANSACTION_AMOUNT = 10.0
MAX_TRANSACTION_AMOUNT = 10000.0

HEURISTIC_WEIGHTS = {
    "w1": 0.35,
    "w2": 0.35,
    "w3": 0.30,
}

HEURISTIC_FRAUD_THRESHOLD = 0.15

GNN_HIDDEN_DIM = 64
GNN_NUM_CLASSES = 2
GNN_DROPOUT = 0.3
GNN_NUM_LAYERS = 2

HIDDEN_DIM = GNN_HIDDEN_DIM
SEED = RANDOM_SEED

LEARNING_RATE = 0.005
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 200
LOG_INTERVAL = 10
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

ALPHA_SWEEP_START = 0.0
ALPHA_SWEEP_END = 1.0
ALPHA_SWEEP_STEP = 0.1

LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(name=None):
    logger = logging.getLogger(name or "fraud_detection")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(LOG_LEVEL)
    return logger


def ensure_dirs():
    dirs = [RAW_DATA_DIR, PROCESSED_DATA_DIR, PLOTS_DIR, RESULTS_DIR, MODELS_DIR]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def setup_dirs():
    ensure_dirs()
