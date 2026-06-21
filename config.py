"""
This script contains configuration variables, constants of the project

Working with a single config.py helps avoid hardcoding!

"""
from pathlib import Path
ROOT_PATH = Path(__file__).resolve().parent
# Paths

RAW_DATA_PATH = ROOT_PATH / "data/raw/Combined_Data.csv"
PROCESSED_DATA_DIR = ROOT_PATH /"data/processed/"
OUTPUTS_DIR = ROOT_PATH / "outputs"
RESULTS_DIR = OUTPUTS_DIR / "results"
MODELS_DIR = OUTPUTS_DIR / "models"
LOGS_DIR = OUTPUTS_DIR / "logs"


# Seeds 
GLOBAL_SEED = 42
HYPERPARAM_SEARCH_SEED = 1234


# Data
NUM_CLASSES = 7
CLASS_NAMES = ['Anxiety', 'Bipolar', 'Depression', 'Normal', 'Personality disorder', 'Stress', 'Suicidal']
LABEL_COLUMN = "status"
ENCODED_LABEL = "encoded_status"
POST_COLUMN = "statement"
PREPROCESS_POSTS_ML = "statement_ml"
PREPROCESS_POSTS_SEQ = "statement_seq"
TRAIN_SIZE, VAL_SIZE, TEST_SIZE =  (0.60,0.20,0.20)
MINORITY_CLASSES_FOR_AUGMENTATION = ["Personality disorder", "Stress", "Bipolar", "Anxiety"]
# TODO: CLASS_NAMES = [...]  (the 7 status labels — confirm the exact order by
#       checking label_encoder.classes_ once you've run Section 3, don't guess)
# TODO: TRAIN_SIZE, VAL_SIZE, TEST_SIZE  (60/20/20 — express as fractions)
# TODO: MINORITY_CLASSES_FOR_AUGMENTATION = [...]  (Personality Disorder,
#       Stress, Bipolar, Anxiety)

# TF-IDF / ML
TFIDF_MAX_FEATURES = 1000

# GRU

GRU_VOCAB_SIZE = 10_000
GRU_MAX_LEN = 200
GRU_BATCH_SIZE = 16
GRU_DROPOUT = 0.3 # this is fixed in the original paper, following the same design

# GRU - HYPERPARAMETER TUNING RANGES
GRU_EMBEDDING_DIM_RANGE = (150,250)
GRU_HIDDEN_DIM_RANGE = (256,768)
GRU_LR_RANGE = (1e-4, 1e-3) 
GRU_EPOCHS_RANGE = (5,10)
GRU_N_TRIALS = 10


# ALBERT
ALBERT_MODEL_NAME = "albert-base-v2"
ALBERT_MAX_LEN = 200
ALBERT_BATCH_SIZE = 16

# HYPERPARAMETER TUNING ALBERT
ALBERT_LR_RANGE = (1e-5, 1e-4)
ALBERT_EPOCHS_RANGE = (3,5)
ALBERT_DROPOUT_RANGE = (0.1,0.5)
ALBERT_N_ITER  = 10

# Evaluation
BOOTSTRAP_N_ITERATIONS = 1000
CONFIDENCE_LEVEL = 0.95

