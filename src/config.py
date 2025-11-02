# Configuration file for the nPulse software pipeline

import os
from pathlib import Path

#=====================================================
# Paths
#=====================================================

BASE_DIR = Path(__file__).parent.parent

# Data paths
DATA_DIR = BASE_DIR / 'data' / 'badr-data'
RAW_DIR = DATA_DIR / 'raw'
CLEAN_DIR = DATA_DIR / 'clean'
PROCESSED_DIR = DATA_DIR / 'proc'  # For further processed data
FEATURES_DIR = DATA_DIR / 'features'  # For engineered features

OUTPUTS_DIR = BASE_DIR / 'outputs'

PREPROC_DATA_PATH = os.path.join(PROCESSED_DIR, "WS_R_ch3_seq4.csv")

# Ensure directories exist
for dir_path in [CLEAN_DIR, PROCESSED_DIR, FEATURES_DIR, OUTPUTS_DIR:
    dir_path.mkdir(parents=True, exist_ok=True)

# Raw data files
FILES = [
    'WS_R_ch3_seq4_250523175746.csv',
    'WS_R_ch3_seq4_250523180012.csv',
    'WS_R_ch3_seq4_250523180128.csv',
    'WS_R_ch3_seq4_250523180233.csv',
    'WS_R_ch3_seq4_250523180349.csv',
    'WS_R_ch3_seq4_250523180502.csv'
]



#=====================================================
# Pipeline flags
#=====================================================

CLEAN = True           
PREPROCESSING = True   # Run preprocessing 
FEATURE_ENGINEERING = False  # Run feature engineering
TRAIN_MODEL = False     # Train model (Random Forest)
TUNE_HYPERPARAMS = False    # Run hyperparameter tuning



#=====================================================
# Tuning
#=====================================================

# Model hyperparameters (example)
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': None,
    'random_state': 42
}

# Other constants
ACTION_TRIM_SECONDS = 0.2  # Seconds to trim from start of each action


