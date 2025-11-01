# Configuration file for the nPulse software pipeline

# Data paths
DATA_PATH = '../data/badr-data'

# Raw data files
RAW_FILES = [
    'WS_R_ch3_seq4_250523175746.csv',
    'WS_R_ch3_seq4_250523180012.csv',
    'WS_R_ch3_seq4_250523180128.csv',
    'WS_R_ch3_seq4_250523180233.csv',
    'WS_R_ch3_seq4_250523180349.csv',
    'WS_R_ch3_seq4_250523180502.csv'
]

# Clean data files (same names as raw)
CLEAN_FILES = RAW_FILES

# Pipeline flags
CLEAN = True  # Whether to run the cleaning step
PREPROCESSING = True  # Whether to run preprocessing
RF = True  # Whether to train Random Forest model


