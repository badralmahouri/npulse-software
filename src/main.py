"""
Main entry point for nPulse EMG signal processing pipeline.

Pipeline stages:
1. Preprocessing: Clean, filter, and normalize raw sensor data
2. Feature Engineering: Extract time and frequency domain features
3. Hyperparameter Tuning: Optimize model parameters (optional)
4. Model Training: Train and evaluate final model
"""

import logging

import config
import preprocessing
import feature_engineering 
import tune
import train

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Execute the complete nPulse ML pipeline."""
    logger.info("Starting nPulse ML Pipeline")
    
    # Stage 1: Preprocessing
    if config.CLEAN or config.PREPROCESSING:
        preprocessing.preprocess()
    
    # Stage 2: Feature Engineering
    if config.FEATURE_ENGINEERING:
        feature_engineering.create_features()
    
    # Stage 3: Hyperparameter Tuning
    if config.TUNE_HYPERPARAMS:
        tune.tune()
    
    # Stage 4: Model Training
    if config.TRAIN_MODEL:
        train.train()
    
    logger.info("nPulse ML Pipeline completed successfully")


if __name__ == "__main__":
    main()







