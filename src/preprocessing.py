"""
Preprocessing module for nPulse EMG signal processing.

This module handles the complete preprocessing pipeline for EMG sensor data,
including data cleaning, noise removal, normalization, and baseline correction.
"""

#=====================================================
# there is a python package that automates the prepocessing.
#=====================================================

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

import config
from utils import save




def remove_invalid_rows(df):
    """
    Remove invalid rows where Action2 is not NaN.
    
    Args:
        df: Raw DataFrame with Action1 and Action2 columns
        
    Returns:
        DataFrame with invalid rows removed
    """
    df = df.copy()
    initial_rows = len(df)
    df = df[df['Action2'].isna()].copy()

    print(f"Removed {initial_rows - len(df)} rows (Action2 not NaN)")
    
    return df


def consolidate_action_columns(df):
    """
    Create unified 'Action' column from Action1 and drop Action1/Action2.
    
    Args:
        df: DataFrame with Action1 and Action2 columns
        
    Returns:
        DataFrame with consolidated Action column
    """
    df = df.copy()
    df['Action'] = df['Action1']
    df = df.drop(columns=['Action1', 'Action2'])
    print(f"Consolidate action columns. Columns: {list(df.columns)}")
    return df


def trim_action_transitions(df, trim_seconds=config.ACTION_TRIM_SECONDS):
    """
    Remove initial seconds from each action segment to eliminate transition artifacts.
    
    Args:
        df : DataFrame with Timestamp and Action columns
        trim_seconds (float): Number of seconds to trim from start of each action
        
    Returns:
        DataFrame with trimmed action transitions
    """
    df = df.copy()
    df['_action_segment'] = df['Action'].ne(df['Action'].shift()).cumsum()
    num_segments = df['_action_segment'].nunique()
    print(f"Identified {num_segments} action segments")
    
    cleaned_groups = []
    for _, group in df.groupby('_action_segment'):
        start_time = group['Timestamp'].iloc[0]
        group_trimmed = group[group['Timestamp'] >= start_time + trim_seconds]
        if not group_trimmed.empty:
            cleaned_groups.append(group_trimmed)
    
    df_trimmed = pd.concat(cleaned_groups, ignore_index=True)
    df_trimmed = df_trimmed.drop(columns=['_action_segment'])
    trimmed_rows = len(df) - len(df_trimmed)
    print(f"Trimmed {trimmed_rows} rows from action transitions ({trim_seconds}s each)")
    return df_trimmed


def bandpass_filter(df, lowcut=config.BANDPASS_LOWCUT, highcut=config.BANDPASS_HIGHCUT, order=config.FILTER_ORDER, sampling_rate=config.SAMPLING_RATE):
    """
    Apply Butterworth bandpass filter to sensor channels.
    
    Removes low-frequency drift and high-frequency noise while preserving
    gesture-relevant signals in the 0.5-20 Hz range.
    
    Args:
        df: DataFrame with sensor channels and Timestamp
        lowcut (float): Lower cutoff frequency in Hz
        highcut (float): Upper cutoff frequency in Hz
        order (int): Filter order
        
    Returns:
        DataFrame with filtered sensor channels
    """
    df = df.copy()
    
    
    # Design Butterworth bandpass filter
    nyquist = 0.5 * sampling_rate
    low_normalized = lowcut / nyquist
    high_normalized = highcut / nyquist
    
    if high_normalized >= 1.0:
        print(f"Highcut frequency {highcut} Hz exceeds Nyquist frequency {nyquist:.2f} Hz")
        high_normalized = 0.99
        #suppress this if, if high cut if well under 1.
    
    # Use second-order sections for numerical stability
    sos = butter(order, [low_normalized, high_normalized], btype='band', output='sos')
    
    # Apply zero-phase filter to each channel using SOS
    for channel in config.SENSOR_CHANNELS:
        if channel in df.columns:
            df[channel] = sosfiltfilt(sos, df[channel].values)

    return df


# def remove_dc_offset(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Remove DC offset from sensor channels by subtracting the mean.
    
#     Args:
#         df: DataFrame with sensor channels
        
#     Returns:
#         DataFrame with DC offset removed
#     """
#     df = df.copy()
    
#     for channel in SENSOR_CHANNELS:
#         if channel in df.columns:
#             channel_mean = df[channel].mean()
#             df[channel] = df[channel] - channel_mean
#             print(f"Removed DC offset from {channel}: {channel_mean:.2f}")
#     return df


def standardize_channels(df):
    """
    Standardize sensor channels to zero mean and unit variance (z-score normalization).
    
    Args:
        df: DataFrame with sensor channels
        
    Returns:
        DataFrame with standardized sensor channels
    """
    df = df.copy()
    
    for channel in config.SENSOR_CHANNELS:
        if channel in df.columns:
            channel_mean = df[channel].mean()
            channel_std = df[channel].std()
            
            if channel_std > 0:
                df[channel] = (df[channel] - channel_mean) / channel_std
                print(f"Standardized {channel}: mean={channel_mean:.2f}, std={channel_std:.2f}")
            else:
                print(f" suppress this line if std is always > 0")
    return df


def combine_data():
    """
    Load and concatenate all cleaned data files into a single DataFrame.
    Returns:
        (pd.DataFrame): Combined DataFrame containing all cleaned data.

    """
    dataframes = []
    for file_name in config.FILES:
        file_path = config.CLEAN_DIR / file_name
        df = pd.read_csv(file_path)
        dataframes.append(df)
        print(f"Loaded {file_name}, shape={df.shape}")
    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"Combined data shape: {combined_df.shape}")
    return combined_df


def preprocess():
    """
    Execute complete preprocessing pipeline.
    
    Pipeline stages:
    1. Clean raw data (if config.CLEAN is True)
       - Remove invalid rows
       - Consolidate action columns
       - Trim action transitions
       
    2. Preprocess cleaned data (if config.PREPROCESSING is True)
       - Apply bandpass filter (0.5-20 Hz)
       - Standardize channels (z-score)
    
    All processed files are saved to config.PROCESSED_DIR.
    """
    
    # Stage 1: Clean raw data
    if config.CLEAN:
        print(f"Cleaning raw data")
        for file_name in config.FILES:
            raw_path = config.RAW_DIR / file_name
            clean_path = config.CLEAN_DIR / file_name

            df = pd.read_csv(raw_path)
            df = remove_invalid_rows(df)
            df = consolidate_action_columns(df)
            df = trim_action_transitions(df)
            save(clean_path, df)
    
    if config.PREPROCESSING:
        print(f"Preprocessing")

        df_all = combine_data()
        df_all = bandpass_filter(df_all)
        #df = remove_dc_offset(df) - Unless you wanna check the effect of only centering 
        df_all = standardize_channels(df_all)
        save(config.PREPROC_DATA_PATH, df_all)
    
    

