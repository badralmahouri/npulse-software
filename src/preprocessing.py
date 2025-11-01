import utils
import pandas as pd
from pathlib import Path

def clean():
    """
    clean the data

    what the function does: 
    1) Creates a new file (with the same name) that will be saved in /clean at the end
    2) The clean file will come from the raw data, in which we would have replaced Action 1 and Action 2 by the column :
        Action: We remove all the rows where Action 2 is not NaN (2776 rows, look at eda.ipynb)
    3) Finally, we remove the 0.2s of the start of a new action (transition from a gesture to a new one).
    """
    
    # Define paths
    raw_path = Path('../data/badr-data/raw')
    clean_path = Path('../data/badr-data/clean')
    clean_path.mkdir(exist_ok=True)
    
    # Process each CSV file in raw directory
    for csv_file in raw_path.glob('*.csv'):
        # Load the raw data
        df = pd.read_csv(csv_file)
        
        # Step 1: Remove rows where Action2 is not NaN
        df = df[df['Action2'].isna()]
        print(df.shape)
        
        # Step 2: Create 'Action' column from Action1 and drop Action1 and Action2
        df['Action'] = df['Action1']
        df = df.drop(columns=['Action1', 'Action2'])
        
        # Step 3: Remove 0.2s from the start of each new action
        # Group by consecutive actions
        df['action_change'] = df['Action'].ne(df['Action'].shift()).cumsum()
        print(df)

        
        cleaned_groups = []
        for _, group in df.groupby('action_change'):
            start_time = group['Timestamp'].iloc[0]
            # Keep only rows where Timestamp >= start_time + 0.2
            group_clean = group[group['Timestamp'] >= start_time + 0.2]
            if not group_clean.empty:
                cleaned_groups.append(group_clean)
        
        # Concatenate cleaned groups
        if cleaned_groups:
            df_clean = pd.concat(cleaned_groups, ignore_index=True)
        else:
            df_clean = pd.DataFrame()  # Empty if no valid groups
        
        # Save the cleaned data to the clean directory with the same filename
        output_file = clean_path / csv_file.name
        df_clean.to_csv(output_file, index=False)
        print(f"Processed and saved: {output_file}")

def preprocess():
    # calls out the functions in the file 
    # orchestrate the preprocessing 
    clean()
    return

if __name__ == "__main__":
    preprocess()

