

def save(output_path, df):
    """
    Args:
        output_path: where it is saved.
        df: DataFrame
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path.name}, shape={df.shape}")