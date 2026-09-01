import pandas as pd


def prepare_generation_sets(df, smiles_col, target_col, generation_bins):
    data = df[[smiles_col, target_col]].copy()
    
    data = data.dropna(subset=[smiles_col, target_col])
    data[smiles_col] = data[smiles_col].astype(str).str.strip()
    data[target_col] = pd.to_numeric(data[target_col], errors="coerce")
    
    data = data.dropna(subset=[target_col])
    data = data[data[smiles_col] != ""]

    subsets = {}
    for low, high in generation_bins:
        subset = data[
            (data[target_col] >= low) & (data[target_col] < high)
        ].copy()
        
        subset = subset.drop_duplicates(subset=[smiles_col]).reset_index(drop=True)
        subsets[(low, high)] = subset
        
    return subsets


def extract_smiles(df, smiles_col):
    """Extract a deduplicated list of valid SMILES strings from a DataFrame."""
    return (
        df[smiles_col]
        .dropna()
        .astype(str)
        .str.strip()
        .drop_duplicates()
        .reset_index(drop=True)
    )
