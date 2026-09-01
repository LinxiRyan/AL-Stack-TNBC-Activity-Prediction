import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem.MolStandardize import rdMolStandardize


def standardize_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(str(smiles).strip())
        if mol is None:
            return None
            
        # Apply standardization pipeline
        mol = rdMolStandardize.Cleanup(mol)
        mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
        mol = rdMolStandardize.Uncharger().uncharge(mol)
        mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
        
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def molecular_similarity(smiles_a, smiles_b, fingerprint_function):
    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)
    
    if mol_a is None or mol_b is None:
        return None
        
    fp_a = fingerprint_function(mol_a)
    fp_b = fingerprint_function(mol_b)
    
    return DataStructs.TanimotoSimilarity(fp_a, fp_b)


def process_generated_pool(
    generated_df, training_df, generated_smiles_col, seed_smiles_col,
    training_smiles_col, source_bin, similarity_threshold, fingerprint_function
):
    data = generated_df.copy()

    data["canonical_smiles"] = data[generated_smiles_col].apply(standardize_smiles)
    data["_seed_canonical"] = data[seed_smiles_col].apply(standardize_smiles)
    data = data.dropna(subset=["canonical_smiles", "_seed_canonical"]).copy()

    data["_similarity"] = data.apply(
        lambda row: molecular_similarity(
            row["canonical_smiles"],
            row["_seed_canonical"],
            fingerprint_function
        ),
        axis=1
    )
    
    data = data.dropna(subset=["_similarity"])
    data = data[data["_similarity"] < similarity_threshold].copy()

    data = data.drop_duplicates(subset=["canonical_smiles"])

    training_smiles_set = set(
        training_df[training_smiles_col].apply(standardize_smiles).dropna()
    )
    data = data[~data["canonical_smiles"].isin(training_smiles_set)].copy()

    data["source_bin"] = source_bin
    
    return (
        data.drop(columns=["_seed_canonical", "_similarity"])
        .reset_index(drop=True)
    )


def combine_generated_pools(pools):
    if not pools:
        return pd.DataFrame()
        
    data = pd.concat(pools, ignore_index=True)
    
    return (
        data.drop_duplicates(subset=["canonical_smiles"])
        .reset_index(drop=True)
    )