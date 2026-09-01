import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import SaltRemover

_remover = SaltRemover.SaltRemover()


def process_nci_data(activity_df, structure_df, cell_name, activity_threshold):
    # Filter by cell line and merge structure data
    df = activity_df[activity_df["CELL_NAME"] == cell_name].copy()
    df = df.merge(structure_df[["NSC", "SMILES"]], on="NSC", how="left")
    
    # Basic cleaning: remove rows without SMILES or non-molar concentrations
    df = df.dropna(subset=["SMILES"])
    df = df[df["CONCENTRATION_UNIT"] == "M"].copy()
    
    # Outlier handling: keep the measurement closest to the median
    median_values = df.groupby("NSC")["AVERAGE"].transform("median")
    df["_distance"] = (df["AVERAGE"] - median_values).abs()
    
    df = (
        df.loc[df.groupby("NSC")["_distance"].idxmin()]
        .drop(columns="_distance")
        .reset_index(drop=True)
    )
    
    # Calculate pGI50 and assign activity labels
    df["pGI50"] = -df["AVERAGE"]
    df["Comment"] = np.where(
        df["pGI50"] > activity_threshold, "active", "inactive"
    )
    
    return df


def standardize_smiles(smiles):
    """Standardize SMILES string and strip salt ions."""
    if pd.isna(smiles):
        return None
        
    try:
        smiles = str(smiles).strip()
        
        # Filter out invalid or placeholder SMILES
        if not smiles or "[R]" in smiles or "[ZH]" in smiles:
            return None
            
        # Parse molecule and strip salts
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
            
        mol = _remover.StripMol(mol)
        if mol is None or mol.GetNumAtoms() == 0:
            return None
            
        # Return canonical SMILES
        return Chem.MolToSmiles(mol, canonical=True)
        
    except Exception:
        return None


def remove_dataset_overlap(reference_df, target_df):
    """Remove molecules from target dataset that already exist in the reference dataset."""
    reference = reference_df.copy()
    target = target_df.copy()
    
    # Standardize SMILES
    reference["_canonical"] = reference["Smiles"].apply(standardize_smiles)
    target["_canonical"] = target["SMILES"].apply(standardize_smiles)
    
    # Drop molecules that failed standardization
    reference = reference.dropna(subset=["_canonical"])
    target = target.dropna(subset=["_canonical"])
    
    # Filter out molecules present in the reference set
    target = target[
        ~target["_canonical"].isin(set(reference["_canonical"]))
    ].copy()
    
    # Clean up temporary column, rename, and reset index
    target = (
        target.drop(columns="_canonical")
        .rename(columns={"SMILES": "Smiles"})
        .reset_index(drop=True)
    )
    
    return target


def process_chembl_data(df):
    """Clean ChEMBL data, extract GI50 metrics, and unify units."""
    # Initial filtering and separation of GI50 vs non-GI50 data
    data = df[
        df["Comment"].astype(str).str.lower().isin(["active", "inactive"])
    ].copy()
    
    gi50 = data[data["Standard Type"] == "GI50"].copy()
    non_gi50 = data[data["Standard Type"] != "GI50"].copy()
    
    # Strict filtering for GI50 data
    gi50["Standard Value"] = pd.to_numeric(gi50["Standard Value"], errors="coerce")
    gi50 = gi50.dropna(subset=["Standard Value"])
    gi50 = gi50[gi50["Standard Value"] >= 0]
    
    # Keep only exact match relations (or missing/empty)
    gi50 = gi50[
        (gi50["Standard Relation"] == "'='") | 
        gi50["Standard Relation"].isna() | 
        (gi50["Standard Relation"] == "")
    ]
    
    # Filter by validity comment and SMILES
    gi50 = gi50[gi50["Data Validity Comment"] != "Outside typical range"]
    gi50 = gi50[gi50["Smiles"].notna() & (gi50["Smiles"] != "")].copy()
    
    # Convert units to mol/L
    gi50["Standard Value (mol/L)"] = np.nan
    
    # Handle nM units
    mask = gi50["Standard Units"] == "nM"
    gi50.loc[mask, "Standard Value (mol/L)"] = (
        gi50.loc[mask, "Standard Value"] * 1e-9
    )
    
    # Handle ug.mL-1 units (requires molecular weight)
    if "Molecular Weight" in gi50.columns:
        mask = (
            (gi50["Standard Units"] == "ug.mL-1") & 
            gi50["Molecular Weight"].notna()
        )
        gi50.loc[mask, "Standard Value (mol/L)"] = (
            gi50.loc[mask, "Standard Value"] / 
            (gi50.loc[mask, "Molecular Weight"] * 1e6)
        )
        
    gi50 = gi50.dropna(subset=["Standard Value (mol/L)"])
    
    # Calculate pGI50 and merge datasets
    gi50["pGI50"] = -np.log10(gi50["Standard Value (mol/L)"])
    
    non_gi50["Standard Value (mol/L)"] = np.nan
    non_gi50["pGI50"] = np.nan
    
    return pd.concat([gi50, non_gi50], ignore_index=True)


def deduplicate_chembl_data(df):
    """Resolve label conflicts and deduplicate molecules in ChEMBL data."""
    data = df.copy()
    
    # Resolve qualitative label conflicts (both active and inactive present)
    comment_types = data.groupby("Molecule ChEMBL ID")["Comment"].apply(
        lambda x: set(x.astype(str).str.lower())
    )
    abnormal_ids = comment_types[comment_types.apply(len) > 1].index
    data = data[~data["Molecule ChEMBL ID"].isin(abnormal_ids)].copy()
    
    # Separate molecules with and without quantitative data (pGI50)
    has_pgi50 = data[data["pGI50"].notna()].copy()
    no_pgi50 = data[data["pGI50"].isna()].copy()

    # Take median pGI50 per molecule ID and deduplicate
    if len(has_pgi50) > 0:
        medians = has_pgi50.groupby("Molecule ChEMBL ID")["pGI50"].median()
        has_pgi50 = has_pgi50.drop_duplicates("Molecule ChEMBL ID").copy()
        has_pgi50["pGI50"] = has_pgi50["Molecule ChEMBL ID"].map(medians)

    # Deduplicate non-pGI50 rows and exclude IDs that conflict with pGI50 rows
    no_pgi50 = no_pgi50.drop_duplicates("Molecule ChEMBL ID")
    conflict_ids = (
        set(has_pgi50["Molecule ChEMBL ID"]) & 
        set(no_pgi50["Molecule ChEMBL ID"])
    )
    no_pgi50 = no_pgi50[~no_pgi50["Molecule ChEMBL ID"].isin(conflict_ids)]
    
    # Merge both subsets
    return pd.concat([has_pgi50, no_pgi50], ignore_index=True)