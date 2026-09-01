import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Avalon import pyAvalonTools


def standardize_mol(mol):
    """Standardize a molecule using RDKit's MolStandardize pipeline."""
    if mol is None:
        return None
        
    try:
        # Basic cleanup (remove fragments, standardize isotopes, etc.)
        mol = rdMolStandardize.Cleanup(mol)
        # Keep only the largest fragment (remove salts/solvents)
        mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
        # Neutralize the molecule
        mol = rdMolStandardize.Uncharger().uncharge(mol)
        # Canonicalize tautomers
        mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
        
        return mol
    except Exception:
        return None


def prepare_mol(smiles):
    """Parse a SMILES string and return a standardized RDKit molecule."""
    try:
        mol = Chem.MolFromSmiles(str(smiles).strip())
        return standardize_mol(mol) if mol is not None else None
    except Exception:
        return None


def bitvect_to_array(fp):
    """Convert an RDKit ExplicitBitVect to a numpy array of uint8."""
    arr = np.zeros(fp.GetNumBits(), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def compute_fingerprints(mol, morgan_radius, morgan_bits, avalon_bits):
    """Compute Morgan, MACCS, and Avalon fingerprints for a molecule."""
    # 1. Morgan (ECFP) fingerprint
    morgan = AllChem.GetMorganFingerprintAsBitVect(
        mol, morgan_radius, nBits=morgan_bits
    )
    
    # 2. MACCS keys (167 bits total)
    maccs = MACCSkeys.GenMACCSKeys(mol)
    
    # 3. Avalon fingerprint
    avalon = pyAvalonTools.GetAvalonFP(mol, avalon_bits)
    
    # Note: MACCS keys have an unused first bit (index 0), so we drop it using [1:]
    return (
        bitvect_to_array(morgan), 
        bitvect_to_array(maccs)[1:], 
        bitvect_to_array(avalon)
    )


def calculate_fingerprints(df, smiles_col, morgan_radius, morgan_bits, avalon_bits):
    """Calculate fingerprints for all valid molecules in a DataFrame."""
    rows = []
    
    for idx, smiles in df[smiles_col].items():
        mol = prepare_mol(smiles)
        if mol is None:
            continue
            
        # Compute the three types of fingerprints
        morgan, maccs, avalon = compute_fingerprints(
            mol, morgan_radius, morgan_bits, avalon_bits
        )
        
        # Build the base row with index and canonical SMILES
        row = {
            "index": idx, 
            "canonical_smiles": Chem.MolToSmiles(mol, canonical=True)
        }
        
        # Expand fingerprint bits into individual columns (e.g., Morgan_0, MACCS_1)
        row.update({f"Morgan_{i}": v for i, v in enumerate(morgan)})
        row.update({f"MACCS_{i}": v for i, v in enumerate(maccs)})
        row.update({f"Avalon_{i}": v for i, v in enumerate(avalon)})
        
        rows.append(row)
        
    return pd.DataFrame(rows)