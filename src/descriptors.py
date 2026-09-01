import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, QED
from rdkit.Chem.MolStandardize import rdMolStandardize

try:
    from rdkit.Contrib.SA_Score import sascorer
except Exception:
    import sascorer

# Define the target descriptor columns for downstream processing
DESCRIPTOR_COLUMNS = [
    "MW", "LogP", "TPSA", "HBA", "HBD", "RotBonds", 
    "AromaticRings", "FractionCsp3", "QED", "MolecularFlexibility", 
    "ESOL_logS", "SA"
]


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


def compute_descriptors(mol):
    """Compute a set of physicochemical and pharmacokinetic descriptors for a molecule."""
    # 1. Basic physicochemical properties
    mw = float(Descriptors.MolWt(mol))
    logp = float(Descriptors.MolLogP(mol))
    tpsa = float(Descriptors.TPSA(mol))
    
    # 2. Lipinski's Rule of Five components
    hba = int(Lipinski.NumHAcceptors(mol))
    hbd = int(Lipinski.NumHDonors(mol))
    rot = int(Lipinski.NumRotatableBonds(mol))
    aromatic = int(Lipinski.NumAromaticRings(mol))
    csp3 = float(Lipinski.FractionCSP3(mol))
    
    # 3. Drug-likeness and flexibility
    qed = float(QED.qed(mol))
    heavy = mol.GetNumHeavyAtoms()
    flexibility = float(rot / heavy) if heavy else 0.0
    
    # 4. ESOL (Estimated SOLubility) calculation
    aromatic_atoms = sum(atom.GetIsAromatic() for atom in mol.GetAtoms())
    aromatic_proportion = aromatic_atoms / heavy if heavy else 0.0
    esol = float(
        0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * rot - 0.74 * aromatic_proportion
    )
    
    # 5. Synthetic Accessibility (SA) score
    sa = float(sascorer.calculateScore(mol))
    
    return {
        "MW": mw, "LogP": logp, "TPSA": tpsa, "HBA": hba, "HBD": hbd, 
        "RotBonds": rot, "AromaticRings": aromatic, "FractionCsp3": csp3, 
        "QED": qed, "MolecularFlexibility": flexibility, "ESOL_logS": esol, "SA": sa
    }


def calculate_descriptors(df, smiles_col):
    """Calculate descriptors for all valid molecules in a DataFrame."""
    rows = []
    
    for idx, smiles in df[smiles_col].items():
        mol = prepare_mol(smiles)
        if mol is None:
            continue
            
        # Build the result row with index and canonical SMILES
        row = {
            "index": idx, 
            "canonical_smiles": Chem.MolToSmiles(mol, canonical=True)
        }
        row.update(compute_descriptors(mol))
        rows.append(row)
        
    return pd.DataFrame(rows)


def scale_descriptors(df, scaler):
    """Scale the descriptor columns using a provided scaler (e.g., StandardScaler)."""
    result = df.copy()
    result[DESCRIPTOR_COLUMNS] = scaler.transform(result[DESCRIPTOR_COLUMNS])
    return result