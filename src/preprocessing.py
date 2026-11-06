from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize

_REMOVER = SaltRemover.SaltRemover()
_FRAGMENT = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)


@dataclass(frozen=True)
class NCIConfig:
    log_tolerance: float = 5e-4  # NCI values are often reported to 4 decimals
    strict_high_concentration: bool = True
    flag_lower_boundary: bool = True
    structure_conflicts: Literal["raise", "first"] = "raise"
    minimum_replicates: int = 1
    lower_concentration_col: str = "LOG_LO_CONCENTRATION"

    def __post_init__(self):
        if not np.isfinite(self.log_tolerance) or self.log_tolerance < 0 or self.log_tolerance > 0.1:
            raise ValueError("log_tolerance must be in [0, 0.1]")
        if self.minimum_replicates < 1:
            raise ValueError("minimum_replicates must be >=1")


@dataclass(frozen=True)
class ChEMBLConfig:
    allow_missing_relation_as_exact: bool = True  # preserves original behavior
    exclude_invalidity_comment: bool = True


@dataclass
class Audit:
    """Structured audit. Every NCI input row receives exactly one decision."""
    summary: dict
    row_ledger: pd.DataFrame

    def write(self, prefix: Path | str) -> None:
        prefix = Path(prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        self.row_ledger.to_csv(f"{prefix}_rows.csv", index=False)
        with open(f"{prefix}_summary.json", "w", encoding="utf-8") as f:
            json.dump(self.summary, f, ensure_ascii=False, indent=2, allow_nan=False)


def _need_columns(data: pd.DataFrame, names, label: str) -> None:
    missing = sorted(set(names).difference(data.columns))
    if missing:
        raise KeyError(f"{label} missing columns: {missing}")


def _canonical_nsc(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Input files sometimes read NSC as integers in one file and strings in another.
    out["NSC"] = out["NSC"].astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    return out


def standardize_smiles(smiles) -> Optional[str]:
    """Deterministic parent-fragment SMILES canonicalization."""
    if pd.isna(smiles):
        return None
    s = str(smiles).strip()
    if not s or any(mark in s for mark in ("[R]", "[ZH]")):
        return None
    try:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return None
        mol = _REMOVER.StripMol(mol, dontRemoveEverything=True)
        if mol is None or not mol.GetNumAtoms():
            return None
        mol = _FRAGMENT.choose(mol)
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def _attach_structures(activities: pd.DataFrame, structures: pd.DataFrame,
                       config: NCIConfig) -> pd.DataFrame:
    _need_columns(structures, ["NSC", "SMILES"], "structure_df")
    structures = _canonical_nsc(structures[["NSC", "SMILES"]])
    structures = structures.copy()
    structures["_canon"] = structures["SMILES"].map(standardize_smiles)
    structures = structures.dropna(subset=["_canon"])
    conflicts = structures.groupby("NSC", dropna=False)["_canon"].nunique()
    bad = conflicts[conflicts > 1]
    if not bad.empty and config.structure_conflicts == "raise":
        raise ValueError(f"{len(bad)} NSC identifiers map to different structures; example: {bad.index[:5].tolist()}")
    structures = structures.sort_values(["NSC", "_canon"], kind="stable").drop_duplicates("NSC")
    merged = activities.merge(structures[["NSC", "_canon"]].rename(columns={"_canon": "SMILES"}),
                              on="NSC", how="left", validate="many_to_one", sort=False)
    return merged


def _join_empirical_lower_bounds(activities: pd.DataFrame,
                                  dose_response_df: Optional[pd.DataFrame]) -> pd.Series:
    """Resolve the actual minimum tested concentration for each experimental series.

    Avoids assuming five ten-fold dilutions for all compounds: some assays use
    a different dilution grid. All available common keys are used to match runs.
    """
    if dose_response_df is None:
        return pd.Series(np.nan, index=activities.index, dtype=float)
    dr = dose_response_df.copy()
    _need_columns(dr, ["NSC", "CONCENTRATION"], "dose_response_df")
    dr = _canonical_nsc(dr)
    keys = [k for k in ("EXPID", "NSC", "CELL_NAME", "PANEL_NUMBER", "CELL_NUMBER",
                        "CONCENTRATION_UNIT", "LOG_HI_CONCENTRATION") if k in activities.columns and k in dr.columns]
    if "NSC" not in keys or ("EXPID" not in keys and "LOG_HI_CONCENTRATION" not in keys):
        raise ValueError("Dose-response rows need matching NSC and EXPID or LOG_HI_CONCENTRATION")
    dr["CONCENTRATION"] = pd.to_numeric(dr["CONCENTRATION"], errors="coerce")
    dr = dr.dropna(subset=["CONCENTRATION"])
    low = dr.groupby(keys, dropna=False, sort=False)["CONCENTRATION"].min().rename("_empirical_low").reset_index()
    left = activities.reset_index().rename(columns={"index": "_idx"})
    merged = left[["_idx", *keys]].merge(low, on=keys, how="left", validate="many_to_one", sort=False)
    return pd.Series(merged["_empirical_low"].to_numpy(dtype=float), index=activities.index)


def _row_decisions(data: pd.DataFrame, cfg: NCIConfig,
                   dose_response_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    out = data.copy()
    out["AVERAGE"] = pd.to_numeric(out["AVERAGE"], errors="coerce")
    out["LOG_HI_CONCENTRATION"] = pd.to_numeric(out["LOG_HI_CONCENTRATION"], errors="coerce")
    low = pd.to_numeric(out[cfg.lower_concentration_col], errors="coerce") if cfg.lower_concentration_col in out else pd.Series(np.nan, index=out.index)
    lower_from_dose = _join_empirical_lower_bounds(out, dose_response_df)
    out["_lower_logM"] = low.where(low.notna(), lower_from_dose)
    hi = out["LOG_HI_CONCENTRATION"]
    val = out["AVERAGE"]
    lo = out["_lower_logM"]
    tol = cfg.log_tolerance
    conditions = [
        out["SMILES"].isna(),
        val.isna(),
        hi.isna(),
        (lo.notna()) & (lo > hi + tol),
        val > hi + tol,
        (val - hi).abs() <= tol,
        (lo.notna()) & (val < lo - tol),
        (lo.notna()) & ((val - lo).abs() <= tol) if cfg.flag_lower_boundary else pd.Series(False, index=out.index),
    ]
    labels = ["invalid_structure", "invalid_activity", "missing_assay_limit",
              "invalid_concentration_series", "above_assay_range", "high_boundary_censored",
              "below_assay_range", "low_boundary_censored"]
    out["censor_status"] = np.select(conditions, labels, default="exact")
    return out


def process_nci_data(activity_df: pd.DataFrame,
                     structure_df: pd.DataFrame,
                     cell_name: str,
                     activity_threshold: float,
                     *,
                     dose_response_df: Optional[pd.DataFrame] = None,
                     config: NCIConfig = NCIConfig(),
                     return_audit: bool = False):
    """Apply row-level censoring policy BEFORE collapsing repeated NSC assays.

    By default, only uncensored measured endpoints are used for regression.
    If return_audit=True, returns (processed_frame, Audit) instead of frame.
    """
    _need_columns(activity_df, ["NSC", "CELL_NAME", "CONCENTRATION_UNIT", "AVERAGE"], "activity_df")
    if "LOG_HI_CONCENTRATION" not in activity_df:
        if config.strict_high_concentration:
            raise KeyError("LOG_HI_CONCENTRATION unavailable; cannot audit per-experiment high-boundary censoring")
        activity_df = activity_df.assign(LOG_HI_CONCENTRATION=np.nan)

    original = _canonical_nsc(activity_df)
    data = original.loc[original["CELL_NAME"].astype(str).str.strip().eq(cell_name)].copy().reset_index(drop=True)
    data["_source_row"] = np.arange(len(data))
    unit = data["CONCENTRATION_UNIT"].astype("string").str.strip().str.upper()
    data = _attach_structures(data, structure_df, config)
    data = _row_decisions(data, config, dose_response_df)
    # Unit checking has priority over concentration-boundary diagnostics.
    data.loc[~unit.eq("M").fillna(False), "censor_status"] = "non_molar_unit"
    data["pGI50"] = -data["AVERAGE"]

    exact = data.loc[data["censor_status"].eq("exact")].copy()
    multiplicity = exact.groupby("NSC")["NSC"].transform("size")
    exact.loc[multiplicity < config.minimum_replicates, "censor_status"] = "insufficient_replicates"
    valid = exact.loc[exact["censor_status"].eq("exact")].copy()

    # Select the actual observation closest to the group median, preserving
    # per-experiment structure and concentration information. Deterministic ties.
    if not valid.empty:
        median = valid.groupby("NSC")["AVERAGE"].transform("median")
        valid["_distance"] = (valid["AVERAGE"] - median).abs()
        valid = valid.sort_values(["NSC", "_distance", "_source_row"], kind="stable")
        chosen = valid.drop_duplicates("NSC", keep="first").drop(columns=["_distance"])
        chosen["Comment"] = np.where(chosen["pGI50"] > float(activity_threshold), "active", "inactive")
    else:
        chosen = valid.assign(Comment=pd.Series(dtype="object"))

    chosen_ids = set(chosen["_source_row"].tolist())
    valid_ids = set(valid["_source_row"].tolist())
    ledger = data.copy()
    insufficient_ids = set(exact.loc[exact["censor_status"].eq("insufficient_replicates"), "_source_row"])
    if insufficient_ids:
        ledger.loc[ledger["_source_row"].isin(insufficient_ids), "censor_status"] = "insufficient_replicates"
    redundant_ids = valid_ids - chosen_ids
    ledger.loc[ledger["_source_row"].isin(redundant_ids), "censor_status"] = "valid_replicate_not_selected"
    ledger.loc[ledger["_source_row"].isin(chosen_ids), "censor_status"] = "retained"

    summary = {
        "cell_line": cell_name,
        "input_rows_for_cell": int(len(data)),
        "retained_molecules": int(len(chosen)),
        "high_limit_available_rows": int(data["LOG_HI_CONCENTRATION"].notna().sum()),
        "low_limit_available_rows": int(data["_lower_logM"].notna().sum()),
        "log_tolerance": float(config.log_tolerance),
        "status_counts": {k: int(v) for k, v in ledger["censor_status"].value_counts(dropna=False).items()},
        "censoring_policy": "exclude_boundary_and_out_of_range_prior_to_replicate_selection",
    }
    audit_cols = [c for c in ("_source_row", "EXPID", "NSC", "CELL_NAME", "CONCENTRATION_UNIT",
                              "LOG_HI_CONCENTRATION", "_lower_logM", "AVERAGE", "pGI50",
                              "SMILES", "censor_status") if c in ledger]
    audit = Audit(summary=summary, row_ledger=ledger[audit_cols].sort_values("_source_row"))
    chosen = chosen.drop(columns=[c for c in ("_source_row", "_lower_logM", "censor_status") if c in chosen])
    chosen = chosen.reset_index(drop=True)
    return (chosen, audit) if return_audit else chosen


def remove_dataset_overlap(reference_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    """Remove identical canonical parent structures from the target set."""
    _need_columns(reference_df, ["Smiles"], "reference_df")
    _need_columns(target_df, ["SMILES"], "target_df")
    ref = reference_df.copy()
    target = target_df.copy()
    ref["_canonical"] = ref["Smiles"].map(standardize_smiles)
    target["_canonical"] = target["SMILES"].map(standardize_smiles)
    ref = ref.dropna(subset=["_canonical"])
    target = target.dropna(subset=["_canonical"])
    target = target.loc[~target["_canonical"].isin(set(ref["_canonical"]))].copy()
    return target.drop(columns=["_canonical"]).rename(columns={"SMILES": "Smiles"}).reset_index(drop=True)


def _normalize_relation(s: pd.Series) -> pd.Series:
    return (s.astype("string").str.strip().str.strip("\"'").str.replace("≤", "<=", regex=False)
            .str.replace("≥", ">=", regex=False))


def _mol_units(values: pd.Series, units: pd.Series, weights: pd.Series) -> pd.Series:
    """Mass concentration ug/mL -> mol/L is value/(MW * 1000)."""
    unit = units.astype("string").str.strip().str.lower().str.replace("µ", "u").str.replace("μ", "u")
    factors = {"m": 1., "mm": 1e-3, "um": 1e-6, "nm": 1e-9, "pm": 1e-12}
    out = pd.Series(np.nan, index=values.index, dtype=float)
    for name, factor in factors.items():
        m = unit.eq(name).fillna(False)
        out.loc[m] = values.loc[m] * factor
    mass = unit.isin(("ug.ml-1", "ug/ml", "ug ml-1", "ug ml^-1", "ug.ml^-1")) & weights.gt(0)
    out.loc[mass] = values.loc[mass] / (weights.loc[mass] * 1000.)
    return out


def process_chembl_data(df: pd.DataFrame,
                        *, config: ChEMBLConfig = ChEMBLConfig(),
                        return_audit: bool = False):
    """Retain exact quantitative GI50 values; keep qualitative non-GI50 rows separately."""
    required = ["Comment", "Standard Type", "Standard Value", "Standard Units",
                "Standard Relation", "Smiles"]
    _need_columns(df, required, "ChEMBL df")
    src = df.copy()
    src["Comment"] = src["Comment"].astype("string").str.strip().str.lower()
    src = src.loc[src["Comment"].isin(["active", "inactive"])].copy()
    gi50 = src.loc[src["Standard Type"].astype("string").str.upper().str.strip().eq("GI50")].copy()
    other = src.loc[~src.index.isin(gi50.index)].copy()

    relation = _normalize_relation(gi50["Standard Relation"])
    is_equal = relation.eq("=")
    if config.allow_missing_relation_as_exact:
        is_equal |= relation.isna() | relation.eq("")
    is_equal = is_equal.fillna(False)
    gi50["Standard Value"] = pd.to_numeric(gi50["Standard Value"], errors="coerce")
    weights = (pd.to_numeric(gi50["Molecular Weight"], errors="coerce")
               if "Molecular Weight" in gi50 else pd.Series(np.nan, index=gi50.index))
    converted = _mol_units(gi50["Standard Value"], gi50["Standard Units"], weights)
    valid = is_equal & converted.gt(0) & np.isfinite(converted)
    if config.exclude_invalidity_comment and "Data Validity Comment" in gi50:
        valid &= ~gi50["Data Validity Comment"].astype("string").str.contains("outside typical range", case=False, na=False)
    valid &= gi50["Smiles"].astype("string").str.strip().ne("").fillna(False) & gi50["Smiles"].notna()

    counts = {"input_quantitative_GI50": int(len(gi50)), "exact_GI50_retained": int(valid.sum()),
              "non_exact_or_unusable_GI50": int((~valid).sum()),
              "non_GI50_qualitative_rows": int(len(other))}
    gi50 = gi50.loc[valid].copy()
    gi50["Standard Value (mol/L)"] = converted.loc[valid]
    gi50["pGI50"] = -np.log10(gi50["Standard Value (mol/L)"])
    other["Standard Value (mol/L)"] = np.nan
    other["pGI50"] = np.nan
    out = pd.concat([gi50, other], ignore_index=True)
    return (out, counts) if return_audit else out


def deduplicate_chembl_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preserve original ID-level qualitative conflict policy and median aggregation."""
    _need_columns(df, ["Molecule ChEMBL ID", "Comment", "pGI50"], "ChEMBL dedup input")
    data = df.copy()
    comment_types = data.groupby("Molecule ChEMBL ID")["Comment"].apply(
        lambda x: set(x.dropna().astype(str).str.lower()))
    invalid_ids = comment_types[comment_types.apply(len) > 1].index
    data = data.loc[~data["Molecule ChEMBL ID"].isin(invalid_ids)]
    measured = data.loc[data["pGI50"].notna()].copy()
    categorical = data.loc[data["pGI50"].isna()].copy()
    if not measured.empty:
        medians = measured.groupby("Molecule ChEMBL ID")["pGI50"].median()
        measured = measured.drop_duplicates("Molecule ChEMBL ID").copy()
        measured["pGI50"] = measured["Molecule ChEMBL ID"].map(medians)
    categorical = categorical.drop_duplicates("Molecule ChEMBL ID")
    categorical = categorical.loc[~categorical["Molecule ChEMBL ID"].isin(set(measured["Molecule ChEMBL ID"]))]
    return pd.concat([measured, categorical], ignore_index=True)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Auditable NCI-60 censor-aware preprocessing")
    parser.add_argument("--activity", required=True, help="DTP GI50.csv")
    parser.add_argument("--structures", required=True, help="NSC to SMILES CSV")
    parser.add_argument("--dose-response", help="Optional matched DTP DOSERESP file (lower concentration)")
    parser.add_argument("--cell-line", default="MDA-MB-231")
    parser.add_argument("--threshold", type=float, default=6.0)
    parser.add_argument("--tolerance", type=float, default=5e-4)
    parser.add_argument("--min-replicates", type=int, default=1)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args(argv)
    cfg = NCIConfig(log_tolerance=args.tolerance, minimum_replicates=args.min_replicates)
    activity = pd.read_csv(args.activity, low_memory=False)
    structures = pd.read_csv(args.structures, low_memory=False)
    dose = pd.read_csv(args.dose_response, low_memory=False) if args.dose_response else None
    cleaned, audit = process_nci_data(activity, structures, args.cell_line,
                                      args.threshold, dose_response_df=dose,
                                      config=cfg, return_audit=True)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(outdir / "nci_processed.csv", index=False)
    audit.write(outdir / "nci_censor_audit")
    provenance = {"configuration": asdict(cfg), "input_sha256": {
        "activity": _sha256(args.activity), "structures": _sha256(args.structures),
        "dose_response": _sha256(args.dose_response) if args.dose_response else None}}
    (outdir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps(audit.summary, indent=2))


if __name__ == "__main__":
    main()
