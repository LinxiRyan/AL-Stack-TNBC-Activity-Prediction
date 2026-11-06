from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

LOG = logging.getLogger("stacking-shap")
BASE_KEYS = ("cat", "xgb", "mlp")
ALIASES = {
    "cat": ("cat", "catboost", "cat_model", "catboost_model"),
    "xgb": ("xgb", "xgboost", "xgb_model", "xgboost_model"),
    "mlp": ("mlp", "nn", "mlp_model", "neural_model"),
    "ridge": ("ridge", "meta", "meta_model", "meta_learner", "final_estimator_"),
}
META_EXCLUDE = {
    "pgi50", "y", "target", "label", "smiles", "canonical_smiles", "source",
    "source_database", "split", "split_index", "dataset", "scaffold",
    "scaffold_id", "fold", "molecule_id", "id", "sample_id", "origin", "is_generated",
}
BINS = [(4.0, 5.0), (5.0, 6.0), (6.0, 7.0), (7.0, 8.0), (8.0, 9.0)]


def _find(bundle: Any, names: tuple[str, ...]):
    if isinstance(bundle, dict):
        return next((bundle[n] for n in names if n in bundle), None)
    return next((getattr(bundle, n) for n in names if hasattr(bundle, n)), None)


def _load(path: Path):
    # SECURITY: joblib/pickle files can execute code. Load only your own artifacts.
    return joblib.load(path)


def _one_dim_prediction(model: Any, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict"):
        raw = model.predict(x)
    else:
        # Optional support for a bare PyTorch regression MLP. Inputs must
        # already have exactly the same numeric preprocessing as training.
        import torch
        if not isinstance(model, torch.nn.Module):
            raise TypeError("Model requires predict(X), or must be a torch.nn.Module")
        model.eval()
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        with torch.no_grad():
            raw = model(torch.as_tensor(x.to_numpy(dtype=np.float32), device=device)).detach().cpu().numpy()
    y = np.asarray(raw, dtype=np.float64)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    if y.shape != (len(x),):
        raise ValueError(f"Model must return 1D regression predictions; got {y.shape}")
    if not np.isfinite(y).all():
        raise ValueError("Model returned non-finite predictions")
    return y


def _match_meta_name(name: str) -> str | None:
    lower = re.sub(r"[^a-z0-9]", "", name.lower())
    possibilities = {
        "cat": ("cat", "catboost"),
        "xgb": ("xgb", "xgboost"),
        "mlp": ("mlp", "neuralnetwork"),
    }
    matched = [k for k, labels in possibilities.items() if any(label in lower for label in labels)]
    return matched[0] if len(matched) == 1 else None


@dataclass
class StackingAdapter:
    models: dict[str, Any]
    ridge: Any
    order: tuple[str, str, str]
    ensemble: Any | None = None

    @classmethod
    def from_artifacts(cls, args):
        bundle = _load(Path(args.bundle)) if args.bundle else None
        paths = {k: getattr(args, f"{k}_path") for k in (*BASE_KEYS, "ridge")}
        models: dict[str, Any] = {}
        for k in BASE_KEYS:
            models[k] = _load(Path(paths[k])) if paths[k] else _find(bundle, ALIASES[k])
            if models[k] is None:
                raise ValueError(f"Missing {k} model: supply --{k}-path or include it in --bundle")
        ridge = _load(Path(paths["ridge"])) if paths["ridge"] else _find(bundle, ALIASES["ridge"])
        if ridge is None:
            raise ValueError("Missing fitted Ridge meta learner (--ridge-path)")
        coef = np.asarray(getattr(ridge, "coef_", None), dtype=float)
        if coef.shape != (3,):
            raise ValueError(f"Expected unfurled three-feature Ridge.coef_ (3,), got {coef.shape}")
        if np.asarray(getattr(ridge, "intercept_", None)).size != 1:
            raise ValueError("Expected scalar Ridge.intercept_")
        if args.meta_order:
            order = tuple(s.strip().lower() for s in args.meta_order.split(","))
        elif hasattr(ridge, "feature_names_in_"):
            order = tuple(_match_meta_name(str(s)) for s in ridge.feature_names_in_)
        else:
            raise ValueError("Ridge input order is unknown: specify --meta-order xgb,cat,mlp (or the ACTUAL order)")
        if len(order) != 3 or set(order) != set(BASE_KEYS):
            raise ValueError(f"Invalid/ambiguous Ridge input order {order}; must be a permutation of {BASE_KEYS}")
        if args.meta_order and hasattr(ridge, "feature_names_in_"):
            recorded = tuple(_match_meta_name(str(s)) for s in ridge.feature_names_in_)
            if None not in recorded and order != recorded:
                raise ValueError(f"--meta-order={order} contradicts Ridge.feature_names_in_={recorded}")
        return cls(models, ridge, order, bundle if hasattr(bundle, "predict") else None)

    @property
    def coefficients(self) -> dict[str, float]:
        return dict(zip(self.order, np.asarray(self.ridge.coef_, dtype=float)))

    @property
    def intercept(self) -> float:
        return float(np.asarray(self.ridge.intercept_, dtype=float).item())

    def base_predictions(self, x: pd.DataFrame) -> dict[str, np.ndarray]:
        return {key: _one_dim_prediction(model, x) for key, model in self.models.items()}

    def ridge_input(self, base_preds: dict[str, np.ndarray]):
        p = np.column_stack([base_preds[key] for key in self.order])
        if hasattr(self.ridge, "feature_names_in_"):
            return pd.DataFrame(p, columns=self.ridge.feature_names_in_)
        return p

    def predictions(self, x: pd.DataFrame) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        base_preds = self.base_predictions(x)
        p = self.intercept + sum(self.coefficients[k] * base_preds[k] for k in BASE_KEYS)
        ridge_direct = np.asarray(self.ridge.predict(self.ridge_input(base_preds))).reshape(-1)
        np.testing.assert_allclose(p, ridge_direct, atol=1e-8, rtol=1e-8,
                                   err_msg="Ridge input order or fusion does not match")
        if self.ensemble is not None:
            published = _one_dim_prediction(self.ensemble, x)
            np.testing.assert_allclose(p, published, atol=1e-5, rtol=1e-6,
                                       err_msg="Fitted ensemble.predict differs: inspect preprocessing/order")
        return p, base_preds


def _names_file(path: Path) -> list[str]:
    if path.suffix.lower() == ".json":
        names = json.loads(path.read_text(encoding="utf-8"))
    elif path.suffix.lower() == ".csv":
        names = pd.read_csv(path).iloc[:, 0].astype(str).tolist()
    else:
        names = [v.strip() for v in path.read_text(encoding="utf-8").splitlines() if v.strip()]
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        raise ValueError("--feature-list must be a JSON array or one feature per line/CSV row")
    if len(names) != len(set(names)):
        raise ValueError("Feature list has duplicate names")
    return names


def _cohorts(raw: list[str]) -> list[tuple[str, Path]]:
    cohorts = []
    for arg in raw:
        if "=" not in arg:
            raise ValueError("Use --cohort TEST=/path/to/test.csv")
        name, path = arg.split("=", 1)
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError(f"Invalid cohort name {name}")
        cohorts.append((name, Path(path)))
    if len({name for name, _ in cohorts}) != len(cohorts):
        raise ValueError("Cohort names must be unique")
    return cohorts


def _read_x(path: Path, names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(path)
    missing = [n for n in names if n not in raw.columns]
    if missing:
        raise ValueError(f"{path}: missing required features: {missing[:12]}")
    x = raw.loc[:, names].apply(pd.to_numeric, errors="raise").astype(np.float64)
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError(f"{path}: missing or infinite feature entries")
    return x, raw


def _infer_names(raw: pd.DataFrame, args, adapter: StackingAdapter) -> list[str]:
    if args.feature_list:
        return _names_file(Path(args.feature_list))
    feat = [c for c in raw.columns if c.lower() not in META_EXCLUDE and
            not c.lower().startswith("unnamed:") and c != args.target_col]
    if not all(pd.api.types.is_numeric_dtype(raw[c]) for c in feat):
        raise ValueError("Non-numeric candidate features: supply explicit --feature-list")
    sizes = {int(getattr(model, "n_features_in_")) for model in adapter.models.values()
             if getattr(model, "n_features_in_", None) is not None}
    if sizes and (len(sizes) > 1 or len(feat) not in sizes):
        raise ValueError(f"Auto-inferred {len(feat)} features, model expected {sizes}; use --feature-list")
    LOG.warning("Feature order auto-inferred from CSV columns; use --feature-list for rigorous provenance")
    return feat


def _as_array(values: Any, n: int, d: int, component: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.shape == (n, d, 1):
        arr = arr[:, :, 0]
    if arr.shape != (n, d):
        raise ValueError(f"{component} SHAP should have shape {(n, d)}; obtained {arr.shape}")
    return arr


def _as_bases(bases: Any, n: int, component: str) -> np.ndarray:
    arr = np.asarray(bases, dtype=float)
    if arr.size == 1:
        return np.full(n, float(arr.reshape(-1)[0]))
    if arr.shape == (n, 1):
        arr = arr[:, 0]
    if arr.shape != (n,):
        raise ValueError(f"{component} baseline shape {arr.shape}, expected {(n,)}")
    return arr


@dataclass
class Attribution:
    phi: np.ndarray
    baseline: np.ndarray
    prediction: np.ndarray
    method: str
    max_residual: float


def _explain_one(key: str, model: Any, x: pd.DataFrame, background: pd.DataFrame,
                 args, expected: np.ndarray) -> Attribution:
    n, d = x.shape
    method = "TreeExplainer(interventional/raw)" if key != "mlp" else "PermutationExplainer(independent/identity)"
    if key != "mlp":
        try:
            exp = shap.TreeExplainer(model, data=background,
                                     feature_perturbation="interventional", model_output="raw")
            # A deliberately independent residual check follows, so no silent SHAP additivity override.
            result = exp(x, check_additivity=False)
        except Exception as exc:
            if args.tree_fallback != "permutation":
                raise RuntimeError(f"{key} TreeExplainer failed; --tree-fallback permutation is available") from exc
            LOG.warning("%s TreeExplainer failed (%r); using a documented model-agnostic fallback", key, exc)
            method = "PermutationExplainer(FALLBACK, independent/identity)"
            exp = shap.PermutationExplainer(lambda z: _one_dim_prediction(
                model, pd.DataFrame(z, columns=x.columns)),
                masker=shap.maskers.Independent(background, max_samples=len(background)),
                feature_names=list(x.columns), seed=args.seed)
            result = exp(x.to_numpy(), max_evals=(2 * d + 1) * args.permutations, silent=True)
    else:
        exp = shap.PermutationExplainer(
            lambda z: _one_dim_prediction(model, pd.DataFrame(z, columns=x.columns)),
            masker=shap.maskers.Independent(background, max_samples=len(background)),
            feature_names=list(x.columns), seed=args.seed)
        result = exp(x.to_numpy(), max_evals=(2 * d + 1) * args.permutations, silent=True)
    phi = _as_array(result.values, n, d, key)
    baseline = _as_bases(result.base_values, n, key)
    bg_expected = float(np.mean(_one_dim_prediction(model, background)))
    bg_gap = float(np.max(np.abs(baseline - bg_expected)))
    # SHAP 0.50 + CatBoost 1.2.x sometimes expose only the tree component's
    # baseline, omitting the fitted constant bias, while phi itself remains
    # additive against the empirical background. Correct ONLY when verified.
    if bg_gap > args.baseline_tol and key == "cat":
        corrected_residual = float(np.max(np.abs(bg_expected + phi.sum(axis=1) - expected)))
        if corrected_residual <= args.additivity_tol:
            LOG.warning("CatBoost SHAP base_value omitted a constant offset; "
                        "verified correction to E_background[f(x)] (gap %.3g)", bg_gap)
            baseline = np.full(n, bg_expected)
            bg_gap = 0.0
    residue = np.abs(baseline + phi.sum(axis=1) - expected)
    max_residual = float(np.max(residue))
    if max_residual > args.additivity_tol:
        raise AssertionError(f"{key} local accuracy violated: max residual {max_residual:.5g}; "
                             f"tolerance {args.additivity_tol}. Check masking/model compatibility")
    if bg_gap > args.baseline_tol:
        raise AssertionError(f"{key} SHAP background mean mismatch {bg_gap:.5g}; "
                             "base explainers must represent the same reference distribution")
    LOG.info("%s (%s): local max residual %.3g, background gap %.3g", key, method,
             max_residual, bg_gap)
    return Attribution(phi, baseline, expected, method, max_residual)


def _sample_uniform(x: pd.DataFrame, raw: pd.DataFrame, limit: int, seed: int):
    if limit == 0 or len(x) <= limit:
        return x, raw
    idx = np.sort(np.random.default_rng(seed).choice(len(x), size=limit, replace=False))
    return x.iloc[idx], raw.iloc[idx]


def _summary(phi: np.ndarray, features: list[str]) -> pd.DataFrame:
    s = np.abs(phi).mean(axis=0)
    return pd.DataFrame({
        "Feature": features,
        "MeanAbsSHAP": s,
        "MeanSignedSHAP": phi.mean(axis=0),
        "FractionPositive": (phi > 0).mean(axis=0),
        "Rank": pd.Series(s).rank(ascending=False, method="first").astype(int).to_numpy(),
    }).sort_values("Rank").reset_index(drop=True)


def _bins(phi: np.ndarray, y: np.ndarray, features: list[str]):
    rows = []
    for lo, hi in BINS:
        mask = ((y >= lo) & (y < hi)) if hi < 9 else ((y >= lo) & (y <= hi))
        if not mask.any():
            continue
        score = np.abs(phi[mask]).mean(axis=0)
        for j, feat in enumerate(features):
            rows.append({"Bin": f"[{lo:g},{hi:g}{')' if hi < 9 else ']'}", "Count": int(mask.sum()),
                         "Feature": feat, "MeanAbsSHAP": score[j],
                         "MeanSignedSHAP": float(phi[mask, j].mean())})
    return pd.DataFrame(rows)


def _pretty_name(name: str) -> str:
    name = re.sub(r"^Morgan_bit_(\d+)$", r"Morgan_\1", name)
    name = re.sub(r"^Avalon_bit_(\d+)$", r"Avalon_\1", name)
    name = re.sub(r"^MACCS_key_(\d+)$", r"MACCS_\1", name)
    if name.lower().replace("_", " ") in {"molecular flexibility", "rotatable bond fraction"}:
        return "RotatableBondFraction"  # comparison/display only; NEVER transform model input values
    return name


def _plots(phi, x, importance, prefix: Path):
    top = importance.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.8, 7))
    ax.barh(top["Feature"].map(_pretty_name), top["MeanAbsSHAP"])
    ax.set(xlabel="Mean absolute ensemble SHAP (predicted pGI50)",
           title="Ridge-weighted signed SHAP fusion")
    fig.tight_layout()
    fig.savefig(str(prefix) + "_bar.png", dpi=220)
    plt.close(fig)
    plt.figure(figsize=(9, 7))
    shap.summary_plot(phi, x, show=False, max_display=min(20, x.shape[1]))
    plt.tight_layout()
    plt.savefig(str(prefix) + "_beeswarm.png", dpi=220, bbox_inches="tight")
    plt.close("all")


def _reference_check(importance: pd.DataFrame, file: str, out: Path):
    ref = pd.read_csv(file)
    if not {"Feature", "MeanAbsSHAP"}.issubset(ref):
        raise ValueError("Reference CSV requires Feature,MeanAbsSHAP columns")
    current = importance.copy()
    current["NormalizedFeature"] = current.Feature.map(_pretty_name)
    ref["NormalizedFeature"] = ref.Feature.map(_pretty_name)
    matched = ref.merge(current[["NormalizedFeature", "MeanAbsSHAP", "Rank"]],
                        on="NormalizedFeature", how="left", suffixes=("_SI", "_recomputed"))
    matched["AbsoluteDifference"] = (matched.MeanAbsSHAP_SI - matched.MeanAbsSHAP_recomputed).abs()
    matched.to_csv(out / "si_table_s20_comparison.csv", index=False)


def _metadata_versions():
    import sklearn, xgboost, catboost, scipy
    return {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "shap": shap.__version__,
            "sklearn": sklearn.__version__, "xgboost": xgboost.__version__,
            "catboost": catboost.__version__, "scipy": scipy.__version__}


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    adapter = StackingAdapter.from_artifacts(args)
    cohorts = _cohorts(args.cohort)
    background_raw = pd.read_csv(args.background_csv)
    features = _infer_names(background_raw, args, adapter)
    if len(features) != 265:
        LOG.warning("Input feature count %d (paper reports 265). Verify preprocessing provenance.", len(features))
    background, _ = _read_x(Path(args.background_csv), features)
    if len(background) > args.background_size:
        idx = np.sort(np.random.default_rng(args.seed).choice(
            len(background), size=args.background_size, replace=False))
        background = background.iloc[idx]
    adapter.predictions(background)  # test Ridge input order and optional published ensemble
    if len(background) < 2:
        raise ValueError("Background must contain >=2 rows")
    model_fingerprints = {key: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                          for key, path in ((key, getattr(args, f"{key}_path"))
                                            for key in (*BASE_KEYS, "ridge")) if path}
    if args.bundle:
        model_fingerprints["bundle"] = hashlib.sha256(Path(args.bundle).read_bytes()).hexdigest()
    meta = {"seed": args.seed, "feature_count": len(features), "features": features,
            "model_file_sha256": model_fingerprints,
            "background_csv": str(args.background_csv), "background_sample_size": len(background),
            "background_sha256": hashlib.sha256(background.to_csv(index=False).encode()).hexdigest(),
            "meta_feature_order": adapter.order, "ridge_coefficients": adapter.coefficients,
            "ridge_intercept": adapter.intercept, "permutations": args.permutations,
            "feature_perturbation": "interventional/independent; common empirical background",
            "software": _metadata_versions(), "cohorts": {}}
    background.to_csv(output / "effective_background.csv", index=False)
    full_importance = {}
    for cohort_name, input_path in cohorts:
        x, raw = _read_x(input_path, features)
        sample_seed = args.seed + int(hashlib.sha256(cohort_name.encode()).hexdigest()[:8], 16)
        x, raw = _sample_uniform(x, raw, args.max_explain, sample_seed)
        if len(x) == 0:
            raise ValueError(f"{cohort_name}: empty input")
        LOG.info("Cohort %s: explaining %d rows x %d columns", cohort_name, *x.shape)
        pred, base_preds = adapter.predictions(x)
        parts = {key: _explain_one(key, adapter.models[key], x, background, args,
                                   base_preds[key]) for key in BASE_KEYS}
        phi = sum(adapter.coefficients[k] * parts[k].phi for k in BASE_KEYS)
        base = np.full(len(x), adapter.intercept, dtype=float)
        for key in BASE_KEYS:
            base += adapter.coefficients[key] * parts[key].baseline
        residual = np.abs(base + phi.sum(axis=1) - pred)
        final_residual = float(residual.max())
        if final_residual > args.additivity_tol:
            raise AssertionError(f"Stacking local accuracy failed: {final_residual:.6g}")
        important = _summary(phi, features)
        important.to_csv(output / f"{cohort_name}_ensemble_importance.csv", index=False)
        full_importance[cohort_name] = important
        for k in BASE_KEYS:
            _summary(parts[k].phi, features).to_csv(
                output / f"{cohort_name}_{k}_importance.csv", index=False)
        np.savez_compressed(output / f"{cohort_name}_signed_shap.npz", ensemble=phi,
                            cat=parts["cat"].phi, xgb=parts["xgb"].phi,
                            mlp=parts["mlp"].phi, features=np.asarray(features),
                            base_values=base, predictions=pred, row_indices=raw.index.to_numpy())
        sample_frame = pd.DataFrame({"SourceRowIndex": raw.index.to_numpy(),
                                     "Predicted_pGI50": pred, "BaseValue": base,
                                     "ReconstructionResidual": residual})
        for k in BASE_KEYS:
            sample_frame[f"{k}_pred"] = base_preds[k]
        if args.target_col in raw:
            y = pd.to_numeric(raw[args.target_col], errors="coerce").to_numpy()
            sample_frame["Observed_pGI50"] = y
            sample_frame["AbsoluteError"] = np.abs(pred - y)
            _bins(phi, y, features).to_csv(output / f"{cohort_name}_bin_importance.csv", index=False)
        sample_frame.to_csv(output / f"{cohort_name}_predictions.csv", index=False)
        _plots(phi, x, important, output / cohort_name)
        meta["cohorts"][cohort_name] = {
            "path": str(input_path),
            "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            "n_explained": len(x),
            "max_stack_reconstruction_residual": final_residual,
            "explainers": {k: v.method for k, v in parts.items()},
            "base_max_reconstruction_residual": {k: v.max_residual for k, v in parts.items()},
        }
        LOG.info("%s: exact Ridge combination, max local error %.3g", cohort_name, final_residual)
    cohort_names = list(full_importance)
    if len(cohort_names) >= 2:
        comparisons = []
        for i, left in enumerate(cohort_names):
            for right in cohort_names[i+1:]:
                a = full_importance[left].set_index("Feature").loc[features]
                b = full_importance[right].set_index("Feature").loc[features]
                rho = spearmanr(a.MeanAbsSHAP, b.MeanAbsSHAP).statistic
                a20 = set(a.nsmallest(20, "Rank").index)
                b20 = set(b.nsmallest(20, "Rank").index)
                comparisons.append({"CohortA": left, "CohortB": right,
                                    "SpearmanImportance": float(rho),
                                    "Top20Overlap": len(a20 & b20),
                                    "Top20Jaccard": len(a20 & b20) / len(a20 | b20)})
        pd.DataFrame(comparisons).to_csv(output / "cohort_stability.csv", index=False)
    if args.si_reference:
        ref_cohort = args.reference_cohort or cohort_names[0]
        _reference_check(full_importance[ref_cohort], args.si_reference, output)
    (output / "run_manifest.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    LOG.info("Completed: %s", output)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--bundle", help="Trusted joblib containing cat/xgb/mlp/ridge or object attributes")
    for key in (*BASE_KEYS, "ridge"):
        p.add_argument(f"--{key}-path", help=f"Trusted serialized fitted {key} model")
    p.add_argument("--meta-order", help="Ridge fitting column order; e.g., xgb,cat,mlp. REQUIRED if not recorded")
    p.add_argument("--background-csv", required=True, help="REAL TRAINING molecules, preprocessed as the model input")
    p.add_argument("--cohort", action="append", required=True,
                   help="NAME=preprocessed.csv; repeat for REAL_TRAIN, TEST, AUGMENTED")
    p.add_argument("--feature-list", help="Canonical 265 feature names in model training order (txt/json/CSV)")
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument("--target-col", default="pGI50", help="Experimental endpoint for bin analysis if present")
    p.add_argument("--background-size", type=int, default=64,
                   help="Max real-training background rows, shared by ALL explainers")
    p.add_argument("--max-explain", type=int, default=256,
                   help="Rows per cohort sampled uniformly; 0 means all rows")
    p.add_argument("--permutations", type=int, default=2,
                   help="Antithetic permutation rounds; raise for better MLP SHAP precision")
    p.add_argument("--seed", type=int, default=20261103)
    p.add_argument("--additivity-tol", type=float, default=1e-3)
    p.add_argument("--baseline-tol", type=float, default=2e-3)
    p.add_argument("--tree-fallback", choices=("error", "permutation"), default="error")
    p.add_argument("--si-reference", help="Optional table S20 reference CSV to audit (NOT force-match) results")
    p.add_argument("--reference-cohort", help="Cohort to compare against SI table")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)
    if args.permutations < 1 or args.background_size < 2 or args.max_explain < 0:
        p.error("permutations >= 1, background-size >=2, max-explain >= 0 required")
    return args


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(message)s")
    run(args)
