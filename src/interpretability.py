import numpy as np
import pandas as pd
import shap


def compute_tree_shap(model, X):
    """Compute SHAP values for a single tree-based model using TreeExplainer."""
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(np.asarray(X))
    return np.asarray(values)


def compute_base_shap(model, X):
    """
    Compute SHAP values for the CatBoost and XGBoost base models 
    inside the custom StackingEnsemble.
    """
    X = np.asarray(X)
    
    # Extract SHAP values from the underlying base models
    cat_values = compute_tree_shap(model.cat, X)
    xgb_values = compute_tree_shap(model.xgb, X)
    
    return {"CatBoost": cat_values, "XGBoost": xgb_values}


def global_importance(shap_values, feature_names):
    """Calculate global feature importance based on mean absolute SHAP values."""
    # Mean of absolute SHAP values across all samples
    values = np.abs(np.asarray(shap_values)).mean(axis=0)
    
    df = pd.DataFrame({
        "Feature": feature_names, 
        "Importance": values
    })
    
    return df.sort_values("Importance", ascending=False).reset_index(drop=True)


def ensemble_importance(shap_results, feature_names, model_weights):
    """Calculate weighted ensemble feature importance from base model SHAP values."""
    # 1. Calculate mean absolute SHAP values for each base model
    cat_imp = np.abs(shap_results["CatBoost"]).mean(axis=0)
    xgb_imp = np.abs(shap_results["XGBoost"]).mean(axis=0)
    
    # 2. Compute weighted average based on provided model weights
    weights = np.asarray(
        [model_weights["CatBoost"], model_weights["XGBoost"]], 
        dtype=float
    )
    importance = np.average(
        np.vstack([cat_imp, xgb_imp]), 
        axis=0, 
        weights=weights
    )
    
    # 3. Build and sort the resulting DataFrame
    df = pd.DataFrame({
        "Feature": feature_names,
        "CatBoost_Importance": cat_imp,
        "XGBoost_Importance": xgb_imp,
        "Ensemble_Importance": importance
    })
    
    return df.sort_values("Ensemble_Importance", ascending=False).reset_index(drop=True)


def importance_by_bins(shap_results, y, feature_names, bins, model_weights):
    """Calculate feature importance stratified by target value bins."""
    y = np.asarray(y)
    rows = []
    
    # Prepare weights for ensemble calculation
    weights = np.asarray(
        [model_weights["CatBoost"], model_weights["XGBoost"]], 
        dtype=float
    )

    for low, high in bins:
        mask = (y >= low) & (y < high)
        
        # Skip empty bins to avoid errors
        if not np.any(mask):
            continue

        # Calculate mean absolute SHAP values for the current bin
        cat_imp = np.abs(shap_results["CatBoost"][mask]).mean(axis=0)
        xgb_imp = np.abs(shap_results["XGBoost"][mask]).mean(axis=0)
        ensemble_imp = np.average(
            np.vstack([cat_imp, xgb_imp]), 
            axis=0, 
            weights=weights
        )

        # Append results for each feature in this specific bin
        for feature, cat_value, xgb_value, ensemble_value in zip(
            feature_names, cat_imp, xgb_imp, ensemble_imp
        ):
            rows.append({
                "lower": low, 
                "upper": high, 
                "Feature": feature,
                "CatBoost_Importance": cat_value,
                "XGBoost_Importance": xgb_value,
                "Ensemble_Importance": ensemble_value
            })

    return pd.DataFrame(rows)


def shap_value_table(shap_values, feature_names):
    """Convert a SHAP values array into a pandas DataFrame with feature names as columns."""
    return pd.DataFrame(np.asarray(shap_values), columns=feature_names)