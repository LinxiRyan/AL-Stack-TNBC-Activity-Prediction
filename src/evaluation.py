import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def regression_metrics(y_true, y_pred):
    """Calculate standard regression metrics (R2, RMSE, MAE)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred))
    }


def evaluate_model(model, df, feature_cols, target_col):
    """Evaluate the model on the entire dataset and return overall metrics."""
    y_true = df[target_col].to_numpy()
    y_pred = model.predict(df[feature_cols].to_numpy())
    
    return regression_metrics(y_true, y_pred)


def evaluate_by_bins(model, df, feature_cols, target_col, bins):
    """
    Evaluate model performance stratified by target value bins.
    Returns a DataFrame with metrics for each bin to analyze performance across different activity ranges.
    """
    y_true = df[target_col].to_numpy()
    y_pred = model.predict(df[feature_cols].to_numpy())
    rows = []

    for low, high in bins:
        mask = (y_true >= low) & (y_true < high)

        # Skip empty bins to avoid division by zero or invalid metric calculations
        if mask.sum() == 0:
            continue

        # Calculate metrics for the current bin
        metrics = regression_metrics(y_true[mask], y_pred[mask])
        rows.append({
            "lower": low,
            "upper": high,
            "count": int(mask.sum()),
            **metrics
        })

    return pd.DataFrame(rows)


def predict_with_uncertainty(model, df, feature_cols):
    """
    Generate predictions and uncertainty estimates, appending them to a copy of the dataframe.
    Requires the model to have a `predict_with_uncertainty` method (e.g., the StackingEnsemble).
    """
    X = df[feature_cols].to_numpy()
    
    # Get both the predicted values and the uncertainty (standard deviation of base models)
    prediction, uncertainty = model.predict_with_uncertainty(X)
    
    result = df.copy()
    result["prediction"] = prediction
    result["uncertainty"] = uncertainty
    
    return result