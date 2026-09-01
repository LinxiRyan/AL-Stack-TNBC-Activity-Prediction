import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression


def balanced_mi(X, y, activity_threshold, n_rounds, random_seed):
    """
    Calculate averaged Mutual Information (MI) using balanced undersampling 
    to handle class imbalance between active and inactive compounds.
    """
    pos_idx = np.where(y >= activity_threshold)[0]
    neg_idx = np.where(y < activity_threshold)[0]
    
    rng = np.random.RandomState(random_seed)
    mi_sum = np.zeros(X.shape[1], dtype=float)
    neg_counts = np.zeros(len(neg_idx), dtype=int)

    for r in range(n_rounds):
        # Calculate sampling weights to ensure uniform coverage of negatives over rounds
        weights = 1.0 / (neg_counts + 1)
        probs = weights / weights.sum()
        
        # Sample negative instances to balance the dataset
        sample_size = min(len(pos_idx), len(neg_idx))
        chosen = rng.choice(
            np.arange(len(neg_idx)), 
            size=sample_size, 
            replace=False, 
            p=probs
        )
        neg_counts[chosen] += 1
        
        # Combine positive and sampled negative indices
        idx = np.concatenate([pos_idx, neg_idx[chosen]])
        
        # Compute MI for this round and accumulate
        mi_sum += mutual_info_regression(
            X[idx], y[idx], discrete_features=True, random_state=r
        )

    return mi_sum / n_rounds


def variance_filter(X, feature_names, variance_threshold):
    """Remove features with variance below the specified threshold."""
    variances = X.var(axis=0)
    mask = variances > variance_threshold
    
    return X[:, mask], [f for f, keep in zip(feature_names, mask) if keep], variances[mask]


def jaccard_filter(X, feature_names, mi_scores, jaccard_threshold):
    """
    Remove redundant features based on Jaccard similarity. 
    Features are evaluated in descending order of their MI scores.
    """
    order = np.argsort(mi_scores)[::-1]
    
    # Binarize the feature matrix (assuming fingerprint bits are 0 or 1)
    X_binary = (X > 0).astype(np.uint8)
    selected = []

    for idx in order:
        if not selected:
            selected.append(idx)
            continue
            
        current = X_binary[:, idx]
        keep = True
        
        # Check similarity against all already selected features
        for selected_idx in selected:
            reference = X_binary[:, selected_idx]
            union = np.sum(current | reference)
            similarity = np.sum(current & reference) / union if union else 0.0
            
            if similarity >= jaccard_threshold:
                keep = False
                break
                
        if keep:
            selected.append(idx)

    return X[:, selected], [feature_names[i] for i in selected]


def mi_filter(X, feature_names, y, activity_threshold, mi_threshold, n_rounds, random_seed):
    """Filter features based on a Mutual Information threshold."""
    scores = balanced_mi(X, y, activity_threshold, n_rounds, random_seed)
    mask = scores >= mi_threshold
    
    return [f for f, keep in zip(feature_names, mask) if keep], scores


def select_features(
    train_df, target_col, fingerprint_cols, descriptor_cols, 
    variance_threshold, jaccard_threshold, mi_threshold, 
    activity_threshold, n_rounds, random_seed
):
    """
    Main feature selection pipeline. 
    Applies variance, Jaccard, and MI filters sequentially on fingerprints.
    """
    X = train_df[fingerprint_cols].to_numpy()
    y = train_df[target_col].to_numpy()

    # 1. Variance filter: remove constant or near-constant features
    X, variance_features, _ = variance_filter(X, fingerprint_cols, variance_threshold)
    
    # 2. Preliminary MI calculation for ordering in Jaccard filter
    preliminary_mi = balanced_mi(X, y, activity_threshold, n_rounds, random_seed)
    
    # 3. Jaccard filter: remove highly correlated/redundant fingerprints
    X, correlation_features = jaccard_filter(
        X, variance_features, preliminary_mi, jaccard_threshold
    )
    
    # 4. Final MI filter: keep only features with sufficient predictive power
    selected_fingerprints, final_mi = mi_filter(
        X, correlation_features, y, activity_threshold, mi_threshold, n_rounds, random_seed
    )

    # Combine selected fingerprints with all descriptor columns
    selected_features = selected_fingerprints + list(descriptor_cols)
    
    return selected_features, selected_fingerprints, final_mi


def apply_selected_features(df, selected_features, target_col=None, meta_cols=None):
    """Subset the dataframe to keep only metadata, selected features, and the target."""
    cols = list(meta_cols or []) + list(selected_features)
    
    if target_col is not None and target_col in df.columns:
        cols.append(target_col)
        
    return df[cols].copy()