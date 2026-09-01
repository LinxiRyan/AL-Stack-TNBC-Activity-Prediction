import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans


def select_representative_samples(df, feature_cols, n_select, cluster_params, random_seed):
    """
    Select representative samples from a dataframe using K-Means clustering.
    Chooses the sample closest to each cluster center.
    """
    if len(df) <= n_select:
        return df.index.tolist()
        
    X = df[feature_cols].to_numpy()
    
    # Configure and fit MiniBatchKMeans
    params = dict(cluster_params)
    params["n_clusters"] = n_select
    params["random_state"] = random_seed
    model = MiniBatchKMeans(**params)
    
    labels = model.fit_predict(X)
    centers = model.cluster_centers_
    selected = []

    # Find the sample closest to the center for each cluster
    for k in range(n_select):
        local = np.where(labels == k)[0]
        if len(local) == 0:
            continue
            
        distances = np.linalg.norm(X[local] - centers[k], axis=1)
        selected.append(df.index[local[np.argmin(distances)]])

    # Return unique indices preserving order
    return list(dict.fromkeys(selected))


def build_initial_training_set(
    train_df, feature_cols, target_col, bins, reference_bin, 
    cluster_params, random_seed
):
    """
    Build an initial training set by stratifying the target variable into bins.
    Downsampling is applied to bins larger than the reference bin.
    """
    data = train_df.copy()
    
    # Determine the size of the reference bin to use as the target count
    ref_low, ref_high = reference_bin
    reference_count = (
        (data[target_col] >= ref_low) & (data[target_col] < ref_high)
    ).sum()
    
    selected_parts, remaining_parts = [], []
    covered = np.zeros(len(data), dtype=bool)

    for low, high in bins:
        mask = (data[target_col] >= low) & (data[target_col] < high)
        covered |= mask.to_numpy()
        part = data[mask].copy()

        # Keep all samples if it's the reference bin or smaller than reference count
        if (low, high) == reference_bin or len(part) <= reference_count:
            selected_parts.append(part)
            continue

        # Downsample larger bins to match the reference count
        idx = select_representative_samples(
            part, feature_cols, reference_count, cluster_params, random_seed
        )
        selected_parts.append(part.loc[idx].copy())
        remaining_parts.append(part.loc[~part.index.isin(idx)].copy())

    # Include any samples that didn't fall into the defined bins
    selected_parts.append(data.loc[~covered].copy())
    
    initial = pd.concat(selected_parts, ignore_index=True)
    remaining = (
        pd.concat(remaining_parts, ignore_index=True) 
        if remaining_parts 
        else pd.DataFrame(columns=data.columns)
    )
    
    return initial, remaining


def allocate_samples(quota_weights, total_samples, priority_order):
    """
    Allocate a total sample quota across different bins based on weights.
    Uses the largest remainder method to distribute fractional allocations.
    """
    weights = np.asarray(
        [quota_weights[b] for b in priority_order], dtype=float
    )
    weights = weights / weights.sum()
    
    raw = weights * total_samples
    allocation = {b: int(v) for b, v in zip(priority_order, np.floor(raw))}
    remaining = total_samples - sum(allocation.values())

    # Distribute remaining samples to bins with the largest fractional parts
    if remaining > 0:
        order = np.argsort(raw - np.floor(raw))[::-1]
        for i in order[:remaining]:
            allocation[priority_order[i]] += 1

    return allocation


def select_samples(
    real_pool, generated_pool, feature_cols, model, allocation, 
    priority_order, selection_modes, source_bin_map, target_col, 
    source_bin_col, uncertainty_threshold, cluster_params, random_seed
):
    """
    Select samples from real and generated pools based on allocated quotas,
    selection modes, and model uncertainty.
    """
    real_pool = real_pool.copy()
    generated_pool = generated_pool.copy()
    selected_parts = []
    carry = 0

    for bin_key in priority_order:
        low, high = bin_key
        quota = allocation.get(bin_key, 0) + carry
        mode = selection_modes[bin_key]
        
        selected_real = pd.DataFrame(columns=real_pool.columns)
        selected_generated = pd.DataFrame(columns=generated_pool.columns)

        if quota <= 0:
            carry = 0
            continue

        # 1. Select from the real pool if allowed by the mode
        if mode in ("real_only", "hybrid"):
            candidates = real_pool[
                (real_pool[target_col] >= low) & (real_pool[target_col] < high)
            ].copy()
            n = min(quota, len(candidates))

            if n > 0:
                if n == len(candidates):
                    idx = candidates.index.tolist()
                else:
                    idx = select_representative_samples(
                        candidates, feature_cols, n, cluster_params, random_seed
                    )
                selected_real = real_pool.loc[idx].copy()
                real_pool = real_pool.drop(idx)
                selected_parts.append(selected_real)

        # 2. Select from the generated pool to fill the remaining quota
        remaining_quota = quota - len(selected_real)

        if remaining_quota > 0 and mode in ("generated_only", "hybrid"):
            candidates = generated_pool[
                generated_pool[source_bin_col] == source_bin_map[bin_key]
            ].copy()

            if len(candidates) > 0:
                # Predict target and uncertainty using the ensemble model
                X_gen = candidates[feature_cols].to_numpy()
                pred, uncertainty = model.predict_with_uncertainty(X_gen)
                
                candidates["_prediction"] = pred
                candidates["_uncertainty"] = uncertainty
                
                # Filter by uncertainty threshold and predicted target bin
                mask = (
                    (candidates["_uncertainty"] <= uncertainty_threshold) & 
                    (candidates["_prediction"] >= low) & 
                    (candidates["_prediction"] < high)
                )
                candidates = candidates[mask].sort_values("_uncertainty")
                
                # Select the most confident samples up to the remaining quota
                selected_generated = candidates.head(remaining_quota).copy()

                if len(selected_generated) > 0:
                    # Assign predicted values as pseudo-labels
                    selected_generated[target_col] = selected_generated["_prediction"]
                    idx = selected_generated.index
                    generated_pool = generated_pool.drop(idx)
                    
                    selected_generated = selected_generated.drop(
                        columns=["_prediction", "_uncertainty"]
                    )
                    selected_parts.append(selected_generated)

        # 3. Calculate carry-over for the next bin if quota wasn't fully met
        total_selected = len(selected_real) + len(selected_generated)
        carry = max(0, quota - total_selected)

    selected = (
        pd.concat(selected_parts, ignore_index=True) 
        if selected_parts 
        else pd.DataFrame()
    )
    
    return selected, real_pool.reset_index(drop=True), generated_pool.reset_index(drop=True)