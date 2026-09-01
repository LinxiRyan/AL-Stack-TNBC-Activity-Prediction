import copy
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sample_selection import build_initial_training_set, allocate_samples, select_samples


def run_iterative_selection(
    train_df, val_df, generated_pool, feature_cols, target_col,
    model_provider, quota_weights, bins, reference_bin, priority_order,
    selection_modes, source_bin_map, source_bin_col,
    total_samples_per_round, uncertainty_threshold, cluster_params,
    max_rounds, minimum_improvement, patience, random_seed
):
    """
    Run an iterative active learning loop to progressively build the training set.
    Selects new samples from real and generated pools based on model performance and uncertainty.
    """
    # 1. Initialize the training set and data pools
    current_train, real_pool = build_initial_training_set(
        train_df, feature_cols, target_col, bins, reference_bin,
        cluster_params, random_seed
    )
    current_generated = generated_pool.copy()
    
    X_val = val_df[feature_cols].to_numpy()
    y_val = val_df[target_col].to_numpy()

    # Tracking variables for early stopping and best model selection
    best_model = None
    best_train = None
    best_score = -np.inf
    previous_score = None
    no_improvement = 0
    history = []

    # 2. Main iterative loop
    for round_index in range(max_rounds):
        X_train = current_train[feature_cols].to_numpy()
        y_train = current_train[target_col].to_numpy()
        
        # Train model using the provided model factory/function
        model = model_provider(X_train, y_train, X_val, y_val)
        score = r2_score(y_val, model.predict(X_val))

        # Record history for this round
        history.append({
            "round": round_index,
            "train_size": len(current_train),
            "validation_r2": score
        })

        # 3. Update the best model if current validation score is higher
        if score > best_score:
            best_score = score
            best_model = copy.deepcopy(model)
            best_train = current_train.copy()

        # 4. Early stopping logic based on patience and minimum improvement
        if previous_score is not None:
            if score - previous_score < minimum_improvement:
                no_improvement += 1
            else:
                no_improvement = 0

        previous_score = score

        if no_improvement >= patience:
            break

        # 5. Allocate sample quotas and select new samples for the next round
        allocation = allocate_samples(
            quota_weights, total_samples_per_round, priority_order
        )
        
        selected, real_pool, current_generated = select_samples(
            real_pool, current_generated, feature_cols, model, allocation,
            priority_order, selection_modes, source_bin_map, target_col,
            source_bin_col, uncertainty_threshold, cluster_params,
            random_seed + round_index
        )

        # Stop if no new samples could be selected
        if len(selected) == 0:
            break

        # 6. Append selected samples to the current training set
        current_train = pd.concat(
            [current_train, selected], ignore_index=True
        )

    # 7. Return the best model, data, and remaining pools
    return {
        "model": best_model,
        "training_data": best_train,
        "validation_r2": best_score,
        "history": pd.DataFrame(history),
        "remaining_real_pool": real_pool,
        "remaining_generated_pool": current_generated
    }