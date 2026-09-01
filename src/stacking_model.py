import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


class StackingEnsemble:
    def __init__(
        self, cat_params, mlp_params, xgb_params, ridge_alphas, 
        ridge_cv, n_splits, random_seed, mlp_transformer
    ):
        # Store base model parameters
        self.cat_params = dict(cat_params)
        self.mlp_params = dict(mlp_params)
        self.xgb_params = dict(xgb_params)
        
        # Store meta-learner and cross-validation parameters
        self.ridge_alphas = ridge_alphas
        self.ridge_cv = ridge_cv
        self.n_splits = n_splits
        self.random_seed = random_seed
        
        # Clone the transformer to avoid modifying the original object in memory
        self.mlp_transformer = clone(mlp_transformer)

    def _new_models(self):
        """Instantiate fresh copies of the three base models."""
        return (
            CatBoostRegressor(**self.cat_params), 
            MLPRegressor(**self.mlp_params), 
            XGBRegressor(**self.xgb_params)
        )

    def fit(self, X, y):
        """
        Fit the stacking ensemble.
        1. Generate out-of-fold (OOF) predictions for the meta-learner.
        2. Train the meta-learner on these OOF predictions.
        3. Refit base models on the full dataset for final inference.
        """
        X, y = np.asarray(X), np.asarray(y)
        
        # 1. Transform features specifically for the MLP model
        self.mlp_transformer.fit(X)
        X_mlp = self.mlp_transformer.transform(X)
        
        # 2. Generate out-of-fold (OOF) predictions using K-Fold CV
        meta_X = np.zeros((len(X), 3), dtype=float)
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_seed)

        for train_idx, val_idx in kf.split(X):
            cat, mlp, xgb = self._new_models()
            
            # Fit base models on the training fold
            cat.fit(X[train_idx], y[train_idx])
            mlp.fit(X_mlp[train_idx], y[train_idx])
            xgb.fit(X[train_idx], y[train_idx])
            
            # Predict on the validation fold and store in the meta-feature matrix
            meta_X[val_idx, 0] = cat.predict(X[val_idx])
            meta_X[val_idx, 1] = mlp.predict(X_mlp[val_idx])
            meta_X[val_idx, 2] = xgb.predict(X[val_idx])

        # 3. Train the meta-learner (RidgeCV) on the OOF predictions
        self.meta = RidgeCV(alphas=self.ridge_alphas, cv=self.ridge_cv)
        self.meta.fit(meta_X, y)
        
        # 4. Refit all base models on the entire training dataset
        self.cat, self.mlp, self.xgb = self._new_models()
        self.cat.fit(X, y)
        self.mlp.fit(X_mlp, y)
        self.xgb.fit(X, y)
        
        return self

    def predict_base(self, X):
        """Get raw predictions from the three base models."""
        X = np.asarray(X)
        X_mlp = self.mlp_transformer.transform(X)
        
        return np.column_stack([
            self.cat.predict(X), 
            self.mlp.predict(X_mlp), 
            self.xgb.predict(X)
        ])

    def predict_with_uncertainty(self, X):
        """
        Predict using the meta-learner and estimate uncertainty.
        Uncertainty is approximated by the standard deviation of base model predictions.
        """
        base_preds = self.predict_base(X)
        final_pred = self.meta.predict(base_preds)
        uncertainty = np.std(base_preds, axis=1)
        
        return final_pred, uncertainty

    def predict(self, X):
        """Return only the final meta-learner prediction."""
        return self.predict_with_uncertainty(X)[0]

    def get_meta_coefficients(self):
        """Extract the learned weights and intercept from the Ridge meta-learner."""
        return np.asarray(self.meta.coef_).copy(), float(self.meta.intercept_)