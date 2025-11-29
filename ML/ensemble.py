"""
Ensemble and Stacking Models for Maximum Accuracy
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import StackingRegressor
import joblib
import warnings
warnings.filterwarnings('ignore')


class EnsemblePredictor:
    """
    Ensemble predictor that combines multiple models
    """
    
    def __init__(self, models_dict, weights=None, method='weighted_average'):
        """
        Args:
            models_dict: Dictionary of {name: model} pairs
            weights: List of weights for each model (None = equal weights)
            method: 'weighted_average', 'median', or 'stacking'
        """
        self.models_dict = models_dict
        self.weights = weights
        self.method = method
        self.meta_model = None
        
        if weights is None:
            self.weights = np.ones(len(models_dict)) / len(models_dict)
        else:
            self.weights = np.array(weights) / np.sum(weights)
    
    def predict(self, X):
        """Make ensemble predictions"""
        predictions = []
        
        for name, model in self.models_dict.items():
            try:
                pred = model.predict(X)
                predictions.append(pred)
                print(f"  {name}: prediction shape {pred.shape}")
            except Exception as e:
                print(f"  Warning: {name} failed - {e}")
        
        predictions = np.array(predictions)
        
        if self.method == 'weighted_average':
            # Weighted average
            ensemble_pred = np.average(predictions, axis=0, weights=self.weights[:len(predictions)])
        elif self.method == 'median':
            # Median (robust to outliers)
            ensemble_pred = np.median(predictions, axis=0)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return ensemble_pred
    
    def evaluate(self, X, y):
        """Evaluate ensemble performance"""
        y_pred = self.predict(X)
        
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        print(f"\nEnsemble Evaluation ({self.method}):")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
        
        return {'rmse': rmse, 'mae': mae, 'r2': r2}
    
    def optimize_weights(self, X, y, cv_folds=5):
        """
        Optimize ensemble weights using cross-validation
        """
        print(f"Optimizing ensemble weights with {cv_folds}-fold CV...")
        
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Get predictions from each model
        model_predictions = {name: [] for name in self.models_dict.keys()}
        y_true_list = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_val = X.iloc[val_idx]
            y_val = y.iloc[val_idx]
            
            for name, model in self.models_dict.items():
                try:
                    pred = model.predict(X_val)
                    model_predictions[name].extend(pred)
                except:
                    pass
            
            y_true_list.extend(y_val)
        
        # Convert to arrays
        y_true = np.array(y_true_list)
        predictions_matrix = np.column_stack([
            np.array(model_predictions[name]) 
            for name in self.models_dict.keys()
        ])
        
        # Grid search for optimal weights
        from scipy.optimize import minimize
        
        def objective(weights):
            weights = np.abs(weights) / np.sum(np.abs(weights))
            ensemble_pred = np.average(predictions_matrix, axis=1, weights=weights)
            return np.sqrt(mean_squared_error(y_true, ensemble_pred))
        
        initial_weights = np.ones(len(self.models_dict)) / len(self.models_dict)
        result = minimize(objective, initial_weights, method='Nelder-Mead')
        
        optimal_weights = np.abs(result.x) / np.sum(np.abs(result.x))
        self.weights = optimal_weights
        
        print(f"\nOptimal weights:")
        for name, weight in zip(self.models_dict.keys(), optimal_weights):
            print(f"  {name}: {weight:.4f}")
        
        return optimal_weights
    
    def save(self, filepath):
        """Save ensemble"""
        from pathlib import Path
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        print(f"Ensemble saved to {filepath}")


class StackingPredictor:
    """
    Stacking predictor with meta-model
    """
    
    def __init__(self, base_models_dict, meta_model=None):
        """
        Args:
            base_models_dict: Dictionary of {name: model} pairs for base models
            meta_model: Meta-learner model (default: Ridge)
        """
        self.base_models_dict = base_models_dict
        self.meta_model = meta_model or Ridge(alpha=1.0)
        self.is_fitted = False
    
    def fit(self, X, y, cv_folds=5):
        """
        Fit stacking model using cross-validation
        """
        print(f"Training stacking model with {len(self.base_models_dict)} base models...")
        
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Generate out-of-fold predictions
        oof_predictions = np.zeros((len(X), len(self.base_models_dict)))
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"  Processing fold {fold + 1}/{cv_folds}...")
            X_train_fold = X.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_train_fold = y.iloc[train_idx]
            
            for i, (name, model) in enumerate(self.base_models_dict.items()):
                try:
                    # Clone model and train on fold
                    from sklearn.base import clone
                    model_fold = clone(model) if hasattr(model, 'get_params') else model
                    
                    # If it's our custom predictor class, use the underlying model
                    if hasattr(model, 'model'):
                        model_fold = model.model
                    
                    # Train and predict
                    model_fold.fit(X_train_fold, y_train_fold)
                    oof_predictions[val_idx, i] = model_fold.predict(X_val_fold)
                except Exception as e:
                    print(f"    Warning: {name} failed on fold {fold + 1} - {e}")
        
        # Train meta-model on out-of-fold predictions
        print("  Training meta-model...")
        self.meta_model.fit(oof_predictions, y)
        self.is_fitted = True
        
        # Evaluate on training data
        train_pred = self.meta_model.predict(oof_predictions)
        train_rmse = np.sqrt(mean_squared_error(y, train_pred))
        print(f"  Stacking train RMSE: {train_rmse:.4f}")
        
        return self
    
    def predict(self, X):
        """Make stacking predictions"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet!")
        
        # Get predictions from base models
        base_predictions = np.zeros((len(X), len(self.base_models_dict)))
        
        for i, (name, model) in enumerate(self.base_models_dict.items()):
            try:
                base_predictions[:, i] = model.predict(X)
            except Exception as e:
                print(f"  Warning: {name} prediction failed - {e}")
        
        # Meta-model prediction
        stacking_pred = self.meta_model.predict(base_predictions)
        
        return stacking_pred
    
    def evaluate(self, X, y):
        """Evaluate stacking performance"""
        y_pred = self.predict(X)
        
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        print(f"\nStacking Evaluation:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
        
        return {'rmse': rmse, 'mae': mae, 'r2': r2}
    
    def save(self, filepath):
        """Save stacking model"""
        from pathlib import Path
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        print(f"Stacking model saved to {filepath}")


def load_all_models(models_dir='../models'):
    """Load all trained models"""
    import os
    from pathlib import Path
    
    models_dir = Path(models_dir)
    models = {}
    
    model_files = {
        'xgboost': 'xgboost_model.pkl',
        'random_forest': 'random_forest_model.pkl',
        'extra_trees': 'extra_trees_model.pkl',
        'lightgbm': 'lightgbm_model.pkl',
        'catboost': 'catboost_model.pkl'
    }
    
    for name, filename in model_files.items():
        filepath = models_dir / filename
        if filepath.exists():
            try:
                models[name] = joblib.load(filepath)
                print(f"Loaded {name} model")
            except Exception as e:
                print(f"Failed to load {name}: {e}")
    
    return models


if __name__ == "__main__":
    # Load configuration
    import config
    config.print_config()
    
    print("="*80)
    print("Ensemble Model Creation")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    try:
        X_train = pd.read_csv('../data/X_train_processed.csv', index_col=0)
        print(f"✓ Loaded preprocessed data: {X_train.shape}")
    except FileNotFoundError:
        print("\n❌ ERROR: X_train_processed.csv not found!")
        print("Please run preprocessing first:")
        print("  cd ML")
        print("  python preprocessing.py")
        import sys
        sys.exit(1)
    y_train = pd.read_csv('../data/y_train.csv', index_col=0)['MathScore']
    
    common_idx = X_train.index.intersection(y_train.index)
    X_train = X_train.loc[common_idx]
    y_train = y_train.loc[common_idx]
    
    print(f"Full data shape: {X_train.shape}")
    
    # Sample data if configured
    X_train, y_train = config.get_sample_data(X_train, y_train)
    
    print(f"Data shape: {X_train.shape}")
    
    # Load all trained models
    print("\n" + "="*80)
    print("Loading trained models...")
    print("="*80)
    models = load_all_models()
    
    if len(models) == 0:
        print("No models found! Please train models first.")
    else:
        # Create simple ensemble
        print("\n" + "="*80)
        print("Creating Weighted Average Ensemble")
        print("="*80)
        ensemble = EnsemblePredictor(models, method='weighted_average')
        ensemble.optimize_weights(X_train, y_train)
        ensemble.evaluate(X_train, y_train)
        ensemble.save('../models/ensemble_weighted.pkl')
        
        # Create median ensemble
        print("\n" + "="*80)
        print("Creating Median Ensemble")
        print("="*80)
        ensemble_median = EnsemblePredictor(models, method='median')
        ensemble_median.evaluate(X_train, y_train)
        ensemble_median.save('../models/ensemble_median.pkl')
        
        # Create stacking ensemble
        print("\n" + "="*80)
        print("Creating Stacking Ensemble")
        print("="*80)
        stacking = StackingPredictor(models, meta_model=Ridge(alpha=10.0))
        stacking.fit(X_train, y_train, cv_folds=5)
        stacking.evaluate(X_train, y_train)
        stacking.save('../models/stacking_model.pkl')
        
        print("\n" + "="*80)
        print("All ensemble models created!")
        print("="*80)

