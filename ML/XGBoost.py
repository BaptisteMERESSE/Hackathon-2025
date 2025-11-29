"""
Advanced XGBoost Model with Hyperparameter Tuning for PISA Math Score Prediction
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
import joblib
import warnings
warnings.filterwarnings('ignore')

class XGBoostPredictor:
    """
    XGBoost model with hyperparameter optimization
    """
    
    def __init__(self, n_trials=50, cv_folds=5, use_gpu=False):
        """
        Args:
            n_trials: Number of hyperparameter tuning trials
            cv_folds: Number of cross-validation folds
            use_gpu: Whether to use GPU acceleration
        """
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.use_gpu = use_gpu
        self.best_params = None
        self.model = None
        self.feature_importance = None
        
    def objective(self, trial, X, y):
        """Optuna objective function for hyperparameter tuning"""
        
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'booster': 'gbtree',
            'tree_method': 'hist',  # 'hist' works on both CPU and GPU
            'device': 'cuda' if self.use_gpu else 'cpu',  # New way to specify device
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'early_stopping_rounds': 50,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Cross-validation
        params_copy = params.copy()
        params_copy['early_stopping_rounds'] = 50
        params_copy['eval_metric'] = 'rmse'
        
        model = xgb.XGBRegressor(**params_copy)
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        scores = []
        for train_idx, val_idx in kf.split(X):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            model.fit(
                X_train_fold, y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                verbose=False
            )
            
            y_pred = model.predict(X_val_fold)
            rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
            scores.append(rmse)
        
        return np.mean(scores)
    
    def tune_hyperparameters(self, X, y):
        """Tune hyperparameters using Optuna"""
        print(f"Starting hyperparameter tuning with {self.n_trials} trials...")
        
        study = optuna.create_study(direction='minimize', study_name='xgboost_optimization')
        study.optimize(lambda trial: self.objective(trial, X, y), n_trials=self.n_trials)
        
        print(f"\nBest trial:")
        print(f"  RMSE: {study.best_trial.value:.4f}")
        print(f"  Params: {study.best_trial.params}")
        
        self.best_params = study.best_trial.params
        self.best_params['objective'] = 'reg:squarederror'
        self.best_params['eval_metric'] = 'rmse'
        self.best_params['tree_method'] = 'hist'
        self.best_params['device'] = 'cuda' if self.use_gpu else 'cpu'
        self.best_params['random_state'] = 42
        self.best_params['n_jobs'] = -1
        
        return self.best_params
    
    def train(self, X, y, params=None):
        """Train the final model"""
        if params is None:
            params = self.best_params
        
        if params is None:
            # Use default good parameters
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'tree_method': 'hist',
                'device': 'cuda' if self.use_gpu else 'cpu',
                'max_depth': 7,
                'learning_rate': 0.05,
                'n_estimators': 1000,
                'min_child_weight': 3,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'colsample_bylevel': 0.8,
                'gamma': 1,
                'reg_alpha': 1,
                'reg_lambda': 1,
                'random_state': 42,
                'n_jobs': -1
            }
        
        print("Training final XGBoost model...")
        self.model = xgb.XGBRegressor(**params)
        self.model.fit(X, y, verbose=0)
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 20 most important features:")
        print(self.feature_importance.head(20))
        
        return self.model
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        """Evaluate model performance"""
        y_pred = self.predict(X)
        
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        print(f"\nModel Evaluation:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
        
        return {'rmse': rmse, 'mae': mae, 'r2': r2}
    
    def save(self, filepath):
        """Save model"""
        from pathlib import Path
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        print(f"Model saved to {filepath}")
    
    @staticmethod
    def load(filepath):
        """Load model"""
        return joblib.load(filepath)


def train_xgboost_with_cv(X, y, n_trials=50, use_gpu=False):
    """
    Train XGBoost with cross-validation and hyperparameter tuning
    """
    predictor = XGBoostPredictor(n_trials=n_trials, cv_folds=5, use_gpu=use_gpu)
    
    # Tune hyperparameters
    best_params = predictor.tune_hyperparameters(X, y)
    
    # Train final model
    model = predictor.train(X, y, best_params)
    
    # Evaluate on full training set
    metrics = predictor.evaluate(X, y)
    
    # Check for overfitting
    from metrics_visualizer import quick_train_val_check
    train_metrics, val_metrics = quick_train_val_check(predictor, X, y, model_name='XGBoost')
    
    return predictor, metrics


if __name__ == "__main__":
    # Load configuration
    import config
    config.print_config()
    
    print("="*80)
    print("XGBoost Model Training")
    print("="*80)
    
    # Load preprocessed data
    print("\nLoading preprocessed data...")
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
    
    # Align indices
    common_idx = X_train.index.intersection(y_train.index)
    X_train = X_train.loc[common_idx]
    y_train = y_train.loc[common_idx]
    
    print(f"Full data shape: {X_train.shape}")
    
    # Sample data if configured
    X_train, y_train = config.get_sample_data(X_train, y_train)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Target shape: {y_train.shape}")
    
    # Check for GPU
    import subprocess
    try:
        subprocess.check_output(['nvidia-smi'])
        use_gpu = True
        print("\nGPU detected! Using GPU acceleration.")
    except:
        use_gpu = False
        print("\nNo GPU detected. Using CPU (hist tree method).")
    
    # Train model
    predictor, metrics = train_xgboost_with_cv(
        X_train, 
        y_train, 
        n_trials=config.N_TRIALS_XGBOOST,
        use_gpu=use_gpu
    )
    
    # Save model
    predictor.save('../models/xgboost_model.pkl')
    
    print("\n" + "="*80)
    print("XGBoost training complete!")
    print("="*80)

