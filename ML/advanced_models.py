"""
Additional Advanced Models: LightGBM, CatBoost, Neural Network
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
import joblib
import warnings
warnings.filterwarnings('ignore')

# LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available. Install with: pip install lightgbm")

# CatBoost
try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not available. Install with: pip install catboost")


class LightGBMPredictor:
    """LightGBM model with hyperparameter optimization"""
    
    def __init__(self, n_trials=50, cv_folds=5, use_gpu=False):
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.use_gpu = use_gpu
        self.best_params = None
        self.model = None
        self.feature_importance = None
        
    def objective(self, trial, X, y):
        """Optuna objective function"""
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'device_type': 'gpu' if self.use_gpu else 'cpu',
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, val_idx in kf.split(X):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train_fold, y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
            
            y_pred = model.predict(X_val_fold)
            rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
            scores.append(rmse)
        
        return np.mean(scores)
    
    def tune_hyperparameters(self, X, y):
        """Tune hyperparameters using Optuna"""
        print(f"Starting LightGBM hyperparameter tuning with {self.n_trials} trials...")
        
        study = optuna.create_study(direction='minimize', study_name='lightgbm_optimization')
        study.optimize(lambda trial: self.objective(trial, X, y), n_trials=self.n_trials)
        
        print(f"\nBest trial:")
        print(f"  RMSE: {study.best_trial.value:.4f}")
        print(f"  Params: {study.best_trial.params}")
        
        self.best_params = study.best_trial.params
        self.best_params.update({
            'objective': 'regression',
            'metric': 'rmse',
            'device_type': 'gpu' if self.use_gpu else 'cpu',
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        })
        
        return self.best_params
    
    def train(self, X, y, params=None):
        """Train the final model"""
        if params is None:
            params = self.best_params or {
                'objective': 'regression',
                'metric': 'rmse',
                'num_leaves': 50,
                'learning_rate': 0.05,
                'n_estimators': 1000,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 1,
                'reg_lambda': 1,
                'random_state': 42,
                'n_jobs': -1
            }
        
        print("Training final LightGBM model...")
        self.model = lgb.LGBMRegressor(**params)
        self.model.fit(X, y)
        
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 20 most important features:")
        print(self.feature_importance.head(20))
        
        return self.model
    
    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        print(f"\nLightGBM Evaluation:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
        
        return {'rmse': rmse, 'mae': mae, 'r2': r2}
    
    def save(self, filepath):
        from pathlib import Path
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        print(f"Model saved to {filepath}")


class CatBoostPredictor:
    """CatBoost model with hyperparameter optimization"""
    
    def __init__(self, n_trials=50, cv_folds=5, use_gpu=False):
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.use_gpu = use_gpu
        self.best_params = None
        self.model = None
        self.feature_importance = None
        
    def objective(self, trial, X, y):
        """Optuna objective function"""
        params = {
            'loss_function': 'RMSE',
            'task_type': 'GPU' if self.use_gpu else 'CPU',
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'iterations': trial.suggest_int('iterations', 100, 2000),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
            'random_strength': trial.suggest_float('random_strength', 0, 10),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'random_state': 42,
            'verbose': False
        }
        
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, val_idx in kf.split(X):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            model = CatBoostRegressor(**params)
            model.fit(
                X_train_fold, y_train_fold,
                eval_set=(X_val_fold, y_val_fold),
                early_stopping_rounds=50,
                verbose=False
            )
            
            y_pred = model.predict(X_val_fold)
            rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
            scores.append(rmse)
        
        return np.mean(scores)
    
    def tune_hyperparameters(self, X, y):
        """Tune hyperparameters using Optuna"""
        print(f"Starting CatBoost hyperparameter tuning with {self.n_trials} trials...")
        
        study = optuna.create_study(direction='minimize', study_name='catboost_optimization')
        study.optimize(lambda trial: self.objective(trial, X, y), n_trials=self.n_trials)
        
        print(f"\nBest trial:")
        print(f"  RMSE: {study.best_trial.value:.4f}")
        print(f"  Params: {study.best_trial.params}")
        
        self.best_params = study.best_trial.params
        self.best_params.update({
            'loss_function': 'RMSE',
            'task_type': 'GPU' if self.use_gpu else 'CPU',
            'random_state': 42,
            'verbose': False
        })
        
        return self.best_params
    
    def train(self, X, y, params=None):
        """Train the final model"""
        if params is None:
            params = self.best_params or {
                'loss_function': 'RMSE',
                'task_type': 'GPU' if self.use_gpu else 'CPU',
                'depth': 7,
                'learning_rate': 0.05,
                'iterations': 1000,
                'l2_leaf_reg': 3,
                'bagging_temperature': 0.5,
                'random_strength': 1,
                'border_count': 128,
                'random_state': 42
            }
        
        print("Training final CatBoost model...")
        self.model = CatBoostRegressor(**params)
        self.model.fit(X, y, verbose=100)
        
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 20 most important features:")
        print(self.feature_importance.head(20))
        
        return self.model
    
    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        print(f"\nCatBoost Evaluation:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
        
        return {'rmse': rmse, 'mae': mae, 'r2': r2}
    
    def save(self, filepath):
        from pathlib import Path
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        print(f"Model saved to {filepath}")


if __name__ == "__main__":
    # Load configuration
    import config
    config.print_config()
    
    print("="*80)
    print("Advanced Models Training (LightGBM & CatBoost)")
    print("="*80)
    
    # Load data
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
    
    common_idx = X_train.index.intersection(y_train.index)
    X_train = X_train.loc[common_idx]
    y_train = y_train.loc[common_idx]
    
    print(f"Full data shape: {X_train.shape}")
    
    # Sample data if configured
    X_train, y_train = config.get_sample_data(X_train, y_train)
    
    print(f"Training data shape: {X_train.shape}")
    
    # Check GPU
    import subprocess
    try:
        subprocess.check_output(['nvidia-smi'])
        use_gpu = True
        print("\nGPU detected!")
    except:
        use_gpu = False
        print("\nNo GPU detected.")
    
    # Train LightGBM
    if LIGHTGBM_AVAILABLE:
        print("\n" + "="*80)
        print("Training LightGBM")
        print("="*80)
        lgb_predictor = LightGBMPredictor(n_trials=config.N_TRIALS_LIGHTGBM, use_gpu=use_gpu)
        lgb_predictor.tune_hyperparameters(X_train, y_train)
        lgb_predictor.train(X_train, y_train)
        lgb_predictor.evaluate(X_train, y_train)
        
        # Check for overfitting
        from metrics_visualizer import quick_train_val_check
        quick_train_val_check(lgb_predictor, X_train, y_train, model_name='LightGBM')
        
        lgb_predictor.save('../models/lightgbm_model.pkl')
    
    # Train CatBoost
    if CATBOOST_AVAILABLE:
        print("\n" + "="*80)
        print("Training CatBoost")
        print("="*80)
        cb_predictor = CatBoostPredictor(n_trials=config.N_TRIALS_CATBOOST, use_gpu=use_gpu)
        cb_predictor.tune_hyperparameters(X_train, y_train)
        cb_predictor.train(X_train, y_train)
        cb_predictor.evaluate(X_train, y_train)
        
        # Check for overfitting
        from metrics_visualizer import quick_train_val_check
        quick_train_val_check(cb_predictor, X_train, y_train, model_name='CatBoost')
        
        cb_predictor.save('../models/catboost_model.pkl')
    
    print("\n" + "="*80)
    print("Advanced models training complete!")
    print("="*80)

