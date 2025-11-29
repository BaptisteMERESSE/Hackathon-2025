"""
Advanced Random Forest Model with Hyperparameter Tuning
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import cross_val_score, KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class RandomForestPredictor:
    """
    Random Forest model with hyperparameter optimization
    """
    
    def __init__(self, n_iter=50, cv_folds=5, use_extra_trees=False):
        """
        Args:
            n_iter: Number of parameter settings sampled
            cv_folds: Number of cross-validation folds
            use_extra_trees: Use ExtraTreesRegressor instead of RandomForest
        """
        self.n_iter = n_iter
        self.cv_folds = cv_folds
        self.use_extra_trees = use_extra_trees
        self.best_params = None
        self.model = None
        self.feature_importance = None
        
    def get_param_distribution(self):
        """Get parameter distribution for random search"""
        param_dist = {
            'n_estimators': [300],
            'max_depth': [50],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt'],
            'bootstrap': [True],
            'max_samples': [0.8] if not self.use_extra_trees else [None]
        }
        return param_dist
    
    def tune_hyperparameters(self, X, y):
        """Tune hyperparameters using RandomizedSearchCV"""
        print(f"Starting hyperparameter tuning with {self.n_iter} iterations...")
        
        if self.use_extra_trees:
            base_model = ExtraTreesRegressor(random_state=42, n_jobs=-1)
            model_name = "ExtraTrees"
        else:
            base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
            model_name = "RandomForest"
        
        print(f"Using {model_name} model")
        
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=self.get_param_distribution(),
            n_iter=self.n_iter,
            cv=self.cv_folds,
            scoring='neg_root_mean_squared_error',
            random_state=42,
            n_jobs=-1,
            verbose=2
        )
        
        random_search.fit(X, y)
        
        print(f"\nBest parameters:")
        print(random_search.best_params_)
        print(f"Best CV RMSE: {-random_search.best_score_:.4f}")
        
        self.best_params = random_search.best_params_
        return self.best_params
    
    def train(self, X, y, params=None):
        """Train the final model"""
        if params is None:
            params = self.best_params
        
        if params is None:
            # Use default good parameters
            params = {
                'n_estimators': 500,
                'max_depth': 30,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'bootstrap': True,
                'max_samples': 0.8,
                'random_state': 42,
                'n_jobs': -1
            }
        
        params['random_state'] = 42
        params['n_jobs'] = -1
        
        print("Training final model...")
        if self.use_extra_trees:
            self.model = ExtraTreesRegressor(**params)
        else:
            self.model = RandomForestRegressor(**params)
        
        self.model.fit(X, y)
        
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


def train_random_forest_with_cv(X, y, n_iter=30, use_extra_trees=False):
    """
    Train Random Forest with cross-validation and hyperparameter tuning
    """
    predictor = RandomForestPredictor(
        n_iter=n_iter, 
        cv_folds=5, 
        use_extra_trees=use_extra_trees
    )
    
    # Tune hyperparameters
    best_params = predictor.tune_hyperparameters(X, y)
    
    # Train final model
    model = predictor.train(X, y, best_params)
    
    # Evaluate on full training set
    metrics = predictor.evaluate(X, y)
    
    # Check for overfitting
    from metrics_visualizer import quick_train_val_check
    train_metrics, val_metrics = quick_train_val_check(predictor, X, y, 
                                                       model_name='Random Forest' if not use_extra_trees else 'ExtraTrees')
    
    return predictor, metrics


if __name__ == "__main__":
    # Load configuration
    import config
    config.print_config()
    
    print("="*80)
    print("Random Forest Model Training")
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
    
    # Train Random Forest
    print("\n" + "="*80)
    print("Training Random Forest")
    print("="*80)
    rf_predictor, rf_metrics = train_random_forest_with_cv(
        X_train, 
        y_train, 
        n_iter=config.N_ITER_RF,
        use_extra_trees=False
    )
    rf_predictor.save('../models/random_forest_model.pkl')
    
    # Train ExtraTrees
    print("\n" + "="*80)
    print("Training ExtraTrees")
    print("="*80)
    et_predictor, et_metrics = train_random_forest_with_cv(
        X_train, 
        y_train, 
        n_iter=config.N_ITER_EXTRATREES,
        use_extra_trees=True
    )
    et_predictor.save('../models/extra_trees_model.pkl')
    
    print("\n" + "="*80)
    print("Random Forest training complete!")
    print("="*80)

