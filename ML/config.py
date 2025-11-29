"""
Configuration file for model training
Modify these parameters to control training behavior
"""

# ============================================================================
# SAMPLING CONFIGURATION
# ============================================================================

# Set to True to use only a sample of the data (faster training for testing)
USE_SAMPLE = True

# Sample size options:
# - If < 1: fraction of data (e.g., 0.1 = 10% of data)
# - If >= 1: absolute number of rows (e.g., 50000 rows)
SAMPLE_SIZE = 0.1  # Use 10% of data for quick testing

# Random seed for reproducible sampling
RANDOM_SEED = 42

# ============================================================================
# HYPERPARAMETER TUNING CONFIGURATION
# ============================================================================

# Number of trials for Optuna hyperparameter tuning
# Reduce for faster testing, increase for better results
N_TRIALS_XGBOOST = 5      # Default: 30 (reduce to 5-10 for quick tests)
N_TRIALS_LIGHTGBM = 5     # Default: 30
N_TRIALS_CATBOOST = 5     # Default: 30

# Number of iterations for Random Forest RandomizedSearchCV
N_ITER_RF = 5             # Default: 20
N_ITER_EXTRATREES = 5     # Default: 20

# Cross-validation folds
CV_FOLDS = 3              # Default: 5 (reduce to 3 for faster training)

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# GPU usage (automatically detected if available)
USE_GPU = True

# ============================================================================
# PREPROCESSING CONFIGURATION
# ============================================================================

# Missing value imputation
USE_KNN_IMPUTER = False   # KNN is slow for large datasets, use False for speed

# Feature selection
FEATURE_SELECTION_K = None  # None = keep all features, or specify number (e.g., 200)

# ============================================================================
# QUICK PRESETS
# ============================================================================

def set_quick_test_mode():
    """Set all parameters for quick testing"""
    global USE_SAMPLE, SAMPLE_SIZE, N_TRIALS_XGBOOST, N_TRIALS_LIGHTGBM
    global N_TRIALS_CATBOOST, N_ITER_RF, N_ITER_EXTRATREES, CV_FOLDS
    
    USE_SAMPLE = True
    SAMPLE_SIZE = 10000  # Use only 10k rows
    N_TRIALS_XGBOOST = 3
    N_TRIALS_LIGHTGBM = 3
    N_TRIALS_CATBOOST = 3
    N_ITER_RF = 3
    N_ITER_EXTRATREES = 3
    CV_FOLDS = 2
    print("⚡ Quick test mode activated!")

def set_full_training_mode():
    """Set all parameters for full training (best accuracy, takes longer)"""
    global USE_SAMPLE, SAMPLE_SIZE, N_TRIALS_XGBOOST, N_TRIALS_LIGHTGBM
    global N_TRIALS_CATBOOST, N_ITER_RF, N_ITER_EXTRATREES, CV_FOLDS
    
    USE_SAMPLE = False
    SAMPLE_SIZE = 1.0
    N_TRIALS_XGBOOST = 50
    N_TRIALS_LIGHTGBM = 50
    N_TRIALS_CATBOOST = 50
    N_ITER_RF = 30
    N_ITER_EXTRATREES = 30
    CV_FOLDS = 5
    print("🚀 Full training mode activated!")

def set_medium_mode():
    """Balanced mode: reasonable speed with good accuracy"""
    global USE_SAMPLE, SAMPLE_SIZE, N_TRIALS_XGBOOST, N_TRIALS_LIGHTGBM
    global N_TRIALS_CATBOOST, N_ITER_RF, N_ITER_EXTRATREES, CV_FOLDS
    
    USE_SAMPLE = True
    SAMPLE_SIZE = 0.3  # Use 30% of data
    N_TRIALS_XGBOOST = 15
    N_TRIALS_LIGHTGBM = 15
    N_TRIALS_CATBOOST = 15
    N_ITER_RF = 10
    N_ITER_EXTRATREES = 10
    CV_FOLDS = 3
    print("⚖️  Medium training mode activated!")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_sample_data(X, y, sample_size=None, random_seed=None):
    """
    Sample data for faster training
    
    Args:
        X: Features dataframe
        y: Target series
        sample_size: If None, uses config SAMPLE_SIZE
        random_seed: If None, uses config RANDOM_SEED
    
    Returns:
        X_sample, y_sample
    """
    import numpy as np
    
    if sample_size is None:
        sample_size = SAMPLE_SIZE
    if random_seed is None:
        random_seed = RANDOM_SEED
    
    if not USE_SAMPLE:
        print(f"Using full dataset: {len(X)} rows")
        return X, y
    
    # Determine sample size
    if sample_size < 1:
        n_samples = int(len(X) * sample_size)
    else:
        n_samples = int(min(sample_size, len(X)))
    
    print(f"Sampling {n_samples} rows from {len(X)} ({n_samples/len(X)*100:.1f}%)")
    
    # Random sampling
    np.random.seed(random_seed)
    sample_idx = np.random.choice(X.index, size=n_samples, replace=False)
    
    X_sample = X.loc[sample_idx]
    y_sample = y.loc[sample_idx]
    
    return X_sample, y_sample

def print_config():
    """Print current configuration"""
    print("\n" + "="*80)
    print("TRAINING CONFIGURATION")
    print("="*80)
    print(f"Use Sample: {USE_SAMPLE}")
    if USE_SAMPLE:
        if SAMPLE_SIZE < 1:
            print(f"Sample Size: {SAMPLE_SIZE*100:.1f}% of data")
        else:
            print(f"Sample Size: {int(SAMPLE_SIZE)} rows")
    print(f"Random Seed: {RANDOM_SEED}")
    print(f"\nHyperparameter Tuning:")
    print(f"  XGBoost trials: {N_TRIALS_XGBOOST}")
    print(f"  LightGBM trials: {N_TRIALS_LIGHTGBM}")
    print(f"  CatBoost trials: {N_TRIALS_CATBOOST}")
    print(f"  Random Forest iterations: {N_ITER_RF}")
    print(f"  CV folds: {CV_FOLDS}")
    print(f"\nPreprocessing:")
    print(f"  Use KNN Imputer: {USE_KNN_IMPUTER}")
    print(f"  Feature Selection K: {FEATURE_SELECTION_K}")
    print("="*80 + "\n")


# ============================================================================
# UNCOMMENT ONE OF THESE TO SET A PRESET
# ============================================================================

# set_quick_test_mode()      # For quick testing (1-2 min per model)
# set_medium_mode()          # Balanced speed/accuracy
# set_full_training_mode()   # For final submission (best accuracy)

