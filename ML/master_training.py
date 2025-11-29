"""
Master Training Script - Runs entire pipeline for maximum accuracy
"""
import sys
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def create_directories():
    """Create necessary directories"""
    Path('../models').mkdir(parents=True, exist_ok=True)
    Path('../results').mkdir(parents=True, exist_ok=True)
    Path('../data').mkdir(parents=True, exist_ok=True)
    print("✓ Directories created")

def run_data_exploration():
    """Run data exploration"""
    print("\n" + "="*80)
    print("STEP 1: DATA EXPLORATION")
    print("="*80)
    try:
        import data_exploration
        print("✓ Data exploration complete")
        return True
    except Exception as e:
        print(f"✗ Data exploration failed: {e}")
        return False

def run_preprocessing():
    """Run preprocessing"""
    print("\n" + "="*80)
    print("STEP 2: DATA PREPROCESSING")
    print("="*80)
    
    from pathlib import Path
    if Path('../data/X_train_processed.csv').exists():
        print("✓ Preprocessed data already exists, skipping preprocessing")
        return True
    
    try:
        import subprocess
        result = subprocess.run(['python', 'preprocessing.py'], 
                              capture_output=False, 
                              text=True)
        if result.returncode == 0:
            print("✓ Preprocessing complete")
            return True
        else:
            print("✗ Preprocessing failed")
            return False
    except Exception as e:
        print(f"✗ Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_xgboost():
    """Train XGBoost model"""
    print("\n" + "="*80)
    print("STEP 3: TRAINING XGBOOST")
    print("="*80)
    try:
        import XGBoost
        print("✓ XGBoost training complete")
        return True
    except Exception as e:
        print(f"✗ XGBoost training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_random_forest():
    """Train Random Forest models"""
    print("\n" + "="*80)
    print("STEP 4: TRAINING RANDOM FOREST")
    print("="*80)
    try:
        import Random_forest
        print("✓ Random Forest training complete")
        return True
    except Exception as e:
        print(f"✗ Random Forest training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_advanced_models():
    """Train advanced models (LightGBM, CatBoost)"""
    print("\n" + "="*80)
    print("STEP 5: TRAINING ADVANCED MODELS")
    print("="*80)
    try:
        import advanced_models
        print("✓ Advanced models training complete")
        return True
    except Exception as e:
        print(f"✗ Advanced models training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_ensemble():
    """Create ensemble models"""
    print("\n" + "="*80)
    print("STEP 6: CREATING ENSEMBLE MODELS")
    print("="*80)
    try:
        import ensemble
        print("✓ Ensemble creation complete")
        return True
    except Exception as e:
        print(f"✗ Ensemble creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run complete training pipeline"""
    print("\n" + "="*80)
    print("PISA MATH SCORE PREDICTION - MASTER TRAINING PIPELINE")
    print("="*80)
    print("This will train multiple models and create ensembles for maximum accuracy")
    
    start_time = time.time()
    
    # Create directories
    create_directories()
    
    # Track success
    results = {}
    
    # Step 1: Data Exploration (optional, can skip if data is large)
    print("\nSkipping data exploration for large dataset...")
    results['exploration'] = True
    
    # Step 2: Preprocessing
    results['preprocessing'] = run_preprocessing()
    if not results['preprocessing']:
        print("\n✗ Pipeline failed at preprocessing step")
        return
    
    # Step 3: XGBoost
    results['xgboost'] = run_xgboost()
    
    # Step 4: Random Forest
    results['random_forest'] = run_random_forest()
    
    # Step 5: Advanced Models
    results['advanced'] = run_advanced_models()
    
    # Step 6: Ensemble
    if any([results['xgboost'], results['random_forest'], results['advanced']]):
        results['ensemble'] = run_ensemble()
    else:
        print("\n✗ No models trained successfully, skipping ensemble")
        results['ensemble'] = False
    
    # Summary
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"Total time: {elapsed_time/60:.2f} minutes")
    print("\nResults:")
    for step, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {step}")
    
    if results.get('ensemble', False):
        print("\n" + "="*80)
        print("SUCCESS! All models trained and ensemble created.")
        print("Best model will likely be the ensemble or stacking model.")
        print("Check ../models/ directory for all trained models.")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("Training completed with some errors.")
        print("Check individual model files in ../models/ directory.")
        print("="*80)

if __name__ == "__main__":
    main()

