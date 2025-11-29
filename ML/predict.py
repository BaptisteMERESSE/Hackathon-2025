"""
Prediction Script - Generate predictions for test data
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_best_model(models_dir='../models'):
    """
    Load the best performing model
    Priority: ensemble > stacking > individual models
    """
    models_dir = Path(models_dir)
    
    # Try ensemble models first
    ensemble_files = [
        'stacking_model.pkl',
        'ensemble_weighted.pkl',
        'ensemble_median.pkl'
    ]
    
    for filename in ensemble_files:
        filepath = models_dir / filename
        if filepath.exists():
            print(f"Loading {filename}...")
            return joblib.load(filepath), filename
    
    # Try individual models
    individual_files = [
        'xgboost_model.pkl',
        #'lightgbm_model.pkl',
        #'catboost_model.pkl',
        'random_forest_model.pkl',
        'extra_trees_model.pkl'
    ]
    
    for filename in individual_files:
        filepath = models_dir / filename
        if filepath.exists():
            print(f"Loading {filename}...")
            return joblib.load(filepath), filename
    
    raise FileNotFoundError("No trained models found!")

def predict_test_data(X_test_path, output_path='../results/predictions.csv', 
                     preprocessor_path='../data/preprocessor.pkl'):
    """
    Generate predictions for test data
    """
    print("="*80)
    print("GENERATING PREDICTIONS")
    print("="*80)
    
    # Load test data
    print(f"\nLoading test data from {X_test_path}...")
    X_test = pd.read_csv(X_test_path, index_col=0)
    print(f"Test data shape: {X_test.shape}")
    
    # Load preprocessor
    print(f"\nLoading preprocessor from {preprocessor_path}...")
    try:
        preprocessor = joblib.load(preprocessor_path)
        
        # Preprocess test data
        print("Preprocessing test data...")
        X_test_processed = preprocessor.transform(X_test)
        
        # Feature engineering
        from preprocessing import create_advanced_features
        X_test_processed = create_advanced_features(X_test_processed, y=None)
        
        print(f"Preprocessed test data shape: {X_test_processed.shape}")
    except FileNotFoundError:
        print("Warning: Preprocessor not found. Using raw test data.")
        X_test_processed = X_test
    
    # Load best model
    print("\nLoading best model...")
    model, model_name = load_best_model()
    print(f"Using model: {model_name}")
    
    # Generate predictions
    print("\nGenerating predictions...")
    predictions = model.predict(X_test_processed)
    
    # Create submission dataframe
    submission = pd.DataFrame({
        'MathScore': predictions
    }, index=X_test.index)
    
    # Save predictions
    print(f"\nSaving predictions to {output_path}...")
    from pathlib import Path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path)
    
    print(f"\n{'='*80}")
    print("PREDICTIONS COMPLETE!")
    print(f"{'='*80}")
    print(f"Predictions saved to: {output_path}")
    print(f"Number of predictions: {len(predictions)}")
    print(f"Prediction statistics:")
    print(f"  Mean: {predictions.mean():.2f}")
    print(f"  Std: {predictions.std():.2f}")
    print(f"  Min: {predictions.min():.2f}")
    print(f"  Max: {predictions.max():.2f}")
    
    return submission

def predict_with_all_models(X_test_path, output_dir='../results'):
    """
    Generate predictions with all available models and save separately
    """
    print("="*80)
    print("GENERATING PREDICTIONS WITH ALL MODELS")
    print("="*80)
    
    # Load and preprocess test data
    print(f"\nLoading test data from {X_test_path}...")
    X_test = pd.read_csv(X_test_path, index_col=0)
    
    try:
        preprocessor = joblib.load('../data/preprocessor.pkl')
        X_test_processed = preprocessor.transform(X_test)
        from preprocessing import create_advanced_features
        X_test_processed = create_advanced_features(X_test_processed, y=None)
    except:
        X_test_processed = X_test
    
    # Load all models
    models_dir = Path('../models')
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    model_files = list(models_dir.glob('*.pkl'))
    
    all_predictions = {}
    
    for model_file in model_files:
        try:
            print(f"\nProcessing {model_file.name}...")
            model = joblib.load(model_file)
            predictions = model.predict(X_test_processed)
            
            # Save individual predictions
            output_path = output_dir / f"predictions_{model_file.stem}.csv"
            pd.DataFrame({
                'MathScore': predictions
            }, index=X_test.index).to_csv(output_path)
            
            all_predictions[model_file.stem] = predictions
            print(f"  Saved to {output_path}")
            
        except Exception as e:
            print(f"  Failed: {e}")
    
    # Create average ensemble of all predictions
    if all_predictions:
        print("\nCreating average ensemble of all predictions...")
        avg_predictions = np.mean(list(all_predictions.values()), axis=0)
        pd.DataFrame({
            'MathScore': avg_predictions
        }, index=X_test.index).to_csv(output_dir / 'predictions_average_all.csv')
        print(f"  Saved to {output_dir / 'predictions_average_all.csv'}")
    
    print(f"\n{'='*80}")
    print(f"Generated predictions with {len(all_predictions)} models")
    print(f"{'='*80}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_X_test.csv>")
        print("Example: python predict.py ../data/X_test.csv")
        sys.exit(1)
    
    test_path = sys.argv[1]
    
    # Generate predictions with best model
    predict_test_data(test_path)
    
    # Optionally generate with all models
    # predict_with_all_models(test_path)

