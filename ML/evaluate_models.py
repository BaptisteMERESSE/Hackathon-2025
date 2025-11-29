"""
Model Evaluation and Comparison Script
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def evaluate_model_cv(model, X, y, cv_folds=5):
    """
    Evaluate model using cross-validation
    """
    print(f"Running {cv_folds}-fold cross-validation...")
    
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    rmse_scores = []
    mae_scores = []
    r2_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]
        
        # Train if needed
        if hasattr(model, 'model') and model.model is None:
            model.train(X_train_fold, y_train_fold)
        
        # Predict
        y_pred = model.predict(X_val_fold)
        
        # Metrics
        rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
        mae = mean_absolute_error(y_val_fold, y_pred)
        r2 = r2_score(y_val_fold, y_pred)
        
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        r2_scores.append(r2)
        
        print(f"  Fold {fold + 1}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
    
    results = {
        'rmse_mean': np.mean(rmse_scores),
        'rmse_std': np.std(rmse_scores),
        'mae_mean': np.mean(mae_scores),
        'mae_std': np.std(mae_scores),
        'r2_mean': np.mean(r2_scores),
        'r2_std': np.std(r2_scores)
    }
    
    print(f"\nCV Results:")
    print(f"  RMSE: {results['rmse_mean']:.4f} ± {results['rmse_std']:.4f}")
    print(f"  MAE: {results['mae_mean']:.4f} ± {results['mae_std']:.4f}")
    print(f"  R²: {results['r2_mean']:.4f} ± {results['r2_std']:.4f}")
    
    return results

def compare_all_models(X, y, models_dir='../models'):
    """
    Load and compare all trained models
    """
    print("="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    models_dir = Path(models_dir)
    model_files = list(models_dir.glob('*.pkl'))
    
    if len(model_files) == 0:
        print("No models found! Please train models first.")
        return
    
    results = {}
    
    for model_file in model_files:
        model_name = model_file.stem
        print(f"\n{'='*80}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*80}")
        
        try:
            model = joblib.load(model_file)
            
            # Simple prediction test
            y_pred = model.predict(X)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            
            results[model_name] = {
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }
            
            print(f"Training Set Performance:")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  R²: {r2:.4f}")
            
        except Exception as e:
            print(f"Failed to evaluate {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    if results:
        print("\n" + "="*80)
        print("SUMMARY - ALL MODELS")
        print("="*80)
        
        results_df = pd.DataFrame(results).T
        results_df = results_df.sort_values('rmse')
        
        print("\nRanked by RMSE (lower is better):")
        print(results_df.to_string())
        
        # Save results
        results_df.to_csv('../results/model_comparison.csv')
        print(f"\nResults saved to ../results/model_comparison.csv")
        
        # Plot comparison
        try:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # RMSE
            results_df.sort_values('rmse').plot(y='rmse', kind='barh', ax=axes[0], legend=False)
            axes[0].set_title('RMSE (lower is better)')
            axes[0].set_xlabel('RMSE')
            
            # MAE
            results_df.sort_values('mae').plot(y='mae', kind='barh', ax=axes[1], legend=False)
            axes[1].set_title('MAE (lower is better)')
            axes[1].set_xlabel('MAE')
            
            # R²
            results_df.sort_values('r2', ascending=False).plot(y='r2', kind='barh', ax=axes[2], legend=False)
            axes[2].set_title('R² (higher is better)')
            axes[2].set_xlabel('R²')
            
            plt.tight_layout()
            plt.savefig('../results/model_comparison.png', dpi=300, bbox_inches='tight')
            print(f"Plot saved to ../results/model_comparison.png")
            
        except Exception as e:
            print(f"Could not create plots: {e}")
    
    return results

def analyze_predictions(X, y, models_dir='../models'):
    """
    Analyze prediction patterns across models
    """
    print("\n" + "="*80)
    print("PREDICTION ANALYSIS")
    print("="*80)
    
    models_dir = Path(models_dir)
    model_files = list(models_dir.glob('*.pkl'))
    
    all_predictions = {}
    
    for model_file in model_files:
        try:
            model = joblib.load(model_file)
            predictions = model.predict(X)
            all_predictions[model_file.stem] = predictions
        except:
            pass
    
    if len(all_predictions) == 0:
        print("No predictions generated")
        return
    
    pred_df = pd.DataFrame(all_predictions)
    
    print(f"\nPrediction Statistics:")
    print(pred_df.describe())
    
    print(f"\nPrediction Correlations:")
    print(pred_df.corr())
    
    # Analyze prediction diversity
    print(f"\nPrediction Diversity:")
    for col in pred_df.columns:
        std = pred_df[col].std()
        print(f"  {col}: std={std:.2f}")
    
    return pred_df

if __name__ == "__main__":
    print("="*80)
    print("MODEL EVALUATION AND COMPARISON")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    try:
        X_train = pd.read_csv('../data/X_train_processed.csv', index_col=0)
        y_train = pd.read_csv('../data/y_train.csv', index_col=0)['MathScore']
        
        common_idx = X_train.index.intersection(y_train.index)
        X_train = X_train.loc[common_idx]
        y_train = y_train.loc[common_idx]
        
        print(f"Data shape: {X_train.shape}")
        
        # Use a sample for faster evaluation if dataset is large
        if len(X_train) > 100000:
            print(f"Using sample of 100000 rows for evaluation...")
            sample_idx = np.random.choice(X_train.index, size=100000, replace=False)
            X_sample = X_train.loc[sample_idx]
            y_sample = y_train.loc[sample_idx]
        else:
            X_sample = X_train
            y_sample = y_train
        
        # Compare all models
        results = compare_all_models(X_sample, y_sample)
        
        # Analyze predictions
        pred_df = analyze_predictions(X_sample, y_sample)
        
        print("\n" + "="*80)
        print("EVALUATION COMPLETE!")
        print("="*80)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

