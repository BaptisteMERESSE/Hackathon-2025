"""
Quick script to check overfitting for all trained models
Run this after training to visualize train/val performance
"""
import pandas as pd
import joblib
from pathlib import Path
from metrics_visualizer import quick_train_val_check

def check_all_models():
    """Check overfitting for all saved models"""
    print("="*80)
    print("OVERFITTING CHECK FOR ALL MODELS")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    X_train = pd.read_csv('../data/X_train_processed.csv', index_col=0)
    y_train = pd.read_csv('../data/y_train.csv', index_col=0)['MathScore']
    
    common_idx = X_train.index.intersection(y_train.index)
    X_train = X_train.loc[common_idx]
    y_train = y_train.loc[common_idx]
    
    # Sample if too large
    if len(X_train) > 100000:
        import numpy as np
        print(f"Sampling 100k rows for faster evaluation...")
        sample_idx = np.random.choice(X_train.index, size=100000, replace=False)
        X_train = X_train.loc[sample_idx]
        y_train = y_train.loc[sample_idx]
    
    print(f"Data shape: {X_train.shape}\n")
    
    # Find all models
    models_dir = Path('../models')
    if not models_dir.exists():
        print("No models directory found. Train models first!")
        return
    
    model_files = list(models_dir.glob('*.pkl'))
    
    if len(model_files) == 0:
        print("No models found. Train models first!")
        return
    
    results = []
    
    for model_file in model_files:
        model_name = model_file.stem.replace('_', ' ').title()
        
        try:
            print(f"\nChecking {model_name}...")
            model = joblib.load(model_file)
            
            train_metrics, val_metrics = quick_train_val_check(
                model, X_train, y_train, 
                model_name=model_name,
                save_plot=True
            )
            
            results.append({
                'Model': model_name,
                'Train_RMSE': train_metrics['rmse'],
                'Val_RMSE': val_metrics['rmse'],
                'RMSE_Gap_%': ((val_metrics['rmse'] - train_metrics['rmse']) / train_metrics['rmse'] * 100),
                'Train_R2': train_metrics['r2'],
                'Val_R2': val_metrics['r2'],
                'R2_Gap_%': ((train_metrics['r2'] - val_metrics['r2']) / train_metrics['r2'] * 100)
            })
            
        except Exception as e:
            print(f"  Failed to check {model_name}: {e}")
    
    # Summary
    if results:
        print("\n" + "="*80)
        print("SUMMARY - OVERFITTING ANALYSIS")
        print("="*80)
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('Val_RMSE')
        
        print("\n" + results_df.to_string(index=False))
        
        # Save summary
        results_df.to_csv('../results/overfitting_summary.csv', index=False)
        print(f"\n📊 Summary saved to ../results/overfitting_summary.csv")
        print(f"📊 Individual plots saved to ../results/")
        
        # Best model
        best_model = results_df.iloc[0]
        print(f"\n🏆 Best Model: {best_model['Model']}")
        print(f"   Validation RMSE: {best_model['Val_RMSE']:.4f}")
        print(f"   Overfitting Gap: {best_model['RMSE_Gap_%']:.1f}%")

if __name__ == "__main__":
    check_all_models()

