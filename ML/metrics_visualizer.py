"""
Quick metrics visualization to detect overfitting
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_train_val_metrics(train_metrics, val_metrics, model_name='Model', save_path=None):
    """
    Plot training vs validation metrics
    
    Args:
        train_metrics: dict with 'rmse', 'mae', 'r2'
        val_metrics: dict with 'rmse', 'mae', 'r2'
        model_name: Name of the model
        save_path: Path to save plot (optional)
    """
    metrics = ['rmse', 'mae', 'r2']
    train_vals = [train_metrics[m] for m in metrics]
    val_vals = [val_metrics[m] for m in metrics]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, train_vals, width, label='Training', alpha=0.8)
    bars2 = ax.bar(x + width/2, val_vals, width, label='Validation', alpha=0.8)
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Values', fontsize=12)
    ax.set_title(f'{model_name} - Training vs Validation', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['RMSE ↓', 'MAE ↓', 'R² ↑'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Add overfitting indicator
    rmse_diff = ((val_metrics['rmse'] - train_metrics['rmse']) / train_metrics['rmse'] * 100)
    r2_diff = ((train_metrics['r2'] - val_metrics['r2']) / train_metrics['r2'] * 100)
    
    overfitting_text = f"RMSE gap: {rmse_diff:+.1f}% | R² gap: {r2_diff:+.1f}%"
    if rmse_diff > 10 or r2_diff > 5:
        color = 'red'
        status = "⚠️ Possible Overfitting"
    elif rmse_diff > 5 or r2_diff > 2:
        color = 'orange'
        status = "⚡ Slight Overfitting"
    else:
        color = 'green'
        status = "✓ Good Generalization"
    
    ax.text(0.5, 0.95, f"{status}\n{overfitting_text}", 
            transform=ax.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.3),
            fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Plot saved to {save_path}")
    
    plt.show()
    plt.close()
    
    return fig

def quick_train_val_check(model, X, y, model_name='Model', test_size=0.2, save_plot=True):
    """
    Quick train/val split check for overfitting
    
    Args:
        model: Trained model with predict() method
        X: Features
        y: Target
        model_name: Name for plot
        test_size: Validation split size
        save_plot: Whether to save plot
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    # Metrics
    train_metrics = {
        'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'mae': mean_absolute_error(y_train, y_train_pred),
        'r2': r2_score(y_train, y_train_pred)
    }
    
    val_metrics = {
        'rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
        'mae': mean_absolute_error(y_val, y_val_pred),
        'r2': r2_score(y_val, y_val_pred)
    }
    
    print(f"\n{'='*60}")
    print(f"OVERFITTING CHECK: {model_name}")
    print(f"{'='*60}")
    print(f"Training Set   - RMSE: {train_metrics['rmse']:.4f} | MAE: {train_metrics['mae']:.4f} | R²: {train_metrics['r2']:.4f}")
    print(f"Validation Set - RMSE: {val_metrics['rmse']:.4f} | MAE: {val_metrics['mae']:.4f} | R²: {val_metrics['r2']:.4f}")
    print(f"{'='*60}\n")
    
    # Plot
    save_path = f'../results/{model_name.lower().replace(" ", "_")}_train_val.png' if save_plot else None
    plot_train_val_metrics(train_metrics, val_metrics, model_name, save_path)
    
    return train_metrics, val_metrics

