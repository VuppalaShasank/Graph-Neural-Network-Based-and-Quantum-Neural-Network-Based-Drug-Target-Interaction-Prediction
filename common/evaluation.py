"""
Evaluation Metrics and Visualization Utilities
Calculates RMSE, MAE, R², Pearson r, Spearman rho, and produces publication-quality comparison plots.
"""

from typing import Dict, Tuple, List, Optional
import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes standard regression and correlation metrics:
    - RMSE
    - MAE
    - R²
    - Pearson r (and p-value)
    - Spearman rho (and p-value)
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    
    # Filter non-finite if any
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    
    if len(y_true) < 2:
        return {'rmse': float('nan'), 'mae': float('nan'), 'r2': float('nan'),
                'pearson_r': float('nan'), 'spearman_rho': float('nan')}
                
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    
    pearson_r, pearson_p = stats.pearsonr(y_true, y_pred)
    spearman_rho, spearman_p = stats.spearmanr(y_true, y_pred)
    
    return {
        'rmse': round(rmse, 4),
        'mae': round(mae, 4),
        'r2': round(r2, 4),
        'pearson_r': round(float(pearson_r), 4),
        'pearson_p': float(pearson_p),
        'spearman_rho': round(float(spearman_rho), 4),
        'spearman_p': float(spearman_p),
        'n_samples': int(len(y_true)),
    }


def plot_actual_vs_predicted(
    actual: np.ndarray,
    predicted: np.ndarray,
    model_name: str,
    output_png_path: str,
    metrics: Optional[Dict[str, float]] = None
) -> None:
    """Generates an Actual vs Predicted scatter plot with identity line."""
    plt.figure(figsize=(7, 6))
    sns.set_theme(style="whitegrid")
    
    actual = np.asarray(actual).ravel()
    predicted = np.asarray(predicted).ravel()
    
    plt.scatter(actual, predicted, alpha=0.35, edgecolors='none', color='#2b5c8f', s=25)
    
    min_val = min(actual.min(), predicted.min()) - 0.5
    max_val = max(actual.max(), predicted.max()) + 0.5
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1.5, label='Ideal (y = x)')
    
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.xlabel('Experimental pKi (-log10 M)', fontsize=12)
    plt.ylabel('Predicted pKi (-log10 M)', fontsize=12)
    plt.title(f'{model_name}: Experimental vs. Predicted Binding Affinity', fontsize=13, fontweight='bold')
    
    if metrics:
        textstr = f"RMSE: {metrics.get('rmse', 0):.3f}\nMAE: {metrics.get('mae', 0):.3f}\nR²: {metrics.get('r2', 0):.3f}\nPearson r: {metrics.get('pearson_r', 0):.3f}\nSpearman ρ: {metrics.get('spearman_rho', 0):.3f}"
        props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85, edgecolor='gray')
        plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                       verticalalignment='top', bbox=props)
                       
    plt.legend(loc='lower right')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
    plt.savefig(output_png_path, dpi=300)
    plt.close()


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    model_name: str,
    output_png_path: str
) -> None:
    """Plots training and validation loss per epoch."""
    plt.figure(figsize=(7, 5))
    sns.set_theme(style="whitegrid")
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-o', markersize=4, label='Train Loss (MSE)')
    plt.plot(epochs, val_losses, 'r-s', markersize=4, label='Val Loss (MSE)')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title(f'{model_name}: Learning Curves', fontsize=13, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
    plt.savefig(output_png_path, dpi=300)
    plt.close()


def plot_model_comparison_bar(
    comparison_df: pd.DataFrame,
    output_png_path: str
) -> None:
    """Generates bar plots comparing metrics across all three models."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    metrics_to_plot = [('rmse', 'RMSE (Lower is better)'),
                       ('mae', 'MAE (Lower is better)'),
                       ('r2', 'R² (Higher is better)'),
                       ('pearson_r', 'Pearson r (Higher is better)')]
                       
    colors = ['#3470a3', '#e07a5f', '#81b29a']
    
    for ax, (metric_col, title) in zip(axes, metrics_to_plot):
        if metric_col in comparison_df.columns:
            sns.barplot(data=comparison_df, x='model', y=metric_col, ax=ax, palette=colors)
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_xlabel('')
            ax.set_ylabel(metric_col.upper())
            for p in ax.patches:
                val = p.get_height()
                ax.annotate(f"{val:.3f}", (p.get_x() + p.get_width() / 2., val / 2),
                            ha='center', va='center', color='white', fontweight='bold', fontsize=10)
                            
    plt.suptitle('Fair Model Performance Comparison (GNN vs QNN vs Hybrid GNN+QNN)', fontsize=14, fontweight='bold', y=1.03)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
    plt.savefig(output_png_path, dpi=300)
    plt.close()

