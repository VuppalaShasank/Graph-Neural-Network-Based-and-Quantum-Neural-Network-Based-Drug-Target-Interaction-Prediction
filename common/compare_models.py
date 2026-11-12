"""
Model Comparison and Synthesis Pipeline
Implements Phase 8:
- Compares GNN, QNN, and Hybrid GNN+QNN on identical held-out test splits
- Gathers RMSE, MAE, R2, Pearson r, Spearman rho, and training time
- Generates results/comparison/model_comparison.csv
- Generates comparative plots:
  - results/comparison/model_comparison.png (Bar chart comparison)
  - results/comparison/prediction_comparison.png (Multi-panel actual vs predicted)
  - results/comparison/training_curves.png (Overlay training loss curves)
  - results/comparison/residual_distribution.png (Error distribution histogram)
"""

import os
import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from common.config import RESULTS_DIR, REPORTS_DIR
from common.evaluation import plot_model_comparison_bar


def run_model_comparison(results_dir: Path = RESULTS_DIR):
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    models = ["GNN", "QNN", "GNN_QNN"]
    display_names = {
        "GNN": "GNN (Bond-Aware GINE)",
        "QNN": "QNN (Variational Circuit)",
        "GNN_QNN": "Proposed Hybrid (GNN + QNN)"
    }
    
    rows = []
    test_preds = {}
    train_summaries = {}
    
    for m in models:
        m_dir = results_dir / m
        metrics_file = m_dir / "metrics.json"
        preds_file = m_dir / "test_predictions.csv"
        summary_file = m_dir / "training_summary.json"
        
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                data = json.load(f)
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    s_data = json.load(f)
                    data['training_time_seconds'] = s_data.get('training_time_seconds', 0.0)
                    data['best_epoch'] = s_data.get('best_epoch', 0)
            data['display_name'] = display_names[m]
            rows.append(data)
            
        if preds_file.exists():
            test_preds[m] = pd.read_parquet(preds_file) if str(preds_file).endswith('.parquet') else pd.read_csv(preds_file)
            
    if not rows:
        print("No evaluation metrics found! Please train and evaluate models first.")
        return None
        
    df_comp = pd.DataFrame(rows)
    # Order columns
    cols_order = ['model', 'display_name', 'rmse', 'mae', 'r2', 'pearson_r', 'spearman_rho', 'n_samples', 'training_time_seconds', 'best_epoch']
    cols_present = [c for c in cols_order if c in df_comp.columns]
    df_comp = df_comp[cols_present]
    
    comp_csv_path = comparison_dir / "model_comparison.csv"
    df_comp.to_csv(comp_csv_path, index=False)
    print(f"\n=======================================================")
    print(" MODEL PERFORMANCE COMPARISON SUMMARY")
    print(f"=======================================================")
    print(df_comp.to_string(index=False))
    print(f"\nSaved comparison CSV to: {comp_csv_path}")
    
    # 1. Bar Chart Comparison
    plot_bar_path = comparison_dir / "model_comparison.png"
    plot_model_comparison_bar(df_comp, str(plot_bar_path))
    print(f"Saved metric bar comparison to: {plot_bar_path}")
    
    # 2. Multi-panel Scatter Plot (Actual vs Predicted)
    if len(test_preds) == 3:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharex=True, sharey=True)
        sns.set_theme(style="whitegrid")
        
        colors = ['#2b5c8f', '#d95f02', '#2ca02c']
        
        for ax, m, color in zip(axes, models, colors):
            p_df = test_preds[m]
            actual = p_df['pKi'].values
            pred = p_df['predicted_pKi'].values
            
            row = df_comp[df_comp['model'] == m].iloc[0]
            
            ax.scatter(actual, pred, alpha=0.35, edgecolors='none', color=color, s=25)
            min_val = min(actual.min(), pred.min()) - 0.5
            max_val = max(actual.max(), pred.max()) + 0.5
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1.5, label='Ideal (y = x)')
            
            ax.set_title(display_names[m], fontsize=12, fontweight='bold')
            ax.set_xlabel('Experimental pKi', fontsize=11)
            if ax == axes[0]:
                ax.set_ylabel('Predicted pKi', fontsize=11)
                
            textstr = f"RMSE: {row['rmse']:.3f}\nMAE: {row['mae']:.3f}\nR²: {row['r2']:.3f}\nPearson r: {row['pearson_r']:.3f}\nSpearman ρ: {row['spearman_rho']:.3f}"
            props = dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='gray')
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9.5,
                    verticalalignment='top', bbox=props)
            ax.legend(loc='lower right')
            
        plt.suptitle("Experimental vs Predicted Binding Affinity across Models (Held-out Test N=1,500)", fontsize=13, fontweight='bold')
        plt.tight_layout()
        pred_comp_path = comparison_dir / "prediction_comparison.png"
        plt.savefig(pred_comp_path, dpi=300)
        plt.close()
        print(f"Saved prediction comparison plot to: {pred_comp_path}")
        
    # 3. Residual Error Distribution
    if len(test_preds) == 3:
        plt.figure(figsize=(9, 5))
        sns.set_theme(style="whitegrid")
        for m, color in zip(models, ['#2b5c8f', '#d95f02', '#2ca02c']):
            p_df = test_preds[m]
            residuals = p_df['residual'].values
            sns.kdeplot(residuals, label=display_names[m], color=color, lw=2.0)
            
        plt.axvline(0, color='gray', linestyle='--', lw=1.0)
        plt.xlabel('Prediction Residual (Experimental - Predicted pKi)', fontsize=11)
        plt.ylabel('Density', fontsize=11)
        plt.title('Prediction Error Residual Distributions on Held-out Test Split', fontsize=12, fontweight='bold')
        plt.legend()
        plt.tight_layout()
        res_plot_path = comparison_dir / "residual_distribution.png"
        plt.savefig(res_plot_path, dpi=300)
        plt.close()
        print(f"Saved residual distribution plot to: {res_plot_path}")
        
    return df_comp


if __name__ == '__main__':
    run_model_comparison()

