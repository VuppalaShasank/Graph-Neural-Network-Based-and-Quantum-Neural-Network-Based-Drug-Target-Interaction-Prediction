"""
4-Model Comparison and Synthesis Pipeline
Compares all 4 models:
1. Model 1: GNN (Bond-Aware GINE)
2. Model 2: QNN (Variational Circuit)
3. Model 3: Proposed Hybrid (GNN + QNN)
4. Model 4: Novel Bi-Partite Quantum Docking Hybrid (BP-QDA)

Saves results to results/comparison_4models/ so that existing results/comparison/ is completely untouched!
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

from common.config import RESULTS_DIR


def run_4model_comparison(results_dir: Path = RESULTS_DIR):
    comparison_dir = results_dir / "comparison_4models"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    models = ["GNN", "QNN", "GNN_QNN", "Novel_Hybrid_BP_QDA"]
    model_dirs = {
        "GNN": "GNN",
        "QNN": "QNN",
        "GNN_QNN": "GNN_QNN",
        "Novel_Hybrid_BP_QDA": "Novel_Hybrid"
    }
    display_names = {
        "GNN": "GNN (Bond-Aware GINE)",
        "QNN": "QNN (Variational Circuit)",
        "GNN_QNN": "Hybrid (GNN + QNN)",
        "Novel_Hybrid_BP_QDA": "Novel Hybrid (BP-QDA)"
    }

    rows = []
    test_preds = {}

    for m in models:
        m_dir = results_dir / model_dirs[m]
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
            test_preds[m] = pd.read_csv(preds_file)

    if not rows:
        print("No evaluation metrics found!")
        return None

    df_comp = pd.DataFrame(rows)
    cols_order = ['model', 'display_name', 'rmse', 'mae', 'r2', 'pearson_r', 'spearman_rho', 'n_samples', 'training_time_seconds', 'best_epoch']
    cols_present = [c for c in cols_order if c in df_comp.columns]
    df_comp = df_comp[cols_present]

    comp_csv_path = comparison_dir / "model_comparison_4models.csv"
    df_comp.to_csv(comp_csv_path, index=False)
    print("\n" + "="*80)
    print(" 4-MODEL COMPREHENSIVE PERFORMANCE COMPARISON")
    print("="*80)
    print(df_comp.to_string(index=False))
    print(f"\nSaved 4-model comparison CSV to: {comp_csv_path}")

    # 1. Bar Chart Comparison
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    metrics_to_plot = [
        ('rmse', 'RMSE (Lower is better)'),
        ('mae', 'MAE (Lower is better)'),
        ('r2', 'R² (Higher is better)'),
        ('pearson_r', 'Pearson r (Higher is better)')
    ]
    colors = ['#3470a3', '#e07a5f', '#81b29a', '#9d4edd']

    for ax, (metric_col, title) in zip(axes, metrics_to_plot):
        if metric_col in df_comp.columns:
            sns.barplot(data=df_comp, x='model', y=metric_col, ax=ax, palette=colors)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('')
            ax.set_ylabel(metric_col.upper())
            ax.tick_params(axis='x', rotation=15)
            for p in ax.patches:
                val = p.get_height()
                ax.annotate(f"{val:.3f}", (p.get_x() + p.get_width() / 2., val / 2),
                            ha='center', va='center', color='white', fontweight='bold', fontsize=10)

    plt.suptitle('4-Model Performance Comparison (GNN vs QNN vs Hybrid vs Novel BP-QDA Hybrid)', fontsize=14, fontweight='bold', y=1.03)
    plt.tight_layout()
    bar_path = comparison_dir / "model_comparison_4models.png"
    plt.savefig(bar_path, dpi=300)
    plt.close()
    print(f"Saved 4-model metric bar comparison to: {bar_path}")

    # 2. Multi-panel Scatter Plot
    if len(test_preds) == 4:
        fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), sharex=True, sharey=True)
        sns.set_theme(style="whitegrid")
        colors = ['#2b5c8f', '#d95f02', '#2ca02c', '#7b2cbf']

        for ax, m, color in zip(axes, models, colors):
            p_df = test_preds[m]
            actual = p_df['pKi'].values
            pred = p_df['predicted_pKi'].values
            row = df_comp[df_comp['model'] == m].iloc[0]

            ax.scatter(actual, pred, alpha=0.35, edgecolors='none', color=color, s=25)
            min_val = min(actual.min(), pred.min()) - 0.5
            max_val = max(actual.max(), pred.max()) + 0.5
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1.5, label='Ideal (y = x)')

            ax.set_title(display_names[m], fontsize=11, fontweight='bold')
            ax.set_xlabel('Experimental pKi', fontsize=10)
            if ax == axes[0]:
                ax.set_ylabel('Predicted pKi', fontsize=10)

            textstr = f"RMSE: {row['rmse']:.3f}\nMAE: {row['mae']:.3f}\nR²: {row['r2']:.3f}\nPearson r: {row['pearson_r']:.3f}\nSpearman ρ: {row['spearman_rho']:.3f}"
            props = dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='gray')
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9, verticalalignment='top', bbox=props)
            ax.legend(loc='lower right', fontsize=8)

        plt.suptitle("Experimental vs Predicted Binding Affinity across 4 Models (Held-out Test N=1,500)", fontsize=13, fontweight='bold')
        plt.tight_layout()
        pred_comp_path = comparison_dir / "prediction_comparison_4models.png"
        plt.savefig(pred_comp_path, dpi=300)
        plt.close()
        print(f"Saved 4-model prediction comparison plot to: {pred_comp_path}")

    # 3. Residual Error Distribution
    if len(test_preds) == 4:
        plt.figure(figsize=(10, 5.5))
        sns.set_theme(style="whitegrid")
        for m, color in zip(models, ['#2b5c8f', '#d95f02', '#2ca02c', '#7b2cbf']):
            p_df = test_preds[m]
            residuals = p_df['residual'].values
            sns.kdeplot(residuals, label=display_names[m], color=color, lw=2.0)

        plt.axvline(0, color='gray', linestyle='--', lw=1.0)
        plt.xlabel('Prediction Residual (Experimental - Predicted pKi)', fontsize=11)
        plt.ylabel('Density', fontsize=11)
        plt.title('Prediction Error Residual Distributions Across 4 Models (Held-out Test Split)', fontsize=12, fontweight='bold')
        plt.legend()
        plt.tight_layout()
        res_plot_path = comparison_dir / "residual_distribution_4models.png"
        plt.savefig(res_plot_path, dpi=300)
        plt.close()
        print(f"Saved 4-model residual distribution plot to: {res_plot_path}")

    return df_comp


if __name__ == '__main__':
    run_4model_comparison()
