"""Publication-quality figures for the paper/README."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    precision_recall_curve, roc_curve,
    average_precision_score, roc_auc_score,
)
from sklearn.calibration import calibration_curve


def plot_calibration(y_true, y_prob_raw, y_prob_cal=None,
                     brier_raw=None, brier_cal=None, out_path=None):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfectly calibrated')
    p_true_r, p_pred_r = calibration_curve(y_true, y_prob_raw, n_bins=10, strategy='quantile')
    lbl = 'Raw model' + (f' (Brier={brier_raw:.3f})' if brier_raw is not None else '')
    ax.plot(p_pred_r, p_true_r, 'o-', label=lbl, linewidth=2, markersize=8)
    if y_prob_cal is not None:
        p_true_c, p_pred_c = calibration_curve(y_true, y_prob_cal, n_bins=10, strategy='quantile')
        lbl_c = 'Platt-scaled' + (f' (Brier={brier_cal:.3f})' if brier_cal is not None else '')
        ax.plot(p_pred_c, p_true_c, 's-', label=lbl_c, linewidth=2, markersize=8)
    ax.set_xlabel('Predicted probability'); ax.set_ylabel('Observed frequency')
    ax.set_title('Calibration curve (reliability diagram)')
    ax.legend(loc='lower right'); ax.grid(alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()
    if out_path: plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_pr_curve(y_true, scores_dict, out_path=None):
    fig, ax = plt.subplots(figsize=(7, 6))
    prevalence = float(np.mean(y_true))
    ax.axhline(prevalence, linestyle='--', color='gray',
               label=f'Baseline (prevalence={prevalence:.2f})')
    for name, scores in scores_dict.items():
        prec, rec, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        ax.plot(rec, prec, label=f'{name} (PR-AUC={ap:.3f})', linewidth=2)
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall curves (Test Set)')
    ax.legend(loc='lower left'); ax.grid(alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    plt.tight_layout()
    if out_path: plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc_multi(y_true, scores_dict, out_path=None):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    for name, scores in scores_dict.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)
        ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', linewidth=2)
    ax.set_xlabel('False Positive Rate (1 − Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.set_title('ROC curves (Test Set)')
    ax.legend(loc='lower right'); ax.grid(alpha=0.3)
    plt.tight_layout()
    if out_path: plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_cv_comparison(cv_df, out_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric in zip(axes, ['roc_auc', 'average_precision']):
        sub = cv_df[cv_df['metric'] == metric]
        data = [row['per_fold'] for _, row in sub.iterrows()]
        labels = sub['model'].tolist()
        bp = ax.boxplot(data, labels=labels, showmeans=True, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('#89CFF0'); patch.set_alpha(0.7)
        pretty = 'ROC-AUC' if metric == 'roc_auc' else 'PR-AUC'
        ax.set_ylabel(pretty)
        ax.set_title(f'5-Fold Stratified CV — {pretty}')
        ax.grid(alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=15)
    plt.suptitle('Cross-Validation Comparison on Training Set',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    if out_path: plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
