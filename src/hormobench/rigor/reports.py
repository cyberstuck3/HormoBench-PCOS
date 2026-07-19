"""Markdown-table generators (paper/README style)."""


PRETTY = {
    'roc_auc':           'ROC-AUC',
    'pr_auc':            'PR-AUC',
    'brier':             'Brier score',
    'precision':         'Precision (PPV)',
    'recall':            'Recall (Sensitivity)',
    'specificity':       'Specificity',
    'f1':                'F1-score',
    'balanced_accuracy': 'Balanced accuracy',
    'mcc':               'Matthews CC',
}
ORDER = ['roc_auc', 'pr_auc', 'brier',
         'recall', 'specificity', 'precision',
         'f1', 'balanced_accuracy', 'mcc']


def metrics_table_md(metrics, title):
    lines = [f"### {title}", "",
             "| Metric | Point estimate | 95% CI |",
             "|--------|---------------:|:------:|"]
    for k in ORDER:
        if k in metrics:
            m = metrics[k]
            lines.append(f"| {PRETTY[k]} | {m['point']:.3f} | [{m['lo']:.3f}, {m['hi']:.3f}] |")
    return '\n'.join(lines) + '\n'


def cv_table_md(cv_df):
    lines = ["### Cross-Validation Results (5-fold stratified, training set)", "",
             "| Model | ROC-AUC (mean ± SD) | PR-AUC (mean ± SD) |",
             "|-------|:-------------------:|:------------------:|"]
    for m in cv_df['model'].unique():
        roc = cv_df[(cv_df['model'] == m) & (cv_df['metric'] == 'roc_auc')].iloc[0]
        pr  = cv_df[(cv_df['model'] == m) & (cv_df['metric'] == 'average_precision')].iloc[0]
        lines.append(f"| {m} | {roc['mean']:.3f} ± {roc['std']:.3f} | {pr['mean']:.3f} ± {pr['std']:.3f} |")
    return '\n'.join(lines) + '\n'


def delong_table_md(pairwise):
    lines = ["### DeLong pairwise ROC-AUC comparison (test set)", "",
             "| Pair | ΔAUC | z | p-value | Significant (α=0.05) |",
             "|------|-----:|--:|--------:|:--------------------:|"]
    for pair, res in pairwise.items():
        sig = "**Yes**" if res['significant_at_0.05'] else "No"
        lines.append(f"| {pair} | {res['delta']:+.4f} | {res['z']:+.2f} | {res['p_value']:.3f} | {sig} |")
    return '\n'.join(lines) + '\n'


def calibration_table_md(cal_raw, cal_after=None):
    has_platt = cal_after is not None
    header = "| Metric | Raw model |" + (" Platt-scaled |" if has_platt else "")
    align  = "|--------|----------:|" + ("-------------:|" if has_platt else "")
    lines = ["### Calibration analysis (test set)", "", header, align]
    def row(name, key):
        s = f"| {name} | {cal_raw[key]:.4f} |"
        if has_platt: s += f" {cal_after[key]:.4f} |"
        return s
    lines.append(row("Brier score", 'brier'))
    lines.append(row("Expected Calibration Error (10 quantile bins)", 'ece'))
    return '\n'.join(lines) + '\n'
