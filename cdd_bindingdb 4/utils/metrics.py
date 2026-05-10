"""Drug discovery evaluation metrics."""

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


def pearson_r(y_true, y_pred):
    if len(y_true) < 2:
        return 0.0
    r, _ = stats.pearsonr(y_true, y_pred)
    return float(r)


def spearman_rho(y_true, y_pred):
    """Rank correlation. Most important metric for virtual screening."""
    if len(y_true) < 2:
        return 0.0
    rho, _ = stats.spearmanr(y_true, y_pred)
    return float(rho)


def roc_auc(y_true, y_pred, threshold=6.0):
    """pIC50 threshold of 6.0 corresponds to IC50 of 1000 nM."""
    y_binary = (np.array(y_true) >= threshold).astype(int)
    if y_binary.sum() == 0 or y_binary.sum() == len(y_binary):
        return 0.5
    return float(roc_auc_score(y_binary, y_pred))


def enrichment_factor(y_true, y_pred, top_fraction=0.01, threshold=6.0):
    """
    Enrichment factor at top X%. EF=10 means the model finds 10x
    more actives in the top X% than random selection would.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n_total = len(y_true)
    n_top = max(1, int(n_total * top_fraction))

    top_idx = np.argsort(-y_pred)[:n_top]
    is_active = (y_true >= threshold)
    hits = is_active[top_idx].sum()
    total_actives = is_active.sum()

    if total_actives == 0:
        return 0.0

    hit_rate_top = hits / n_top
    hit_rate_random = total_actives / n_total
    return float(hit_rate_top / hit_rate_random) if hit_rate_random > 0 else 0.0


def evaluate_all(y_true, y_pred, threshold=6.0):
    """Compute all metrics in one call."""
    return {
        'rmse': rmse(y_true, y_pred),
        'mae': mae(y_true, y_pred),
        'pearson_r': pearson_r(y_true, y_pred),
        'spearman_rho': spearman_rho(y_true, y_pred),
        'roc_auc': roc_auc(y_true, y_pred, threshold),
        'enrichment_1pct': enrichment_factor(y_true, y_pred, 0.01, threshold),
        'enrichment_5pct': enrichment_factor(y_true, y_pred, 0.05, threshold),
        'n_total': len(y_true),
        'n_active': int((np.array(y_true) >= threshold).sum()),
    }


def print_results(metrics):
    print("\nResults")
    print(f"  Samples:          {metrics['n_total']:>8,}")
    print(f"  Active:           {metrics['n_active']:>8,}")
    print(f"  RMSE (pIC50):     {metrics['rmse']:>8.4f}")
    print(f"  MAE (pIC50):      {metrics['mae']:>8.4f}")
    print(f"  Pearson R:        {metrics['pearson_r']:>8.4f}")
    print(f"  Spearman rho:     {metrics['spearman_rho']:>8.4f}")
    print(f"  ROC-AUC:          {metrics['roc_auc']:>8.4f}")
    print(f"  Enrichment @1%:   {metrics['enrichment_1pct']:>8.2f}x")
    print(f"  Enrichment @5%:   {metrics['enrichment_5pct']:>8.2f}x")
