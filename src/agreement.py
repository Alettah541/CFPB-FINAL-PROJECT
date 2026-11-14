"""Shared agreement helpers used by final_project.ipynb and backtest_governance.ipynb."""
import numpy as np
from sklearn.metrics import cohen_kappa_score

THRESHOLD_SDS = 1


def mdd_from(diffs, threshold_sds=THRESHOLD_SDS):
    """MDD: mean rater disagreement + threshold_sds * SD."""
    return np.mean(diffs) + threshold_sds * np.std(diffs)


def classification_kappa(rater_a, rater_b, labels):
    """Unweighted Cohen's kappa between two raters' category labels."""
    return cohen_kappa_score(rater_a, rater_b, labels=labels)
