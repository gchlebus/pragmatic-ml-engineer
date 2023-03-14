from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def compute_metrics(y_true, y_pred, average="macro", round=3):
    """Compute metrics for a classification task"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average=average)
    }
    if round:
        for k, v in metrics.items():
            metrics[k] = np.round(v, round)
    return metrics