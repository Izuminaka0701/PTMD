"""Evaluation metrics for G1/G2/G3 classification."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

GROUP_NAMES = ["G1_GetProcAddress", "G2_PEB_Traversal", "G3_Hash_Encrypted"]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_per_class": {
            GROUP_NAMES[i]: float(f1_score(y_true, y_pred, labels=[i], average="macro", zero_division=0))
            for i in range(len(GROUP_NAMES))
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }


def classification_report_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    report = classification_report(
        y_true, y_pred, target_names=GROUP_NAMES, output_dict=True, zero_division=0
    )
    return report
