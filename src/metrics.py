"""
Evaluation metrics, confusion matrix computation, and error analysis.
Implemented strictly in pure NumPy and Matplotlib (zero scikit-learn dependency).
"""

from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
from src.dataset import CLASS_CHARS, IDX_TO_CHAR


def compute_accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Compute multiclass classification accuracy.

    Parameters
    ----------
    y_pred : np.ndarray
        Predicted class indices of shape (N,) or probabilities of shape (N, C).
    y_true : np.ndarray
        Ground-truth class indices of shape (N,) or one-hot targets of shape (N, C).

    Returns
    -------
    float
        Fraction of correctly classified samples in [0.0, 1.0].
    """
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=-1)
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=-1)

    return float(np.mean(y_pred == y_true))


def compute_confusion_matrix(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    num_classes: int = 35,
) -> np.ndarray:
    """
    Compute a (num_classes, num_classes) confusion matrix strictly in NumPy.
    Row i denotes true class, Column j denotes predicted class.

    Parameters
    ----------
    y_pred : np.ndarray
        Predicted class indices of shape (N,).
    y_true : np.ndarray
        Ground-truth class indices of shape (N,).
    num_classes : int, optional
        Total number of classes (default 35).

    Returns
    -------
    np.ndarray
        Integer confusion matrix of shape (num_classes, num_classes).
    """
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=-1)
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=-1)

    # Vectorized indexing using 1D bin counting
    flat_indices = num_classes * y_true.astype(np.int64) + y_pred.astype(np.int64)
    cm = np.bincount(flat_indices, minlength=num_classes**2)
    return cm.reshape(num_classes, num_classes)


def compute_classification_metrics(
    cm: np.ndarray,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """
    Compute Macro-Averaged Precision, Recall, and F1-Score from confusion matrix.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix of shape (C, C).
    eps : float, optional
        Small epsilon to prevent division by zero.

    Returns
    -------
    Dict[str, float]
        Dictionary with 'accuracy', 'macro_precision', 'macro_recall', and 'macro_f1'.
    """
    num_classes = cm.shape[0]
    tp = np.diag(cm).astype(np.float64)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp

    precision_per_class = tp / (tp + fp + eps)
    recall_per_class = tp / (tp + fn + eps)
    f1_per_class = 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class + eps)

    total_samples = np.sum(cm)
    accuracy = float(np.sum(tp) / total_samples) if total_samples > 0 else 0.0

    return {
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precision_per_class)),
        "macro_recall": float(np.mean(recall_per_class)),
        "macro_f1": float(np.mean(f1_per_class)),
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = CLASS_CHARS,
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix (35 Classes)",
) -> None:
    """
    Render and optionally save a high-resolution confusion matrix heatmap.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix of shape (35, 35).
    class_names : List[str]
        List of 35 character symbols.
    save_path : str, optional
        Path to save rendered image.
    title : str, optional
        Plot title.
    """
    fig, ax = plt.subplots(figsize=(14, 12), dpi=300)
    cax = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title, fontsize=16, pad=16, fontweight="bold")
    plt.colorbar(cax, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)

    ax.set_ylabel("True Class", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Class", fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[Metrics] Saved confusion matrix heatmap to: {save_path}")
    plt.close()


def find_misclassified_samples(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    top_k: int = 5,
) -> List[Dict]:
    """
    Identify misclassified samples with highest model confidence.

    Parameters
    ----------
    X : np.ndarray
        Images array of shape (N, 784).
    y_true : np.ndarray
        Ground-truth class indices of shape (N,).
    y_pred_probs : np.ndarray
        Predicted probabilities of shape (N, 35).
    top_k : int, optional
        Number of incorrect cases to extract (default 5).

    Returns
    -------
    List[Dict]
        List of dicts containing image, true_char, pred_char, and confidence values.
    """
    y_pred = np.argmax(y_pred_probs, axis=-1)
    incorrect_mask = (y_pred != y_true)
    incorrect_indices = np.where(incorrect_mask)[0]

    if len(incorrect_indices) == 0:
        return []

    # Sort incorrect predictions by highest confidence in the erroneous class
    pred_confidences = np.max(y_pred_probs[incorrect_indices], axis=-1)
    sorted_order = np.argsort(-pred_confidences)
    chosen_indices = incorrect_indices[sorted_order[:top_k]]

    results = []
    for idx in chosen_indices:
        p_class = int(y_pred[idx])
        t_class = int(y_true[idx])
        results.append({
            "sample_index": int(idx),
            "image": X[idx],
            "true_idx": t_class,
            "true_char": IDX_TO_CHAR[t_class],
            "pred_idx": p_class,
            "pred_char": IDX_TO_CHAR[p_class],
            "pred_confidence": float(y_pred_probs[idx, p_class]),
            "true_confidence": float(y_pred_probs[idx, t_class]),
        })

    return results
