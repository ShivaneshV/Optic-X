"""
Optic-X: Character Recognition Neural Network built from scratch in pure NumPy.
Custom implementation for 35-class handwritten character recognition (A-Z, 1-9).
"""

from src.activations import ReLU, Softmax, Tanh
from src.dataset import (
    CHAR_TO_IDX,
    CLASS_CHARS,
    IDX_TO_CHAR,
    load_and_preprocess_emnist,
    one_hot_encode,
    train_val_test_split,
)
from src.loss import CategoricalCrossEntropy
from src.metrics import (
    compute_accuracy,
    compute_classification_metrics,
    compute_confusion_matrix,
    find_misclassified_samples,
    plot_confusion_matrix,
)
from src.model import NeuralNetwork
from src.optimizer import SGDOptimizer

__version__ = "1.0.0"

__all__ = [
    "ReLU",
    "Tanh",
    "Softmax",
    "CategoricalCrossEntropy",
    "NeuralNetwork",
    "SGDOptimizer",
    "CLASS_CHARS",
    "CHAR_TO_IDX",
    "IDX_TO_CHAR",
    "load_and_preprocess_emnist",
    "train_val_test_split",
    "one_hot_encode",
    "compute_accuracy",
    "compute_confusion_matrix",
    "compute_classification_metrics",
    "find_misclassified_samples",
    "plot_confusion_matrix",
]
