"""
Loss functions and gradient computations for multiclass classification.
Implemented strictly in pure NumPy.
"""

import numpy as np


class CategoricalCrossEntropy:
    """Multiclass Categorical Cross-Entropy Loss with epsilon smoothing."""

    def __init__(self, eps: float = 1e-15):
        """
        Initialize loss with numerical stability epsilon.

        Parameters
        ----------
        eps : float, optional
            Small float to clip probabilities avoiding log(0), default is 1e-15.
        """
        self.eps = eps
        self.y_pred = None
        self.y_true = None

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute mean categorical cross-entropy loss over the batch.

        Parameters
        ----------
        y_pred : np.ndarray
            Predicted class probabilities of shape (batch_size, num_classes).
        y_true : np.ndarray
            Ground truth one-hot encoded targets of shape (batch_size, num_classes)
            or 1D class indices of shape (batch_size,).

        Returns
        -------
        float
            Scalar cross-entropy loss averaged across the batch.
        """
        self.y_pred = y_pred
        batch_size = y_pred.shape[0]

        # Clip probabilities to prevent log(0) and numerical instability
        y_pred_clipped = np.clip(y_pred, self.eps, 1.0 - self.eps)

        if y_true.ndim == 1:
            # y_true is integer class indices
            self.y_true = np.zeros_like(y_pred)
            self.y_true[np.arange(batch_size), y_true] = 1.0
            log_likelihood = -np.log(y_pred_clipped[np.arange(batch_size), y_true])
            return float(np.mean(log_likelihood))
        else:
            # y_true is one-hot encoded
            self.y_true = y_true
            loss = -np.sum(y_true * np.log(y_pred_clipped)) / batch_size
            return float(loss)

    def backward(self) -> np.ndarray:
        """
        Compute gradient of cross-entropy loss w.r.t predicted probabilities.

        Returns
        -------
        np.ndarray
            Gradient dL/dy_pred of shape (batch_size, num_classes).
        """
        batch_size = self.y_pred.shape[0]
        y_pred_clipped = np.clip(self.y_pred, self.eps, 1.0 - self.eps)
        return -(self.y_true / y_pred_clipped) / batch_size

    def gradient_wrt_logits(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Compute exact gradient of Cross-Entropy + Softmax w.r.t pre-activation logits Z.
        Exploits the analytical simplification: dL/dZ = (P - Y) / batch_size.

        Parameters
        ----------
        y_pred : np.ndarray
            Softmax output probabilities of shape (batch_size, num_classes).
        y_true : np.ndarray
            One-hot targets of shape (batch_size, num_classes) or 1D indices (batch_size,).

        Returns
        -------
        np.ndarray
            Gradient dL/dZ of shape (batch_size, num_classes).
        """
        batch_size = y_pred.shape[0]
        if y_true.ndim == 1:
            target_one_hot = np.zeros_like(y_pred)
            target_one_hot[np.arange(batch_size), y_true] = 1.0
        else:
            target_one_hot = y_true

        return (y_pred - target_one_hot) / batch_size
