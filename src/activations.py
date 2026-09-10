"""
Activation functions and their analytical derivatives for neural network layers.
Implemented strictly in pure NumPy.
"""

import numpy as np


class ReLU:
    """Rectified Linear Unit activation function."""

    def __init__(self):
        self.Z = None

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """
        Compute forward pass of ReLU.

        Parameters
        ----------
        Z : np.ndarray
            Linear pre-activation values of shape (batch_size, features).

        Returns
        -------
        np.ndarray
            Activated values of shape (batch_size, features), where A = max(0, Z).
        """
        self.Z = Z
        return np.maximum(0.0, Z)

    def backward(self, dA: np.ndarray) -> np.ndarray:
        """
        Compute backward pass of ReLU derivative.

        Parameters
        ----------
        dA : np.ndarray
            Upstream gradient of loss w.r.t activation output, shape (batch_size, features).

        Returns
        -------
        np.ndarray
            Gradient w.r.t linear pre-activation Z, shape (batch_size, features).
        """
        dZ = dA.copy()
        dZ[self.Z <= 0] = 0.0
        return dZ


class Tanh:
    """Hyperbolic Tangent activation function."""

    def __init__(self):
        self.A = None

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """
        Compute forward pass of Tanh.

        Parameters
        ----------
        Z : np.ndarray
            Linear pre-activation values of shape (batch_size, features).

        Returns
        -------
        np.ndarray
            Activated values of shape (batch_size, features) bounded in (-1, 1).
        """
        self.A = np.tanh(Z)
        return self.A

    def backward(self, dA: np.ndarray) -> np.ndarray:
        """
        Compute backward pass of Tanh derivative.

        Parameters
        ----------
        dA : np.ndarray
            Upstream gradient of loss w.r.t activation output, shape (batch_size, features).

        Returns
        -------
        np.ndarray
            Gradient w.r.t linear pre-activation Z, shape (batch_size, features),
            where dZ = dA * (1 - A^2).
        """
        return dA * (1.0 - np.square(self.A))


class Softmax:
    """Numerically stable Softmax activation for multiclass probability outputs."""

    def __init__(self):
        self.probs = None

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """
        Compute forward pass of Softmax with numerical stability shift.

        Parameters
        ----------
        Z : np.ndarray
            Logits of shape (batch_size, num_classes).

        Returns
        -------
        np.ndarray
            Normalized probabilities of shape (batch_size, num_classes),
            where sum along axis=-1 equals 1.0.
        """
        # Subtract max along class axis to avoid exponential overflow
        shifted_z = Z - np.max(Z, axis=-1, keepdims=True)
        exp_z = np.exp(shifted_z)
        self.probs = exp_z / np.sum(exp_z, axis=-1, keepdims=True)
        return self.probs

    def backward(self, dA: np.ndarray) -> np.ndarray:
        """
        Compute backward pass for Softmax layer given arbitrary upstream gradient dA.

        Parameters
        ----------
        dA : np.ndarray
            Upstream gradient of shape (batch_size, num_classes).

        Returns
        -------
        np.ndarray
            Gradient w.r.t pre-activation logits, shape (batch_size, num_classes).
        """
        # Batch Jacobian-vector product: dZ_i = P_i * (dA_i - sum(dA_k * P_k))
        sum_p_da = np.sum(self.probs * dA, axis=-1, keepdims=True)
        return self.probs * (dA - sum_p_da)
