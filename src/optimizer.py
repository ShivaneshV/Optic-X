"""
Optimization algorithms for updating neural network parameters.
Implemented strictly in pure NumPy.
"""

from typing import Dict, List
import numpy as np


class SGDOptimizer:
    """
    Mini-Batch Stochastic Gradient Descent with optional momentum and learning rate decay.
    """

    def __init__(
        self,
        learning_rate: float = 0.05,
        momentum: float = 0.9,
        lr_decay: float = 0.98,
        decay_step: int = 5,
    ):
        """
        Initialize SGD optimizer.

        Parameters
        ----------
        learning_rate : float, optional
            Initial step size (default is 0.05).
        momentum : float, optional
            Momentum factor in [0, 1). Set to 0.0 for standard gradient descent (default 0.9).
        lr_decay : float, optional
            Multiplicative factor of learning rate decay (default 0.98).
        decay_step : int, optional
            Decay learning rate every `decay_step` epochs (default 5).
        """
        self.initial_lr = learning_rate
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.lr_decay = lr_decay
        self.decay_step = decay_step
        self.velocities: Dict[str, np.ndarray] = {}

    def update(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]) -> None:
        """
        Update model parameters in-place using accumulated gradients.

        Parameters
        ----------
        params : Dict[str, np.ndarray]
            Dictionary of trainable parameter arrays (e.g., 'W1', 'b1', 'W2', 'b2').
        grads : Dict[str, np.ndarray]
            Dictionary of gradient arrays matching params keys (e.g., 'dW1', 'db1').
        """
        for key in params.keys():
            grad_key = f"d{key}"
            if grad_key not in grads:
                continue

            grad = grads[grad_key]

            # Initialize velocity buffer for momentum
            if key not in self.velocities:
                self.velocities[key] = np.zeros_like(params[key])

            if self.momentum > 0.0:
                # Velocity update: v = momentum * v + lr * grad
                self.velocities[key] = self.momentum * self.velocities[key] + self.learning_rate * grad
                params[key] -= self.velocities[key]
            else:
                # Standard gradient descent: param = param - lr * grad
                params[key] -= self.learning_rate * grad

    def step_decay(self, epoch: int) -> None:
        """
        Decay learning rate at specified epoch intervals.

        Parameters
        ----------
        epoch : int
            Current 0-indexed training epoch.
        """
        if (epoch + 1) % self.decay_step == 0:
            self.learning_rate *= self.lr_decay
            print(f"[Optimizer] Decayed learning rate to: {self.learning_rate:.6f}")
