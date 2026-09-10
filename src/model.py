"""
Neural Network implementation for multiclass character recognition.
Built strictly from scratch using pure NumPy matrix operations.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from src.activations import ReLU, Softmax, Tanh


class NeuralNetwork:
    """
    Multilayer Perceptron (MLP) for 35-class character recognition.

    Architecture Design Justification:
    A 784 -> 256 -> 128 -> 35 architecture is selected over a shallow single-layer network
    to establish a 2-stage hierarchical representation. The first hidden layer (256 units)
    models low-level stroke primitives (edges, loops, crossbars), while the second hidden
    layer (128 units) combines these into structural character compositions (differentiating
    visually similar pairs like '5' vs 'S', '1' vs 'I', or '8' vs 'B').

    Parameters are initialized using He (Kaiming) normal initialization when paired with ReLU,
    maintaining variance Var(W) = 2 / n_in to guard against vanishing/exploding gradients.
    """

    def __init__(
        self,
        layer_dims: Optional[List[int]] = None,
        activation: str = "relu",
        random_seed: int = 42,
    ):
        """
        Initialize network weights, biases, and activation layers.

        Parameters
        ----------
        layer_dims : List[int], optional
            Sequence of layer dimensions [input_dim, hidden_1, ..., output_dim].
            Default is [784, 256, 128, 35].
        activation : str, optional
            Hidden layer activation ('relu' or 'tanh'), default is 'relu'.
        random_seed : int, optional
            Seed for reproducible parameter initialization.
        """
        if layer_dims is None:
            layer_dims = [784, 256, 128, 35]

        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims) - 1
        self.activation_name = activation.lower()
        self.random_seed = random_seed

        # Initialize activation modules
        self.hidden_activations: List[Union[ReLU, Tanh]] = []
        for _ in range(self.num_layers - 1):
            if self.activation_name == "relu":
                self.hidden_activations.append(ReLU())
            elif self.activation_name == "tanh":
                self.hidden_activations.append(Tanh())
            else:
                raise ValueError(f"Unsupported activation: {activation}. Choose 'relu' or 'tanh'.")

        self.softmax = Softmax()

        # Parameter dictionaries
        self.params: Dict[str, np.ndarray] = {}
        self.cache: Dict[str, np.ndarray] = {}
        self._init_parameters()

    def _init_parameters(self) -> None:
        """
        Initialize weights and biases using He or Xavier normal distributions.
        W shape: (n_in, n_out), b shape: (1, n_out).
        """
        rng = np.random.default_rng(self.random_seed)

        for l in range(1, len(self.layer_dims)):
            n_in = self.layer_dims[l - 1]
            n_out = self.layer_dims[l]

            if self.activation_name == "relu" and l < self.num_layers:
                # He (Kaiming) normal initialization for ReLU
                std = np.sqrt(2.0 / n_in)
            else:
                # Xavier (Glorot) normal initialization for Tanh or final layer
                std = np.sqrt(1.0 / n_in)

            self.params[f"W{l}"] = rng.normal(loc=0.0, scale=std, size=(n_in, n_out)).astype(np.float32)
            self.params[f"b{l}"] = np.zeros((1, n_out), dtype=np.float32)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Execute forward propagation through hidden layers and output Softmax.

        Parameters
        ----------
        X : np.ndarray
            Input feature batch of shape (batch_size, 784).

        Returns
        -------
        np.ndarray
            Predicted class probabilities of shape (batch_size, 35).
        """
        self.cache = {"A0": X}
        current_A = X

        # Hidden dense layers + activations
        # e.g., layer 1: (batch_size, 784) x (784, 256) -> (batch_size, 256)
        #       layer 2: (batch_size, 256) x (256, 128) -> (batch_size, 128)
        for l in range(1, self.num_layers):
            W = self.params[f"W{l}"]
            b = self.params[f"b{l}"]

            # Z shape: (batch_size, hidden_dim)
            Z = np.dot(current_A, W) + b
            current_A = self.hidden_activations[l - 1].forward(Z)

            self.cache[f"Z{l}"] = Z
            self.cache[f"A{l}"] = current_A

        # Output dense layer (logits)
        W_out = self.params[f"W{self.num_layers}"]
        b_out = self.params[f"b{self.num_layers}"]
        Z_out = np.dot(current_A, W_out) + b_out  # shape: (batch_size, 35)

        # Softmax normalization
        probs = self.softmax.forward(Z_out)  # shape: (batch_size, 35)

        self.cache[f"Z{self.num_layers}"] = Z_out
        self.cache[f"A{self.num_layers}"] = probs
        return probs

    def backward(self, y_true: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Execute backpropagation to compute analytical gradients for all layers.

        Parameters
        ----------
        y_true : np.ndarray
            Ground-truth labels as 1D class indices (batch_size,) or one-hot (batch_size, 35).

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing parameter gradients: 'dW1', 'db1', 'dW2', 'db2', etc.
        """
        batch_size = self.cache["A0"].shape[0]
        y_pred = self.cache[f"A{self.num_layers}"]
        grads: Dict[str, np.ndarray] = {}

        # 1. Output layer error: dZ = (P - Y) / batch_size
        if y_true.ndim == 1:
            Y = np.zeros_like(y_pred)
            Y[np.arange(batch_size), y_true] = 1.0
        else:
            Y = y_true

        dZ = (y_pred - Y) / batch_size  # shape: (batch_size, 35)

        # 2. Output layer parameter gradients
        A_prev = self.cache[f"A{self.num_layers - 1}"]
        grads[f"dW{self.num_layers}"] = np.dot(A_prev.T, dZ)  # (128, 35)
        grads[f"db{self.num_layers}"] = np.sum(dZ, axis=0, keepdims=True)  # (1, 35)

        # 3. Propagate backwards through hidden layers
        for l in range(self.num_layers - 1, 0, -1):
            W_next = self.params[f"W{l + 1}"]
            # Upstream gradient arriving at layer l activation
            dA = np.dot(dZ, W_next.T)  # (batch_size, hidden_dim)

            # Local gradient through activation derivative
            dZ = self.hidden_activations[l - 1].backward(dA)  # (batch_size, hidden_dim)

            A_prev = self.cache[f"A{l - 1}"]
            grads[f"dW{l}"] = np.dot(A_prev.T, dZ)  # (prev_dim, current_dim)
            grads[f"db{l}"] = np.sum(dZ, axis=0, keepdims=True)  # (1, current_dim)

        return grads

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class indices for given input samples.

        Parameters
        ----------
        X : np.ndarray
            Input samples of shape (N, 784).

        Returns
        -------
        np.ndarray
            Predicted class indices of shape (N,).
        """
        probs = self.forward(X)
        return np.argmax(probs, axis=-1)

    def save_weights(self, filepath: str) -> None:
        """Save network parameters to a compressed .npz archive."""
        np.savez_compressed(filepath, **self.params)
        print(f"[Model] Saved weights to {filepath}")

    def load_weights(self, filepath: str) -> None:
        """Load network parameters from a .npz archive."""
        data = np.load(filepath)
        for key in self.params.keys():
            if key in data:
                self.params[key] = data[key]
        print(f"[Model] Successfully restored weights from {filepath}")
