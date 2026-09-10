"""
Training pipeline for the 35-class character recognition neural network.
Executes the mandated training loop sequence, validation loop, and metric logging.
Implemented strictly in pure NumPy and Matplotlib.
"""

import argparse
import os
import time
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np

from src.dataset import (
    CLASS_CHARS,
    load_and_preprocess_emnist,
    one_hot_encode,
    train_val_test_split,
)
from src.loss import CategoricalCrossEntropy
from src.metrics import compute_accuracy
from src.model import NeuralNetwork
from src.optimizer import SGDOptimizer


def train_network(
    model: NeuralNetwork,
    optimizer: SGDOptimizer,
    criterion: CategoricalCrossEntropy,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 25,
    batch_size: int = 128,
) -> Dict[str, List[float]]:
    """
    Execute training loop adhering to the strict sequential requirements:
    1. Forward propagation
    2. Backpropagation
    3. Activation computation
    4. Softmax output layer
    5. Cross-entropy loss
    6. Gradient descent update
    Followed by the epoch validation loop.

    Parameters
    ----------
    model : NeuralNetwork
        Multi-layer perceptron instance.
    optimizer : SGDOptimizer
        Gradient descent optimizer.
    criterion : CategoricalCrossEntropy
        Loss criterion.
    X_train : np.ndarray
        Training features of shape (N_train, 784).
    y_train : np.ndarray
        Training labels of shape (N_train,).
    X_val : np.ndarray
        Validation features of shape (N_val, 784).
    y_val : np.ndarray
        Validation labels of shape (N_val,).
    epochs : int
        Number of training epochs.
    batch_size : int
        Mini-batch size.

    Returns
    -------
    Dict[str, List[float]]
        Training history dictionary containing loss and accuracy trajectories.
    """
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    num_samples = X_train.shape[0]
    num_batches = int(np.ceil(num_samples / batch_size))
    Y_train_onehot = one_hot_encode(y_train, num_classes=len(CLASS_CHARS))
    Y_val_onehot = one_hot_encode(y_val, num_classes=len(CLASS_CHARS))

    best_val_acc = 0.0

    print("\n" + "=" * 78)
    print(f"Starting Training: {epochs} Epochs | Batch Size: {batch_size} | Samples: {num_samples}")
    print("=" * 78)

    for epoch in range(epochs):
        start_time = time.time()

        # Shuffle training set at the start of each epoch
        shuffle_idx = np.random.permutation(num_samples)
        X_shuffled = X_train[shuffle_idx]
        Y_shuffled = Y_train_onehot[shuffle_idx]
        y_labels_shuffled = y_train[shuffle_idx]

        running_loss = 0.0
        running_correct = 0

        # Mini-batch gradient descent loop
        for b in range(num_batches):
            b_start = b * batch_size
            b_end = min(b_start + batch_size, num_samples)

            xb = X_shuffled[b_start:b_end]
            yb = Y_shuffled[b_start:b_end]
            yb_int = y_labels_shuffled[b_start:b_end]

            # -------------------------------------------------------------
            # 1. Forward Propagation & 4. Softmax Layer
            # (Propagates through hidden layers, ReLU/Tanh, and Softmax)
            # -------------------------------------------------------------
            y_pred_probs = model.forward(xb)

            # -------------------------------------------------------------
            # 5. Cross-Entropy Loss
            # -------------------------------------------------------------
            batch_loss = criterion.forward(y_pred_probs, yb)
            running_loss += batch_loss * (b_end - b_start)

            # Track batch training accuracy
            preds = np.argmax(y_pred_probs, axis=-1)
            running_correct += int(np.sum(preds == yb_int))

            # -------------------------------------------------------------
            # 2. Backpropagation & 3. Activation Derivatives
            # (Computes gradients dL/dZ, dW, db through layers)
            # -------------------------------------------------------------
            grads = model.backward(yb)

            # -------------------------------------------------------------
            # 6. Gradient Descent Parameter Update
            # -------------------------------------------------------------
            optimizer.update(model.params, grads)

        # Decay learning rate if scheduled
        optimizer.step_decay(epoch)

        epoch_train_loss = running_loss / num_samples
        epoch_train_acc = running_correct / num_samples

        # -----------------------------------------------------------------
        # Validation Loop
        # -----------------------------------------------------------------
        val_probs = model.forward(X_val)
        val_loss = criterion.forward(val_probs, Y_val_onehot)
        val_acc = compute_accuracy(val_probs, y_val)

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch + 1:02d}/{epochs:02d} | "
            f"Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc * 100:.2f}% | "
            f"Time: {elapsed:.2f}s"
        )

        # Checkpoint best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc

    print("=" * 78)
    print(f"Training Complete! Peak Validation Accuracy: {best_val_acc * 100:.2f}%\n")
    return history


def plot_learning_curves(history: Dict[str, List[float]], save_path: str) -> None:
    """
    Plot and save training and validation loss & accuracy trajectories.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # Loss trajectory
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2)
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="#ff7f0e", linestyle="--", linewidth=2)
    ax1.set_title("Cross-Entropy Loss vs. Epochs", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=10)

    # Accuracy trajectory
    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], label="Train Accuracy", color="#2ca02c", linewidth=2)
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], label="Val Accuracy", color="#d62728", linestyle="--", linewidth=2)
    ax2.set_title("Classification Accuracy (%) vs. Epochs", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    print(f"[Training] Saved learning curves to: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train pure NumPy Character Recognition Neural Network")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=0.06, help="Initial learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum coefficient")
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "tanh"], help="Activation function")
    parser.add_argument("--samples_per_class", type=int, default=1000, help="Samples per class (>=50 required)")
    parser.add_argument("--dataset_path", type=str, default="DataSets/emnist-balanced-train.csv", help="Dataset path")
    parser.add_argument("--artifacts_dir", type=str, default="artifacts", help="Directory to save model & curves")
    args = parser.parse_args()

    os.makedirs(args.artifacts_dir, exist_ok=True)

    # 1. Load and prepare dataset (35 classes)
    X, y = load_and_preprocess_emnist(
        csv_path=args.dataset_path,
        samples_per_class=args.samples_per_class,
        random_seed=42,
    )

    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(
        X, y, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42
    )

    # Save test partition for standalone evaluate.py
    test_data_path = os.path.join(args.artifacts_dir, "test_data.npz")
    np.savez_compressed(test_data_path, X_test=X_test, y_test=y_test)
    print(f"[Data] Cached test partition ({X_test.shape[0]} samples) to: {test_data_path}")

    # 2. Build model architecture: 784 -> 256 -> 128 -> 35
    model = NeuralNetwork(
        layer_dims=[784, 256, 128, 35],
        activation=args.activation,
        random_seed=42,
    )

    optimizer = SGDOptimizer(
        learning_rate=args.lr,
        momentum=args.momentum,
        lr_decay=0.96,
        decay_step=5,
    )

    criterion = CategoricalCrossEntropy()

    # 3. Train network
    history = train_network(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    # 4. Save model weights and learning curves
    weights_path = os.path.join(args.artifacts_dir, "model_weights.npz")
    model.save_weights(weights_path)

    curves_path = os.path.join(args.artifacts_dir, "loss_accuracy_curves.png")
    plot_learning_curves(history, curves_path)


if __name__ == "__main__":
    main()
