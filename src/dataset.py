"""
Dataset pipeline for EMNIST character classification (35 classes).
Extracts, transposes, normalizes, and splits uppercase letters A-Z and digits 1-9.
Implemented strictly in pure NumPy and standard library.
"""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


# 35 target class symbols: 9 digits ('1'-'9') + 26 uppercase letters ('A'-'Z')
CLASS_CHARS: List[str] = [str(d) for d in range(1, 10)] + [chr(c) for c in range(ord("A"), ord("Z") + 1)]
CHAR_TO_IDX: Dict[str, int] = {char: idx for idx, char in enumerate(CLASS_CHARS)}
IDX_TO_CHAR: Dict[int, str] = {idx: char for idx, char in enumerate(CLASS_CHARS)}

# EMNIST Balanced raw mapping:
# 1..9 correspond to ASCII 49..57 ('1'..'9') -> mapped to 0..8
# 10..35 correspond to ASCII 65..90 ('A'..'Z') -> mapped to 9..34
EMNIST_LABEL_TO_IDX: Dict[int, int] = {
    raw_lbl: (raw_lbl - 1) if raw_lbl <= 9 else (raw_lbl - 1)
    for raw_lbl in range(1, 36)
}


def load_and_preprocess_emnist(
    csv_path: str,
    samples_per_class: Optional[int] = 1000,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load EMNIST CSV, filter strictly to 35 classes, correct orientation, and normalize.

    Parameters
    ----------
    csv_path : str
        Path to EMNIST balanced CSV file (e.g., 'DataSets/emnist-balanced-train.csv').
    samples_per_class : int, optional
        Number of samples to retain per class for balanced training/evaluation.
        If None, all available samples for the 35 classes are retained.
    random_seed : int, optional
        Random seed for reproducible stratified sampling.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        X: Preprocessed float32 image array of shape (N, 784), values in [0.0, 1.0].
        y: Integer class labels of shape (N,), values in [0, 34].
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"EMNIST CSV not found at: {csv_path}")

    # Read CSV using pandas for fast chunk/file I/O
    print(f"Loading EMNIST dataset from {csv_path}...")
    df = pd.read_csv(csv_path, header=None)

    raw_labels = df.iloc[:, 0].to_numpy(dtype=np.int32)
    raw_pixels = df.iloc[:, 1:].to_numpy(dtype=np.float32)

    # Filter strictly to target classes (raw labels 1 through 35)
    valid_mask = (raw_labels >= 1) & (raw_labels <= 35)
    filtered_labels = raw_labels[valid_mask]
    filtered_pixels = raw_pixels[valid_mask]

    # Map raw EMNIST labels (1..35) to zero-indexed contiguous classes (0..34)
    mapped_labels = np.array([EMNIST_LABEL_TO_IDX[lbl] for lbl in filtered_labels], dtype=np.int64)

    # Stratified downsampling if samples_per_class is requested
    if samples_per_class is not None:
        rng = np.random.default_rng(random_seed)
        selected_indices = []
        for class_idx in range(len(CLASS_CHARS)):
            class_matches = np.where(mapped_labels == class_idx)[0]
            if len(class_matches) < samples_per_class:
                raise ValueError(
                    f"Class {class_idx} ('{IDX_TO_CHAR[class_idx]}') has only "
                    f"{len(class_matches)} samples, requested {samples_per_class}."
                )
            chosen = rng.choice(class_matches, size=samples_per_class, replace=False)
            selected_indices.append(chosen)

        selected_indices = np.concatenate(selected_indices)
        rng.shuffle(selected_indices)
        mapped_labels = mapped_labels[selected_indices]
        filtered_pixels = filtered_pixels[selected_indices]

    num_samples = filtered_pixels.shape[0]

    # Transposition fix:
    # EMNIST raw images are stored column-major (swapped axes).
    # Reshaping each image to (28, 28), transposing (.T), and flattening restores upright orientation.
    images_2d = filtered_pixels.reshape(num_samples, 28, 28)
    images_transposed = np.transpose(images_2d, (0, 2, 1))
    X_normalized = images_transposed.reshape(num_samples, 784) / 255.0

    print(f"Loaded {num_samples} samples across 35 classes. Feature shape: {X_normalized.shape}")
    return X_normalized.astype(np.float32), mapped_labels


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform stratified split into train, validation, and test partitions using pure NumPy.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (N, 784).
    y : np.ndarray
        Labels of shape (N,).
    train_ratio : float
        Fraction of data for training (default 0.70).
    val_ratio : float
        Fraction of data for validation (default 0.15).
    test_ratio : float
        Fraction of data for testing (default 0.15).
    random_seed : int
        Random seed for shuffling.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {total}")

    rng = np.random.default_rng(random_seed)
    num_classes = len(np.unique(y))

    train_idx, val_idx, test_idx = [], [], []

    for c in range(num_classes):
        c_indices = np.where(y == c)[0]
        rng.shuffle(c_indices)
        n_c = len(c_indices)

        n_train = int(n_c * train_ratio)
        n_val = int(n_c * val_ratio)

        train_idx.append(c_indices[:n_train])
        val_idx.append(c_indices[n_train : n_train + n_val])
        test_idx.append(c_indices[n_train + n_val :])

    train_idx = np.concatenate(train_idx)
    val_idx = np.concatenate(val_idx)
    test_idx = np.concatenate(test_idx)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return (
        X[train_idx],
        y[train_idx],
        X[val_idx],
        y[val_idx],
        X[test_idx],
        y[test_idx],
    )


def one_hot_encode(y: np.ndarray, num_classes: int = 35) -> np.ndarray:
    """
    Convert 1D class labels to one-hot encoded matrix.

    Parameters
    ----------
    y : np.ndarray
        Integer class indices of shape (N,).
    num_classes : int
        Total number of classes (default 35).

    Returns
    -------
    np.ndarray
        One-hot matrix of shape (N, num_classes).
    """
    one_hot = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    one_hot[np.arange(y.shape[0]), y] = 1.0
    return one_hot
