"""
Evaluation script for test set performance, confusion matrix generation,
and comprehensive error analysis of misclassified character predictions.
Implemented strictly in pure NumPy and Matplotlib.
"""

import argparse
import json
import os
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np

from src.dataset import CLASS_CHARS, IDX_TO_CHAR, load_and_preprocess_emnist
from src.metrics import (
    compute_accuracy,
    compute_classification_metrics,
    compute_confusion_matrix,
    find_misclassified_samples,
    plot_confusion_matrix,
)
from src.model import NeuralNetwork


def render_error_samples(errors: List[Dict], save_path: str) -> None:
    """
    Render and save side-by-side visual plots of misclassified characters
    with ground truth, predicted class, and probability confidence.

    Parameters
    ----------
    errors : List[Dict]
        List of error dictionaries from find_misclassified_samples.
    save_path : str
        File path to save the generated figure.
    """
    num_samples = len(errors)
    fig, axes = plt.subplots(1, num_samples, figsize=(3.2 * num_samples, 3.8), dpi=300)

    if num_samples == 1:
        axes = [axes]

    for i, err in enumerate(errors):
        ax = axes[i]
        # Reshape flat 784 array back into 28x28 image
        img = err["image"].reshape(28, 28)
        ax.imshow(img, cmap="gray", interpolation="nearest")
        ax.axis("off")

        title_text = (
            f"True: '{err['true_char']}'\n"
            f"Pred: '{err['pred_char']}' ({err['pred_confidence'] * 100:.1f}%)\n"
            f"True Conf: {err['true_confidence'] * 100:.1f}%"
        )
        ax.set_title(title_text, fontsize=11, pad=8, color="#b30000", fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    print(f"[Evaluation] Saved error analysis visual grid to: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate Character Recognition Model on Test Set")
    parser.add_argument("--weights_path", type=str, default="artifacts/model_weights.npz", help="Trained weights path")
    parser.add_argument("--test_data_path", type=str, default="artifacts/test_data.npz", help="Test partition path")
    parser.add_argument("--artifacts_dir", type=str, default="artifacts", help="Directory to save evaluation artifacts")
    args = parser.parse_args()

    os.makedirs(args.artifacts_dir, exist_ok=True)

    # 1. Load test data
    if os.path.exists(args.test_data_path):
        data = np.load(args.test_data_path)
        X_test = data["X_test"]
        y_test = data["y_test"]
        print(f"[Evaluation] Loaded test partition from {args.test_data_path}: {X_test.shape[0]} samples.")
    else:
        print("[Evaluation] Cached test split not found. Loading test set from raw CSV...")
        X_test, y_test = load_and_preprocess_emnist("DataSets/emnist-balanced-test.csv", samples_per_class=150)

    # 2. Instantiate model and restore trained parameters
    model = NeuralNetwork(layer_dims=[784, 256, 128, 35], activation="relu")
    model.load_weights(args.weights_path)

    # 3. Compute predictions and test probabilities
    y_pred_probs = model.forward(X_test)
    y_pred = np.argmax(y_pred_probs, axis=-1)

    # 4. Compute metrics and confusion matrix
    test_acc = compute_accuracy(y_pred, y_test)
    cm = compute_confusion_matrix(y_pred, y_test, num_classes=len(CLASS_CHARS))
    metrics = compute_classification_metrics(cm)

    print("\n" + "=" * 65)
    print("TEST SET EVALUATION METRICS")
    print("=" * 65)
    print(f"Total Test Samples:   {X_test.shape[0]}")
    print(f"Overall Accuracy:     {test_acc * 100:.2f}%")
    print(f"Macro Precision:      {metrics['macro_precision'] * 100:.2f}%")
    print(f"Macro Recall:         {metrics['macro_recall'] * 100:.2f}%")
    print(f"Macro F1-Score:       {metrics['macro_f1'] * 100:.2f}%")
    print("=" * 65)

    # 5. Plot confusion matrix
    cm_path = os.path.join(args.artifacts_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, class_names=CLASS_CHARS, save_path=cm_path)

    # 6. Extract and analyze top misclassified cases
    errors = find_misclassified_samples(X_test, y_test, y_pred_probs, top_k=5)
    error_plot_path = os.path.join(args.artifacts_dir, "error_analysis_samples.png")
    render_error_samples(errors, error_plot_path)

    # Structural error descriptions
    typographical_explanations = {
        ("1", "I"): "Topological ambiguity: sans-serif vertical stroke for digit '1' is geometrically indistinguishable from capital letter 'I'.",
        ("I", "1"): "Sans-serif stroke: vertical straight bar lack crossbars, causing high-confidence confusion between 'I' and '1'.",
        ("5", "S"): "Curvature smoothing: handwriting rounds the sharp top corner of digit '5', merging into the smooth sigmoid shape of 'S'.",
        ("S", "5"): "Angular stroke: fast handwriting adds sharp inflection points to letter 'S', resembling digit '5'.",
        ("8", "B"): "Closed straight spine: when loop closures of digit '8' align with a flat left boundary, it mimics the dual-loop geometry of uppercase 'B'.",
        ("B", "8"): "Curved left spine: imperfectly vertical back of letter 'B' resembles the continuous figure-8 loops of '8'.",
        ("2", "Z"): "Acute angle inflection: digit '2' written with a sharp bottom vertex instead of a smooth base curve resembles 'Z'.",
        ("Z", "2"): "Soft diagonal transition: letter 'Z' written with rounded transitions mimics the arched top of digit '2'.",
        ("O", "D"): "Vertical straightness: letter 'O' with an accidental flat left edge resembles uppercase 'D'.",
        ("U", "V"): "Vertex sharpness: handwritten 'U' with a pointed bottom curve mimics the acute angle of letter 'V'.",
        ("V", "U"): "Vertex rounding: handwritten 'V' with a softened bottom base resembles letter 'U'.",
        ("9", "Q"): "Ascender loop geometry: digit '9' loop with a low descender stroke mirrors uppercase 'Q' with a small tail.",
    }

    print("\n" + "=" * 65)
    print("DEEP DIVE: TOP 5 MISCLASSIFIED EXAMPLES")
    print("=" * 65)
    for i, err in enumerate(errors, 1):
        pair = (err["true_char"], err["pred_char"])
        explanation = typographical_explanations.get(
            pair,
            f"Handwriting stroke distortion and overlapping visual features between '{err['true_char']}' and '{err['pred_char']}'."
        )
        print(f"\nCase {i}: Sample Index #{err['sample_index']}")
        print(f"  Ground Truth: '{err['true_char']}' (Class #{err['true_idx']})")
        print(f"  Predicted:    '{err['pred_char']}' (Class #{err['pred_idx']})")
        print(f"  Confidence:   {err['pred_confidence'] * 100:.2f}% (Ground Truth Confidence: {err['true_confidence'] * 100:.2f}%)")
        print(f"  Root Cause Analysis: {explanation}")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()
