"""
Quick visual inference demonstration script.
Loads trained weights and runs character prediction on random or selected test images.
"""

import argparse
import os
import numpy as np
from src.dataset import CLASS_CHARS, IDX_TO_CHAR, load_and_preprocess_emnist
from src.model import NeuralNetwork


def main():
    parser = argparse.ArgumentParser(description="Optic-X Quick Inference Demo")
    parser.add_argument("--weights", type=str, default="artifacts/model_weights.npz", help="Weights path")
    parser.add_argument("--test_data", type=str, default="artifacts/test_data.npz", help="Test data path")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of random test predictions")
    args = parser.parse_args()

    if not os.path.exists(args.weights):
        print(f"Error: Weights file '{args.weights}' not found. Please run train.py first.")
        return

    # Load test samples
    if os.path.exists(args.test_data):
        data = np.load(args.test_data)
        X_test = data["X_test"]
        y_test = data["y_test"]
    else:
        print("Loading test data directly from CSV...")
        X_test, y_test = load_and_preprocess_emnist("DataSets/emnist-balanced-test.csv", samples_per_class=50)

    # Initialize model
    model = NeuralNetwork(layer_dims=[784, 256, 128, 35], activation="relu")
    model.load_weights(args.weights)

    # Pick random indices
    rng = np.random.default_rng()
    sample_indices = rng.choice(len(y_test), size=args.num_samples, replace=False)

    print("\n" + "=" * 60)
    print("OPTIC-X INFERENCE DEMONSTRATION")
    print("=" * 60)

    for i, idx in enumerate(sample_indices, 1):
        x = X_test[idx : idx + 1]
        true_label = IDX_TO_CHAR[int(y_test[idx])]

        probs = model.forward(x)[0]
        top3_indices = np.argsort(-probs)[:3]

        pred_label = IDX_TO_CHAR[top3_indices[0]]
        status = "CORRECT [OK]" if pred_label == true_label else "MISMATCH [X]"

        print(f"\nSample {i} (Index #{idx}): {status}")
        print(f"  Ground Truth: '{true_label}'")
        print(f"  Predictions (Top-3):")
        for rank, p_idx in enumerate(top3_indices, 1):
            print(f"    {rank}. '{IDX_TO_CHAR[p_idx]}' -> {probs[p_idx] * 100:.2f}%")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
