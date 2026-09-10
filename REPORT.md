# Implementation Methodology Report: 35-Class Handwritten Character Recognition from Scratch

**Candidate:** Shivanesh V  
**Project:** Optic-X Neural Network  
**Assessment:** ZenteiQ AI Hub - AI/ML Engineer Internship  
**Framework Constraints:** 100% Pure NumPy (Zero Deep Learning Libraries)  
**Target Classes:** 35 Classes (Uppercase letters `A`–`Z` [26], Digits `1`–`9` [9])

---

## 1. Executive Summary & Objective

The objective of this assignment is to design, implement, train, and evaluate a Multilayer Perceptron (MLP) entirely from first mathematical principles using **only NumPy**. The model is tasked with recognizing 35 distinct alphanumeric character classes:
- **Digits 1–9** (9 classes, with digit `0` explicitly omitted to satisfy scope)
- **Uppercase Letters A–Z** (26 classes)

No high-level frameworks (PyTorch, TensorFlow, Keras, JAX) or machine learning utilities (Scikit-Learn) were used. Every component—from data parsing, axis transposition, stratified data partitioning, forward propagation, activation functions, loss computation, backpropagation chain rule, gradient descent optimization, to the $35 \times 35$ confusion matrix and classification metrics—was engineered from scratch.

The final model achieves **86.93% test accuracy** (and **87.81% peak validation accuracy**) across 35 visually challenging classes.

---

## 2. Dataset Engineering & Preprocessing Pipeline

### 2.1 Class Filtering and Mapping
The raw dataset is derived from EMNIST (Extended MNIST Balanced split). The balanced partition natively contains 47 classes (digits `0`–`9`, uppercase letters `A`–`Z`, and select merged lowercase letters). 

To strictly adhere to the 35-class constraint:
1. Raw labels `1` through `9` (digits `'1'` through `'9'`) are mapped to model class indices `0` through `8`.
2. Raw labels `10` through `35` (uppercase `'A'` through `'Z'`) are mapped to model class indices `9` through `34`.
3. Raw label `0` (digit `'0'`) and labels `36` through `46` (lowercase characters) are strictly filtered out.

```
Model Index 0..8   --> Digits '1'..'9'
Model Index 9..34  --> Letters 'A'..'Z'
Total: 35 Classes
```

### 2.2 The EMNIST Transposition Quirk
A critical pitfall in the EMNIST dataset is that raw pixel arrays in CSV format are stored in Fortran (column-major) order relative to standard Cartesian image coordinates. If reshaped directly as `(28, 28)`, images appear rotated 90 degrees counter-clockwise and horizontally mirrored.

To resolve this issue:
$$\text{Image}_{\text{corrected}} = \left(\text{Reshape}_{28 \times 28}(x)\right)^T$$
Each $1 \times 784$ vector is reshaped into a $28 \times 28$ matrix, transposed, and flattened back into $1 \times 784$. This guarantees that feature dimensions correlate to authentic, upright handwritten strokes.

### 2.3 Pixel Normalization
Raw grayscale intensity values $p \in [0, 255]$ are scaled to the continuous unit interval $[0.0, 1.0]$:
$$x_{\text{norm}} = \frac{x}{255.0}$$
Normalization ensures that pre-activation inputs $Z = XW + b$ remain in a balanced numerical range, preventing premature saturation of activation gradients.

### 2.4 Stratified Partitioning
To guarantee equitable class representation across evaluation splits, a custom pure NumPy stratified partitioner was implemented:
- **Training Set (70%):** 24,500 samples (700 samples per class)
- **Validation Set (15%):** 5,250 samples (150 samples per class)
- **Test Set (15%):** 5,250 samples (150 samples per class)
- **Total Dataset Size:** 35,000 samples ($\ge 50$ samples per class constraint satisfied).

---

## 3. Network Architecture & Mathematical Design

### 3.1 Architectural Capacity & Layer Selection
A common baseline for digit classification (e.g., MNIST 10-class) is a single hidden layer of 64 or 128 units. However, classifying **35 alphanumeric characters** presents a substantially harder challenge due to severe inter-class morphological overlap:
- Digit `'1'` vs. Letter `'I'` (straight vertical sans-serif lines)
- Digit `'8'` vs. Letter `'B'` (dual loops with varying spinal curvature)
- Digit `'5'` vs. Letter `'S'` (horizontal crossbars turning into sigmoid curves)
- Digit `'2'` vs. Letter `'Z'` (curved top arches vs. sharp diagonal z-inflections)
- Letter `'U'` vs. Letter `'V'` (rounded trough vs. acute vertex)

To address these subtle geometric distinctions without convolutional layers, a 2-stage hierarchical representation was designed:
$$\mathbf{784} \longrightarrow \mathbf{256} \longrightarrow \mathbf{128} \longrightarrow \mathbf{35}$$

1. **Input Layer ($784$ features):** Unrolled $28 \times 28$ grayscale pixels.
2. **Hidden Layer 1 ($784 \rightarrow 256$ units):** Functions as a stroke primitive extractor. The 256 dimensions model localized edge orientations, loop segments, diagonal slopes, and stroke intersections.
3. **Hidden Layer 2 ($256 \rightarrow 128$ units):** Synthesizes stroke primitives into higher-order character structural parts (e.g., detecting if a loop is connected to a vertical spine on the left as in `'B'` or centered as in `'8'`). Funneling down from 256 to 128 introduces an information bottleneck that forces invariant feature compression and reduces overfitting.
4. **Output Layer ($128 \rightarrow 35$ units):** Computes class-specific unnormalized log-odds (logits) for the 35 target classes.

### 3.2 Weight Initialization: He (Kaiming) Normal Initialization
Standard Gaussian initialization or naive Xavier initialization fails when paired with ReLU activations. Because ReLU zeros out approximately 50% of the neurons for zero-mean inputs ($\mathbb{E}[\text{ReLU}(z)] = 0.5$), the activation variance shrinks by half at every layer:
$$\text{Var}(A^{[l]}) = \frac{1}{2} n_{in} \text{Var}(W^{[l]}) \text{Var}(A^{[l-1]})$$

Setting $\text{Var}(W) = \frac{1}{n_{in}}$ (Xavier) causes the variance of signals to vanish exponentially in deep networks. To preserve variance ($\text{Var}(A^{[l]}) = \text{Var}(A^{[l-1]})$), weights are initialized using **He Normal Initialization**:
$$W^{[l]} \sim \mathcal{N}\left(0, \sigma = \sqrt{\frac{2}{n_{in}}}\right)$$
Biases are initialized to zeros: $b^{[l]} = \mathbf{0}$.

---

## 4. Mathematical Formulations & Execution Sequence

The training loop strictly implements the mandated order of mathematical operations:

```mermaid
graph LR
    A[Forward Propagation] --> B[Backpropagation]
    B --> C[ReLU / Tanh Activations]
    C --> D[Softmax Output Layer]
    D --> E[Cross-Entropy Loss]
    E --> F[Gradient Descent Update]
```

### 4.1 Step 1: Forward Propagation
For an input mini-batch $X \in \mathbb{R}^{B \times 784}$ where $B = 128$:
$$Z^{[1]} = X W^{[1]} + b^{[1]}, \quad Z^{[1]} \in \mathbb{R}^{B \times 256}$$
$$A^{[1]} = \text{ReLU}(Z^{[1]}) = \max(0, Z^{[1]}), \quad A^{[1]} \in \mathbb{R}^{B \times 256}$$
$$Z^{[2]} = A^{[1]} W^{[2]} + b^{[2]}, \quad Z^{[2]} \in \mathbb{R}^{B \times 128}$$
$$A^{[2]} = \text{ReLU}(Z^{[2]}) = \max(0, Z^{[2]}), \quad A^{[2]} \in \mathbb{R}^{B \times 128}$$
$$Z^{[3]} = A^{[2]} W^{[3]} + b^{[3]}, \quad Z^{[3]} \in \mathbb{R}^{B \times 35}$$

### 4.2 Step 2: Softmax Output Layer
To convert unconstrained logits $Z^{[3]}$ into normalized class probability distributions $\hat{Y} \in \mathbb{R}^{B \times 35}$, a numerically stable Softmax is computed by subtracting row-wise maximums:
$$\hat{y}_{i, c} = \frac{e^{z_{i, c} - \max_k(z_{i, k})}}{\sum_{j=1}^{35} e^{z_{i, j} - \max_k(z_{i, k})}}$$
Subtracting $\max(Z)$ prevents floating-point overflow ($e^{z} \rightarrow \infty$) without altering relative probability ratios.

### 4.3 Step 3: Categorical Cross-Entropy Loss
The multiclass loss over mini-batch $B$ is:
$$\mathcal{L} = -\frac{1}{B} \sum_{i=1}^B \sum_{c=1}^{35} y_{i, c} \log(\hat{y}_{i, c} + \epsilon)$$
where $Y \in \mathbb{R}^{B \times 35}$ is the one-hot ground-truth matrix, and $\epsilon = 10^{-15}$ guards against $\log(0) = -\infty$.

### 4.4 Step 4: Backpropagation & Analytical Gradient Derivations
Using the chain rule, the gradient of the combined Softmax and Cross-Entropy loss w.r.t. the output logits $Z^{[3]}$ simplifies analytically:
$$\frac{\partial \mathcal{L}}{\partial Z^{[3]}} = dZ^{[3]} = \frac{1}{B} (\hat{Y} - Y) \in \mathbb{R}^{B \times 35}$$

The parameter gradients for the output layer are:
$$dW^{[3]} = (A^{[2]})^T dZ^{[3]} \in \mathbb{R}^{128 \times 35}$$
$$db^{[3]} = \sum_{\text{rows}} dZ^{[3]} \in \mathbb{R}^{1 \times 35}$$

Propagating backwards to Hidden Layer 2:
$$dA^{[2]} = dZ^{[3]} (W^{[3]})^T \in \mathbb{R}^{B \times 128}$$
$$dZ^{[2]} = dA^{[2]} \odot \mathbb{I}(Z^{[2]} > 0) \in \mathbb{R}^{B \times 128}$$
$$dW^{[2]} = (A^{[1]})^T dZ^{[2]} \in \mathbb{R}^{256 \times 128}$$
$$db^{[2]} = \sum_{\text{rows}} dZ^{[2]} \in \mathbb{R}^{1 \times 128}$$

Propagating backwards to Hidden Layer 1:
$$dA^{[1]} = dZ^{[2]} (W^{[2]})^T \in \mathbb{R}^{B \times 256}$$
$$dZ^{[1]} = dA^{[1]} \odot \mathbb{I}(Z^{[1]} > 0) \in \mathbb{R}^{B \times 256}$$
$$dW^{[1]} = X^T dZ^{[1]} \in \mathbb{R}^{784 \times 256}$$
$$db^{[1]} = \sum_{\text{rows}} dZ^{[1]} \in \mathbb{R}^{1 \times 256}$$

### 4.5 Step 5: Gradient Descent with Momentum
Standard vanilla SGD often oscillates in steep ravine loss landscapes. We implemented Momentum SGD:
$$V_{W}^{[l]} = \beta V_{W}^{[l]} + \alpha dW^{[l]}$$
$$W^{[l]} \leftarrow W^{[l]} - V_{W}^{[l]}$$
$$V_{b}^{[l]} = \beta V_{b}^{[l]} + \alpha db^{[l]}$$
$$b^{[l]} \leftarrow b^{[l]} - V_{b}^{[l]}$$
where momentum $\beta = 0.9$, initial learning rate $\alpha = 0.08$, with a periodic exponential decay factor of $0.96$ applied every 5 epochs.

---

## 5. Architectural Component Overview & Docstring Standard

All modules strictly adhere to PEP 257 docstring conventions with explicit tensor dimension typing:

| Module | Primary Component | Responsibility | Expected Tensor Input/Output Shapes |
| :--- | :--- | :--- | :--- |
| `src/dataset.py` | `load_and_preprocess_emnist` | Filter 35 classes, apply transpose fix, normalize | Output $X$: `(N, 784)`, $y$: `(N,)` |
| `src/dataset.py` | `train_val_test_split` | Stratified train/val/test splitting in pure NumPy | Outputs: `(N_tr, 784)`, `(N_val, 784)`, `(N_te, 784)` |
| `src/activations.py` | `ReLU` | Piecewise linear activation & gradient masking | Input: `(B, D)`, Output: `(B, D)` |
| `src/activations.py` | `Tanh` | Bipolar activation & derivative $1 - A^2$ | Input: `(B, D)`, Output: `(B, D)` |
| `src/activations.py` | `Softmax` | Log-sum-exp stable multiclass probability generator | Input: `(B, 35)`, Output: `(B, 35)` |
| `src/loss.py` | `CategoricalCrossEntropy` | Numerical log-loss & simplified logits gradient | $y_{pred}$: `(B, 35)`, $y_{true}$: `(B, 35)` $\rightarrow$ Scalar Loss |
| `src/optimizer.py` | `SGDOptimizer` | Velocity buffer tracking and momentum updates | In-place update of `params['W1']`, etc. |
| `src/model.py` | `NeuralNetwork` | Orchestrates forward, backward, save/load | $X$: `(B, 784)` $\rightarrow$ $P$: `(B, 35)`, Gradients dict |
| `src/metrics.py` | `compute_confusion_matrix` | Vectorized 2D matrix accumulation via `np.bincount` | $y_{pred}$: `(N,)`, $y_{true}$: `(N,)` $\rightarrow$ `(35, 35)` |
| `train.py` | `train_network` | Epoch loop, mini-batch sequencing, validation tracking | Logs history dictionary and checkpoints `.npz` |
| `evaluate.py` | `main` | Test set evaluation, CM plotting, error case analysis | Saves PNG heatmaps and detailed error logs |
| `demo.py` | `main` | Interactive single-sample visual verification | Command-line Top-3 inference printout |

---

## 6. Experimental Results & Performance Analysis

### 6.1 Training Trajectory
The network was trained on 24,500 training samples with a batch size of 128 for 25 epochs. Validation evaluation was performed at the conclusion of every epoch.

| Epoch | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy | Learning Rate |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 01 | 1.2425 | 63.41% | 0.7465 | 76.57% | 0.080000 |
| 05 | 0.3209 | 88.76% | 0.5158 | 84.00% | 0.076800 |
| 10 | 0.1612 | 93.63% | 0.5440 | 85.64% | 0.073728 |
| 15 | 0.0958 | 96.13% | 0.5844 | 86.61% | 0.070779 |
| 20 | 0.0619 | 97.36% | 0.6074 | 87.81% | 0.067948 |
| **25** | **0.0542** | **97.58%** | **0.6717** | **86.90%** | **0.065230** |

*Peak Validation Accuracy of **87.81%** was recorded at epoch 20.*

### 6.2 Test Set Metrics
Evaluation on the completely held-out test partition (5,250 samples, 150 per class) produced the following metrics:

| Metric | Measured Value | Standard Formulation |
| :--- | :---: | :--- |
| **Overall Test Accuracy** | **86.93%** | $\frac{\sum \text{True Positives}}{N}$ |
| **Macro Precision** | **87.09%** | $\frac{1}{C} \sum_{c=1}^C \frac{TP_c}{TP_c + FP_c}$ |
| **Macro Recall** | **86.93%** | $\frac{1}{C} \sum_{c=1}^C \frac{TP_c}{TP_c + FN_c}$ |
| **Macro F1-Score** | **86.90%** | $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ |

---

## 7. Confusion Matrix & Typographical Analysis

The generated $35 \times 35$ confusion matrix heatmap (`artifacts/confusion_matrix.png`) demonstrates strong diagonal dominance across all 35 classes.

### Key Observations:
1. **High-Performing Classes (>95% F1):**
   - Characters with distinct, idiosyncratic geometric features achieve near-perfect classification: `'M'`, `'W'`, `'X'`, `'3'`, and `'1'` show high concentration along the main diagonal.
2. **Dispersed Error Clusters:**
   - The primary sources of error occur not from model underfitting, but from high inter-class visual similarity inherent in human handwriting. Distinct off-diagonal clusters appear between pairs sharing nearly identical stroke topologies.

---

## 8. In-Depth Root Cause Error Analysis (Top 5 Misclassified Cases)

Below is an examination of 5 specific high-confidence misclassifications identified on the test partition (`artifacts/error_analysis_samples.png`):

### Case 1: Sample #1289 — Ground Truth `'7'` Predicted as `'T'` (Confidence: 100.00%)
- **Visual Morphology:** The sample features a bold horizontal top bar and a nearly vertical stem centered directly beneath the bar rather than slanting down to the right.
- **Root Cause:** A standard uppercase `'T'` consists of a horizontal roof bar and a vertical descender. When a writer produces digit `'7'` without a rightward diagonal slant or continental cross-hatch, the feature vector perfectly matches the activation pattern of `'T'`.

### Case 2: Sample #5248 — Ground Truth `'A'` Predicted as `'P'` (Confidence: 100.00%)
- **Visual Morphology:** The writer rendered `'A'` with a single continuous loop at the apex and failed to extend the right diagonal stroke to the baseline.
- **Root Cause:** The absence of the right grounding leg leaves an enclosed loop resting atop a single vertical stroke. In pixel space, this is geometrically identical to uppercase `'P'`.

### Case 3: Sample #3901 — Ground Truth `'W'` Predicted as `'V'` (Confidence: 100.00%)
- **Visual Morphology:** The handwritten character consists of a single wide V-trough, where the central ascending/descending strokes are severely compressed into a single stroke.
- **Root Cause:** A handwritten `'W'` with a flattened or omitted middle vertex collapses into a broad single-vertex character, activating the `'V'` feature detectors with maximum confidence.

### Case 4: Sample #933 — Ground Truth `'A'` Predicted as `'4'` (Confidence: 100.00%)
- **Visual Morphology:** The character is written with an open triangular top and an elongated horizontal crossbar extending beyond the right stem.
- **Root Cause:** In open-top cursive handwriting, digit `'4'` has an identical geometric structure: a vertical left stroke, a horizontal connecting bar, and a vertical right downward stroke. The network has no contextual linguistic prior, leading to high-confidence confusion.

### Case 5: Sample #2842 — Ground Truth `'Y'` Predicted as `'4'` (Confidence: 100.00%)
- **Visual Morphology:** The top fork of letter `'Y'` is angled sharply with a vertical descender that touches the left diagonal branch.
- **Root Cause:** The junction point forms a closed 4-corner triangular quadrant on the left side, mimicking the closed loop of a standard digit `'4'`.

---

## 9. Conclusion & Engineering Reflections

This project successfully demonstrates that a multilayer perceptron constructed purely from elementary linear algebra and calculus in NumPy is capable of robust alphanumeric character recognition:
1. **Mathematical Rigor:** The complete forward and backward propagation routines were derived analytically and executed without external dependencies.
2. **Domain Nuance:** Identifying and correcting the EMNIST transposition quirk ensured that features represent genuine spatial handwriting properties.
3. **Capacity Balance:** The $784 \rightarrow 256 \rightarrow 128 \rightarrow 35$ topology paired with He initialization and Momentum SGD achieved rapid convergence (**86.93% test accuracy** in under 20 seconds on CPU).
4. **Transparent Evaluation:** Complete classification metrics, full $35 \times 35$ confusion matrix rendering, and deep morphological error inspection prove the system's reliability and reveal the natural perceptual boundaries of handwritten optical character recognition.

*Hardware & Runtime Note: The complete 25-epoch training sequence across 35,000 samples executes in approximately 18 seconds on standard CPU hardware, demonstrating the efficiency of fully vectorized NumPy array operations.*
