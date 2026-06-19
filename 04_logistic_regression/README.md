# Logistic Regression from Scratch

## What is it
Binary classification algorithm using sigmoid
function to map predictions to probabilities (0 to 1).

## Math
Sigmoid: σ(z) = 1 / (1 + e^-z)

Loss (Binary Cross-Entropy):
L = -(1/m) Σ[y·log(ŷ) + (1-y)·log(1-ŷ)]

Gradient:
dL/dw = (1/m) Xᵀ(ŷ - y)

Weight update:
w = w - lr × dL/dw

## Dataset
Breast Cancer dataset (sklearn) — 569 samples, 30 features
Binary classification: malignant vs benign

## Results
| Model            | Accuracy |
|-------------------|----------|
| Scratch (2500 epochs) | 0.9649 |
| Sklearn               | 0.9649 |

Scratch matches Sklearn exactly after sufficient epochs ✓

## Key Insight
At 1000 epochs, scratch model underperformed (0.9386)
because gradient descent hadn't fully converged.
Increasing to 2500 epochs closed the gap completely —
proving the gradient derivation was correct,
just needed more iterations than sklearn's
optimized solver (LBFGS).

## Why Feature Scaling Matters
StandardScaler applied before training — without it,
gradient descent converges very slowly or not at all
due to features being on different scales
(e.g. mean radius vs mean smoothness).