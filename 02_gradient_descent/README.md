# Gradient Descent — Linear Regression from Scratch

## What is Gradient Descent
Iterative optimization algorithm to minimize loss
by updating weights in direction of negative gradient.

## Batch Gradient Descent
Uses entire dataset to compute gradients each epoch.
Smooth convergence but slow for large datasets.

## Math
Prediction: ŷ = Xm + b

Loss (MSE): L = (1/n) Σ(y - ŷ)²

Gradients:
dL/dm = (-2/n) Xᵀ(y - ŷ)
dL/db = (-2/n) Σ(y - ŷ)

Weight update:
m = m - lr × dL/dm
b = b - lr × dL/db

## Results
| Model | R² Score |
|-------|----------|
| Batch GD Scratch (1000 epochs) | 0.7657 |
| Sklearn (Normal Equation)      | 0.7664 |

Difference of 0.0007 — GD is iterative,
Normal Equation is exact closed-form solution.

## Why Difference Exists
Normal Equation → exact solution in one step
Gradient Descent → approximates, needs enough epochs

## When to Use GD over Normal Equation
Normal Equation → O(n³) — slow for large datasets
Gradient Descent → scales well with large data

## Coming Next
- SGD (Stochastic Gradient Descent)
- Mini-batch Gradient Descent