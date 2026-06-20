# L1 (Lasso) and L2 (Ridge) Regression from Scratch

## What is it
Linear Regression with regularization penalty
added to prevent overfitting by controlling
weight magnitude.

## Ridge Regression (L2)
Loss = MSE + λ Σw²
Gradient = (-2/n) Xᵀ(y-ŷ) + λw

Shrinks weights toward zero but never exactly zero.

## Lasso Regression (L1)
Loss = MSE + λ Σ|w|
Gradient = (-2/n) Xᵀ(y-ŷ) + λ·sign(w)

Can shrink weights to exactly zero —
performs automatic feature selection.

## Results
| Model | R² Score |
|-------|----------|
| Ridge (Scratch, λ=1)    | 0.7578 |
| Lasso (Scratch, λ=0.01) | 0.7717 |

## Key Debugging Lesson
Initial Lasso implementation reset weights to
zero inside the training loop every epoch —
this prevented any learning (weights never
accumulated updates). Removing that line fixed
convergence completely.

## Key Insight — Lambda Sensitivity
Lasso is far more sensitive to learning rate
and lambda than Ridge:
- lr=0.1 with λ=1 caused divergence (R² = -15.1)
- lr=0.01 with λ=1 still too strong (R² = 0.12)
- lr=0.01 with λ=0.01 gave best result (R² = 0.77)

This demonstrates why hyperparameter tuning
matters more for L1 — the non-smooth |w| term
makes optimization landscape harder than L2's
smooth w² term.

## When to Use Which
Ridge → many small/medium important features
Lasso → suspect only a few features matter,
        want automatic feature selection