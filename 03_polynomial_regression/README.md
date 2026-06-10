# Polynomial Regression from Scratch

## What is it
Extension of Linear Regression — transforms features
into polynomial form before fitting.
[x] → [x, x², x³, ...]

## Math
Feature transformation:
X_poly = [x, x², x³, ..., xᵈ]

Normal Equation (same as Multiple LR):
β = (XᵀX)⁻¹ Xᵀy

## Results
| Degree | R² Score |
|--------|----------|
| 1      | 0.7444   |
| 2      | 0.7459   |
| 3      | 0.7474   |
| 5      | lower    |
| 10     | lower    |

## Key Insight — Overfitting
Low degree  → underfitting — misses pattern
Degree 2-3  → sweet spot — best generalization
High degree → overfitting — memorizes training data,
              fails on test data

## Why This Leads to Regularization
High degree polynomials overfit badly.
L1/L2 regularization controls this —
penalizes large coefficients to reduce overfitting.