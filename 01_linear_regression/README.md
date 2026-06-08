# Linear Regression from Scratch

## Implementations
1. Single Linear Regression — OLS formula
2. Multiple Linear Regression — Normal Equation
3. Comparison with scikit-learn

## Math

### Single LR — OLS
m = Σ(xi - x̄)(yi - ȳ) / Σ(xi - x̄)²
b = ȳ - m·x̄

### Multiple LR — Normal Equation
β = (XᵀX)⁻¹ Xᵀy

Bias term added by inserting column of 1s:
X → [1 | X]

## Results
| Model | Features | Scratch R² | Sklearn R² |
|-------|----------|------------|------------|
| Single LR | CGPA only | 0.7444 | 0.7444 |
| Multiple LR | All 7 | 0.7664 | 0.7664 |

Scratch matches Sklearn exactly ✓

## Key Insight
Normal equation gives exact closed-form solution.
No iterations needed — direct matrix computation.
Limitation: slow for large datasets (matrix inverse is O(n³))
For large data → Gradient Descent is preferred.