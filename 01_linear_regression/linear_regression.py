import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv('admission_data.csv')

# ── Single Linear Regression ──────────────────────────
class SingleLR:
    """
    Simple Linear Regression using OLS formula.
    m = Σ(xi - x̄)(yi - ȳ) / Σ(xi - x̄)²
    b = ȳ - m·x̄
    """
    def __init__(self):
        self.m = None
        self.b = None

    def fit(self, X_train, y_train):
        X = X_train.values.flatten()
        y = y_train.values

        # Vectorized OLS
        self.m = np.sum((X - X.mean()) * (y - y.mean())) / \
                 np.sum((X - X.mean()) ** 2)
        self.b = y.mean() - (self.m * X.mean())

    def predict(self, X_test):
        return self.m * X_test.values.flatten() + self.b


# ── Multiple Linear Regression ────────────────────────
class MultipleLR:
    """
    Multiple Linear Regression using Normal Equation.
    β = (XᵀX)⁻¹ Xᵀy
    """
    def __init__(self):
        self.coef_      = None
        self.intercept_ = None

    def fit(self, X_train, y_train):
        X = np.insert(X_train, 0, 1, axis=1)  # add bias column
        betas           = np.linalg.inv(X.T @ X) @ X.T @ y_train
        self.intercept_ = betas[0]
        self.coef_      = betas[1:]

    def predict(self, X_test):
        return X_test @ self.coef_ + self.intercept_


# ── Data Preparation ──────────────────────────────────

# Single LR — 1 feature (CGPA)
X_single = df[['CGPA']]
y        = df.iloc[:, -1]
X_train_s, X_test_s, y_train, y_test = train_test_split(
    X_single, y, test_size=0.2, random_state=0
)

# Multiple LR — all features
X_multi = df.iloc[:, :-1]
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multi, y, test_size=0.2, random_state=0
)

# ── Train and Evaluate ────────────────────────────────

# Single LR — Scratch
single_model = SingleLR()
single_model.fit(X_train_s, y_train)
y_pred_single = single_model.predict(X_test_s)

# Single LR — Sklearn
sk_single = LinearRegression()
sk_single.fit(X_train_s, y_train)
y_pred_sk_single = sk_single.predict(X_test_s)

# Multiple LR — Scratch
multi_model = MultipleLR()
multi_model.fit(X_train_m.values, y_train_m.values)
y_pred_multi = multi_model.predict(X_test_m.values)

# Multiple LR — Sklearn
sk_multi = LinearRegression()
sk_multi.fit(X_train_m, y_train_m)
y_pred_sk_multi = sk_multi.predict(X_test_m)

# ── Results ───────────────────────────────────────────
print("=" * 45)
print("Single Linear Regression (CGPA only)")
print(f"  Scratch R²  : {r2_score(y_test, y_pred_single):.4f}")
print(f"  Sklearn R²  : {r2_score(y_test, y_pred_sk_single):.4f}")
print(f"  m = {single_model.m[0]:.6f}, b = {single_model.b[0]:.6f}")

print("=" * 45)
print("Multiple Linear Regression (all features)")
print(f"  Scratch R²  : {r2_score(y_test_m, y_pred_multi):.4f}")
print(f"  Sklearn R²  : {r2_score(y_test_m, y_pred_sk_multi):.4f}")
print(f"  Coefficients: {multi_model.coef_}")
print(f"  Intercept   : {multi_model.intercept_:.6f}")
print("=" * 45)