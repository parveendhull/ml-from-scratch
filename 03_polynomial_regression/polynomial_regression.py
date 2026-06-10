import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# Load data
df = pd.read_csv('admission_data.csv')

X = df[['CGPA']]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)


class PolynomialRegression:
    """
    Polynomial Regression from scratch using Normal Equation.
    Transforms features: [x] → [x, x², x³, ...]
    Then applies: β = (XᵀX)⁻¹ Xᵀy
    """
    def __init__(self, degree=2):
        self.degree      = degree
        self.coef_       = None
        self.intercept_  = None

    def _transform(self, X):
        return np.column_stack([X ** i for i in range(1, self.degree + 1)])

    def fit(self, X_train, y_train):
        X_poly  = self._transform(X_train)
        X_poly  = np.insert(X_poly, 0, 1, axis=1)
        betas   = np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T @ y_train
        self.intercept_ = betas[0]
        self.coef_      = betas[1:]

    def predict(self, X_test):
        X_poly = self._transform(X_test)
        return X_poly @ self.coef_ + self.intercept_


# Compare degrees
results = {}
degrees = [1, 2, 3, 5, 10]

for d in degrees:
    model = PolynomialRegression(degree=d)
    model.fit(X_train.values, y_train.values)
    y_pred = model.predict(X_test.values)
    results[d] = r2_score(y_test, y_pred)
    print(f'Degree {d:2d} — Scratch R²: {results[d]:.4f}')

# Sklearn comparison
print('\n--- Sklearn Comparison ---')
for d in degrees:
    sk = Pipeline([
        ('poly', PolynomialFeatures(degree=d)),
        ('lr',   LinearRegression())
    ])
    sk.fit(X_train, y_train)
    y_pred_sk = sk.predict(X_test)
    print(f'Degree {d:2d} — Sklearn R²: {r2_score(y_test, y_pred_sk):.4f}')

# Plot R² vs Degree
plt.figure(figsize=(8, 4))
plt.plot(degrees, list(results.values()), marker='o', color='blue')
plt.xlabel('Polynomial Degree')
plt.ylabel('R² Score')
plt.title('Polynomial Regression — R² vs Degree')
plt.tight_layout()
plt.show()