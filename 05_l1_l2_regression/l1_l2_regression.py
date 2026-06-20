import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge, Lasso

# Load data
df = pd.read_csv('admission_data.csv')
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)


class RidgeRegressionScratch:
    """
    Linear Regression with L2 (Ridge) penalty.
    Loss = MSE + λ Σw²
    Gradient = (-2/n) Xᵀ(y-ŷ) + λw
    """
    def __init__(self, lr=0.1, epochs=1000, lambda_=1):
        self.lr      = lr
        self.epochs  = epochs
        self.lambda_ = lambda_
        self.m       = None
        self.b       = 0

    def fit(self, X_train, y_train):
        n = len(X_train)
        y_train = np.array(y_train)
        self.m  = np.zeros(X_train.shape[1])

        for i in range(self.epochs):
            y_pred  = np.dot(X_train, self.m) + self.b
            error   = y_train - y_pred
            slope_m = (-2/n) * np.dot(X_train.T, error) + self.lambda_ * self.m
            slope_b = (-2/n) * np.sum(error)

            self.m = self.m - self.lr * slope_m
            self.b = self.b - self.lr * slope_b

    def predict(self, X_test):
        return np.dot(np.array(X_test), self.m) + self.b


class LassoRegressionScratch:
    """
    Linear Regression with L1 (Lasso) penalty.
    Loss = MSE + λ Σ|w|
    Gradient = (-2/n) Xᵀ(y-ŷ) + λ·sign(w)
    """
    def __init__(self, lr=0.01, epochs=1000, lambda_=0.01):
        self.lr      = lr
        self.epochs  = epochs
        self.lambda_ = lambda_
        self.m       = None
        self.b       = 0

    def fit(self, X_train, y_train):
        n = len(X_train)
        y_train = np.array(y_train)
        self.m  = np.zeros(X_train.shape[1])

        for i in range(self.epochs):
            y_pred  = np.dot(X_train, self.m) + self.b
            error   = y_train - y_pred
            slope_m = (-2/n) * np.dot(X_train.T, error) + self.lambda_ * np.sign(self.m)
            slope_b = (-2/n) * np.sum(error)

            self.m = self.m - self.lr * slope_m
            self.b = self.b - self.lr * slope_b

    def predict(self, X_test):
        return np.dot(np.array(X_test), self.m) + self.b


# Train Ridge
ridge = RidgeRegressionScratch(lr=0.1, epochs=1000, lambda_=1)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)
print(f'Ridge (Scratch) R² : {r2_score(y_test, y_pred_ridge):.4f}')

# Train Lasso
lasso = LassoRegressionScratch(lr=0.01, epochs=1000, lambda_=0.01)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)
print(f'Lasso (Scratch) R² : {r2_score(y_test, y_pred_lasso):.4f}')

# Sklearn comparison
sk_ridge = Ridge(alpha=1)
sk_ridge.fit(X_train, y_train)
print(f'Ridge (Sklearn) R² : {r2_score(y_test, sk_ridge.predict(X_test)):.4f}')

sk_lasso = Lasso(alpha=0.01)
sk_lasso.fit(X_train, y_train)
print(f'Lasso (Sklearn) R² : {r2_score(y_test, sk_lasso.predict(X_test)):.4f}')
