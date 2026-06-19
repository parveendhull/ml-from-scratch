import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# Load data
data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)


class LogisticRegressionScratch:
    """
    Logistic Regression from scratch using Gradient Descent.

    Sigmoid: σ(z) = 1 / (1 + e^-z)
    Loss (BCE): L = -(1/m) Σ[y·log(ŷ) + (1-y)·log(1-ŷ)]
    Gradient: dL/dw = (1/m) Xᵀ(ŷ - y)
    """
    def __init__(self, epochs=2500, lr=0.01):
        self.epochs    = epochs
        self.lr        = lr
        self.coef_     = None
        self.intercept_= None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)  # bias column
        weights = np.ones(X.shape[1])
        m = X.shape[0]

        for i in range(self.epochs):
            z      = np.dot(X, weights)
            y_pred = self.sigmoid(z)
            error  = y_pred - y
            dw     = (1/m) * np.dot(X.T, error)
            weights = weights - self.lr * dw

        self.intercept_ = weights[0]
        self.coef_      = weights[1:]

    def predict_prob(self, X):
        z = np.dot(X, self.coef_.T) + self.intercept_
        return self.sigmoid(z)

    def predict(self, X):
        prob = self.predict_prob(X)
        return np.where(prob >= 0.5, 1, 0)


# Train scratch model
model = LogisticRegressionScratch(epochs=2500, lr=0.01)
model.fit(X_train, y_train)
y_pred_scratch = model.predict(X_test)
acc_scratch = accuracy_score(y_test, y_pred_scratch)
print(f'Accuracy (Scratch)  : {acc_scratch:.4f}')

# Sklearn comparison
sk_model = LogisticRegression()
sk_model.fit(X_train, y_train)
y_pred_sklearn = sk_model.predict(X_test)
acc_sklearn = accuracy_score(y_test, y_pred_sklearn)
print(f'Accuracy (Sklearn)  : {acc_sklearn:.4f}')