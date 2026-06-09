import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

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


class BatchGD_LR:
    """
    Linear Regression using Batch Gradient Descent.
    Uses entire dataset to compute gradients each epoch.

    Gradients:
    dL/dm = (-2/n) Xᵀ(y - ŷ)
    dL/db = (-2/n) Σ(y - ŷ)
    """
    def __init__(self, lr=0.1, epochs=1000):
        self.lr      = lr
        self.epochs  = epochs
        self.m       = None
        self.b       = 0
        self.losses  = []

    def fit(self, X_train, y_train):
        n          = len(X_train)
        y_train    = np.array(y_train)
        self.m     = np.zeros(X_train.shape[1])

        for i in range(self.epochs):
            y_pred   = np.dot(X_train, self.m) + self.b
            error    = y_train - y_pred
            loss     = np.mean(error ** 2)
            self.losses.append(loss)

            slope_m  = (-2/n) * np.dot(X_train.T, error)
            slope_b  = (-2/n) * np.sum(error)

            self.m   = self.m - self.lr * slope_m
            self.b   = self.b - self.lr * slope_b

    def predict(self, X_test):
        return np.dot(np.array(X_test), self.m) + self.b


# Train
model = BatchGD_LR(lr=0.1, epochs=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f'Batch GD R²  : {r2_score(y_test, y_pred):.4f}')

# Sklearn comparison
sk = LinearRegression()
sk.fit(X_train, y_train)
y_pred_sk = sk.predict(X_test)
print(f'Sklearn R²   : {r2_score(y_test, y_pred_sk):.4f}')

# Loss curve
plt.plot(model.losses)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Batch GD — Loss Curve')
plt.tight_layout()
plt.show()