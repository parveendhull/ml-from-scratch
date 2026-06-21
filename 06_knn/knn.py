import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# Load data
data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)


class KNNScratch:
    """
    K-Nearest Neighbors Classifier from scratch.

    Algorithm:
    1. Compute distance from test point to all training points
    2. Find k nearest neighbors
    3. Predict majority class among those neighbors
    """
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y).flatten()

    def predict(self, X_test):
        X_test = np.array(X_test)
        predictions = []

        for x in X_test:
            # Euclidean distance to all training points
            distances = np.linalg.norm(self.X_train - x, axis=1)

            # Get indices of k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]
            k_labels  = self.y_train[k_indices]

            # Majority vote
            predicted_label = np.bincount(k_labels).argmax()
            predictions.append(predicted_label)

        return np.array(predictions)


# Train scratch model
knn_scratch = KNNScratch(k=3)
knn_scratch.fit(X_train, y_train)
y_pred_scratch = knn_scratch.predict(X_test)
acc_scratch = accuracy_score(y_test, y_pred_scratch)
print(f'Accuracy (Scratch) : {acc_scratch:.4f}')

# Sklearn comparison
knn_sklearn = KNeighborsClassifier(n_neighbors=3)
knn_sklearn.fit(X_train, y_train)
y_pred_sklearn = knn_sklearn.predict(X_test)
acc_sklearn = accuracy_score(y_test, y_pred_sklearn)
print(f'Accuracy (Sklearn) : {acc_sklearn:.4f}')