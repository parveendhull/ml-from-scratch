# K-Nearest Neighbors (KNN) from Scratch

## What is it
Instance-based, non-parametric classification algorithm.
No training phase — classifies a new point based on
the majority class among its k nearest neighbors
in the training data.

## Algorithm
1. For a test point, compute distance to every training point
2. Sort distances, pick k closest points
3. Take majority class among those k neighbors as prediction

## Math — Euclidean Distance
d(x, y) = √Σ(xi - yi)²

Implemented using np.linalg.norm for vectorized speed.

## Dataset
Iris dataset — 150 samples, 4 features, 3 classes
StandardScaler applied — KNN is distance-based,
so feature scaling is essential (unscaled features
with larger ranges would dominate the distance calculation).

## Results
| Model | Accuracy |
|-------|----------|
| Scratch (k=3) | 0.9667 |
| Sklearn (k=3) | 0.9667 |

Exact match — confirms correct implementation.

## Key Insight
KNN has no "training" — it simply stores the data.
All computation happens at prediction time, which
makes it slow for large datasets (must compare
against every training point for each prediction).

## Why Feature Scaling Matters
Without scaling, a feature with a large numeric range
(e.g. petal length in cm) would dominate the distance
calculation over a feature with a smaller range,
even if both are equally important.